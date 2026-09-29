import numpy as np
import xarray as xr


def lag_corr(x,y,max_lag):
    """Cross-correlation of ``x`` leading ``y`` at lags -max_lag..max_lag.

    Parameters
    ----------
    x, y : numpy.ndarray
        Input time series (same length, same sampling interval).
    max_lag : int
        Maximum lag, in the same units as the series' sampling interval.

    Returns
    -------
    lags : numpy.ndarray
        Lag values, -max_lag..max_lag.
    corr : numpy.ndarray
        Cross-correlation at each lag. Positive lag means ``x`` leads ``y``:
        ``corr[lag] = corr(x(t), y(t+lag))``, so a peak at lag>0 means x's value at t
        best matches y ``lag`` steps later (x happened first).
    """
    lags = np.arange(-max_lag,max_lag+1)
    xc = (x-x.mean())/x.std()
    yc = (y-y.mean())/y.std()
    corr = np.array([
        np.mean(xc[:len(xc)-lag]*yc[lag:]) if lag>0
        else (np.mean(xc[-lag:]*yc[:len(yc)+lag]) if lag<0 else np.mean(xc*yc))
        for lag in lags
    ])
    return lags,corr


def xr_lag_corr_1d(x,y,max_lag,dim="time"):
    """``lag_corr`` for a 1-D ``y`` (e.g. index vs. index), looping over lag in Python.

    ``y.shift({dim: -lag})`` brings y(t+lag) to index t, so ``xc * y.shift({dim: -lag})``
    matches ``lag_corr``'s ``corr[lag] = corr(x(t), y(t+lag))`` convention (positive lag:
    x leads y). ``mean(dim=dim)`` skips the NaNs the shift introduces at the edges,
    matching the overlap-only averaging (dividing by n - |lag|, not by n) of ``lag_corr``.
    This scales with the size of ``y``, so for a full lat/lon field use ``xr_lag_corr``.

    Parameters
    ----------
    x, y : xarray.DataArray
        Input time series, 1-D along ``dim``.
    max_lag : int
        Maximum lag, in the same units as ``dim``'s sampling interval.
    dim : str, optional
        Dimension along which the correlation is calculated.

    Returns
    -------
    xarray.DataArray
        Cross-correlation with dimension ``lag`` (-max_lag..max_lag).
    """
    lags = np.arange(-max_lag,max_lag+1)
    xc = (x-x.mean(dim=dim))/x.std(dim=dim)
    yc = (y-y.mean(dim=dim))/y.std(dim=dim)
    corrs = [(xc*yc.shift({dim:-int(lag)})).mean(dim=dim) for lag in lags]
    result = xr.concat(corrs,dim="lag")
    return result.assign_coords(lag=lags)


def _lag_corr_values(x,y,max_lag):
    _,corr = lag_corr(x,y,max_lag)
    return corr


def xr_lag_corr(x,y,max_lag,dim="time"):
    """``lag_corr`` applied along ``dim`` via ``apply_ufunc``, broadcasting over any
    other dimensions (e.g. a 1-D index ``x`` against a full ``(time, lat, lon)`` field
    ``y``).

    ``lag_corr`` slices ``x``/``y`` directly (like ``.isel``) rather than padding with
    NaNs, so no ``skipna`` handling is needed; each broadcast point (e.g. grid cell) is
    computed by its own ``apply_ufunc`` call, which is faster than looping over lag and
    vectorizing over space because each call's working set fits in cache.

    Parameters
    ----------
    x, y : xarray.DataArray
        Input time series along ``dim``; either may carry extra dimensions
        (e.g. ``lat``, ``lon``) that are broadcast over.
    max_lag : int
        Maximum lag, in the same units as ``dim``'s sampling interval.
    dim : str, optional
        Dimension along which the correlation is calculated.

    Returns
    -------
    xarray.DataArray
        Cross-correlation with dimension ``lag`` (-max_lag..max_lag), broadcasting over
        any other dimensions of ``x``/``y``.
    """
    result = xr.apply_ufunc(
        _lag_corr_values,
        x,
        y,
        input_core_dims=[[dim],[dim]],
        output_core_dims=[["lag"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes":{"lag":2*max_lag+1}},
        kwargs={"max_lag":max_lag},
    )
    return result.assign_coords(lag=np.arange(-max_lag,max_lag+1))


def auto_corr(x,max_lag):
    """Sample autocorrelation of ``x`` at lags 0..max_lag (biased/N-normalized estimator)."""
    x = np.asarray(x,dtype="float64")
    x = x-x.mean()
    n = x.size
    var = np.dot(x,x)/n
    return np.array([np.dot(x[:n-lag],x[lag:])/n/var for lag in range(max_lag+1)])


def xr_auto_corr(x,max_lag,dim="time"):
    """``auto_corr`` applied along ``dim``, broadcasting over any other dimensions.

    Parameters
    ----------
    x : xarray.DataArray
        Input time series along ``dim``.
    max_lag : int
        Maximum lag, in the same units as ``dim``'s sampling interval.
    dim : str, optional
        Dimension along which the autocorrelation is calculated.

    Returns
    -------
    xarray.DataArray
        Autocorrelation with dimension ``lag`` (0..max_lag).
    """
    result = xr.apply_ufunc(
        auto_corr,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[["lag"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes":{"lag":max_lag+1}},
        kwargs={"max_lag":max_lag},
    )
    return result.assign_coords(lag=np.arange(max_lag+1))


def eff_dof(x,y,max_lag=None):
    """Effective number of independent pairs for a cross-correlation of ``x``, ``y``.

    Bretherton et al. (1999), eq. 30: N_eff = N / sum_{k=-(N-1)}^{N-1} rho_x(k) rho_y(k),
    approximated here by truncating the lag sum at ``max_lag`` (default N/5, a common
    rule of thumb) once the product of the two autocorrelation functions becomes
    negligible.

    Parameters
    ----------
    x, y : numpy.ndarray
        Input time series.
    max_lag : int, optional
        Lag at which to truncate the sum. Defaults to ``max(1, min(len(x), len(y)) // 5)``.

    Returns
    -------
    float
        Effective sample size, capped at ``min(len(x), len(y))``.
    """
    n = min(x.size,y.size)
    if max_lag is None:
        max_lag = max(1,n//5)
    rho_x = auto_corr(x,max_lag)
    rho_y = auto_corr(y,max_lag)
    # symmetric in lag: sum_{k=-max_lag}^{max_lag} rho_x(k) rho_y(k) = rho_x(0)*rho_y(0) + 2*sum_{k=1}^{max_lag}
    denom = rho_x[0]*rho_y[0]+2.0*np.sum(rho_x[1:]*rho_y[1:])
    denom = max(denom,1.0)  # a negative/near-zero sum would only inflate N_eff past N
    return float(min(n,n/denom))


def xr_eff_dof(x,y,max_lag=120,dim="time"):
    """``eff_dof`` applied along ``dim``, broadcasting over any other dimensions.

    Unlike a direct ``apply_ufunc`` wrap of ``eff_dof``, ``rho_x`` is computed once via
    ``xr_auto_corr(x, ...)`` instead of once per broadcast point (e.g. per lat/lon),
    which matters when ``x`` has no extra dimensions beyond ``dim`` (e.g. a reference
    index) and ``y`` is a full 3-D field: that redundant recomputation roughly doubled
    the runtime.

    Parameters
    ----------
    x, y : xarray.DataArray
        Input time series along ``dim``; either may carry extra dimensions
        (e.g. ``lat``, ``lon``) that are broadcast over.
    max_lag : int, optional
        Lag at which to truncate the sum (see ``eff_dof``).
    dim : str, optional
        Dimension along which the effective sample size is calculated.

    Returns
    -------
    xarray.DataArray
        Effective sample size, broadcasting over any other dimensions of ``x``/``y``.
    """
    n = min(x.sizes[dim],y.sizes[dim])
    rho_x = xr_auto_corr(x,max_lag,dim=dim)
    rho_y = xr_auto_corr(y,max_lag,dim=dim)
    denom = rho_x.isel(lag=0)*rho_y.isel(lag=0)+2.0*(
        rho_x.isel(lag=slice(1,None))*rho_y.isel(lag=slice(1,None))
    ).sum(dim="lag")
    denom = xr.where(denom>1.0,denom,1.0)  # a negative/near-zero sum would only inflate N_eff past N
    n_eff = n/denom
    return xr.where(n_eff<n,n_eff,float(n))
