import warnings

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


def eff_dof4corr(x,y,max_lag=None):
    """Effective number of independent pairs for a cross-correlation of ``x``, ``y``.

    Bretherton et al. (1999), eq. 30: N_eff = N / sum_{k=-(N-1)}^{N-1} rho_x(k) rho_y(k),
    approximated here by truncating the lag sum at ``max_lag`` (default N/5, a common
    rule of thumb) once the product of the two autocorrelation functions becomes
    negligible.

    The null hypothesis assumes x and y are independent as entire processes,
    not merely uncorrelated at zero lag. Each series may be autocorrelated.
    Inputs should be stationary and regularly sampled. This is not an estimator
    for the uncertainty of a mean; use eff_dof4mean for that purpose.

    Parameters
    ----------
    x, y : numpy.ndarray
        Input time series.
    max_lag : int, optional
        Lag at which to truncate the sum. Defaults to ``max(1, min(len(x), len(y)) // 5)``.

    Returns
    -------
    float
        Effective sample size, capped at ``min(len(x), len(y))``. This is not
        the Student-t degrees of freedom (approximately N_eff - 2 for correlation).
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


def xr_eff_dof4corr(x,y,max_lag=120,dim="time"):
    """``eff_dof4corr`` applied along ``dim``, broadcasting over any other dimensions.

    Assumes x and y are independent as entire processes under the null;
    autocorrelation within either series is allowed. See eff_dof4corr.

    Unlike a direct ``apply_ufunc`` wrap of ``eff_dof4corr``, ``rho_x`` is computed once via
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
        Lag at which to truncate the sum (see ``eff_dof4corr``).
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



def eff_dof4mean(x,max_lag=None,cap_at_n=True):
    """Effective sample size for the mean of a stationary, regular 1-D series.

    Estimate N_eff = N / (1 + 2 * sum(rho[k], k=1..max_lag)). The existing
    auto_corr uses N-normalized autocovariances, which already contain the
    finite-record factor (1-k/N) relative to overlap-normalized estimates;
    do not apply that factor a second time. No detrending or deseasonalization
    is performed. Selected, nonconsecutive months must not be treated as a
    regularly sampled series.

    Parameters
    ----------
    x : array_like
        One-dimensional time series. Missing/nonfinite values and constant
        series return NaN; dropping gaps would change the sampling interval.
    max_lag : int, optional
        Truncate the autocorrelation sum at this lag, in sampling intervals.
        Defaults to max(1, N//5). Must be in [0, N-1]. Choose a cutoff after
        physical correlations decay but before noisy tail estimates dominate.
        Summing all lags of a demeaned sample can cause cancellation.
    cap_at_n : bool, optional
        Default True conservatively caps N_eff at N, as in eff_dof4corr.
        False permits N_eff > N for negative autocorrelation; a nonpositive
        estimated denominator then returns NaN rather than an invalid size.

    Returns
    -------
    float
        Effective sample size for SE(mean) approximately std(x)/sqrt(N_eff),
        not Student-t degrees of freedom. Using N_eff-1 for a one-sample test
        is an approximation, not an exact t distribution under autocorrelation.

    References
    ----------
    https://mc-stan.org/docs/reference-manual/analysis.html
    (Effective sample size; this function uses a fixed lag cutoff.)
    """
    x = np.asarray(x,dtype=np.float64)
    n = x.size
    if max_lag is None:
        max_lag = max(1,n//5)
    if not np.isfinite(x).all() or np.all(x == x[0]):
        return float("nan")
    rho = auto_corr(x,max_lag)
    denom = 1.0 + 2.0*np.sum(rho[1:])
    if not np.isfinite(denom):
        return float("nan")
    if cap_at_n:
        denom = max(denom,1.0)
    elif denom <= 0:
        return float("nan")
    return float(n/denom)


def xr_eff_dof4mean(x,max_lag=None,dim="time",cap_at_n=True):
    """Apply eff_dof4mean along dim, preserving other coordinates and laziness.

    x is a regularly sampled xarray.DataArray; see eff_dof4mean for assumptions,
    the lag cutoff, missing-data policy, and the optional conservative N cap.
    Dask-backed inputs must have a single chunk along dim (e.g. chunk(time=-1));
    spatial dimensions may be chunked freely. Returns effective sample size,
    not Student-t degrees of freedom, with the sample dimension removed.
    """
    return xr.apply_ufunc(
        eff_dof4mean, x, input_core_dims=[[dim]], output_core_dims=[[]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64],
        kwargs={"max_lag":max_lag,"cap_at_n":cap_at_n},
    )


def eff_dof(x,y,max_lag=None):
    """Deprecated correlation-only alias; use eff_dof4corr."""
    warnings.warn("eff_dof is correlation-only; use eff_dof4corr (or eff_dof4mean for a mean)",
                  DeprecationWarning,stacklevel=2)
    return eff_dof4corr(x,y,max_lag=max_lag)


def xr_eff_dof(x,y,max_lag=120,dim="time"):
    """Deprecated correlation-only alias; use xr_eff_dof4corr."""
    warnings.warn("xr_eff_dof is correlation-only; use xr_eff_dof4corr (or xr_eff_dof4mean for a mean)",
                  DeprecationWarning,stacklevel=2)
    return xr_eff_dof4corr(x,y,max_lag=max_lag,dim=dim)
