import numpy as np
import xarray as xr


def lanczos_weights(cutoff_period,num_weights):
    """Symmetric Lanczos low-pass filter weights (Duchon 1979).

    ``cutoff_period`` is in the same time units as the series' sampling interval (e.g.
    months for a monthly series with a cutoff given in months); ``num_weights`` is the
    number of weights on each side of the centre (filter length = 2*num_weights + 1).
    """
    fc = 1.0/cutoff_period
    k = np.arange(1,num_weights+1)
    sigma = np.sin(np.pi*k/num_weights)/(np.pi*k/num_weights)
    firideal = np.sin(2.0*np.pi*fc*k)/(np.pi*k)
    half = firideal*sigma
    weights = np.concatenate([half[::-1],[2.0*fc],half])
    return weights/weights.sum()


def lanczos_lowpass(x,cutoff_period,num_weights):
    """Low-pass ``x`` at ``cutoff_period`` with a Lanczos filter; edges (num_weights on
    each side) are set to NaN since the filter is undefined there without padding.
    """
    weights = lanczos_weights(cutoff_period,num_weights)
    x = np.asarray(x,dtype="float64")
    filtered = np.convolve(x,weights,mode="valid")
    out = np.full(x.size,np.nan)
    out[num_weights:num_weights+filtered.size] = filtered
    return out


def xr_lanczos_lowpass(x,cutoff_period,num_weights,dim="time"):
    """``lanczos_lowpass`` applied along ``dim``, broadcasting over any other dimensions
    (e.g. a full ``(time, lat, lon)`` field), via ``apply_ufunc``.
    """
    filtered = xr.apply_ufunc(
        lanczos_lowpass,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        kwargs={"cutoff_period":cutoff_period,"num_weights":num_weights},
    )
    return filtered.transpose(*x.dims)


def lowpass_filter(var,ww,cutoff_period,npoint=1,dim='time'):
    '''
    window length for filters
    window = 31

    cut-off period [yr], [day],...
    cutoff_period = 10

    number of points per unit time
    6-hourly data with a cutoff period 8[day] --> npoint = 4
    yearly data with with a cutoff period 10[year] --> npoint = 1
    monthly data with with a cutoff period 10[year] --> npoint = 12
    npoint = 1
    '''
    num_weights = ww//2
    var_lp = xr_lanczos_lowpass(var,cutoff_period*npoint,num_weights,dim=dim)
    return var_lp.isel({dim:slice(num_weights,-num_weights)})


def highpass_filter(var,ww,cop,npt=1,dim='time'):
    var_lp = lowpass_filter(var,ww,cop,npt,dim=dim)
    var_trimmed = var.isel({dim:slice(ww//2,-(ww//2))})
    return var_trimmed-var_lp
