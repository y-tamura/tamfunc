import numpy as np
import xarray as xr
from scipy import stats
from scipy.signal import windows

def _get_time_dim(timeseries, dim=None, axis=0):
    if dim is not None:
        return dim
    if hasattr(timeseries, "dims") and "time" in timeseries.dims:
        return "time"
    if hasattr(timeseries, "dims"):
        return timeseries.dims[axis]
    return None

def _power_spectrum_1d(timeseries,deltat,ave_width):
    N=timeseries.shape[-1]
    hannwin = windows.hann(N)
    power = 2*np.abs(np.fft.fft(timeseries*hannwin)[...,:N//2])**2*(deltat/N)*8/3
    power[0] /= 2
    if ave_width>1:
        power_ave = np.full_like(power,np.nan,dtype=float)
        start = ave_width//2
        valid = np.convolve(power,np.ones(ave_width)/ave_width,'valid')
        power_ave[start:start+len(valid)] = valid
        return power_ave
    else:
        return power
def xr_power_spectrum(timeseries,deltat,ave_width=1,axis=0,dim=None):
    """Calculate a power spectrum along one dimension of a DataArray.

    Parameters
    ----------
    timeseries : xarray.DataArray
        Input data. It may have any number of dimensions.
    deltat : float
        Sampling interval of the time axis.
    ave_width : int, optional
        Width of the running mean applied in frequency space. No smoothing is
        applied when this is 1.
    axis : int, optional
        Axis number used as the transform dimension when ``dim`` is not given
        and ``timeseries`` has no ``time`` dimension.
    dim : str, optional
        Dimension name along which the spectrum is calculated. If omitted, the
        ``time`` dimension is used when present; otherwise ``axis`` is used.

    Returns
    -------
    xarray.DataArray
        Power spectrum with the transform dimension replaced by ``freq``.
        All other input dimensions are preserved.
    """
    dim = _get_time_dim(timeseries,dim,axis)
    N=timeseries.sizes[dim]
    freq = np.fft.fftfreq(N,deltat)[:N//2]
    power = xr.apply_ufunc(
        _power_spectrum_1d,
        timeseries,
        kwargs={"deltat":deltat,"ave_width":ave_width},
        input_core_dims=[[dim]],
        output_core_dims=[["freq"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes":{"freq":N//2}},
    )
    return power.assign_coords(freq=freq)
def np_power_spectrum(timeseries,deltat,ave_width):
    N=len(timeseries)
    hannwin = windows.hann(N)
    power = 2*np.abs(np.fft.fft(timeseries*hannwin)[:N//2])**2*(deltat/N)*8/3
    power[0] /= 2
    if ave_width>1:
        power_ave = np.zeros_like(power)
        power_ave[:ave_width//2] = np.nan
        power_ave[-(ave_width//2):] = np.nan
        power_ave[ave_width//2:-(ave_width//2)] = np.convolve(power,np.ones(ave_width)/ave_width,'valid')
        return power_ave
    else:
        return power
def red_power_spectrum(timeseries,deltat,freq):
    r1 = np.corrcoef(timeseries[1:],timeseries[:-1])[0,1]
    var_w = np.var(timeseries).values*(1-r1**2)
    return 2*deltat*var_w/(1+r1**2-2*r1*np.cos(2*np.pi*freq*deltat))
def red_conf_int(red_power,dof,alpha):
    # from scipy import stats
    return dof*red_power/stats.chi2.ppf(alpha/2,df=dof)
def psd_confint_chi2(psd,dof,alpha):
    upper=dof*psd/stats.chi2.ppf(alpha/2,df=dof)
    lower=dof*psd/stats.chi2.ppf(1-alpha/2,df=dof)
    return upper,lower
def compute_psd_confint(timeseries,deltat,avewidth,alpha=0.1):
    n = len(timeseries)
    edof = 1.9*avewidth
    freq = np.fft.fftfreq(n,deltat)[:n//2]
    freq[0] = np.nan
    psd = xr_power_spectrum(timeseries,deltat,avewidth)
    conf_upper,conf_lower = psd_confint_chi2(psd,edof,alpha)
    return freq,psd,conf_upper,conf_lower
def plot_psd_confint(ax,freq,psd,conf_upper,conf_lower,c,label,lw=1.5,ls='-'):
    ax.plot(1/freq,psd,c=c,lw=lw,ls=ls,label=label)
    ax.fill_between(1/freq,conf_upper,conf_lower,color=c,alpha=0.15)
