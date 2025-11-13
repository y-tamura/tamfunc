import numpy as np
from scipy import stats
from scipy.signal import windows

def xr_power_spectrum(timeseries,deltat,ave_width=1,ndims=1,axis=0):
    # from scipy.signal import windows
    N=len(timeseries.time)
    # windowshape=[1]*ndims
    # windowshape[axis]=N
    hannwin = windows.hann(N)#.reshape(windowshape)
    power = 2*np.abs(np.fft.fft(timeseries.values*hannwin,axis=axis)[:N//2])**2*(deltat/N)*8/3
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
