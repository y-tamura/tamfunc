# 
import xarray as xr
import numpy as np
import datetime as dt
from scipy import stats
import dask
import dask.array as da
import cftime
def reg_mean_tser(dataarray,
                x_min,x_max,y_min,y_max):
    # dataarrayはmakeDJFdataで作成したもの
    sel_ds = dataarray.sel(
        lon=slice(x_min,x_max),
        lat=slice(y_min,y_max))
    return sel_ds.mean('lat').mean('lon')

def hg_concat(res,variable='ts',yr_start = 1950, yr_end=2050):
    datalist=[]
    dic_path={'HH':'v20180927','MM':'v20171010','LL':'v20170927'}
    fpath='/l1/cmip6/HighResMIP/control-1950/Amon/{0}/HadGEM3-GC31-{1}/r1i1p1f1/gn/{2}/'.format(variable,res,dic_path[res])
    for yr in range(yr_start,yr_end+1):
        fname=fpath+"{2}_Amon_HadGEM3-GC31-{1}_control-1950_r1i1p1f1_gn_{0}01-{0}12.nc".format(str(yr),res,variable)
        da=xr.open_dataset(fname)
        datalist.append(da)
    return xr.concat(datalist,dim='time')

def corr_bigdata(x,y,dim):
    corr=(x*y).mean(dim)-(x.mean(dim)*y.mean(dim))
    corr/=x.std(dim)*y.std(dim)
    return corr

def hrz_polyfit(y_da,x_da,full=False):
    y_da_with_x=y_da.assign_coords({'index':('time',x_da.values)})
    ds_coef = y_da_with_x.swap_dims({"time":"index"}).polyfit("index",1,full=True)
    if full:
        return ds_coef['polyfit_coefficients'].sel(degree=1),ds_coef['polyfit_residuals'],
    else:
        return ds_coef['polyfit_coefficients'].sel(degree=1)

def lag_corr_3D(x, y, dof,lagx=0, lagy=0):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression
    between the two datasets along their aligned time dimension.  
    Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount. 
    https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')
        #Next important step is to re-align the two datasets so that y adjusts to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)
 
    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n     = x.shape[0]
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)
    
    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    #slope     = cov/(xstd**2)
    #intercept = ymean - xmean*slope  
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(dof-2)/np.sqrt(1-cor**2)
    #stderr = slope/tstats
    
    #from scipy.stats import t
    #pval   = t.sf(tstats, deg_free)*2
    #pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cor,tstats

def lag_linreg_3D(x, y, dof,lagx=0, lagy=0):
    #1. Ensure that the data are properly alinged to each other. 
    # x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). 
        # Hence it needs to be dropped
        x   = x.shift(time = -lagx)
        if lagx>0:
            x = x[:-lagx]
        else:
            x = x[-lagx:]

    if lagy!=0:
        y   = y.shift(time = -lagy)
        if lagy>0:
            y = y[:-lagy]
        else:
            y = y[-lagy:]
    # Lag and truncate data
    # x = x.shift(time=-lagx).dropna('time')
    # y = y.shift(time=-lagy).dropna('time')
    
    # Ensure data is aligned
    x, y = xr.align(x, y)
 
    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n     = x.shape[0]
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  ((x - xmean)*(y - ymean)).sum(axis=0)/(n)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope
    resid = y - (slope*x+intercept)
    rss = (resid**2).sum(axis=0)
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = tval(slope, rss, x, dof)

    return slope, tstats

def xr_regression(x_da,y_da,dof_da,dim='time',xr_out=False,tval_out=True):
    slope = xr.cov(x_da,y_da,dim=dim,ddof=0)/x_da.var(dim)
    
    intercept = y_da.mean(dim)-x_da.mean(dim)*slope
    rss = ((y_da-(slope*x_da+intercept))**2).sum(dim)
    if tval_out:
        if xr_out:
            return slope, tval(slope,rss,x_da,dof_da)
        else:
            return slope.values, tval(slope,rss,x_da,dof_da).values
    else:
        if xr_out:
            return slope
        else:
            return slope.values

def lag_corr_r(da1,da2,lag):
    """_summary_

    Args:
        da1 (_type_): _description_
        da2 (_type_): _description_
        lag (_type_): lag steps of da2 to da1

    Returns:
        _type_: _description_
    """
    
    if lag>0:
        da1_=da1[:-lag]
        da2_=da2[lag:]
    elif lag<0:
        da1_=da1[-lag:]
        da2_=da2[:lag]
    else:
        da1_=da1
        da2_=da2
    da1_=da1_.values
    da2_=da2_.values
    return np.corrcoef(da1_,da2_)[0,1]

def lag_corr_dofs(da1,da2,lag):
    """_summary_

    Args:
        da1 (_type_): _description_
        da2 (_type_): _description_
        lag (_type_): lag steps of da2 respect to da1

    Returns:
        tuple: corrcoefs, dofs
    """
    
    if lag>0:
        da1_=da1[:-lag]
        da2_=da2[lag:]
    elif lag<0:
        da1_=da1[-lag:]
        da2_=da2[:lag]
    else:
        da1_=da1
        da2_=da2
    dof = eff_dof(da1_,da2_)
    da1_=da1_.values
    da2_=da2_.values
    # dof = eff_dof(da1_,da2_)
    return np.corrcoef(da1_,da2_)[0,1], int(np.round(dof))

def eff_dof(idx1,idx2):
    if len(idx1)!=len(idx2):
        raise('Lengths of indexes are not same !')
    r1 = lag_corr_r(idx1,idx1,1)
    r2 = lag_corr_r(idx2,idx2,1)
    return len(idx1)*(1-r1*r2)/(1+r1*r2)

def lag_eff_dof(idx1,idx2,lag):
    if len(idx1)!=len(idx2):
        raise('Lengths of indexes are not same !')
    if lag==0:
        return eff_dof(idx1,idx2)
    elif lag > 0:
        r1 = lag_corr_r(idx1[:-lag],idx1[:-lag],1)
        r2 = lag_corr_r(idx2[lag:],idx2[lag:],1)
    elif lag < 0:
        r1 = lag_corr_r(idx1[-lag:],idx1[-lag:],1)
        r2 = lag_corr_r(idx2[:lag],idx2[:lag],1)
    return len(idx1)*(1-r1*r2)/(1+r1*r2)

def xr_eff_dof_hrz(idx1,da2,dim='time'):
    """_summary_

    Args:
        idx1 (xr.da): 1d dataarray
        idx2 (xr.da): 3d dataarray

    Returns:
        xr.da: 2d dataarray
    """
    
    # r1 = lag_corr_r(idx1,idx1,1)
    # v2 = da2.values
    # r2 = np.sum(v2[:-1]*v2[1:],axis=0)/np.sqrt(np.sum(v2[:-1]**2,axis=0)*np.sum(v2[1:]**2,axis=0))
    r1=xr.corr(idx1,idx1.shift({dim:1}),dim)
    r2=xr.corr(da2,da2.shift({dim:1}),dim)
    dof_da = len(idx1[dim])*(1-r1*r2)/(1+r1*r2)
    # dof_da = xr.DataArray(dof, coords = da2[0].coords)
    return dof_da

def idx_lagcorr(idx1,idx2,lagmin,lagmax,lagint=1):
    rs = np.arange(lagmin,lagmax+lagint,lagint,dtype=float)
    for i,lag in enumerate(range(lagmin,lagmax+lagint,lagint)):
        rs[i] = lag_corr_r(idx1,idx2,lag)
    return rs

def idx_lagdofs(idx1,idx2,lagmin,lagmax,lagint=1):
    dofs = np.arange(lagmin,lagmax+lagint,lagint,dtype=float)
    for i,lag in enumerate(range(lagmin,lagmax+lagint,lagint)):
        dofs[i] = lag_eff_dof(idx1,idx2,lag)
    return dofs

def xr_idx_lagcorr(x,y,lagmin,lagmax,lagint=1):
    """calculate lag-correlation b/w xarray.DataArrays
    lag is defined as how much y is lagged behind x
    Args:
        x (xarray.DataArray): time series x
        y (xarray.DataArray): time series y
    """
    coefs=np.arange(lagmin,lagmax+lagint,lagint,dtype=float)
    coefs[:] = np.nan
    for ilag,lag in enumerate(range(lagmin,lagmax+lagint,lagint)):
        coefs[ilag]=xr.corr(x,y.shift(time=-lag))
    return coefs

def xr_idx_lagreg(x,y,dofs,lagmin,lagmax,lagint=1):
    """calculate lag-regression coeffs b/w xarray.DataArrays
    lag is defined as how much y is lagged behind x
    Args:
        x (xarray.DataArray): time series x
        y (xarray.DataArray): time series y
    """
    coefs=np.arange(lagmin,lagmax+lagint,lagint,dtype=float)
    coefs[:] = np.nan
    tvals=np.zeros_like(coefs)
    x/=x.std()
    tstr,tend=x.time[0],x.time[-1]
    for ilag,lag in enumerate(range(lagmin,lagmax+lagint,lagint)):
        x_ilag=x.shift(time=-lag).sel(time=slice(tstr,tend))
        coefs_tmp=xr_regression(x.shift(time=-lag),y,dofs,xr_out=False,tval_out=False)
        coefs[ilag]=coefs_tmp
        intercept = y.mean()-x.mean()*coefs_tmp
        rss = ((y-(coefs_tmp*x+intercept))**2).sum()
        tvals[ilag]=coefs_tmp/np.sqrt(rss/(dofs[ilag]-2)/np.sum((x-x.mean())**2))
    return coefs,tvals

def xr_idx_lagdofs(x,y,lagmin,lagmax,lagint=1):
    dofs = np.arange(lagmin,lagmax+lagint,lagint,dtype=float)
    for i,lag in enumerate(range(lagmin,lagmax+lagint,lagint)):
        dofs[i] = xr_eff_dof_hrz(x,y.shift(time=-lag))
    return dofs

def ppfs_idxcorr(dofs,alpha=0.95):
    """

    Args:
        dofs (_type_): 2 will be subtracted from dofs in this function.
        alpha (float, optional): alpha will be devided by 2 in this function. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    if alpha>0.5:
        t_alpha = stats.t.ppf(1-(1-alpha)/2,dofs-2)
    else:
        t_alpha = stats.t.ppf(alpha/2,dofs-2)
    r_alpha = t_alpha/np.sqrt(dofs-2+t_alpha**2)
    return r_alpha

def tval(coef_da,rss,x_arr,dof):
    return coef_da/np.sqrt(rss/(dof-2)/np.sum((x_arr-x_arr.mean())**2)) 

def xr_tscore(x_sample,mu=0,dim="time"):
    xmean=x_sample.mean(dim)
    s=x_sample.std(dim,ddof=1)
    return (xmean-mu)/(s/np.sqrt(len(x_sample[dim])))

def xr_tscore_diff(x1_sample,x2_sample,mu1=0,mu2=0,dim="time"):
    x1mean=x1_sample.mean(dim)
    x2mean=x2_sample.mean(dim)
    n1=len(x1_sample[dim]);n2=len(x2_sample[dim])
    s=np.sqrt(((n1-1)*x1_sample.var(dim,ddof=1)+(n2-1)*x2_sample.var(dim,ddof=1))/(n1+n2-2))
    return (x1mean-x2mean-mu1+mu2)/(s*np.sqrt(1/n1+1/n2))

def tval_hrz(coef_da,rss,x_arr,dof_da):
    return coef_da/np.sqrt(rss/(dof_da-2)/np.sum((x_arr-x_arr.mean())**2))

def tcval_hrz(shape,dof_da,alpha,coords=None,xr_out=True):
    tcval = np.zeros(shape)
    tcval_1d = np.ravel(tcval)
    dof_1d = np.ravel(dof_da.values)
    
    for i in range(len(tcval_1d)):
        tcval_1d[i] = stats.t.ppf(1-(1-alpha)/2,dof_1d[i]-2)
    
    tcval = tcval_1d.reshape(shape)
    if xr_out:
        coords=dof_da.coords
        tcval_da = xr.DataArray(tcval,coords=coords)
        return tcval_da
    else:
        return tcval
#------------------------------------------------------------------------------
import string
def abc_ls(n=1):
    return list(string.ascii_lowercase)*n

def create_month_str(month_list):
    inimonstr_ls = ['J','F','M','A','M','J','J','A','S','O','N','D']
    months_str=""
    for m in month_list:
        months_str+=inimonstr_ls[m-1]
    return months_str
def wrap360(da, lon='lon'):
    '''
    da.drop_selは関数内では行わない
    da: 経度座標が東経-180から180であるdataarray
    Purpose: 経度座標を0 -> 360という順に揃える
    '''
    # da = da.drop_sel(lon=-180)
    return da.assign_coords({lon:np.mod(da[lon],360)}).sortby(lon)

def detrend(y_da,key="time"):
    ''' 
    Purpose: remove linear trend
    How to: make x_da (simple number array which has "key" dims)
    and regress y_da to x_da
    '''
    x_da = xr.DataArray(np.arange(len(y_da[key])),
                    coords=y_da[key].coords)
    # regcoef_da,tval,dof_da = get_trend(y_da,key)
    regcoef_da = xr.cov(x_da,y_da,dim=key,ddof=0)/x_da.var(key)
    segment_da = y_da.mean(key) - regcoef_da*x_da.mean(key)
    return y_da - (regcoef_da*x_da + segment_da), regcoef_da*x_da + segment_da

def get_trend(y_da,key="time",xr_out=True):
    ''' 
    Purpose: remove linear trend
    How to: make x_da (simple number array which has "key" dims)
    and regress y_da to x_da
    '''
    x_da = xr.DataArray(np.arange(len(y_da[key])),
                    coords=y_da[key].coords)
    dof_da=xr_eff_dof_hrz(x_da,y_da,key)
    trend,tval=xr_regression(x_da,y_da,dof_da,key,xr_out)#xr.cov(x_da,y_da,dim=key,ddof=0)/x_da.var(key)
    return trend,tval,dof_da

def xr_trend(y_da,dim='time'):
    x_da=xr.DataArray(np.arange(len(y_da[dim])),coords={dim:y_da[dim].values})
    return xr.cov(y_da,x_da,dim=dim,ddof=0)/x_da.var(dim)*len(x_da)

def rm_monthlyclim(var):
    clim = var.groupby('time.month').mean('time')
    return var.groupby('time.month') - clim

def calc_anom(var):
    var_det,trend = detrend(var)
    return rm_monthlyclim(var_det)

def detrend_dask(y_da, dim="time"):
    # y_daをDask配列に変換し、チャンクサイズを指定します。
    # このチャンクサイズはメモリ使用量と計算速度に影響を与えます。
    # 最適な値は具体的な状況やハードウェアによります。
    y_da = y_da.chunk({'lat':'auto','lon':'auto'})

    x_da = xr.DataArray(da.arange(len(y_da[dim])),
                        coords=y_da[dim].coords)
    
    cov_da = xr.cov(y_da, x_da, dim)
    varx_da = x_da.var(dim)
    regcoef_da = cov_da/varx_da
    segment_da = y_da.mean(dim) - regcoef_da*x_da.mean(dim)
    detrended_y_da = y_da - (regcoef_da*x_da + segment_da)

    # 計算を予約してメモリにすぐにロードせず、
    # 必要になったときに計算を実行するためにDaskを使用します。
    # detrended_y_da = detrended_y_da.persist()
    return detrended_y_da, regcoef_da*x_da + segment_da

def calc_anom_dask(y_da,dim="time"):
    # y_dask = y_da.chunk({'lat':'auto','lon':'auto'})
    # y_da = y_da.chunk({'auto'})
    y_det, resid = detrend_dask(y_da,dim)
    return (y_det.compute().groupby('time.month') - y_det.groupby('time.month').mean().compute())#.compute()

def xr_month2yearly_selmonths(da,months,dim="time",irep=-1):
    nmon=len(months)
    if irep>=0:
        wshift=-(nmon-1+irep)
    else:
        wshift=irep+1
    nmonmean_da=da.shift({dim:wshift}).rolling({dim:nmon},center=False).mean()
    repmon=months[irep]
    yearly_selmonths_da=nmonmean_da.where(nmonmean_da[dim].month==repmon,drop=True)
    return yearly_selmonths_da.groupby(dim+".year").mean(dim)

def date2state(date,dts_state,nam_state):
    for ist,dts in enumerate(dts_state):
        if date in dts:
            return nam_state[ist]

def return_1st_of_prevmonth_cftime(dt_cftime,calendar="noleap"):
    y,m=dt_cftime.year,dt_cftime.month
    if m==1:
        dt_1st_prevmonth=cftime.datetime(y-1,12,1,calendar=calendar)
    else:
        dt_1st_prevmonth=cftime.datetime(y,m-1,1,calendar=calendar)
    return dt_1st_prevmonth

def return_1st_of_nextmonth_cftime(dt_cftime,calendar="noleap"):
    y,m=dt_cftime.year,dt_cftime.month
    if m==12:
        dt_1st_nextmonth=cftime.datetime(y+1,1,1,calendar=calendar)
    else:
        dt_1st_nextmonth=cftime.datetime(y,m+1,1,calendar=calendar)
    return dt_1st_nextmonth

def return_nmonafter_cftime(dt_cftime,nmon,calendar="noleap"):
    y,m=dt_cftime.year,dt_cftime.month
    yr=y+(m+nmon-1)//12
    mr=(m+nmon-1)%12+1
    return cftime.datetime(yr,mr,dt_cftime.day,calendar=calendar)

def return_end_of_month_cftime(dt_cftime,calendar="noleap"):
    y,m=dt_cftime.year,dt_cftime.month
    if m==12:
        dt_end_month=cftime.datetime(y+1,1,1,calendar=calendar)-dt.timedelta(days=1)
    else:
        dt_end_month=cftime.datetime(y,m+1,1,calendar=calendar)-dt.timedelta(days=1)
    return dt_end_month
    
def rm_monthlyclim_dask(y_da):
    y_dask = y_da.chunk({'lat':'auto','lon':'auto'})
    # y_da = y_da.chunk({'auto'})
    return (y_da.groupby('time.month') - y_dask.groupby('time.month').mean().compute())#.compute()

def w_area_mean(da,ndims=3,lataxis=1,mdims=("lat","lon")):
    var_cos_da = w_area(da,ndims,lataxis)
    # if taxis != None:
    #     aveaxis.remove(taxis)
    return var_cos_da.mean(mdims)

def w_area_sum(da,ndims=3,lataxis=1,sdims=('lat','lon')):
    var_cos_da = w_area(da,ndims,lataxis)
    return var_cos_da.sum(sdims)

def w_area(da,ndims=3,lataxis=1):
    var = da.values
    lat = da.lat.values
    shape = [1 for i in range(ndims)]
    for i in range(ndims):
        if i==lataxis:
            shape[i] = len(lat)
    lat_ = lat.reshape(shape)
    var_cos = var*np.cos(np.radians(lat_))/np.cos(np.deg2rad(np.mean(lat)))
    # aveaxis = [i for i in range(ndims)]
    var_cos_da = xr.DataArray(var_cos,
                              coords=da.coords)
    # if taxis != None:
    #     aveaxis.remove(taxis)
    return var_cos_da
#
def uv_g(ssh_da,):
    ssh = ssh_da.values
    lat = np.deg2rad(ssh_da.lat.values)
    lon = ssh_da.lon.values
    dlon = np.deg2rad(lon[1]-lon[0])
    dlat = lat[1]-lat[0]
    Omega = 7.292*10**-5 # [rad]
    R = 6371*10**3 # [m]
    g = 9.81
    coriolis_param = 2*Omega*np.sin(lat)
    dh_dx = (ssh[:,:,2:] - ssh[:,:,:-2])/(R*np.cos(lat.reshape((1,len(lat),1)))*2*dlon)
    dh_dy = (ssh[:,2:,:] - ssh[:,:-2,:])/(R*2*dlat)
    
    ugsurf = xr.DataArray(-g/coriolis_param[1:-1].reshape((1,len(lat)-2,1))*dh_dy,
                          coords={'time':ssh_da.time.values,'lat':ssh_da.lat.values[1:-1],'lon':lon})
    
    vgsurf = xr.DataArray(g/coriolis_param.reshape((1,len(lat),1))*dh_dx,
                         coords={'time':ssh_da.time.values,'lat':ssh_da.lat.values,'lon':lon[1:-1]})
    return ugsurf, vgsurf

def hrz_lowpass_boxcar_deg(var,wlat,wlon,min_periods,dim1='lat',dim2='lon',):
    # var_ = var.chunk({'time':chunksize})
    # wlat = int(wlat_deg//(var.lat.values[1]-var.lat.values[0]))
    # wlon = int(wlon_deg//(var.lon.values[1]-var.lon.values[0]))
    return var.rolling({dim1:wlat,dim2:wlon},center=True,min_periods=min_periods).mean()
    # return 0.5*(var_.rolling({dim1:wlat},center=True,).mean()+var_.rolling({'lon':wlon},center=True,).mean()).compute()

def hrz_highpass_boxcar_deg(var,wlat,wlon,min_periods,dim1='lat',dim2='lon'):
    # var_ = var.chunk({'time':'auto'})
    # wlat = int(wlat_deg//(var.lat.values[1]-var.lat.values[0]))
    # wlon = int(wlon_deg//(var.lon.values[1]-var.lon.values[0]))
    return var-hrz_lowpass_boxcar_deg(var,wlat,wlon,min_periods,dim1,dim2)

def eke_hrzfil_from_uv(u,v,wlat,wlon,min_periods,dim1='lat',dim2='lon'):
    u_hrzfil = hrz_highpass_boxcar_deg(u,wlat,wlon,min_periods,dim1,dim2)
    v_hrzfil = hrz_highpass_boxcar_deg(v,wlat,wlon,min_periods,dim1,dim2)
    # return 0.5*(hrz_highpass_boxcar_deg(u,wlat_deg,wlon_deg)**2 + hrz_highpass_boxcar_deg(v,wlat_deg,wlon_deg)**2)
    return 0.5*(u_hrzfil**2 + v_hrzfil**2)

def eke_tmpfil_from_uv(u,v,ww=13):
    u_lp = u.chunk({'lat':200,'lat':200}).rolling({'time':ww},center=True,min_periods=ww//2).mean().compute()#[ww//2:-(ww//2)]
    v_lp = v.chunk({'lat':200,'lat':200}).rolling({'time':ww},center=True,min_periods=ww//2).mean().compute()#[ww//2:-(ww//2)]
    return 0.5*((u - u_lp)**2 +(v - v_lp)**2)
# 
def extract_intervals(da, critical_val, n_conseq,upper=True,calendar="noleap"):
    # boolen array about larger or lower than critical_val
    if upper:
        boolen_critical = da > critical_val
    else:
        boolen_critical = da < critical_val
    
    # starting and ending indices of interval with continuous True values
    true_starts = []
    true_ends = []
    in_interval = False
    for i, val in enumerate(boolen_critical):
        if val and not in_interval:
            # 連続区間の開始
            start_idx = i
            in_interval = True
        elif not val and in_interval:
            # 連続区間の終了
            if i - start_idx >= n_conseq:
                true_starts.append(start_idx)
                true_ends.append(i - 1)
            in_interval = False
    
    # 最後の区間がリストの終わりまで続いていた場合
    if in_interval and len(boolen_critical) - start_idx >= n_conseq:
        true_starts.append(start_idx)
        true_ends.append(len(boolen_critical) - 1)
    
    # 開始時刻と終了時刻のリストを作成
    intervals = [(da.time.values[start], return_end_of_month_cftime(da.time.values[end],calendar=calendar)) for start, end in zip(true_starts, true_ends)]
    return intervals

def extract_intervals_from_boolarray(bool_array, n_conseq,calendar="noleap"):
    # starting and ending indices of interval with continuous True values
    true_starts = []
    true_ends = []
    in_interval = False
    for i, val in enumerate(bool_array):
        if val and not in_interval:
            # 連続区間の開始
            start_idx = i
            in_interval = True
        elif not val and in_interval:
            # 連続区間の終了
            if i - start_idx >= n_conseq:
                true_starts.append(start_idx)
                true_ends.append(i - 1)
            in_interval = False
    
    # 最後の区間がリストの終わりまで続いていた場合
    if in_interval and len(bool_array) - start_idx >= n_conseq:
        true_starts.append(start_idx)
        true_ends.append(len(bool_array) - 1)
    
    # 開始時刻と終了時刻のリストを作成
    intervals = [(bool_array.time.values[start], return_end_of_month_cftime(bool_array.time.values[end],calendar=calendar)) for start, end in zip(true_starts, true_ends)]
    return intervals

def extract_long_events(times,length_criterion,int_limit,ratio_event=0.6):
    """extract initial and final dates of long-lasting events from series of dates

    Args:
        times (array_like): array of event dates
        length_criterion (timedelta): minimum length of events
        int_limit (timedelta): maximum pause of events to be considered as continuous

    Returns:
        list: list of tuples of initial and final dates of events
    """
    diff_times=times[1:]-times[:-1]
    periods_selected=[]
    start_temp=times[0]
    for it,t in zip(range(1,len(times)),times[1:]):
        if diff_times[it-1]>int_limit:
            if times[it-1]-start_temp>=length_criterion:
                end_temp=times[it-1]
                plength=len(xr.cftime_range(start_temp,end_temp,freq="MS"))
                num_evemonths=len(times[(times>=start_temp)&(times<=end_temp)])
                if num_evemonths/plength>=ratio_event:
                    periods_selected.append((start_temp,end_temp))
                start_temp=t
            else:
                start_temp=t
        else:
            continue
    return periods_selected

def extract_short_events(times,times_long,length_criterion,int_limit,ratio_event=0.6):
    """extract initial and final dates of long-lasting events from series of dates

    Args:
        times (array_like): array of event dates
        length_criterion (timedelta): minimum length of events
        int_limit (timedelta): maximum pause of events to be considered as continuous

    Returns:
        list: list of tuples of initial and final dates of events
    """
    times_short=np.array([t for t in times if t not in times_long])
    diff_times_short=times_short[1:]-times_short[:-1]
    periods_selected=[]
    start_temp=times_short[0]
    for it,t in zip(range(1,len(times_short)),times_short[1:]):
        if diff_times_short[it-1]>int_limit:
            if times_short[it-1]-start_temp>=length_criterion:
                end_temp=times_short[it-1]
                plength=len(xr.cftime_range(start_temp,end_temp,freq="MS"))
                num_evemonths=len(times_short[(times_short>=start_temp)&(times_short<=end_temp)])
                if num_evemonths/plength>=ratio_event:
                    periods_selected.append((start_temp,end_temp))
                start_temp=t
            else:
                start_temp=t
        else:
            continue
    return periods_selected

def xr_calc_vorticity(u,v,rad_earth=6371e3):
    """calculate vorticity from u and v in single precision

    Args:
        u (xr.DataArray): zonal velocity [m/s]
        v (xr.DataArray): meridional velocity [m/s]
        lat (array_like): latitude
        rad_earth (float, optional): radius of earth. Defaults to 6371*10**3.
    Returns:
        xr.DataArray: vorticity [1/s]
    """
    lat_rad = np.deg2rad(u.lat)
    lon_rad = np.deg2rad(u.lon)
    nlat,nlon=u.lat.shape[0],u.lon.shape[0]
    dlat=lat_rad[1]-lat_rad[0]
    dlon=lon_rad[1]-lon_rad[0]
    
    # Calculate the grid spacing (assuming regular grid)
    dx = (rad_earth * np.cos(lat_rad) * dlon * xr.ones_like(u.lon)).astype(np.float32)
    dy = (rad_earth * dlat * xr.ones_like(u.lat)).astype(np.float32)
    
    # Calculate derivatives using central difference
    du_dy = (u.shift(lat=-1) - u.shift(lat=1)) / (2 * dy)
    dv_dx = (v.shift(lon=-1) - v.shift(lon=1)) / (2 * dx)
    
    # Calculate vorticity
    vorticity = dv_dx - du_dy
    vorticity=vorticity.assign_attrs({'units':'1/s'})
    return vorticity

def np_calculate_vorticity(u, v, lat, lon, rad_earth=6371e3):
    """
    Calculate the vorticity from u and v wind components using central difference.
    
    Parameters:
    u (numpy.ndarray): Zonal wind component (time, lat, lon)
    v (numpy.ndarray): Meridional wind component (time, lat, lon)
    lat (numpy.ndarray): Array of latitude values
    lon (numpy.ndarray): Array of longitude values
    rad_earth (float): Radius of the Earth in meters (default is 6371e3)
    
    Returns:
    numpy.ndarray: Vorticity (time, lat, lon)
    """
    
    # Convert latitude to radians
    lat_rad = np.deg2rad(lat)
    
    # Calculate the grid spacing (assuming regular grid)
    dx = rad_earth * np.cos(lat_rad[:, np.newaxis]) * np.gradient(lon)
    dy = rad_earth * np.gradient(lat_rad)
    
    # Initialize the derivatives
    du_dy = np.full_like(u, np.nan)
    dv_dx = np.full_like(v, np.nan)
    
    # Calculate derivatives using central difference
    du_dy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy[:, np.newaxis])
    dv_dx[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dx[np.newaxis, :])
    
    # Handle the edges with NaN
    du_dy[:, 0, :] = np.nan
    du_dy[:, -1, :] = np.nan
    dv_dx[:, :, 0] = np.nan
    dv_dx[:, :, -1] = np.nan
    
    # Calculate vorticity
    vorticity = dv_dx - du_dy
    
    return vorticity