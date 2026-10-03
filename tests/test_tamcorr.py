"""Effective sample size: statistical behavior and NumPy/xarray parity."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import xarray as xr
from tamfunc import tamcorr


class EffectiveSampleSizeTests(unittest.TestCase):
    def test_ar1_mean_and_correlation_are_distinct(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=50000)
        for i in range(1,len(x)):
            x[i] += .8*x[i-1]
        mean = tamcorr.eff_dof4mean(x,max_lag=40)
        corr = tamcorr.eff_dof4corr(x,x,max_lag=40)
        # Long-record AR(1) limits, not another implementation of the code.
        self.assertAlmostEqual(mean / (len(x)/9),1,delta=.15)
        self.assertAlmostEqual(corr / (len(x)*.36/1.64),1,delta=.15)
        self.assertLess(mean,corr)

    def test_independent_noise(self):
        x = np.random.default_rng(123).normal(size=30000)
        self.assertAlmostEqual(tamcorr.eff_dof4mean(x,20)/len(x),1,delta=.1)
        self.assertEqual(tamcorr.eff_dof4mean(x,0),len(x))

    def test_xarray_coordinates_nan_and_dask(self):
        x = np.random.default_rng(7).normal(size=(3,400))
        x[1] = 2
        x[2,40] = np.nan
        da = xr.DataArray(x,dims=['lat','month'],coords={'lat':[20,30,40]})
        expected = np.array([tamcorr.eff_dof4mean(row,20) for row in x])
        actual = tamcorr.xr_eff_dof4mean(da,20,dim='month')
        np.testing.assert_allclose(actual,expected,equal_nan=True)
        self.assertEqual(actual.dims,('lat',))
        xr.testing.assert_equal(actual.lat,da.lat)
        lazy = tamcorr.xr_eff_dof4mean(da.chunk({'lat':1,'month':-1}),20,dim='month')
        self.assertTrue(hasattr(lazy.data,'dask'))
        xr.testing.assert_allclose(lazy.compute(),actual)

    def test_negative_correlation_and_invalid_data(self):
        x = np.tile([1.,-1.],100)
        self.assertEqual(tamcorr.eff_dof4mean(x,2),len(x))
        self.assertEqual(tamcorr.eff_dof4mean(x,2,cap_at_n=False),len(x))
        self.assertEqual(tamcorr.eff_dof4mean(x,1,cap_at_n=False),len(x))
        for x in [np.ones(20),np.full(20,np.nan),np.array([0.,1.,np.inf])]:
            self.assertTrue(np.isnan(tamcorr.eff_dof4mean(x,1)))
        for x,lag in [([1],0),([[1,2]],0),([1,2],2),([1,2],-1),([1,2],1.5)]:
            with self.assertRaises(ValueError):
                tamcorr.eff_dof4mean(x,lag)

    def test_full_acf_then_first_nonpositive_cutoff(self):
        x = np.arange(20.)
        # Later positive values must not be resumed after zero/negative lags.
        for rho, retained in [([1., .5, 0., .4], .5),
                              ([1., .5, -.2, .4], .5),
                              ([1., -.1, .3, .4], 0.),
                              ([1., .5, .3, .2], 1.),
                              ([1.], 0.)]:
            with self.subTest(rho=rho):
                with patch.object(tamcorr, 'auto_corr', return_value=np.array(rho)) as acf:
                    actual = tamcorr.eff_dof4mean(x, max_lag=len(rho)-1)
                    acf.assert_called_once_with(x, len(rho)-1)
                self.assertAlmostEqual(actual, len(x)/(1+2*retained))

    def test_xarray_pointwise_cutoffs(self):
        x = xr.DataArray(np.arange(60.).reshape(3,20), dims=['lat','time'],
                         coords={'lat':[20,30,40]})
        acfs = [np.array([1.,-.1,.2,.3]), np.array([1.,.5,0.,.3]),
                np.array([1.,.5,.3,.2])]
        with patch.object(tamcorr, 'auto_corr', side_effect=acfs) as acf:
            result = tamcorr.xr_eff_dof4mean(x,max_lag=3)
            self.assertEqual(acf.call_count,3)
            self.assertTrue(all(call.args[1] == 3 for call in acf.call_args_list))
        np.testing.assert_allclose(result,[20.,10.,20/3])
        xr.testing.assert_equal(result.lat,x.lat)

    def test_correlation_formula_and_compatibility(self):
        rng = np.random.default_rng(99)
        x,y = rng.normal(size=(2,400))
        rx,ry = tamcorr.auto_corr(x,30),tamcorr.auto_corr(y,30)
        expected = len(x)/max(1.,rx[0]*ry[0]+2*np.dot(rx[1:],ry[1:]))
        actual = tamcorr.eff_dof4corr(x,y,30)
        self.assertAlmostEqual(actual,expected)
        dx,dy = xr.DataArray(x,dims=['time']),xr.DataArray(y,dims=['time'])
        self.assertAlmostEqual(float(tamcorr.xr_eff_dof4corr(dx,dy,30)),actual)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',DeprecationWarning)
            self.assertEqual(tamcorr.eff_dof(x,y,30),actual)
            xr.testing.assert_equal(tamcorr.xr_eff_dof(dx,dy,30),
                                    tamcorr.xr_eff_dof4corr(dx,dy,30))
            self.assertEqual(len(caught),2)


if __name__ == '__main__':
    unittest.main()
