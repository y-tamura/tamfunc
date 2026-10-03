"""Compatibility and coordinate-aware effective sample sizes for t scores."""
import unittest

import numpy as np
import xarray as xr
from scipy import stats
import tamfunc as tf


class TScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(35)
        self.x = xr.DataArray(rng.normal(1, 2, (24, 3)), dims=['month', 'lat'],
                              coords={'month': list(range(24)), 'lat': [20, 30, 40]})
        self.y = xr.DataArray(rng.normal(0, 3, (36, 3)), dims=['month', 'lat'],
                              coords={'month': list(range(24, 60)), 'lat': [20, 30, 40]})

    def test_legacy_positional_calls_match_scipy(self):
        np.testing.assert_allclose(tf.xr_tscore(self.x, .3, 'month'),
                                   stats.ttest_1samp(self.x, .3, axis=0).statistic)
        np.testing.assert_allclose(tf.xr_tscore_diff(self.x, self.y, .3, .1, 'month'),
                                   stats.ttest_ind(self.x-.3, self.y-.1, axis=0).statistic)

    def test_single_sample_effective_count_scaling_and_lazy_arrays(self):
        n = xr.DataArray([6., 12., 24.], dims=['lat'], coords={'lat': [20, 30, 40]})
        expected = tf.xr_tscore(self.x, dim='month') * np.sqrt(n/24)
        actual = tf.xr_tscore(self.x.chunk({'month': 6}), dim='month', dof=n)
        self.assertTrue(hasattr(actual.data, 'dask'))
        xr.testing.assert_allclose(actual.compute(), expected)

    def test_two_sample_matches_scipy_summary_statistics(self):
        n1 = xr.DataArray([6., 12., 24.], dims=['lat'], coords={'lat': [20, 30, 40]})
        for n2 in [None, 9.]:
            expected = stats.ttest_ind_from_stats(
                self.x.mean('month'), self.x.std('month', ddof=1), n1,
                self.y.mean('month'), self.y.std('month', ddof=1), 36 if n2 is None else n2)
            result = tf.xr_tscore_diff(self.x, self.y, dim='month', dof1=n1, dof2=n2)
            np.testing.assert_allclose(result, expected.statistic)
            xr.testing.assert_equal(result.lat, self.x.lat)

    def test_invalid_dof_dimensions_and_coordinates(self):
        for bad in [xr.DataArray([3., 4.], dims=['month']),
                    xr.DataArray([3., 4.], dims=['unknown']),
                    xr.DataArray([3., 4., 5.], dims=['lat'], coords={'lat': [0, 1, 2]})]:
            with self.assertRaises(ValueError):
                tf.xr_tscore(self.x, dim='month', dof=bad)
            with self.assertRaises(ValueError):
                tf.xr_tscore_diff(self.x, self.y, dim='month', dof2=bad)


if __name__ == '__main__':
    unittest.main()
