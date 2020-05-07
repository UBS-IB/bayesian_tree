import itertools
from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from bayesian_decision_tree.utils import hypercube_to_hypersphere_surface


class UtilsTest(TestCase):
    def test_hypercube_to_hypersphere_surface_2D_full_single_point(self):
        hc = np.array([0.2, 0.9])
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=False)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 1)
        self.assertEqual(hs.shape, (3,))
        assert_almost_equal(np.linalg.norm(hs), 1)

    def test_hypercube_to_hypersphere_surface_1D_full(self):
        n_points = 11
        hc = np.linspace(0, 1, n_points).reshape(-1, 1)
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=False)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 2))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # check uniformity
        expected_cos = np.dot(hs[0], hs[1])
        for i in range(1, n_points):
            cos = np.dot(hs[i-1], hs[i])
            assert_almost_equal(cos, expected_cos)

        cos = np.dot(hs[0], hs[-2])
        assert_almost_equal(cos, expected_cos)

        cos = np.dot(hs[0], hs[-1])
        assert_almost_equal(cos, 1.0)

    def test_hypercube_to_hypersphere_surface_1D_half(self):
        n_points = 11
        hc = np.linspace(0, 1, n_points).reshape(-1, 1)
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=True)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 2))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # check uniformity
        expected_cos = np.dot(hs[0], hs[1])
        for i in range(1, n_points):
            cos = np.dot(hs[i-1], hs[i])
            assert_almost_equal(cos, expected_cos)

        cos = np.dot(hs[0], hs[-2])
        assert_almost_equal(cos, -expected_cos)

        cos = np.dot(hs[0], hs[-1])
        assert_almost_equal(cos, -1.0)

    def test_hypercube_to_hypersphere_surface_2D_full(self):
        n_points_per_dim = 1000
        n_points = n_points_per_dim**2
        grid = np.linspace(0, 1, n_points_per_dim)
        x, y = np.meshgrid(grid, grid)
        hc = np.array([x.flatten(), y.flatten()]).T
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=False)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 2+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.01
        for quadrant_signs in itertools.product([-1, 1], [-1, 1], [-1, 1]):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            min = n_points / 2**(2+1) * (1-tolerance_fraction)
            max = n_points / 2**(2+1) * (1+tolerance_fraction)
            msg = f'Expected a value between {min:.0f} and {max:.0f}, but was {in_quadrant}'
            self.assertTrue(min <= np.sum(in_quadrant) <= max, msg=msg)

    def test_hypercube_to_hypersphere_surface_2D_half(self):
        n_points_per_dim = 1000
        n_points = n_points_per_dim**2
        grid = np.linspace(0, 1, n_points_per_dim)
        x, y = np.meshgrid(grid, grid)
        hc = np.array([x.flatten(), y.flatten()]).T
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=True)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 2+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.01
        for quadrant_signs in itertools.product([-1, 1], [-1, 1], [-1, 1]):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            if quadrant_signs[0] == -1:
                self.assertEqual(np.sum(in_quadrant), 0)
            else:
                min = n_points / 2**2 * (1-tolerance_fraction)
                max = n_points / 2**2 * (1+tolerance_fraction)
                msg = f'Expected a value between {min:.0f} and {max:.0f} in quadrant {quadrant_signs}, but was {in_quadrant}'
                self.assertTrue(min <= np.sum(in_quadrant) <= max, msg)

    def test_hypercube_to_hypersphere_surface_5D_full(self):
        n_points = 1_000_000
        np.random.seed(666)
        hc = np.random.uniform(0, 1, (n_points, 5))
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=False)
        # hs = np.random.normal(0, 1, hs.shape)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 5+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.02
        for quadrant_signs in itertools.product(*list(np.tile([-1, 1], (5+1, 1)))):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            min = n_points / 2**(5+1) * (1-tolerance_fraction)
            max = n_points / 2**(5+1) * (1+tolerance_fraction)
            msg = f'Expected a value between {min:.0f} and {max:.0f}, but was {in_quadrant}'
            self.assertTrue(min <= np.sum(in_quadrant) <= max, msg=msg)

    def test_hypercube_to_hypersphere_surface_5D_half(self):
        n_points = 1_000_000
        np.random.seed(666)
        hc = np.random.uniform(0, 1, (n_points, 5))
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=True)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 5+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.01
        for quadrant_signs in itertools.product(*list(np.tile([-1, 1], (5+1, 1)))):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            if quadrant_signs[0] == -1:
                self.assertEqual(np.sum(in_quadrant), 0)
            else:
                min = n_points / 2**5 * (1-tolerance_fraction)
                max = n_points / 2**5 * (1+tolerance_fraction)
                msg = f'Expected a value between {min:.0f} and {max:.0f} in quadrant {quadrant_signs}, but was {in_quadrant}'
                self.assertTrue(min <= np.sum(in_quadrant) <= max, msg)

    def test_hypercube_to_hypersphere_surface_6D_full(self):
        n_points = 1_000_000
        np.random.seed(666)
        hc = np.random.uniform(0, 1, (n_points, 6))
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=False)
        # hs = np.random.normal(0, 1, hs.shape)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 6+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.03
        for quadrant_signs in itertools.product(*list(np.tile([-1, 1], (6+1, 1)))):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            min = n_points / 2**(6+1) * (1-tolerance_fraction)
            max = n_points / 2**(6+1) * (1+tolerance_fraction)
            msg = f'Expected a value between {min:.0f} and {max:.0f}, but was {in_quadrant}'
            self.assertTrue(min <= np.sum(in_quadrant) <= max, msg=msg)

    def test_hypercube_to_hypersphere_surface_6D_half(self):
        n_points = 1_000_000
        np.random.seed(666)
        hc = np.random.uniform(0, 1, (n_points, 6))
        hs = hypercube_to_hypersphere_surface(hc, half_hypersphere=True)

        # check dimensionality and norms
        self.assertEqual(hs.ndim, 2)
        self.assertEqual(hs.shape, (n_points, 6+1))
        assert_almost_equal(np.linalg.norm(hs, axis=1), 1)

        # make sure all quadrants contain approximately the same number of data points
        tolerance_fraction = 0.03
        for quadrant_signs in itertools.product(*list(np.tile([-1, 1], (6+1, 1)))):
            in_quadrant = np.all(hs * quadrant_signs > 0, axis=1).sum()
            if quadrant_signs[0] == -1:
                self.assertEqual(np.sum(in_quadrant), 0)
            else:
                min = n_points / 2**6 * (1-tolerance_fraction)
                max = n_points / 2**6 * (1+tolerance_fraction)
                msg = f'Expected a value between {min:.0f} and {max:.0f} in quadrant {quadrant_signs}, but was {in_quadrant}'
                self.assertTrue(min <= np.sum(in_quadrant) <= max, msg)
