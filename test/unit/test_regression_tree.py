from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from bayesian_decision_tree.regression import RegressionNode


class RegressionNodeTest(TestCase):
    def test_one_partition(self):
        mu = 0
        sd_prior = 1
        prior_obs = 0.01
        kappa = prior_obs
        alpha = prior_obs/2
        var_prior = sd_prior**2
        tau_prior = 1/var_prior
        beta = alpha/tau_prior

        prior = (mu, kappa, alpha, beta)

        Xy = np.array([
            [0.0, 0],
            [0.1, 1],
            [0.9, 0],
            [1.0, 1],
            [1.0, 0],
        ])

        root = RegressionNode(0.5, prior)
        X = Xy[:, :-1]
        y = Xy[:, -1]
        root.fit(X, y)
        print(root)

        self.assertEqual(root.depth_and_leaves(), (0, 1))

        self.assertIsNone(root.child1)
        self.assertIsNone(root.child2)

        self.assertEqual(root.split_dimension, -1)
        self.assertEqual(root.split_value, None)
        self.assertEqual(root.split_index, -1)

        n = len(y)
        mean = y.mean()
        mu, kappa, alpha, beta = prior
        kappa_post = kappa + n
        mu_post = (kappa*mu + n*mean) / kappa_post

        self.assertEqual(root.predict(0.0), mu_post)
        self.assertEqual(root.predict(0.49), mu_post)
        self.assertEqual(root.predict(0.51), mu_post)
        self.assertEqual(root.predict(1.0), mu_post)

        assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), np.array([mu_post, mu_post, mu_post, mu_post]))
        assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), np.array([mu_post, mu_post, mu_post, mu_post]))

        assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), np.array([mu_post, mu_post, mu_post, mu_post]))
        assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([mu_post, mu_post, mu_post, mu_post]))

    def test_decreasing_mse_for_increased_partition_prior(self):
        mu = 0
        sd_prior = 1
        prior_obs = 0.01
        kappa = prior_obs
        alpha = prior_obs/2
        var_prior = sd_prior**2
        tau_prior = 1/var_prior
        beta = alpha/tau_prior

        prior = (mu, kappa, alpha, beta)

        x = np.linspace(-np.pi/2, np.pi/2, 20)
        y = np.sin(x)

        mse_list = []
        for partition_prior in [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]:
            root = RegressionNode(partition_prior, prior)
            root.fit(x.reshape(-1, 1), y)
            mse = np.sum((y - root.predict(x))**2)/len(y)
            mse_list.append(mse)

        self.assertTrue(mse_list[-1] < mse_list[0])
        for i in range(0, len(mse_list)-1):
            self.assertTrue(mse_list[i+1] <= mse_list[i])
