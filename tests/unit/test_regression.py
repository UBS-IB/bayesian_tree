from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from bayesian_decision_tree.regression import PerpendicularRegressionNode
from tests.unit.helper import data_matrix_transforms, create_regression_models


class RegressionNodeTest(TestCase):
    def test_cannot_predict_before_training(self):
        mu = 0
        sd_prior = 1
        prior_obs = 0.01
        kappa = prior_obs
        alpha = prior_obs/2
        var_prior = sd_prior**2
        tau_prior = 1/var_prior
        beta = alpha/tau_prior

        prior = np.array([mu, kappa, alpha, beta])

        for model in create_regression_models(prior, 0.5):
            # can't predict yet
            try:
                model.predict([])
                self.fail()
            except ValueError:
                pass

    def test_cannot_predict_with_bad_input_dimensions(self):
        mu = 0
        sd_prior = 1
        prior_obs = 0.01
        kappa = prior_obs
        alpha = prior_obs/2
        var_prior = sd_prior**2
        tau_prior = 1/var_prior
        beta = alpha/tau_prior

        prior = np.array([mu, kappa, alpha, beta])

        for data_matrix_transform in data_matrix_transforms:
            for model in create_regression_models(prior, 0.5):
                Xy = np.array([
                    [0.0, 0.0, 0],
                    [0.0, 1.0, 1],
                    [1.0, 1.0, 0],
                    [1.0, 0.0, 1],
                    [1.0, 0.0, 0],
                ])
                X = Xy[:, :-1]
                y = Xy[:, -1]

                X = data_matrix_transform(X)

                print('Testing {}'.format(type(model).__name__))
                model.fit(X, y)
                print(model)

                model.predict([0, 0])

                try:
                    model.predict(0)
                    self.fail()
                except ValueError:
                    pass

                try:
                    model.predict([0])
                    self.fail()
                except ValueError:
                    pass

                try:
                    model.predict([0, 0, 0])
                    self.fail()
                except ValueError:
                    pass

    def test_no_split(self):
        for data_matrix_transform in data_matrix_transforms:
            mu = 0
            sd_prior = 1
            prior_obs = 0.01
            kappa = prior_obs
            alpha = prior_obs/2
            var_prior = sd_prior**2
            tau_prior = 1/var_prior
            beta = alpha/tau_prior

            prior = np.array([mu, kappa, alpha, beta])

            for model in create_regression_models(prior, 0.5):
                Xy = np.array([
                    [0.0, 0.0, 0],
                    [0.1, 0.1, 1.3],
                    [0.9, 0.9, 0],
                    [1.0, 1.0, 1.2],
                    [1.0, 1.0, 0],
                ])
                X = Xy[:, :-1]
                y = Xy[:, -1]

                X = data_matrix_transform(X)

                print('Testing {}'.format(type(model).__name__))
                model.fit(X, y)
                print(model)

                self.assertEqual(model.depth_and_leaves(), (0, 1))

                self.assertIsNone(model.child1)
                self.assertIsNone(model.child2)

                if isinstance(model, PerpendicularRegressionNode):
                    self.assertEqual(model.split_dimension, -1)
                    self.assertEqual(model.split_value, None)
                else:
                    self.assertEqual(model.best_hyperplane_origin, None)

                n = len(y)
                mean = y.mean()
                mu, kappa, alpha, beta = prior
                kappa_post = kappa + n
                mu_post = (kappa*mu + n*mean) / kappa_post

                expected = np.array([mu_post, mu_post, mu_post, mu_post])
                self.assertEqual(model.predict([0.0, 0.5]), np.expand_dims(expected[0], 0))
                self.assertEqual(model.predict([0.49, 0.5]), np.expand_dims(expected[1], 0))
                self.assertEqual(model.predict([0.51, 0.5]), np.expand_dims(expected[2], 0))
                self.assertEqual(model.predict([1.0, 0.5]), np.expand_dims(expected[3], 0))

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_equal(model.predict(data_matrix_transform2([[0.0, 0.5], [0.49, 0.5], [0.51, 0.5], [1.0, 0.5]])), expected)

    def test_decreasing_mse_for_increased_partition_prior(self):
        for data_matrix_transform in data_matrix_transforms:
            mu = 0
            sd_prior = 1
            prior_obs = 0.01
            kappa = prior_obs
            alpha = prior_obs/2
            var_prior = sd_prior**2
            tau_prior = 1/var_prior
            beta = alpha/tau_prior

            prior = np.array([mu, kappa, alpha, beta])

            x = np.linspace(-np.pi/2, np.pi/2, 20)
            y = np.linspace(-np.pi/2, np.pi/2, 20)
            X = np.array([x, y]).T
            y = np.sin(x) + 3*np.cos(y)

            X = data_matrix_transform(X)

            for i_model in range(len(create_regression_models(prior, 0.5))):
                mse_list = []
                for partition_prior in [0.1, 0.5, 0.9, 0.99]:
                    model = create_regression_models(prior, partition_prior)[i_model]
                    print('Testing {}'.format(type(model).__name__))
                    model.fit(X, y)
                    print(model)
                    mse = np.sum((y - model.predict(X))**2)/len(y)
                    mse_list.append(mse)

                self.assertTrue(mse_list[-1] < mse_list[0])
                for i in range(0, len(mse_list)-1):
                    self.assertTrue(mse_list[i+1] <= mse_list[i])
