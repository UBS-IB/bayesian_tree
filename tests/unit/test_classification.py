from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import normal, randint
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bayesian_decision_tree.classification import PerpendicularClassificationTree
from tests.unit.helper import data_matrix_transforms, create_classification_trees


class ClassificationTreeTest(TestCase):
    def test_cannot_fit_with_bad_dimensions(self):
        for model in create_classification_trees(np.array([1, 1]), 0.5):
            np.random.seed(6666)

            for good_X in [normal(0, 1, [10, 10])]:
                for bad_y in [randint(0, 2, []), randint(0, 2, [10, 10]), randint(0, 2, [11]), randint(0, 2, [10, 10, 10])]:
                    try:
                        model.fit(good_X, bad_y)
                        self.fail()
                    except ValueError:
                        pass

            for bad_X in [normal(0, 1, [10, 10, 10])]:
                for good_y in [randint(0, 2, [10])]:
                    try:
                        model.fit(bad_X, good_y)
                        self.fail()
                    except ValueError:
                        pass

    def test_cannot_predict_before_training(self):
        for model in create_classification_trees(np.array([1, 1]), 0.5):
            # can't predict yet
            try:
                model.predict([])
                self.fail()
            except ValueError:
                pass

            # can't predict probability yet
            try:
                model.predict_proba([])
                self.fail()
            except ValueError:
                pass

    def test_cannot_predict_with_bad_input_dimensions(self):
        for data_matrix_transform in data_matrix_transforms:
            for model in create_classification_trees(np.array([1, 1]), 0.5):
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

    def test_print_empty_model(self):
        for model in create_classification_trees(np.array([1, 1]), 0.5):
            print(model)

    def test_no_split(self):
        for data_matrix_transform in data_matrix_transforms:
            for model in create_classification_trees(np.array([1, 1]), 0.5):
                Xy = np.array([
                    [0.0, 0, 0],
                    [0.0, 1, 1],
                    [1.0, 2, 0],
                    [1.0, 3, 1],
                    [1.0, 4, 0],
                ])
                X = Xy[:, :-1]
                y = Xy[:, -1]

                X = data_matrix_transform(X)

                print('Testing {}'.format(type(model).__name__))
                model.fit(X, y)
                print(model)

                self.assertEqual(model.get_depth(), 0)
                self.assertEqual(model.get_n_leaves(), 1)
                self.assertEqual(model.n_data, 5)

                self.assertIsNone(model.child1)
                self.assertIsNone(model.child2)

                if isinstance(model, PerpendicularClassificationTree):
                    self.assertEqual(model.split_dimension, -1)
                    self.assertEqual(model.split_value, None)
                else:
                    self.assertEqual(model.best_hyperplane_origin, None)
                    self.assertEqual(model.best_hyperplane_normal, None)

                expected = np.array([0, 0, 0, 0])
                self.assertEqual(model.predict([0, 0]), expected[0])
                self.assertEqual(model.predict([0, 1]), expected[1])
                self.assertEqual(model.predict([1, 0]), expected[2])
                self.assertEqual(model.predict([1, 1]), expected[3])

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_equal(model.predict(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 1]])), expected)

                expected = np.array([[4/7, 3/7], [4/7, 3/7], [4/7, 3/7], [4/7, 3/7], ])
                assert_array_almost_equal(model.predict_proba([0, 0]), np.expand_dims(expected[0], 0))
                assert_array_almost_equal(model.predict_proba([0, 1]), np.expand_dims(expected[1], 0))
                assert_array_almost_equal(model.predict_proba([1, 0]), np.expand_dims(expected[2], 0))
                assert_array_almost_equal(model.predict_proba([1, 1]), np.expand_dims(expected[3], 0))

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_almost_equal(model.predict_proba(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 1]])), expected)

                if isinstance(model, PerpendicularClassificationTree):
                    # TODO: also add for hyperplane version
                    expected_paths = [
                        [],
                        [],
                        [],
                        [],
                    ]
                    self.assertEqual(model.prediction_paths([0, 0]), [expected_paths[0]])
                    self.assertEqual(model.prediction_paths([0, 1]), [expected_paths[1]])
                    self.assertEqual(model.prediction_paths([1, 0]), [expected_paths[2]])
                    self.assertEqual(model.prediction_paths([1, 1]), [expected_paths[3]])

                    for data_matrix_transform2 in data_matrix_transforms:
                        self.assertEqual(model.prediction_paths(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 1]])), expected_paths)

    def test_one_split(self):
        for data_matrix_transform in data_matrix_transforms:
            for model in create_classification_trees(np.array([1, 1]), 0.7):
                Xy = np.array([
                    [0.0, 0, 0],
                    [0.1, 1, 0],

                    [0.9, 0, 1],
                    [1.0, 1, 1],
                ])
                X = Xy[:, :-1]
                y = Xy[:, -1]

                X = data_matrix_transform(X)

                print('Testing {}'.format(type(model).__name__))
                model.fit(X, y)
                print(model)

                self.assertEqual(model.get_depth(), 1)
                self.assertEqual(model.get_n_leaves(), 2)
                self.assertEqual(model.n_data, 4)

                self.assertIsNotNone(model.child1)
                self.assertIsNone(model.child1.child1)
                self.assertIsNone(model.child1.child2)
                self.assertEqual(model.child1.n_data, 2)

                self.assertIsNotNone(model.child2)
                self.assertIsNone(model.child2.child1)
                self.assertIsNone(model.child2.child2)
                self.assertEqual(model.child1.n_data, 2)

                if isinstance(model, PerpendicularClassificationTree):
                    self.assertEqual(model.split_dimension, 0)
                    self.assertEqual(model.split_value, 0.5)
                else:
                    self.assertTrue(0.1 < model.best_hyperplane_origin[0] < 0.9)

                expected = np.array([0, 0, 1, 1])
                self.assertEqual(model.predict([0, 0]), expected[0])
                self.assertEqual(model.predict([0, 1]), expected[1])
                self.assertEqual(model.predict([1, 0]), expected[2])
                self.assertEqual(model.predict([1, 1]), expected[3])

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_equal(model.predict(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 0]])), expected)

                expected = np.array([[3/4, 1/4], [3/4, 1/4], [1/4, 3/4], [1/4, 3/4]])
                assert_array_almost_equal(model.predict_proba([0, 0]), np.expand_dims(expected[0], 0))
                assert_array_almost_equal(model.predict_proba([0, 1]), np.expand_dims(expected[1], 0))
                assert_array_almost_equal(model.predict_proba([1, 0]), np.expand_dims(expected[2], 0))
                assert_array_almost_equal(model.predict_proba([1, 1]), np.expand_dims(expected[3], 0))

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_almost_equal(model.predict_proba(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 0]])), expected)

    def test_two_splits(self):
        for data_matrix_transform in data_matrix_transforms:
            for model in create_classification_trees(np.array([1, 1]), 0.9):
                Xy = np.array([
                    [0.0, 0.0, 0],
                    [0.1, 1.0, 0],
                    [0.2, 0.01, 0],
                    [0.3, 0.99, 0],

                    [0.7, 0.02, 1],
                    [0.8, 0.98, 1],
                    [0.9, 0.03, 1],
                    [1.0, 0.97, 1],

                    [2.0, 0.04, 0],
                    [2.1, 0.96, 0],
                ])
                X = Xy[:, :-1]
                y = Xy[:, -1]

                X = data_matrix_transform(X)

                print('Testing {}'.format(type(model).__name__))
                model.fit(X, y, prune=True)
                print(model)

                if isinstance(model, PerpendicularClassificationTree):
                    self.assertEqual(model.get_depth(), 2)
                    self.assertEqual(model.get_n_leaves(), 3)
                    self.assertEqual(model.n_data, 10)

                    self.assertIsNotNone(model.child1)
                    self.assertEqual(model.child1.n_data, 4)
                    self.assertIsNone(model.child1.child1)
                    self.assertIsNone(model.child1.child2)

                    self.assertIsNotNone(model.child2)
                    self.assertEqual(model.child2.n_data, 6)
                    self.assertIsNotNone(model.child2.child1)
                    self.assertEqual(model.child2.child1.n_data, 4)
                    self.assertIsNotNone(model.child2.child2)
                    self.assertEqual(model.child2.child2.n_data, 2)

                    self.assertIsNone(model.child2.child1.child1)
                    self.assertIsNone(model.child2.child1.child2)
                    self.assertIsNone(model.child2.child2.child1)
                    self.assertIsNone(model.child2.child2.child2)

                    self.assertEqual(model.split_dimension, 0)
                    self.assertEqual(model.split_value, 0.5)

                    self.assertEqual(model.child2.split_dimension, 0)
                    self.assertEqual(model.child2.split_value, 1.5)
                else:
                    self.assertEqual(model.get_depth(), 2)
                    self.assertEqual(model.get_n_leaves(), 3)
                    self.assertEqual(model.n_data, 10)

                    self.assertTrue(0.3 < model.best_hyperplane_origin[0] < 0.7)
                    if model.child1.best_hyperplane_origin is not None:
                        self.assertTrue(1.0 < model.child1.best_hyperplane_origin[0] < 2.0)
                        self.assertEqual(model.child1.n_data, 6)
                        self.assertEqual(model.child2.n_data, 4)
                    else:
                        self.assertTrue(1.0 < model.child2.best_hyperplane_origin[0] < 2.0)
                        self.assertEqual(model.child1.n_data, 4)
                        self.assertEqual(model.child2.n_data, 6)

                expected = np.array([0, 0, 1, 1, 0, 0])
                self.assertEqual(model.predict([0, 0.5]), expected[0])
                self.assertEqual(model.predict([0.4, 0.5]), expected[1])
                self.assertEqual(model.predict([0.6, 0.5]), expected[2])
                self.assertEqual(model.predict([1.4, 0.5]), expected[3])
                self.assertEqual(model.predict([1.6, 0.5]), expected[4])
                self.assertEqual(model.predict([100, 0.5]), expected[5])

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_equal(model.predict(data_matrix_transform2(
                        [[0.0, 0.5], [0.4, 0.5], [0.6, 0.5], [1.4, 0.5], [1.6, 0.5], [100, 0.5]])
                    ), expected)

                expected = np.array([[5/6, 1/6], [5/6, 1/6], [1/6, 5/6], [1/6, 5/6], [3/4, 1/4], [3/4, 1/4]])
                assert_array_almost_equal(model.predict_proba([0, 0.5]), np.expand_dims(expected[0], 0))
                assert_array_almost_equal(model.predict_proba([0.4, 0.5]), np.expand_dims(expected[1], 0))
                assert_array_almost_equal(model.predict_proba([0.6, 0.5]), np.expand_dims(expected[2], 0))
                assert_array_almost_equal(model.predict_proba([1.4, 0.5]), np.expand_dims(expected[3], 0))
                assert_array_almost_equal(model.predict_proba([1.6, 0.5]), np.expand_dims(expected[4], 0))
                assert_array_almost_equal(model.predict_proba([100, 0.5]), np.expand_dims(expected[5], 0))

                for data_matrix_transform2 in data_matrix_transforms:
                    assert_array_equal(model.predict_proba(data_matrix_transform2(
                        [[0.0, 0.5], [0.4, 0.5], [0.6, 0.5], [1.4, 0.5], [1.6, 0.5], [100, 0.5]])
                    ), expected)

                if isinstance(model, PerpendicularClassificationTree):
                    # TODO: also add for hyperplane version
                    feature_names = X.columns if isinstance(X, pd.DataFrame) else ['x{}'.format(i) for i in range(X.shape[1])]
                    expected_paths = [
                        [(0, feature_names[0], 0.5, False)],
                        [(0, feature_names[0], 0.5, False)],
                        [(0, feature_names[0], 0.5, True), (0, feature_names[0], 1.5, False)],
                        [(0, feature_names[0], 0.5, True), (0, feature_names[0], 1.5, False)],
                        [(0, feature_names[0], 0.5, True), (0, feature_names[0], 1.5, True)],
                        [(0, feature_names[0], 0.5, True), (0, feature_names[0], 1.5, True)],
                    ]
                    self.assertEqual(model.prediction_paths([0, 0.5]), [expected_paths[0]])
                    self.assertEqual(model.prediction_paths([0.4, 0.5]), [expected_paths[1]])
                    self.assertEqual(model.prediction_paths([0.6, 0.5]), [expected_paths[2]])
                    self.assertEqual(model.prediction_paths([1.4, 0.5]), [expected_paths[3]])
                    self.assertEqual(model.prediction_paths([1.6, 0.5]), [expected_paths[4]])
                    self.assertEqual(model.prediction_paths([100, 0.5]), [expected_paths[5]])

                    for data_matrix_transform2 in data_matrix_transforms:
                        self.assertEqual(model.prediction_paths(data_matrix_transform2(
                            [[0.0, 0.5], [0.4, 0.5], [0.6, 0.5], [1.4, 0.5], [1.6, 0.5], [100, 0.5]])
                        ), expected_paths)

    def test_prune(self):
        models = lambda: create_classification_trees(np.array([10, 10]), 0.9)
        for model1, model2 in zip(models(), models()):
            np.random.seed(666)

            X = np.vstack([
                normal(0, 1, [100, 2]),
                normal(10, 1, [100, 2]),
                normal(14, 1, [100, 2]),
            ])
            y = np.hstack([
                0 * np.ones(100),
                1 * np.ones(100),
                np.minimum(1, randint(0, 3, 100)),  # about two thirds should be 1's
            ])

            # make sure  model1 finds two splits at 5 and 12 and that model2 only finds one (because everything >= 5 has target 1)
            model1.fit(X, y, prune=False)
            model2.fit(X, y, prune=True)
            self.assertEqual(model1.get_depth(), 2)
            self.assertEqual(model1.get_n_leaves(), 3)
            self.assertEqual(model2.get_depth(), 1)
            self.assertEqual(model2.get_n_leaves(), 2)

            # now make sure the node that is the result of pruning two children is consistent
            c1 = model1.child2.child1
            c2 = model1.child2.child2
            c12 = model2.child2
            assert_array_equal(c12.posterior, c1.posterior + c2.posterior - c12.prior)
