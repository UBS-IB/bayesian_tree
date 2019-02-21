from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csc_matrix, csr_matrix

from bayesian_decision_tree.classification import BinaryClassificationNode
from tests.unit.helper import data_matrix_transforms


class BinaryClassificationNodeTest(TestCase):
    def test_one_partition(self):
        for data_matrix_transform in data_matrix_transforms:
            Xy = np.array([
                [0.0, 0],
                [0.1, 1],
                [0.9, 0],
                [1.0, 1],
                [1.0, 0],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            root = BinaryClassificationNode(0.5, (1, 1))
            root.fit(X, y)
            print(root)

            self.assertEqual(root.depth_and_leaves(), (0, 1))

            self.assertIsNone(root.child1)
            self.assertIsNone(root.child2)

            self.assertEqual(root.split_dimension, -1)
            self.assertEqual(root.split_value, None)

            expected = np.array([0, 0, 0, 0])
            self.assertEqual(root.predict(0.0), expected[0])
            self.assertEqual(root.predict(0.49), expected[1])
            self.assertEqual(root.predict(0.51), expected[2])
            self.assertEqual(root.predict(1.0), expected[3])

            assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), expected)
            assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), expected)

            assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), expected)
            assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), expected)

            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]]).to_sparse()), expected)
            assert_array_equal(root.predict(csc_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_equal(root.predict(csr_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)

            expected = np.array([[(1+3)/7, (1+2)/7], [(1+3)/7, (1+2)/7], [(1+3)/7, (1+2)/7], [(1+3)/7, (1+2)/7], ])
            assert_array_almost_equal(root.predict_proba(0.0), np.expand_dims(expected[0], 0))
            assert_array_almost_equal(root.predict_proba(0.49), np.expand_dims(expected[1], 0))
            assert_array_almost_equal(root.predict_proba(0.51), np.expand_dims(expected[2], 0))
            assert_array_almost_equal(root.predict_proba(1.0), np.expand_dims(expected[3], 0))
            assert_array_almost_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.0])), expected)
            assert_array_almost_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.0]])), expected)

            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]]).to_sparse()), expected)
            assert_array_almost_equal(root.predict_proba(csc_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_almost_equal(root.predict_proba(csr_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)

    def test_two_partitions(self):
        for data_matrix_transform in data_matrix_transforms:
            Xy = np.array([
                [0.0, 0],
                [0.1, 0],
                [0.9, 1],
                [1.0, 1],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            root = BinaryClassificationNode(0.5, (1, 1))
            root.fit(X, y)
            print(root)

            self.assertEqual(root.depth_and_leaves(), (1, 2))

            self.assertIsNotNone(root.child1)
            self.assertIsNone(root.child1.child1)
            self.assertIsNone(root.child1.child2)

            self.assertIsNotNone(root.child2)
            self.assertIsNone(root.child2.child1)
            self.assertIsNone(root.child2.child2)

            self.assertEqual(root.split_dimension, 0)
            self.assertEqual(root.split_value, 0.5)

            expected = np.array([0, 0, 1, 1])
            self.assertEqual(root.predict(0.0), expected[0])
            self.assertEqual(root.predict(0.49), expected[1])
            self.assertEqual(root.predict(0.51), expected[2])
            self.assertEqual(root.predict(1.0), expected[3])

            assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), expected)
            assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), expected)

            assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), expected)
            assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), expected)

            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]]).to_sparse()), expected)
            assert_array_equal(root.predict(csc_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_equal(root.predict(csr_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)

            expected = np.array([[(1+2)/4, 1/4], [(1+2)/4, 1/4], [1/4, (1+2)/4], [1/4, (1+2)/4]])
            assert_array_almost_equal(root.predict_proba(0.0), np.expand_dims(expected[0], 0))
            assert_array_almost_equal(root.predict_proba(0.49), np.expand_dims(expected[1], 0))
            assert_array_almost_equal(root.predict_proba(0.51), np.expand_dims(expected[2], 0))
            assert_array_almost_equal(root.predict_proba(1.0), np.expand_dims(expected[3], 0))
            assert_array_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.0])), expected)
            assert_array_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.0]])), expected)

            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.0]]).to_sparse()), expected)
            assert_array_almost_equal(root.predict_proba(csc_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)
            assert_array_almost_equal(root.predict_proba(csr_matrix([[0.0], [0.49], [0.51], [1.0]])), expected)

    def test_three_partitions(self):
        for data_matrix_transform in data_matrix_transforms:
            Xy = np.array([
                [0.0, 0],
                [0.1, 0],
                [0.2, 0],
                [0.3, 0],
                [0.7, 1],
                [0.9, 1],
                [1.0, 1],
                [2.0, 0],
                [2.1, 0],
                [2.2, 0],
                [2.3, 0],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            root = BinaryClassificationNode(0.9, (1, 1))
            root.fit(X, y)
            print(root)

            self.assertEqual(root.depth_and_leaves(), (2, 3))

            self.assertIsNotNone(root.child1)
            self.assertIsNone(root.child1.child1)
            self.assertIsNone(root.child1.child2)

            self.assertIsNotNone(root.child2)
            self.assertIsNotNone(root.child2.child1)
            self.assertIsNotNone(root.child2.child2)

            self.assertIsNone(root.child2.child1.child1)
            self.assertIsNone(root.child2.child1.child2)
            self.assertIsNone(root.child2.child2.child1)
            self.assertIsNone(root.child2.child2.child2)

            self.assertEqual(root.split_dimension, 0)
            self.assertEqual(root.split_value, 0.5)

            self.assertEqual(root.child2.split_dimension, 0)
            self.assertEqual(root.child2.split_value, 1.5)

            expected = np.array([0, 0, 1, 1, 0, 0])
            self.assertEqual(root.predict(0.0), expected[0])
            self.assertEqual(root.predict(0.49), expected[1])
            self.assertEqual(root.predict(0.51), expected[2])
            self.assertEqual(root.predict(1.49), expected[3])
            self.assertEqual(root.predict(1.51), expected[4])
            self.assertEqual(root.predict(100), expected[5])

            assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.49, 1.51, 100]), expected)
            assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.49], [1.51], [100]]), expected)

            assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.49, 1.51, 100])), expected)
            assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)

            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
            assert_array_equal(root.predict(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.49], [1.51], [100]]).to_sparse()), expected)
            assert_array_equal(root.predict(csr_matrix([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
            assert_array_equal(root.predict(csc_matrix([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)

            expected = np.array([[(1+4)/6, 1/6], [(1+4)/6, 1/6], [1/5, (1+3)/5], [1/5, (1+3)/5], [(1+4)/6, 1/6], [(1+4)/6, 1/6]])
            assert_array_almost_equal(root.predict_proba(0.0), np.expand_dims(expected[0], 0))
            assert_array_almost_equal(root.predict_proba(0.49), np.expand_dims(expected[1], 0))
            assert_array_almost_equal(root.predict_proba(0.51), np.expand_dims(expected[2], 0))
            assert_array_almost_equal(root.predict_proba(1.49), np.expand_dims(expected[3], 0))
            assert_array_almost_equal(root.predict_proba(1.51), np.expand_dims(expected[4], 0))
            assert_array_almost_equal(root.predict_proba(100), np.expand_dims(expected[4], 0))
            assert_array_almost_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.49, 1.51, 100])), expected)
            assert_array_almost_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
            assert_array_almost_equal(root.predict_proba(csc_matrix([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)

            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
            assert_array_almost_equal(root.predict_proba(pd.DataFrame(data=[[0.0], [0.49], [0.51], [1.49], [1.51], [100]]).to_sparse()), expected)
            assert_array_almost_equal(root.predict_proba(csc_matrix([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
            assert_array_almost_equal(root.predict_proba(csr_matrix([[0.0], [0.49], [0.51], [1.49], [1.51], [100]])), expected)
