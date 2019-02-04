from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bayesian_decision_tree.classification import MultiClassificationNode


class MultiClassificationNodeTest(TestCase):
    def test_one_partition(self):
        Xy = np.array([
            [0.0, 0],
            [0.1, 1],
            [0.9, 0],
            [1.0, 1],
            [1.0, 0],
        ])

        root = MultiClassificationNode(0.5, (1, 1))
        root.fit(Xy[:, :-1], Xy[:, -1])
        print(root)

        self.assertEqual(root.depth_and_leaves(), (0, 1))

        self.assertIsNone(root.child1)
        self.assertIsNone(root.child2)

        self.assertEqual(root.split_dimension, -1)
        self.assertEqual(root.split_value, None)
        self.assertEqual(root.split_index, -1)

        self.assertEqual(root.predict(0.0), 0)
        self.assertEqual(root.predict(0.49), 0)
        self.assertEqual(root.predict(0.51), 0)
        self.assertEqual(root.predict(1.0), 0)

        assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), np.array([0, 0, 0, 0]))
        assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), np.array([0, 0, 0, 0]))

        assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), np.array([0, 0, 0, 0]))
        assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([0, 0, 0, 0]))

        assert_array_almost_equal(root.predict_proba(0.0), np.array([[(1+3)/7, (1+2)/7]]))
        assert_array_almost_equal(root.predict_proba(0.49), np.array([[(1+3)/7, (1+2)/7]]))
        assert_array_almost_equal(root.predict_proba(0.51), np.array([[(1+3)/7, (1+2)/7]]))
        assert_array_almost_equal(root.predict_proba(1.0), np.array([[(1+3)/7, (1+2)/7]]))
        assert_array_almost_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.0])), np.array([
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
        ]))
        assert_array_almost_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
            [(1+3)/7, (1+2)/7],
        ]))

    def test_two_partitions(self):
        Xy = np.array([
            [0.0, 0],
            [0.1, 0],
            [0.9, 1],
            [1.0, 1],
        ])

        root = MultiClassificationNode(0.5, (1, 1))
        root.fit(Xy[:, :-1], Xy[:, -1])
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
        self.assertEqual(root.split_index, 2)

        self.assertEqual(root.predict(0.0), 0)
        self.assertEqual(root.predict(0.49), 0)
        self.assertEqual(root.predict(0.51), 1)
        self.assertEqual(root.predict(1.0), 1)

        assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), np.array([0, 0, 1, 1]))
        assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), np.array([0, 0, 1, 1]))

        assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), np.array([0, 0, 1, 1]))
        assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([0, 0, 1, 1]))

        assert_array_equal(root.predict_proba(0.0), np.array([[(1+2)/4, 1/4]]))
        assert_array_equal(root.predict_proba(0.49), np.array([[(1+2)/4, 1/4]]))
        assert_array_equal(root.predict_proba(0.51), np.array([[1/4, (1+2)/4]]))
        assert_array_equal(root.predict_proba(1.0), np.array([[1/4, (1+2)/4]]))
        assert_array_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.0])), np.array([
            [(1+2)/4, 1/4],
            [(1+2)/4, 1/4],
            [1/4, (1+2)/4],
            [1/4, (1+2)/4]
        ]))
        assert_array_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([
            [(1+2)/4, 1/4],
            [(1+2)/4, 1/4],
            [1/4, (1+2)/4],
            [1/4, (1+2)/4]
        ]))

    def test_three_partitions(self):
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

        root = MultiClassificationNode(0.9, (1, 1))
        root.fit(Xy[:, :-1], Xy[:, -1])
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
        self.assertEqual(root.split_index, 4)

        self.assertEqual(root.child2.split_dimension, 0)
        self.assertEqual(root.child2.split_value, 1.5)
        self.assertEqual(root.child2.split_index, 3)

        self.assertEqual(root.predict(0.0), 0)
        self.assertEqual(root.predict(0.49), 0)
        self.assertEqual(root.predict(0.51), 1)
        self.assertEqual(root.predict(1.49), 1)
        self.assertEqual(root.predict(1.51), 0)
        self.assertEqual(root.predict(100), 0)

        assert_array_equal(root.predict([0.0, 0.49, 0.51, 1.0]), np.array([0, 0, 1, 1]))
        assert_array_equal(root.predict([[0.0], [0.49], [0.51], [1.0]]), np.array([0, 0, 1, 1]))

        assert_array_equal(root.predict(np.array([0.0, 0.49, 0.51, 1.0])), np.array([0, 0, 1, 1]))
        assert_array_equal(root.predict(np.array([[0.0], [0.49], [0.51], [1.0]])), np.array([0, 0, 1, 1]))

        assert_array_almost_equal(root.predict_proba(0.0), np.array([[(1+4)/6, 1/6]]))
        assert_array_almost_equal(root.predict_proba(0.49), np.array([[(1+4)/6, 1/6]]))
        assert_array_almost_equal(root.predict_proba(0.51), np.array([[1/5, (1+3)/5]]))
        assert_array_almost_equal(root.predict_proba(1.49), np.array([[1/5, (1+3)/5]]))
        assert_array_almost_equal(root.predict_proba(1.51), np.array([[(1+4)/6, 1/6]]))
        assert_array_almost_equal(root.predict_proba(np.array([0.0, 0.49, 0.51, 1.49, 1.51])), np.array([
            [(1+4)/6, 1/6],
            [(1+4)/6, 1/6],
            [1/5, (1+3)/5],
            [1/5, (1+3)/5],
            [(1+4)/6, 1/6]
        ]))
        assert_array_almost_equal(root.predict_proba(np.array([[0.0], [0.49], [0.51], [1.49], [1.51]])), np.array([
            [(1+4)/6, 1/6],
            [(1+4)/6, 1/6],
            [1/5, (1+3)/5],
            [1/5, (1+3)/5],
            [(1+4)/6, 1/6]
        ]))
