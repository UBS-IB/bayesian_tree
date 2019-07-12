from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.base import BaseEstimator


class BaseTree(ABC, BaseEstimator):
    """
    Abstract base class of all Bayesian decision tree models (classification and regression). Performs all
    high-level fitting and prediction tasks and outsources the medium- and low-level work to subclasses.

    Implementation note: This class hierarchy is diamond-shaped: The four concrete model classes each
    inherit from two superclasses which in turn inherit from this class.
    """

    def __init__(self, partition_prior, prior, child_type, is_regression, level):
        if not isinstance(prior, np.ndarray):
            raise TypeError('\'prior\' must be a numpy array')

        self.partition_prior = partition_prior
        self.prior = prior
        self.child_type = child_type
        self.is_regression = is_regression
        self.level = level

        # to be set later
        self.n_dim = None
        self.posterior = None
        self.n_data = None
        self.child1 = None
        self.child2 = None
        self.log_p_data_no_split = None
        self.best_log_p_data_split = None

    def fit(self, X, y, delta=0.0, prune=False, verbose=False, feature_names=None):
        """
        Trains this classification or regression tree using the training set (X, y).

        Parameters
        ----------
        X : array-like, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, pandas.DataFrame or pandas.SparseDataFrame, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values. In case of binary classification only the
            integers 0 and 1 are permitted. In case of multiclass classification
            only the integers 0, 1, ..., {n_classes-1} are permitted. In case of
            regression all finite float values are permitted.

        delta : float, default=0.0
            Determines the strengthening of the prior as the tree grows deeper,
            see [1]. Must be a value between 0.0 and 1.0.

        prune : boolean, default=False
            Prunes the tree after fitting if `True` by removing all splits that don't add information,
            i.e., where the predictions of both children are identical. It's usually sensible to set
            this to `True` in the classification case if you're only interested in class predictions
            (`predict(X)`), but it makes sense to set it to `False` if you're looking for class
            probabilities (`predict_proba(X)`). It can safely be set to 'True' in the regression case
            because it will only merge children if their predictions are identical.

        verbose : bool, default=False
            Prints fitting progress.

        feature_names: array-lie, shape = [n_features]
            An optional sequence of feature names. If not provided then 'x0', 'x1', ... is used
            if X is a matrix, or the column headers if X is a DataFrame.

        References
        ----------

        .. [1] https://arxiv.org/abs/1901.03214
        """

        # validation
        if self.level == 0:
            self._check_target(y)

        if delta < 0.0 or delta > 1.0:
            raise ValueError('Delta must be between 0.0 and 1.0 but was {}.'.format(delta))

        # input transformation
        X, feature_names = self._normalize_data_and_feature_names(X, feature_names)

        if isinstance(y, list):
            y = np.array(y)

        y = y.squeeze()

        # fit
        self._fit(X, y, delta, verbose, feature_names)

        if prune:
            self._prune()

        return self

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, pandas.DataFrame or pandas.SparseDataFrame, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # input transformation and checks
        X, _ = self._normalize_data_and_feature_names(X)
        self._ensure_is_fitted(X)

        prediction = self._predict(X, predict_class=True)
        if not isinstance(prediction, np.ndarray):
            # if the tree consists of a single leaf only then we have to cast that single float back to an array
            prediction = self._create_merged_predictions_array(X, True, prediction)

        return prediction

    def feature_importance(self):
        """
        Compute and return feature importance of this tree after having fitted it to data. Feature
        importance for a given feature dimension is defined as the sum of all increases in the
        marginal data log-likelihood across splits of that dimension. Finally, the feature
        importance vector is normalized to sum to 1.

        Returns
        -------
        feature_importance: array of floats
            The feature importance.
        """

        self._ensure_is_fitted()

        feature_importance = np.zeros(self.n_dim)
        self._update_feature_importance(feature_importance)
        feature_importance /= feature_importance.sum()

        return feature_importance

    def _predict(self, X, predict_class):
        if self.is_leaf():
            prediction = self._predict_leaf() if predict_class else self._compute_posterior_mean().reshape(1, -1)
            predictions = self._create_merged_predictions_array(X, predict_class, prediction)
            predictions[:] = prediction
            return predictions
        else:
            # query children and then re-assemble

            dense = isinstance(X, np.ndarray)
            if not dense and isinstance(X, csr_matrix):
                # column accesses coming up, so convert to CSC sparse matrix format
                X = csc_matrix(X)

            # query both children, let them predict their side, and then re-assemble
            indices1, indices2 = self._compute_child1_and_child2_indices(X, dense)

            predictions_merged = None

            if len(indices1) > 0:
                X1 = X[indices1]
                predictions1 = self.child1._predict(X1, predict_class)
                predictions_merged = self._create_merged_predictions_array(X, predict_class, predictions1)
                predictions_merged[indices1] = predictions1

            if len(indices2) > 0:
                X2 = X[indices2]
                predictions2 = self.child2._predict(X2, predict_class)
                if predictions_merged is None:
                    predictions_merged = self._create_merged_predictions_array(X, predict_class, predictions2)

                predictions_merged[indices2] = predictions2

            return predictions_merged

    @abstractmethod
    def _update_feature_importance(self, feature_importance):
        pass

    @staticmethod
    def _normalize_data_and_feature_names(X, feature_names=None):
        if isinstance(X, pd.SparseDataFrame):
            # we cannot directly access the sparse underlying data,
            # but we can convert it to a sparse scipy matrix
            if feature_names is None:
                feature_names = X.columns

            X = csc_matrix(X.to_coo())
        elif isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns

            X = X.values
        else:
            if isinstance(X, list):
                X = np.array(X)
            elif np.isscalar(X):
                X = np.array([X])

            if X.ndim == 1:
                X = np.expand_dims(X, 0)

            if feature_names is None:
                feature_names = ['x{}'.format(i) for i in range(X.shape[1])]

        if X.ndim != 2:
            raise ValueError('X should have 2 dimensions but has {}'.format(X.ndim))

        return X, feature_names

    def _ensure_is_fitted(self, X=None):
        if self.posterior is None:
            raise ValueError('Cannot predict on an untrained model; call .fit() first')

        if X is not None and X.shape[1] != self.n_dim:
            raise ValueError('Bad input dimensions: Expected {}, got {}'.format(self.n_dim, X.shape[1]))

    @staticmethod
    def _create_merged_predictions_array(X, predict_class, predictions_child):
        # class predictions: 1D array
        # probability predictions: 2D array
        len_X = 1 if X is None or np.isscalar(X) else X.shape[0]
        return np.zeros(len_X) if predict_class else np.zeros((len_X, predictions_child.shape[1]))

    def get_depth(self):
        """Computes and returns the tree depth.

        Returns
        -------
        depth : int
            The tree depth.
        """

        return self._update_depth(0)

    def get_n_leaves(self):
        """Computes and returns the total number of leaves of this tree.

        Returns
        -------
        n_leaves : int
            The number of leaves.
        """

        return self._update_n_leaves(0)

    def _update_depth(self, depth):
        if self.is_leaf():
            return max(depth, self.level)
        else:
            if self.child1 is not None:
                depth = self.child1._update_depth(depth)
                depth = self.child2._update_depth(depth)

            return depth

    def _update_n_leaves(self, n_leaves):
        if self.is_leaf():
            return n_leaves+1
        else:
            if self.child1 is not None:
                n_leaves = self.child1._update_n_leaves(n_leaves)
                n_leaves = self.child2._update_n_leaves(n_leaves)

            return n_leaves

    @abstractmethod
    def is_leaf(self):
        pass

    @abstractmethod
    def _check_target(self, y):
        pass

    @abstractmethod
    def _compute_log_p_data_no_split(self, y):
        pass

    @abstractmethod
    def _compute_log_p_data_split(self, y, split_indices, n_dim):
        pass

    @abstractmethod
    def _compute_posterior(self, y, delta=1):
        pass

    @abstractmethod
    def _compute_posterior_mean(self):
        pass

    @abstractmethod
    def _predict_leaf(self):
        pass

    @abstractmethod
    def _fit(self, X, y, delta, verbose, feature_names):
        pass

    @abstractmethod
    def _prune(self):
        pass

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _compute_child1_and_child2_indices(self, X, dense):
        pass
