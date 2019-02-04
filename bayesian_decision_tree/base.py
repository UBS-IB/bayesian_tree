from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Node(ABC):
    """
    The base class for all Bayesian decision tree algorithms (classification and regression). Performs all the high-level fitting
    and prediction tasks and outsources the low-level work to the subclasses.
    """

    def __init__(self, partition_prior, prior, posterior, level, child_type, is_regression):
        self.partition_prior = partition_prior
        self.prior = np.array(prior)
        self.posterior = np.array(posterior) if posterior is not None else np.array(prior)
        self.level = level
        self.child_type = child_type
        self.is_regression = is_regression

        # to be set later
        self.split_dimension = -1
        self.split_index = -1
        self.split_value = None
        self.split_feature_name = None
        self.child1 = None
        self.child2 = None

    def fit(self, X, y, delta=0.0, verbose=False):
        """
        Trains this classification or regression tree using the training set (X, y).

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values. In case of binary classification only the
            integers 0 and 1 are permitted. In case of multi-class classification
            only the integers 0, 1, ..., {n_classes-1} are permitted. In case of
            regression all finite float values are permitted.

        delta : float, default=0.0
            Determines the strengthening of the prior as the tree grows deeper,
            see [1].

        verbose : bool, default=False
            Prints fitting progress.

        References
        ----------

        .. [1] https://arxiv.org/abs/1901.03214
        """

        feature_names = None
        if type(X) is pd.DataFrame:
            feature_names = X.columns
            X = X.values

        if type(y) == list:
            y = np.array(y)

        y = y.squeeze()

        if self.level == 0:
            self.check_target(y)

        return self._fit(X, y, delta, verbose, feature_names)

    def _fit(self, X, y, delta, verbose, feature_names):
        if verbose:
            print('Training level {} with {:10} data points'.format(self.level, len(y)))

        # compute data likelihood of not splitting and remember it as the best option so far
        log_p_data_post_best = self.compute_log_p_data_post_no_split(y)

        # compute data likelihoods of all possible splits along all data dimensions
        n_dim = X.shape[1]
        best_split_index = -1       # index of best split
        best_split_dimension = -1   # dimension of best split
        for dim in range(n_dim):
            X_dim = X[:, dim]
            sort_indices = np.argsort(X_dim)
            X_dim_sorted = X_dim[sort_indices]
            split_indices = 1 + np.where(np.diff(X_dim_sorted) != 0)[0]  # we can only split between *different* data points
            if len(split_indices) == 0:
                # no split possible along this dimension
                continue

            y_sorted = y[sort_indices]

            # compute data likelihoods of all possible splits along this dimension and find split with highest data likelihood
            log_p_data_post_split = self.compute_log_p_data_post_split(y_sorted, split_indices, n_dim)
            i_max = log_p_data_post_split.argmax()
            if log_p_data_post_split[i_max] > log_p_data_post_best:
                # remember new best split
                log_p_data_post_best = log_p_data_post_split[i_max]
                best_split_index = split_indices[i_max]  # data index of best split
                best_split_dimension = dim

        # did we find a split that has a higher likelihood than the no-split likelihood?
        if best_split_index > 0:
            # split data and target to recursively train children
            sort_indices = np.argsort(X[:, best_split_dimension])
            X_sorted = X[sort_indices]
            y_sorted = y[sort_indices]
            X1 = X_sorted[:best_split_index]
            X2 = X_sorted[best_split_index:]
            y1 = y_sorted[:best_split_index]
            y2 = y_sorted[best_split_index:]

            # compute posteriors of children and priors for further splitting
            posterior1 = self.compute_posterior(y1)
            posterior2 = self.compute_posterior(y2)
            prior_child1 = self.compute_posterior(y1, delta)
            prior_child2 = self.compute_posterior(y1, delta)

            # store split info, create children and continue training them if there's data left to split
            self.split_dimension = best_split_dimension
            self.split_index = best_split_index
            self.split_feature_name = feature_names[best_split_dimension] if feature_names is not None else None
            self.split_value = 0.5 * (X_sorted[best_split_index-1, best_split_dimension] + X_sorted[best_split_index, best_split_dimension])

            self.child1 = self.child_type(self.partition_prior, prior_child1, posterior1, self.level+1)
            self.child2 = self.child_type(self.partition_prior, prior_child2, posterior2, self.level+1)

            if len(X1) > 1:
                self.child1._fit(X1, y1, delta, verbose, feature_names)
            if len(X2) > 1:
                self.child2._fit(X2, y2, delta, verbose, feature_names)
        else:
            # no, so this node is a leaf node
            self.posterior = self.compute_posterior(y)

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        if type(X) is pd.DataFrame:
            X = X.values

        if type(X) == list:
            X = np.array(X)

        prediction = self._predict(X, predict_class=True)
        if type(prediction) != np.ndarray:
            # if the tree consists of a single leaf only then we have to cast that single float back to an array
            prediction = self._create_predictions_merged(X, True, prediction)

        return prediction

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        if type(X) is pd.DataFrame:
            X = X.values

        return self._predict(X, predict_class=False)

    def _predict(self, X, predict_class):
        if self.is_regression and not predict_class:
            # probability prediction for regressions makes no sense
            raise ValueError('Cannot predict probabilities for regression trees')

        if self.is_leaf():
            prediction = self._predict_leaf() if predict_class else self.compute_posterior_mean().reshape(1, -1)
            predictions = self._create_predictions_merged(X, predict_class, prediction)
            predictions[:] = prediction
            return predictions
        else:
            # query children and then re-assemble

            # convert X to a correctly shaped numpy array
            if np.isscalar(X):
                X = np.array(X).reshape(-1, 1)
            elif type(X) is list:
                X = np.array(X).reshape(-1, 1)
            elif len(X.shape) == 1:
                X = X.reshape(-1, 1)

            # query both children, let them predict their side, and then re-assemble
            indices1 = np.where(X[:, self.split_dimension] < self.split_value)[0]
            indices2 = np.where(X[:, self.split_dimension] >= self.split_value)[0]
            predictions_merged = None
            if len(indices1) > 0:
                predictions1 = self.child1._predict(X[indices1], predict_class)
                predictions_merged = self._create_predictions_merged(X, predict_class, predictions1)
                predictions_merged[indices1] = predictions1

            if len(indices2) > 0:
                predictions2 = self.child2._predict(X[indices2], predict_class)
                if predictions_merged is None:
                    predictions_merged = self._create_predictions_merged(X, predict_class, predictions2)
                predictions_merged[indices2] = predictions2

            return predictions_merged

    @staticmethod
    def _create_predictions_merged(X, predict_class, predictions_child):
        # class predictions: 1D array
        # probability predictions: 2D array
        len_X = len(X) if type(X) == np.ndarray else 1
        return np.zeros(len_X) if predict_class else np.zeros((len_X, predictions_child.shape[1]))

    def depth_and_leaves(self):
        """Compute and return the tree depth and the number of leaves.

        Returns
        -------
        depth_and_leaves : tuple of (int, int)
            The tree depth and the number of leaves.
        """

        return self._update_depth_and_leaves(0, 0)

    def _update_depth_and_leaves(self, depth, leaves):
        if self.is_leaf():
            return max(depth, self.level), leaves+1
        else:
            if self.child1 is not None:
                depth, leaves = self.child1._update_depth_and_leaves(depth, leaves)

            if self.child2 is not None:
                depth, leaves = self.child2._update_depth_and_leaves(depth, leaves)

        return depth, leaves

    def is_leaf(self):
        return self.split_index == -1

    @abstractmethod
    def check_target(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_no_split(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_split(self, y, split_indices, n_dim):
        pass

    @abstractmethod
    def compute_posterior(self, y, delta=1):
        pass

    @abstractmethod
    def compute_posterior_mean(self):
        pass

    @abstractmethod
    def _predict_leaf(self):
        pass

    def __str__(self):
        return self._str([], self.split_value, None)

    def __str__(self):
        return self._str([], self.split_value, '\u2523', '\u2517', '\u2503', '\u2265', None)

    def _str(self, anchor, parent_split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_first_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_first_child is not None:
            s += anchor_str + ' {}{}: '.format('<' if is_first_child else GEQ, parent_split_value)

        if self.is_leaf():
            s += 'y={}'.format(self._predict_leaf())
            if not self.is_regression:
                s += ', p(y)={}'.format(self.predict_proba(parent_split_value)[0])
        else:
            split_feature_name = self.split_feature_name if self.split_feature_name is not None else 'x{}'.format(self.split_dimension)
            s += '{}={}'.format(split_feature_name, self.split_value)

            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_first_child else '  '), VERT_RIGHT])
            s += self.child1._str(anchor_child1, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_first_child else '  '), DOWN_RIGHT])
            s += self.child2._str(anchor_child2, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s

    def __repr__(self):
        return self.__str__()
