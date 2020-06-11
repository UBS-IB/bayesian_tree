import numpy as np
from abc import ABC
from scipy.sparse import csr_matrix, csc_matrix

from bayesian_decision_tree.base import BaseTree


class BasePerpendicularTree(BaseTree, ABC):
    """
    Abstract base class of all Bayesian tree models using splits perpendicular to a single feature axis
    (classification and regression). Performs  medium-level fitting and prediction tasks and outsources
    the low-level work to subclasses.
    """

    def __init__(self, partition_prior, prior, delta, prune, child_type, is_regression, split_precision, level):
        BaseTree.__init__(self, partition_prior, prior, delta, prune, child_type, is_regression, split_precision, level)

    def prediction_paths(self, X):
        """Returns the prediction paths for X.

        Parameters
        ----------
        X : array-like, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix or pandas.DataFrame,
            shape = [n_samples, n_features]

            The input samples.

        Returns
        -------
        prediction_paths : array-like, shape = [n_samples, 4]

            The prediction paths, each row containing the following fields:
            split dimension, split feature name, split value, True if greater than the split value and False otherwise
        """

        # input transformation and checks
        X, _ = self._normalize_data_and_feature_names(X)
        self._ensure_is_fitted(X)

        paths = [[] for i in range(X.shape[0])]
        self._update_prediction_paths(X, np.arange(X.shape[0]), paths)

        return paths

    def _update_prediction_paths(self, X, indices, paths):
        if not self.is_leaf():
            dense = isinstance(X, np.ndarray)
            if not dense and isinstance(X, csr_matrix):
                # column accesses coming up, so convert to CSC sparse matrix format
                X = csc_matrix(X)

            indices1, indices2 = self._compute_child1_and_child2_indices(X, indices, dense)

            if len(indices1) > 0:
                step = (self.split_dimension_, self.split_feature_name_, self.split_value_, False)
                for i in indices1:
                    paths[i].append(step)

            if len(indices2) > 0:
                step = (self.split_dimension_, self.split_feature_name_, self.split_value_, True)
                for i in indices2:
                    paths[i].append(step)

            if len(indices1) > 0 and not self.child1_.is_leaf():
                paths1 = [paths[i] for i in indices1]
                self.child1_._update_prediction_paths(X, indices1, paths1)

            if len(indices2) > 0 and not self.child2_.is_leaf():
                paths2 = [paths[i] for i in indices2]
                self.child2_._update_prediction_paths(X, indices2, paths2)

    @staticmethod
    def _create_merged_paths_array(n_rows):
        return np.zeros((n_rows, 4))

    def _fit(self, X, y, verbose, feature_names, side_name, sort_indices_by_dim=None):
        n_data = sort_indices_by_dim.shape[1] if sort_indices_by_dim is not None else X.shape[0]

        if verbose:
            name = 'level {} {}'.format(self.level, side_name)
            print('Training {} with {:10} data points'.format(name, n_data))

        dense = isinstance(X, np.ndarray)
        if not dense and isinstance(X, csr_matrix):
            # column accesses coming up, so convert to CSC sparse matrix format
            X = csc_matrix(X)

        n_dim = X.shape[1]

        # compute sort indices (only done once at the start)
        if sort_indices_by_dim is None:
            dtype = np.uint16 if n_data < (1 << 16) else np.uint32 if n_data < (1 << 32) else np.uint64
            sort_indices_by_dim = np.zeros(X.shape[::-1], dtype=dtype)
            for dim in range(n_dim):
                X_dim = X[:, dim]
                if not dense:
                    X_dim = self._to_array(X_dim)

                sort_indices_by_dim[dim] = np.argsort(X_dim)

        # compute data likelihood of not splitting and remember it as the best option so far
        prior = self._get_prior(n_data, n_dim)
        log_p_data_no_split = self._compute_log_p_data_no_split(y[sort_indices_by_dim[0]], prior)  # any dim works as the order doesn't matter
        best_log_p_data_split = log_p_data_no_split

        # compute data likelihoods of all possible splits along all data dimensions
        best_split_index = -1       # index of best split
        best_split_dimension = -1   # dimension of best split
        for dim in range(n_dim):
            sort_indices = sort_indices_by_dim[dim]
            X_dim_sorted = X[sort_indices, dim]
            if not dense:
                X_dim_sorted = self._to_array(X_dim_sorted)

            split_indices = 1 + np.where(np.abs(np.diff(X_dim_sorted)) > self.split_precision)[0]  # we can only split between *different* data points
            if len(split_indices) == 0:
                # no split possible along this dimension
                continue

            y_sorted = y[sort_indices]

            # compute data likelihoods of all possible splits along this dimension and find split with highest data likelihood
            log_p_data_split = self._compute_log_p_data_split(y_sorted, prior, n_dim, split_indices)
            i_max = log_p_data_split.argmax()
            if log_p_data_split[i_max] > best_log_p_data_split:
                # remember new best split
                best_log_p_data_split = log_p_data_split[i_max]
                best_split_index = split_indices[i_max]  # data index of best split
                best_split_dimension = dim

        # did we find a split that has a higher likelihood than the no-split likelihood?
        if best_split_index > 0:
            # split data and target to recursively train children
            indices1 = sort_indices_by_dim[best_split_dimension, :best_split_index]
            indices2 = sort_indices_by_dim[best_split_dimension, best_split_index:]

            # compute posteriors of children and priors for further splitting
            prior = self._get_prior(n_data, n_dim)
            prior_child1 = tuple(self._compute_posterior(y[indices1], prior, self.delta)) if self.delta != 0 else prior
            prior_child2 = tuple(self._compute_posterior(y[indices2], prior, self.delta)) if self.delta != 0 else prior

            # store split info, create children and continue training them if there's data left to split
            self.split_dimension_ = best_split_dimension
            self.split_feature_name_ = feature_names[best_split_dimension]
            self.split_value_ = 0.5 * (
                    X[indices1[-1], best_split_dimension]
                    + X[indices2[0], best_split_dimension]
            )
            self.log_p_data_no_split_ = log_p_data_no_split
            self.best_log_p_data_split_ = best_log_p_data_split

            self.child1_ = self.child_type(self.partition_prior, prior_child1, self.delta,
                                           self.prune, self.split_precision, self.level+1)
            self.child2_ = self.child_type(self.partition_prior, prior_child2, self.delta,
                                           self.prune, self.split_precision, self.level+1)
            self.child1_._erase_split_info_base()
            self.child2_._erase_split_info_base()
            self.child1_._erase_split_info()
            self.child2_._erase_split_info()

            # fit children if there is more than one data point (i.e., there is
            # something to split) and if the targets differ (no point otherwise)
            sort_indices_by_dim_1 = sort_indices_by_dim[np.isin(sort_indices_by_dim, indices1)].reshape(n_dim, -1)
            n_data1 = sort_indices_by_dim_1.shape[1]
            y1 = y[indices1]
            if n_data1 > 1 and len(np.unique(y1)) > 1:
                self.child1_._fit(X, y, verbose, feature_names, 'LHS', sort_indices_by_dim_1)
            else:
                self.child1_.posterior_ = self._compute_posterior(y1, prior)
                self.child1_.n_data_ = n_data1

            sort_indices_by_dim_2 = sort_indices_by_dim[np.isin(sort_indices_by_dim, indices2)].reshape(n_dim, -1)
            n_data2 = sort_indices_by_dim_2.shape[1]
            y2 = y[indices2]
            if n_data2 > 1 and len(np.unique(y2)) > 1:
                self.child2_._fit(X, y, verbose, feature_names, 'RHS', sort_indices_by_dim_2)
            else:
                self.child2_.posterior_ = self._compute_posterior(y2, prior)
                self.child2_.n_data_ = n_data2
        else:
            self._erase_split_info_base()
            self._erase_split_info()

        # compute posterior
        self.n_dim_ = n_dim
        self.n_data_ = n_data
        self.posterior_ = self._compute_posterior(y[sort_indices_by_dim[0]], prior)  # any dim works as the order doesn't matter

    def _compute_child1_and_child2_indices(self, X, indices, dense):
        X_split = X[indices, self.split_dimension_]
        if not dense:
            X_split = self._to_array(X_split)

        indices1 = np.where(X_split < self.split_value_)[0]
        indices2 = np.where(X_split >= self.split_value_)[0]

        return indices1, indices2

    def is_leaf(self):
        self._ensure_is_fitted()
        return self.split_value_ is None

    def _update_feature_importance(self, feature_importance):
        if self.is_leaf():
            return
        else:
            log_p_gain = self.best_log_p_data_split_ - self.log_p_data_no_split_
            feature_importance[self.split_dimension_] += log_p_gain
            if self.child1_ is not None:
                self.child1_._update_feature_importance(feature_importance)
                self.child2_._update_feature_importance(feature_importance)

    def _erase_split_info(self):
        self.split_dimension_ = -1
        self.split_value_ = None
        self.split_feature_name_ = None

    @staticmethod
    def _to_array(sparse_array):
        array = sparse_array.toarray()
        return array[0] if array.shape == (1, 1) else array.squeeze()

    def __str__(self):
        if not self.is_fitted():
            return 'Unfitted model'

        return self._str([], self.split_value_, '\u251C', '\u2514', '\u2502', '\u2265', None)

    def _str(self, anchor, parent_split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_left_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_left_child is not None:
            s += anchor_str + ' {}{}: '.format('<' if is_left_child else GEQ, parent_split_value)

        if self.is_leaf():
            s += 'y={}, n={}'.format(self._predict_leaf(), self.n_data_)
            if not self.is_regression:
                s += ', p(y)={}'.format(self._compute_posterior_mean())
        else:
            s += '{}={}'.format(self.split_feature_name_, self.split_value_)

            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_left_child else '  '), VERT_RIGHT])
            s += self.child1_._str(anchor_child1, self.split_value_, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_left_child else '  '), DOWN_RIGHT])
            s += self.child2_._str(anchor_child2, self.split_value_, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s
