from abc import ABC

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from bayesian_decision_tree.base import BaseTree


class BasePerpendicularTree(BaseTree, ABC):
    """
    Abstract base class of all Bayesian tree models using splits perpendicular to a single feature axis
    (classification and regression). Performs  medium-level fitting and prediction tasks and outsources
    the low-level work to subclasses.
    """

    def __init__(self, partition_prior, prior, child_type, is_regression, level):
        BaseTree.__init__(self, partition_prior, prior, child_type, is_regression, level)

        # to be set later
        self.split_dimension = -1
        self.split_value = None
        self.split_feature_name = None

    def _fit(self, X, y, delta, verbose, feature_names):
        if verbose:
            print('Training level {} with {:10} data points'.format(self.level, len(y)))

        dense = isinstance(X, np.ndarray)
        if not dense and isinstance(X, csr_matrix):
            # column accesses coming up, so convert to CSC sparse matrix format
            X = csc_matrix(X)

        # compute data likelihood of not splitting and remember it as the best option so far
        log_p_data_no_split = self._compute_log_p_data_no_split(y)
        best_log_p_data_split = log_p_data_no_split

        # compute data likelihoods of all possible splits along all data dimensions
        n_dim = X.shape[1]
        best_split_index = -1       # index of best split
        best_split_dimension = -1   # dimension of best split
        for dim in range(n_dim):
            X_dim = X[:, dim]
            if not dense:
                X_dim = X_dim.toarray().squeeze()

            sort_indices = np.argsort(X_dim)
            X_dim_sorted = X_dim[sort_indices]

            split_indices = 1 + np.where(np.diff(X_dim_sorted) != 0)[0]  # we can only split between *different* data points
            if len(split_indices) == 0:
                # no split possible along this dimension
                continue

            y_sorted = y[sort_indices]

            # compute data likelihoods of all possible splits along this dimension and find split with highest data likelihood
            log_p_data_split = self._compute_log_p_data_split(y_sorted, split_indices, n_dim)
            i_max = log_p_data_split.argmax()
            if log_p_data_split[i_max] > best_log_p_data_split:
                # remember new best split
                best_log_p_data_split = log_p_data_split[i_max]
                best_split_index = split_indices[i_max]  # data index of best split
                best_split_dimension = dim

        # did we find a split that has a higher likelihood than the no-split likelihood?
        if best_split_index > 0:
            # split data and target to recursively train children
            X_best_split = X[:, best_split_dimension]
            if not dense:
                X_best_split = X_best_split.toarray().squeeze()

            sort_indices = np.argsort(X_best_split)
            y_sorted = y[sort_indices]
            X_sorted = X[sort_indices]
            if not dense:
                # row accesses coming up, so convert to CSR sparse matrix format
                X_sorted = csr_matrix(X_sorted)

            X1 = X_sorted[:best_split_index]
            X2 = X_sorted[best_split_index:]
            y1 = y_sorted[:best_split_index]
            y2 = y_sorted[best_split_index:]

            # compute posteriors of children and priors for further splitting
            prior_child1 = self._compute_posterior(y1, delta)
            prior_child2 = self._compute_posterior(y2, delta)

            # store split info, create children and continue training them if there's data left to split
            self.split_dimension = best_split_dimension
            self.split_feature_name = feature_names[best_split_dimension]
            if dense:
                self.split_value = 0.5 * (
                        X_sorted[best_split_index-1, best_split_dimension]
                        + X_sorted[best_split_index, best_split_dimension]
                )
            else:
                self.split_value = 0.5 * (
                        X_sorted[best_split_index-1, :].toarray()[0][best_split_dimension]
                        + X_sorted[best_split_index, :].toarray()[0][best_split_dimension]
                )
            self.log_p_data_no_split = log_p_data_no_split
            self.best_log_p_data_split = best_log_p_data_split

            self.child1 = self.child_type(self.partition_prior, prior_child1, self.level+1)
            self.child2 = self.child_type(self.partition_prior, prior_child2, self.level+1)

            if X1.shape[0] > 1:
                self.child1._fit(X1, y1, delta, verbose, feature_names)
            else:
                self.child1.posterior = self._compute_posterior(y1)

            if X2.shape[0] > 1:
                self.child2._fit(X2, y2, delta, verbose, feature_names)
            else:
                self.child2.posterior = self._compute_posterior(y2)

        # compute posterior
        self.n_dim = X.shape[1]
        self.posterior = self._compute_posterior(y)

    def _compute_child1_and_child2_indices(self, X, dense):
        X_split = X[:, self.split_dimension]
        if not dense:
            X_split = X_split.toarray().squeeze()

        indices1 = np.where(X_split < self.split_value)[0]
        indices2 = np.where(X_split >= self.split_value)[0]

        return indices1, indices2

    def is_leaf(self):
        self._ensure_is_fitted()
        return self.split_value is None

    def _update_feature_importance(self, feature_importance):
        if self.is_leaf():
            return
        else:
            log_p_gain = self.best_log_p_data_split - self.log_p_data_no_split
            feature_importance[self.split_dimension] += log_p_gain
            if self.child1 is not None:
                self.child1._update_feature_importance(feature_importance)
                self.child2._update_feature_importance(feature_importance)

    def _prune(self):
        depth_start = self.get_depth()
        n_leaves_start = self.get_n_leaves()

        if self.is_leaf():
            return

        if self.child1.is_leaf() and self.child2.is_leaf():
            if self.child1._predict_leaf() == self.child2._predict_leaf():
                # same prediction (class if classification, value if regression) -> no need to split
                self.child1 = None
                self.child2 = None
                self.log_p_data_no_split = None
                self.best_log_p_data_split = None

                self.split_dimension = -1
                self.split_value = None
                self.split_feature_name = None
        else:
            self.child1._prune()
            self.child2._prune()

        if depth_start != self.get_depth() or n_leaves_start != self.get_n_leaves():
            # we did some pruning somewhere down this sub-tree -> prune again
            self._prune()

    def __str__(self):
        return self._str([], self.split_value, '\u2523', '\u2517', '\u2503', '\u2265', None)

    def _str(self, anchor, parent_split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_left_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_left_child is not None:
            s += anchor_str + ' {}{}: '.format('<' if is_left_child else GEQ, parent_split_value)

        if self.is_leaf():
            s += 'y={}'.format(self._predict_leaf())
            if not self.is_regression:
                s += ', p(y)={}'.format(self._predict(None, predict_class=False)[0])
        else:
            s += '{}={}'.format(self.split_feature_name, self.split_value)

            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_left_child else '  '), VERT_RIGHT])
            s += self.child1._str(anchor_child1, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_left_child else '  '), DOWN_RIGHT])
            s += self.child2._str(anchor_child2, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s
