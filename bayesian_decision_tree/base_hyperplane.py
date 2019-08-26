from abc import ABC

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.sparse import csc_matrix, csr_matrix

from bayesian_decision_tree.base import BaseTree
from bayesian_decision_tree.hyperplane_optimization import HyperplaneOptimizationFunction, ScipyOptimizer


class BaseHyperplaneTree(BaseTree, ABC):
    """
    Abstract base class of all Bayesian decision tree models using arbitrarily-oriented hyperplane splits
    (classification and regression). Performs medium-level fitting and prediction tasks and outsources
    the low-level work to subclasses.
    """

    def __init__(self, partition_prior, prior, child_type, is_regression, optimizer, level):
        BaseTree.__init__(self, partition_prior, prior, child_type, is_regression, level)

        if optimizer is None:
            # default to 'Differential Evolution' which works well and is reasonably fast
            optimizer = ScipyOptimizer(DifferentialEvolutionSolver, 666)

        self.optimizer = optimizer

        # to be set later
        self.best_hyperplane_normal = None
        self.best_hyperplane_origin = None

    def _fit(self, X, y, delta, verbose, feature_names):
        if verbose:
            print('Training level {} with {:10} data points'.format(self.level, len(y)))

        dense = isinstance(X, np.ndarray)
        if not dense and isinstance(X, csr_matrix):
            # column accesses coming up, so convert to CSC sparse matrix format
            X = csc_matrix(X)

        log_p_data_no_split = self._compute_log_p_data_no_split(y)

        # the function to optimize (depends on X and y, hence we need to instantiate it for every data set anew)
        optimization_function = HyperplaneOptimizationFunction(
            X,
            y,
            self._compute_log_p_data_split,
            log_p_data_no_split,
            self.optimizer.search_space_is_unit_hypercube)

        # create and run optimizer
        self.optimizer.solve(optimization_function)

        self.optimization_function = optimization_function

        # retrieve best hyperplane split from optimization function
        if optimization_function.best_hyperplane_normal is not None:
            # split data and target to recursively train children
            projections = X @ optimization_function.best_hyperplane_normal \
                          - np.dot(optimization_function.best_hyperplane_normal, optimization_function.best_hyperplane_origin)
            indices1 = np.where(projections < 0)[0]
            indices2 = np.where(projections >= 0)[0]

            if len(indices1) > 0 and len(indices2) > 0:
                """
                Note: The reason why indices1 or indices2 could be empty is that the optimizer might find a
                'split' that puts all data one one side and nothing on the other side, and that 'split' has
                a higher log probability than 'log_p_data_no_split' because of the partition prior
                overwhelming the data likelihoods (which are of course identical between the 'all data' and
                the 'everything on one side split' scenarios)s.
                """
                X1 = X[indices1]
                X2 = X[indices2]
                y1 = y[indices1]
                y2 = y[indices2]

                # compute posteriors of children and priors for further splitting
                prior_child1 = self._compute_posterior(y1, 0)
                prior_child2 = self._compute_posterior(y2, 0)

                # store split info, create children and continue training them if there's data left to split
                self.best_hyperplane_normal = optimization_function.best_hyperplane_normal
                self.best_hyperplane_origin = optimization_function.best_hyperplane_origin

                self.log_p_data_no_split = optimization_function.log_p_data_no_split
                self.best_log_p_data_split = optimization_function.best_log_p_data_split

                self.child1 = self.child_type(self.partition_prior, prior_child1, self.optimizer, self.level + 1)
                self.child2 = self.child_type(self.partition_prior, prior_child2, self.optimizer, self.level + 1)

            # fit children if there is more than one data point (i.e., there is
            # something to split) and if the targets differ (no point otherwise)
                if X1.shape[0] > 1 and len(np.unique(y1)) > 1:
                    self.child1._fit(X1, y1, delta, verbose, feature_names)
                else:
                    self.child1.posterior = self._compute_posterior(y1)
                    self.child1.n_data = X1.shape[0]

                if X2.shape[0] > 1 and len(np.unique(y2)) > 1:
                    self.child2._fit(X2, y2, delta, verbose, feature_names)
                else:
                    self.child2.posterior = self._compute_posterior(y2)
                    self.child2.n_data = X2.shape[0]

        # compute posterior
        self.n_dim = X.shape[1]
        self.n_data = X.shape[0]
        self.posterior = self._compute_posterior(y)

    def _compute_child1_and_child2_indices(self, X, dense):
        projections = X @ self.best_hyperplane_normal - np.dot(self.best_hyperplane_normal, self.best_hyperplane_origin)
        indices1 = np.where(projections < 0)[0]
        indices2 = np.where(projections >= 0)[0]

        return indices1, indices2

    def is_leaf(self):
        self._ensure_is_fitted()
        return self.best_hyperplane_normal is None

    def feature_importance(self):
        self._ensure_is_fitted()

        feature_importance = np.zeros(self.n_dim)
        self._update_feature_importance(feature_importance)
        feature_importance /= feature_importance.sum()

        return feature_importance

    def _update_feature_importance(self, feature_importance):
        if self.is_leaf():
            return
        else:
            log_p_gain = self.best_log_p_data_split - self.log_p_data_no_split
            hyperplane_normal = self.best_hyperplane_normal

            # the more the normal vector is oriented along a given dimension's axis the more
            # important that dimension is, so weight log_p_gain with hyperplane_normal[i_dim]
            feature_importance += log_p_gain * hyperplane_normal
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

                self.best_hyperplane_normal = None
                self.best_hyperplane_origin = None
        else:
            self.child1._prune()
            self.child2._prune()

        if depth_start != self.get_depth() or n_leaves_start != self.get_n_leaves():
            # we did some pruning somewhere down this sub-tree -> prune again
            self._prune()

    def __str__(self):
        if self.posterior is None:
            return 'Unfitted model'

        return self._str([], '\u251C', '\u2514', '\u2502', '\u2265', None)

    def _str(self, anchor, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_back_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_back_child is not None:
            s += anchor_str + ' {:5s}: '.format('back' if is_back_child else 'front')

        if self.is_leaf():
            s += 'y={}, n={}'.format(self._predict_leaf(), self.n_data)
            if not self.is_regression:
                s += ', p(y)={}'.format(self._compute_posterior_mean())
        else:
            s += 'HP(origin={}, normal={})'.format(self.best_hyperplane_origin, self.best_hyperplane_normal)

            # 'back' child (the child that is on the side of the hyperplane opposite to the normal vector, or projection < 0)
            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_back_child else '  '), VERT_RIGHT])
            s += self.child1._str(anchor_child1, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            # 'front' child (the child that is on same side of the hyperplane as the normal vector, or projection >= 0)
            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_back_child else '  '), DOWN_RIGHT])
            s += self.child2._str(anchor_child2, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s
