import numpy as np
from abc import ABC
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

    def __init__(self, partition_prior, prior, delta, prune, child_type, is_regression, optimizer, split_precision, level):
        BaseTree.__init__(self, partition_prior, prior, delta, prune, child_type, is_regression, split_precision, level)

        self.optimizer = optimizer

    def _fit(self, X, y, verbose, feature_names, side_name):
        n_data = X.shape[0]
        n_dim = X.shape[1]
        prior = self._get_prior(n_data, n_dim)

        if verbose:
            name = 'level {} {}'.format(self.level, side_name)
            print('Training {} with {:10} data points'.format(name, n_data))

        dense = isinstance(X, np.ndarray)
        if not dense and isinstance(X, csr_matrix):
            # column accesses coming up, so convert to CSC sparse matrix format
            X = csc_matrix(X)

        log_p_data_no_split = self._compute_log_p_data_no_split(y, prior)

        optimizer = self.optimizer
        if optimizer is None:
            # default to 'Differential Evolution' which works well and is reasonably fast
            optimizer = ScipyOptimizer(DifferentialEvolutionSolver, 666)

        # the function to optimize (depends on X and y, hence we need to instantiate it for every data set anew)
        optimization_function = HyperplaneOptimizationFunction(
            X,
            y,
            prior,
            self._compute_log_p_data_split,
            log_p_data_no_split,
            optimizer.search_space_is_unit_hypercube,
            self.split_precision)

        # create and run optimizer
        optimizer.solve(optimization_function)

        self.optimization_function = optimization_function

        # retrieve best hyperplane split from optimization function
        self._erase_split_info_base()
        self._erase_split_info()
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

                n_data1 = X1.shape[0]
                n_data2 = X2.shape[0]

                # compute posteriors of children and priors for further splitting
                prior_child1 = self._compute_posterior(y1, prior, delta=0)
                prior_child2 = self._compute_posterior(y2, prior, delta=0)

                # store split info, create children and continue training them if there's data left to split
                self.best_hyperplane_normal_ = optimization_function.best_hyperplane_normal
                self.best_hyperplane_origin_ = optimization_function.best_hyperplane_origin

                self.log_p_data_no_split_ = optimization_function.log_p_data_no_split
                self.best_log_p_data_split_ = optimization_function.best_log_p_data_split

                self.child1_ = self.child_type(self.partition_prior, prior_child1, self.delta,
                                               self.prune, optimizer, self.split_precision, self.level+1)
                self.child2_ = self.child_type(self.partition_prior, prior_child2, self.delta,
                                               self.prune, optimizer, self.split_precision, self.level+1)
                self.child1_._erase_split_info_base()
                self.child2_._erase_split_info_base()
                self.child1_._erase_split_info()
                self.child2_._erase_split_info()

                # fit children if there is more than one data point (i.e., there is
                # something to split) and if the targets differ (no point otherwise)
                if n_data1 > 1 and len(np.unique(y1)) > 1:
                    self.child1_._fit(X1, y1, verbose, feature_names, 'back ')
                else:
                    self.child1_.posterior_ = self._compute_posterior(y1, prior)
                    self.child1_.n_data_ = n_data1

                if n_data2 > 1 and len(np.unique(y2)) > 1:
                    self.child2_._fit(X2, y2, verbose, feature_names, 'front')
                else:
                    self.child2_.posterior_ = self._compute_posterior(y2, prior)
                    self.child2_.n_data_ = n_data2

        # compute posterior
        self.n_dim_ = X.shape[1]
        self.n_data_ = n_data
        self.posterior_ = self._compute_posterior(y, prior)

    def _compute_child1_and_child2_indices(self, X, dense):
        projections = X @ self.best_hyperplane_normal_ - np.dot(self.best_hyperplane_normal_, self.best_hyperplane_origin_)
        indices1 = np.where(projections < 0)[0]
        indices2 = np.where(projections >= 0)[0]

        return indices1, indices2

    def is_leaf(self):
        self._ensure_is_fitted()
        return self.best_hyperplane_normal_ is None

    def feature_importance(self):
        self._ensure_is_fitted()

        feature_importance = np.zeros(self.n_dim_)
        self._update_feature_importance(feature_importance)
        feature_importance /= feature_importance.sum()

        return feature_importance

    def _update_feature_importance(self, feature_importance):
        if self.is_leaf():
            return
        else:
            log_p_gain = self.best_log_p_data_split_ - self.log_p_data_no_split_
            hyperplane_normal = self.best_hyperplane_normal_

            # the more the normal vector is oriented along a given dimension's axis the more
            # important that dimension is, so weight log_p_gain with hyperplane_normal[i_dim]
            # (its absolute value in fact because the sign of the direction is irrelevant)
            feature_importance += log_p_gain * np.abs(hyperplane_normal)
            if self.child1_ is not None:
                self.child1_._update_feature_importance(feature_importance)
                self.child2_._update_feature_importance(feature_importance)

    def _erase_split_info(self):
        self.best_hyperplane_normal_ = None
        self.best_hyperplane_origin_ = None

    def __str__(self):
        if not self.is_fitted():
            return 'Unfitted model'

        return self._str([], '\u251C', '\u2514', '\u2502', '\u2265', None)

    def _str(self, anchor, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_back_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_back_child is not None:
            s += anchor_str + ' {:5s}: '.format('back' if is_back_child else 'front')

        if self.is_leaf():
            s += 'y={}, n={}'.format(self._predict_leaf(), self.n_data_)
            if not self.is_regression:
                s += ', p(y)={}'.format(self._compute_posterior_mean())
        else:
            s += 'HP(origin={}, normal={})'.format(self.best_hyperplane_origin_, self.best_hyperplane_normal_)

            # 'back' child (the child that is on the side of the hyperplane opposite to the normal vector, or projection < 0)
            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_back_child else '  '), VERT_RIGHT])
            s += self.child1_._str(anchor_child1, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            # 'front' child (the child that is on same side of the hyperplane as the normal vector, or projection >= 0)
            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_back_child else '  '), DOWN_RIGHT])
            s += self.child2_._str(anchor_child2, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s
