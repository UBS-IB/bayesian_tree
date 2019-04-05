from abc import ABC

import numpy as np
from scipy import optimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import norm

from bayesian_decision_tree.base import BaseNode
from bayesian_decision_tree.hyperplane_optimization import HyperplaneOptimizationFunction, ScipyOptimizer
from bayesian_decision_tree.utils import multivariate_betaln


class BaseHyperplaneNode(BaseNode, ABC):
    """
    The base class for all Bayesian decision tree algorithms (classification and regression). Performs all the high-level fitting
    and prediction tasks and outsources the low-level work to the subclasses.
    """
    def __init__(self, partition_prior, prior, child_type, is_regression, optimizer, level):
        BaseNode.__init__(self, partition_prior, prior, child_type, is_regression, level)

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
            X = csc_matrix(X)

        log_p_data_post_all = self.compute_log_p_data_post_no_split(y)

        # the function to optimize (depends on X and y, hence we need to instantiate it for every data set anew)
        optimization_function = HyperplaneOptimizationFunction(
            X,
            y,
            self.compute_log_p_data_post_split,
            log_p_data_post_all,
            isinstance(self.optimizer, ScipyOptimizer))

        # create and run optimizer
        self.optimizer.solve(optimization_function)

        # retrieve best hyperplane split from optimization function
        if optimization_function.best_hyperplane_normal is not None:
            # split data and target to recursively train children
            projections = X @ optimization_function.best_hyperplane_normal \
                          - np.dot(optimization_function.best_hyperplane_normal, optimization_function.best_hyperplane_origin)
            indices1 = np.where(projections < 0)[0]
            indices2 = np.where(projections >= 0)[0]
            X1 = X[indices1]
            X2 = X[indices2]
            y1 = y[indices1]
            y2 = y[indices2]

            # compute posteriors of children and priors for further splitting
            prior_child1 = self.compute_posterior(y1, 0)
            prior_child2 = self.compute_posterior(y2, 0)

            # store split info, create children and continue training them if there's data left to split
            self.best_hyperplane_normal = optimization_function.best_hyperplane_normal
            self.best_hyperplane_origin = optimization_function.best_hyperplane_origin

            self.child1 = self.child_type(self.partition_prior, prior_child1, self.optimizer, self.level + 1)
            self.child2 = self.child_type(self.partition_prior, prior_child2, self.optimizer, self.level + 1)

            if X1.shape[0] > 1:
                self.child1._fit(X1, y1, delta, verbose, feature_names)
            else:
                self.child1.posterior = self.compute_posterior(y1)

            if X2.shape[0] > 1:
                self.child2._fit(X2, y2, delta, verbose, feature_names)
            else:
                self.child2.posterior = self.compute_posterior(y2)

        # compute posterior
        self.posterior = self.compute_posterior(y)

    def compute_child1_and_child2_indices(self, X, dense):
        projections = X @ self.best_hyperplane_normal - np.dot(self.best_hyperplane_normal, self.best_hyperplane_origin)
        indices1 = np.where(projections < 0)[0]
        indices2 = np.where(projections >= 0)[0]

        return indices1, indices2

    def is_leaf(self):
        return self.best_hyperplane_normal is None

    def __str__(self):
        return "TODO"  # TODO
