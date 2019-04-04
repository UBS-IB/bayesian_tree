from abc import ABC

import numpy as np
from scipy import optimize
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import norm

from bayesian_decision_tree.base import BaseNode
from bayesian_decision_tree.utils import multivariate_betaln


class BaseHyperplaneNode(BaseNode, ABC):
    """
    The base class for all Bayesian decision tree algorithms (classification and regression). Performs all the high-level fitting
    and prediction tasks and outsources the low-level work to the subclasses.
    """
    def __init__(self, partition_prior, prior, child_type, is_regression, optimizer, n_mc, use_polar, level):
        super().__init__(partition_prior, prior, child_type, is_regression, level)

        self.optimizer = optimizer
        self.n_mc = n_mc
        self.use_polar = use_polar

        # to be set later
        self.best_hyperplane_normal = None
        self.best_hyperplane_origin = None

    def _fit(self, X, y, delta, verbose, feature_names):
        if verbose:
            print('Training level {} with {:10} data points'.format(self.level, len(y)))

        dense = isinstance(X, np.ndarray)
        if not dense and isinstance(X, csr_matrix):
            X = csc_matrix(X)

        n_obs = X.shape[0]
        n_dim = X.shape[1]

        if n_obs <= 2:
            return

        log_p_data_post_all = self.compute_log_p_data_post_no_split(y)

        optimization_function = OptimizationFunction(
            X,
            y,
            self.compute_log_p_data_post_split,
            log_p_data_post_all,
            self.optimizer,
            self.use_polar)

        if self.optimizer == 'scipy':
            if self.use_polar:
                bounds = np.vstack((np.zeros(n_dim-1), np.pi * np.ones(n_dim-1))).T
                bounds[0, 1] = 2*np.pi
            else:
                bounds = np.vstack((np.zeros(n_dim), np.ones(n_dim))).T  # unit hypercube
                # bounds[:, 0] += 1e-6  # avoid numerical problems
                # bounds[:, 1] -= 1e-6  # avoid numerical problems
                bounds[0, 0] = 0.5  # we need only evaluate half the hypersphere

            # if self.level == 0:
            #     print('Dual Annealing')
            # result = optimize.dual_annealing(func=optimization_function, bounds=bounds, seed=666)

            # if self.level == 0:
            #     print('Differential Evolution')
            result = optimize.differential_evolution(func=optimization_function.compute, bounds=bounds, seed=666)

            # print('level={}: n_eval={}'.format(self.level, result.nfev))
        elif self.optimizer == '2-point':
            if len(set(y)) > 1:  # only evaluate splitting if we have more than one class
                n_classes = int(y.max()) + 1
                class_indices = []
                for i in range(n_classes):
                    class_indices.append(np.where(y == i)[0])

                for i in range(self.n_mc):
                    # choose two random points from different classes
                    indices1 = []
                    indices2 = []
                    while len(indices1) == 0 or len(indices2) == 0:
                        c1 = np.random.randint(0, n_classes)
                        indices1 = class_indices[c1]
                        c2 = c1
                        while c2 == c1:
                            c2 = np.random.randint(0, n_classes)

                        indices2 = class_indices[c2]

                    p1 = X[indices1[np.random.randint(0, len(indices1))]]
                    p2 = X[indices2[np.random.randint(0, len(indices2))]]
                    normal = p2-p1
                    optimization_function.compute(normal)
        elif self.optimizer == 'random-hyperplane':
            # random hyperplanes
            for i in range(self.n_mc):
                normal = np.random.normal(0, 1, n_dim)
                optimization_function.compute(normal)
        else:
            raise Exception('Unknown optimizer: {}'.format(self.optimizer))

        self.posterior = self.compute_posterior(y)
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
            prior_child2 = self.compute_posterior(y1, 0)

            # store split info, create children and continue training them if there's data left to split
            self.best_hyperplane_normal = optimization_function.best_hyperplane_normal
            self.best_hyperplane_origin = optimization_function.best_hyperplane_origin

            self.child1 = self.child_type(self.partition_prior, prior_child1, self.optimizer, self.n_mc, self.use_polar, self.level + 1)
            self.child2 = self.child_type(self.partition_prior, prior_child2, self.optimizer, self.n_mc, self.use_polar, self.level + 1)

            if X1.shape[0] > 1:
                self.child1.fit(X1, y1)
            else:
                self.child1.posterior = self.compute_posterior(y1)

            if X2.shape[0] > 1:
                self.child2.fit(X2, y2)
            else:
                self.child2.posterior = self.compute_posterior(y2)

    def compute_child1_and_child2_indices(self, X, dense):
        projections = X @ self.best_hyperplane_normal - np.dot(self.best_hyperplane_normal, self.best_hyperplane_origin)
        indices1 = np.where(projections < 0)[0]
        indices2 = np.where(projections >= 0)[0]

        return indices1, indices2

    def _compute_log_p_data(self, k, betaln_prior):
        alphas = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
        # which can be expressed as a fraction of beta functions
        return multivariate_betaln(alphas+k) - betaln_prior

    def is_leaf(self):
        return self.best_hyperplane_normal is None

    def compute_posterior_mean(self):
        alphas = self.posterior
        return alphas / np.sum(alphas)


class OptimizationFunction:
    def __init__(self, X, y, compute_log_p_data_post_split, log_p_data_post_all, optimizer, use_polar):
        self.X = X
        self.y = y
        self.compute_log_p_data_post_split = compute_log_p_data_post_split
        self.log_p_data_post_all = log_p_data_post_all
        self.optimizer = optimizer
        self.use_polar = use_polar

        self.best_log_p_data_post = log_p_data_post_all
        self.best_cumulative_distances = 0
        self.best_hyperplane_normal = None
        self.best_hyperplane_origin = None

    def compute(self, hyperplane_normal):
        hyperplane_normal = np.nan_to_num(hyperplane_normal)

        if self.optimizer == 'scipy':
            # convert unit hypercube to unit hypersphere to get equal probability in any direction
            if self.use_polar:
                theta = hyperplane_normal
                n = len(theta)
                x = np.zeros(n+1)
                if n <= 2:
                    x[0] = np.cos(theta[0]) * np.prod(np.sin(theta[1:]))
                    x[1] = np.sin(theta[0]) * np.prod(np.sin(theta[1:]))

                if n > 2:
                    for i in range(2, n):
                        x[i] = np.cos(theta[i-1]) * np.prod(np.sin(theta[i:]))

                    x[n] = np.cos(theta[n-1])

                hyperplane_normal = x
            else:
                hyperplane_normal = norm.ppf(hyperplane_normal)  # 0..1 -> standard normal

                if np.all(hyperplane_normal == 0):
                    hyperplane_normal[0] = 1

                hyperplane_normal /= np.linalg.norm(hyperplane_normal)

        # compute distance of all points to the hyperplane: https://mathinsight.org/distance_point_plane
        projections = self.X @ hyperplane_normal  # up to an additive constant which doesn't matter to distance ordering
        sort_indices = np.argsort(projections)
        split_indices = 1 + np.where(np.diff(projections) != 0)[0]  # we can only split between *different* data points
        if len(split_indices) == 0:
            # no split possible along this dimension
            return -self.log_p_data_post_all

        y_sorted = self.y[sort_indices]

        # compute data likelihoods of all possible splits along this projection and find split with highest data likelihood
        n_dim = self.X.shape[1]
        log_p_data_post_split = self.compute_log_p_data_post_split(y_sorted, split_indices, n_dim)
        i_max = log_p_data_post_split.argmax()
        if log_p_data_post_split[i_max] >= self.best_log_p_data_post:
            best_split_index = split_indices[i_max]
            p1 = self.X[sort_indices[best_split_index-1]]
            p2 = self.X[sort_indices[best_split_index]]
            hyperplane_origin = 0.5 * (p1 + p2)  # middle between the points that are being split
            projections_with_origin = projections - np.dot(hyperplane_normal, hyperplane_origin)
            cumulative_distances = np.sum(np.abs(projections_with_origin))

            is_log_p_better_or_same_but_with_better_distance = False
            if log_p_data_post_split[i_max] > self.best_log_p_data_post:
                is_log_p_better_or_same_but_with_better_distance = True
            else:
                # accept new split with same log(p) only if it increases the cumulative distance of all points to the hyperplane
                is_log_p_better_or_same_but_with_better_distance = cumulative_distances > self.best_cumulative_distances

            if is_log_p_better_or_same_but_with_better_distance:
                self.best_log_p_data_post = log_p_data_post_split[i_max]
                self.best_cumulative_distances = cumulative_distances
                self.best_hyperplane_normal = hyperplane_normal
                self.best_hyperplane_origin = hyperplane_origin
                # print('New best origin:            {}'.format(self.best_hyperplane_origin))
                # print('New best hyperplane normal: {}'.format(self.best_hyperplane_normal))
                # print('New best log(p):            {}'.format(self.best_log_p_data_post))
                # print()

        return -log_p_data_post_split[i_max]
