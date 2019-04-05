from abc import ABC, abstractmethod

import numpy as np
from numpy.random import RandomState
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import norm


class HyperplaneOptimizationFunction:
    """
    The function to optimize for hyperplane trees. This is a function of `n_dim` variables representing
    the normal vector of a hyperplane in `n_dim` dimensions. Given such a hyperplane normal the function
    computes the optimum split location (i.e., the origin of the hyperplane) in the data such that the
    data likelihood is maximized.

    """
    def __init__(self, X, y, compute_log_p_data_post_split, log_p_data_post_all, is_scipy_optimizer):
        self.X = X
        self.y = y
        self.compute_log_p_data_post_split = compute_log_p_data_post_split
        self.log_p_data_post_all = log_p_data_post_all
        self.is_scipy_optimizer = is_scipy_optimizer

        # results of the optimization - to be set later during the actual optimization
        self.best_log_p_data_post = log_p_data_post_all
        self.best_cumulative_distances = 0
        self.best_hyperplane_normal = None
        self.best_hyperplane_origin = None

    def compute(self, hyperplane_normal):
        if self.is_scipy_optimizer:
            # If scipy optimizers tried to find hyperplane normal vectors directly in cartesian
            # coordinate space (e.g. by providing bounds of the form -1 <= x[i] <= 1) the search space
            # close to 'corners' of the hypercube would be larger than close to the axes. For a
            # 2-D problem for example searching for normal vectors in the -1..1 unit square leads to
            # more normal vector candidates close to diagonal angles compared to angles close to the
            # vertical or horizontal for obvious reasons. In higher dimensions this problem becomes
            # amplified, leading to significant distortion of the search space.
            #
            # In order to avoid this problem we do the following instead:
            # - Scipy optimizers operate on the following unit hypercube space: 0 <= x[i] <= 1 [1]
            # - Vectors from this space are transformed in the following way:
            #   1. Each x[i] is converted to a standard normal using the inverse CDF of the standard
            #      normal
            #   2. The resulting vector is normalized to unit length
            #
            # This leads to a unit vector that, somewhat surprisingly, has *uniform* distribution
            # on the hypersphere, see [2] for an explanation on why this is the case. This means
            # that changes of a certain magnitude in the original vectors that scipy optimizers
            # operate on lead to similarly-sized changes in angle space after the transformation,
            # avoiding any over- or underweighting of certain directions.
            #
            # The disadvantage of this approach is that a hypersphere in N-dimensional space is
            # being parameterized by a vector of size N, whereas such a hypersphere surface only has
            # N-1 dimensions, so theoretically a vector of size N-1 is sufficient. This means that
            # the optimizers are operating in a search space that is 1 dimension larger than
            # necessary, meaning that an infinite number of vectors in their search space lead
            # to the same solution (all vectors that, after the inverse CDF mapping described above,
            # lead to the same vector).
            #
            # [1] In fact, we only provide *half* of a unit hypercube as bounds because it's
            #     unnecessary to evaluate both the hyperplane normal vectors `x` and `-x` as they
            #     represent the same hyperplane. We arbitrarily choose to limit the cartesian
            #     space of the first dimension to be positive:
            #
            #         0.5 <= x[0] <= 1         <- leads to standard normal values >= 0
            #                                     after transformation
            #
            #         0 <= x[i] <= 1, i > 0    <- all other dimensions may take any standard
            #                                     normal value after transformation
            #
            # [2] See 'Alternative method 1': http://corysimon.github.io/articles/uniformdistn-on-sphere/

            hyperplane_normal = norm.ppf(hyperplane_normal)  # 0..1 -> Normal(0, 1)

        # catch some special cases and normalize to unit length
        hyperplane_normal = np.nan_to_num(hyperplane_normal)
        if np.all(hyperplane_normal == 0):
            hyperplane_normal[0] = 1

        hyperplane_normal /= np.linalg.norm(hyperplane_normal)

        dense = isinstance(self.X, np.ndarray)
        if not dense and isinstance(self.X, csr_matrix):
            self.X = csc_matrix(self.X)

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
            if not dense:
                p1 = p1.toarray()[0]
                p2 = p2.toarray()[0]

            hyperplane_origin = 0.5 * (p1 + p2)  # middle between the points that are being split
            projections_with_origin = projections - np.dot(hyperplane_normal, hyperplane_origin)
            cumulative_distances = np.sum(np.abs(projections_with_origin))

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

        return -log_p_data_post_split[i_max]


class HyperplaneOptimizer(ABC):
    @abstractmethod
    def solve(self, optimization_function):
        pass


class ScipyOptimizer(HyperplaneOptimizer):
    """An optimizer using one of the scipy global optimizers, see [1].

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization
    """

    def __init__(self, solver_type, seed, **extra_solver_kwargs):
        self.solver_type = solver_type
        self.seed = seed
        self.extra_solver_kwargs = extra_solver_kwargs

    def solve(self, optimization_function):
        # bounds for scipy optimizers: half unit hypercube (will be mapped to
        # half hypersphere uniformly later on)
        X = optimization_function.X
        n_dim = X.shape[1]
        half_unit_hypercube_bounds = np.vstack((np.zeros(n_dim), np.ones(n_dim))).T
        half_unit_hypercube_bounds[0, 0] = 0.5  # we need only evaluate half the hypersphere to cover all angles

        solver = self.solver_type(
            func=optimization_function.compute,
            bounds=half_unit_hypercube_bounds,
            seed=self.seed,
            **self.extra_solver_kwargs)

        solver.solve()


class RandomTwoPointOptimizer:
    def __init__(self, n_mc, seed):
        self.n_mc = n_mc
        self.seed = seed

    def solve(self, optimization_function):
        rand = RandomState(self.seed)

        X = optimization_function.X
        y = optimization_function.y

        if np.any(np.round(y) != y):
            raise TypeError('Cannot use {} for regression problems as there are no classes to pick points from'.format(
                RandomTwoPointOptimizer.__name__))

        dense = isinstance(X, np.ndarray)

        if len(set(y)) <= 1:
            # can't pick two points of different classes if there aren't at least two classes
            return

        # find indices of each class
        n_classes = int(y.max()) + 1
        class_indices = [np.where(y == i)[0] for i in range(n_classes)]

        # evaluate 'n_mc' hyperplane normals passing through two random points form different classes
        for i in range(self.n_mc):
            indices1 = []
            indices2 = []

            while len(indices1) == 0 or len(indices2) == 0:
                class1 = rand.randint(0, n_classes)
                indices1 = class_indices[class1]

                class2 = class1
                while class2 == class1:
                    class2 = rand.randint(0, n_classes)

                indices2 = class_indices[class2]

            p1 = X[indices1[rand.randint(0, len(indices1))]]
            p2 = X[indices2[rand.randint(0, len(indices2))]]
            if not dense:
                p1 = p1.toarray()[0]
                p2 = p2.toarray()[0]

            normal = p2-p1
            if normal[0] < 0:
                normal *= -1  # make sure the first coordinate is positive to match the scipy search space

            optimization_function.compute(normal)


class RandomHyperplaneOptimizer:
    def __init__(self, n_mc, seed):
        self.n_mc = n_mc
        self.seed = seed

    def solve(self, optimization_function):
        rand = RandomState(self.seed)

        X = optimization_function.X
        n_dim = X.shape[1]

        for i in range(self.n_mc):
            normal = rand.normal(0, 1, n_dim)
            optimization_function.compute(normal)
