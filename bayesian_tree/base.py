from abc import ABC, abstractmethod

import numpy as np



class Node(ABC):
    """A node class with either no children, in which case it's a leaf node, or exactly two children."""

    def __init__(self, name, partition_prior, prior, posterior, level, child_type):
        self.name = name
        self.partition_prior = partition_prior
        self.prior = np.array(prior)
        self.posterior = np.array(posterior) if posterior is not None else np.array(prior)
        self.level = level
        self.child_type = child_type

        # to be set later
        self.split_dimension = -1
        self.split_index = -1
        self.split_value = None
        self.child1 = None
        self.child2 = None

    def fit(self, X, y, delta):
        """Trains this node with the given feature data matrix X and the given target vector y."""
        if self.level == 0:
            self.check_target(y)

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
            log_p_data_post_split = self.compute_log_p_data_post_split(split_indices, y_sorted)
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
            self.split_value = 0.5 * (X_sorted[best_split_index-1, best_split_dimension] + X_sorted[best_split_index, best_split_dimension])

            self.child1 = self.child_type(self.name + '-child1', self.partition_prior, prior_child1, posterior1, self.level+1)
            self.child2 = self.child_type(self.name + '-child2', self.partition_prior, prior_child2, posterior2, self.level+1)

            if len(X1) > 1:
                self.child1.fit(X1, y1, delta)
            if len(X2) > 1:
                self.child2.fit(X2, y2, delta)

    def predict(self, X):
        if self.child1 is not None:
            # query children and then re-assemble
            indices1 = np.where(X[:, self.split_dimension] < self.split_value)[0]
            indices2 = np.where(X[:, self.split_dimension] >= self.split_value)[0]
            predictions1 = self.child1.predict(X[indices1])
            predictions2 = self.child2.predict(X[indices2])

            predictions = np.zeros(len(X))
            predictions[indices1] = predictions1
            predictions[indices2] = predictions2

            return predictions
        else:
            # no children -> predict mean
            return self.predict_leaf()

    @abstractmethod
    def check_target(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_no_split(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_split(self, split_indices, y):
        pass

    @abstractmethod
    def compute_posterior(self, y, delta=1):
        pass

    @abstractmethod
    def compute_posterior_mean(self):
        pass

    @abstractmethod
    def predict_leaf(self):
        pass

    def __str__(self):
        if self.split_index == -1:
            # nothing to show
            return ''

        s = '{}{}'.format(self.level * '  ', self.name)
        s += ': x{}={}'.format(self.split_dimension+1, self.split_value)
        child1_str = str(self.child1)
        child2_str = str(self.child2)
        if len(child1_str) > 0:
            s += '\n{}'.format(self.child1)
        if len(child2_str) > 0:
            s += '\n{}'.format(self.child2)
        return s

    def __repr__(self):
        return self.__str__()
