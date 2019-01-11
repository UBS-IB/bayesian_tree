from abc import ABC, abstractmethod

import numpy as np
from scipy.special import betaln, gammaln


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
            return self.predict_leaf() * np.ones(len(X))

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
    def compute_posterior(self, y, delta):
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


class BinaryClassificationNode(Node):
    """Concrete node implementation for binary classification using a Beta(alpha, beta) prior."""

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, BinaryClassificationNode)
        assert len(self.prior) == 2, 'Expected a Beta(alpha, beta) prior, i.e., a sequence with two entries, but got {}'.format(prior)
        assert len(self.posterior) == 2, 'Expected a Beta(alpha, beta) posterior, i.e., a sequence with two entries, but got {}'.format(posterior)

    def check_target(self, y):
        assert y.min() == 0 and y.max() == 1,\
            'Expected target values 0..1 but found {}..{}'.format(y.min(), y.max())

    def compute_log_p_data_post_no_split(self, y):
        alpha, beta = self.prior
        alpha_post, beta_post = self.compute_posterior(y)

        betaln_prior = betaln(alpha, beta)
        log_p_prior = np.log(1-self.partition_prior**(1+self.level))
        log_p_data = betaln(alpha_post, beta_post) - betaln_prior

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, split_indices, y):
        n = len(y)
        n_splits = len(split_indices)

        n1 = split_indices
        n2 = n - n1
        k1 = y.cumsum()[split_indices - 1]
        k2 = y.sum() - k1

        alpha, beta = self.prior

        betaln_prior = betaln(alpha, beta)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / n_splits)
        log_p_data1 = self._compute_log_p_data(n1, k1, betaln_prior)
        log_p_data2 = self._compute_log_p_data(n2, k2, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        alpha, beta = self.prior
        if delta == 0:
            return alpha, beta

        n = len(y)
        k = y.sum()
        alpha_post = alpha + delta*(n-k)
        beta_post = beta + delta*k

        return alpha_post, beta_post

    def compute_posterior_mean(self):
        alpha, beta = self.posterior
        return beta/(alpha+beta)

    def predict_leaf(self):
        return np.argmax(self.posterior)

    def _compute_log_p_data(self, n, k, betaln_prior):
        alpha, beta = self.prior
        return betaln(alpha+(n-k), beta+k) - betaln_prior


def multivariate_betaln(alphas):
    if len(alphas) == 2:
        return betaln(alphas[0], alphas[1])
    else:
        # see https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function
        return np.sum([gammaln(alpha) for alpha in alphas], axis=0) - gammaln(alphas.sum())


class MultiClassificationNode(Node):
    """Concrete node implementation for multi-class classification using a Dirichlet prior."""

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, MultiClassificationNode)
        assert len(self.prior) == len(self.posterior)

    def check_target(self, y):
        n_classes = len(self.prior)
        assert y.min() == 0 and y.max() == n_classes - 1,\
            'Expected target values 0..{} but found {}..{}'.format(n_classes - 1, y.min(), y.max())

    def compute_log_p_data_post_no_split(self, y):
        alphas = self.prior
        alphas_post = self.compute_posterior(y)

        betaln_prior = multivariate_betaln(alphas)
        log_p_prior = np.log(1-self.partition_prior**(1+self.level))
        log_p_data = multivariate_betaln(alphas_post) - betaln_prior

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, split_indices, y):
        n_splits = len(split_indices)

        alphas = self.prior
        n_classes = len(alphas)
        k1 = np.array(n_classes * [None])
        k2 = np.array(n_classes * [None])
        for i in range(n_classes):
            k1_plus_sum = (y == i).cumsum()
            sum = k1_plus_sum[-1]
            k1[i] = k1_plus_sum[split_indices-1]
            k2[i] = sum - k1[i]

        betaln_prior = multivariate_betaln(alphas)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / n_splits)
        log_p_data1 = self._compute_log_p_data(k1, betaln_prior)
        log_p_data2 = self._compute_log_p_data(k2, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        alphas = self.prior
        if delta == 0:
            return alphas

        k = np.zeros(len(alphas))
        for i in range(len(k)):
            k[i] = (y == i).sum()
        alphas_post = alphas + delta*k

        return alphas_post

    def compute_posterior_mean(self):
        alphas = self.posterior
        means = alphas / np.sum(alphas)
        posterior_mean = 0
        for i in range(len(means)):
            posterior_mean += i * means[i]
        return posterior_mean

    def predict_leaf(self):
        return np.argmax(self.posterior)

    def _compute_log_p_data(self, k, betaln_prior):
        alphas = self.prior
        return multivariate_betaln(alphas+k) - betaln_prior


class RegressionNode(Node):
    """
    Concrete node implementation for regression using a Normal-Gamma(mu, kappa, alpha, beta) prior
    for unknown mean and unknown variance.
    """

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, RegressionNode)

    def check_target(self, y):
        pass

    def compute_log_p_data_post_no_split(self, y):
        n = len(y)
        mean = y.mean()

        y_minus_mean_sq_sum = ((y - mean)**2).sum()
        mu_post, kappa_post, alpha_post, beta_post = self._compute_posterior_internal(n, mean, y_minus_mean_sq_sum)
        log_p_prior = np.log(1 - self.partition_prior**(1 + self.level))
        log_p_data = self._compute_log_p_data(alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, split_indices, y):
        n = len(y)
        n_splits = len(split_indices)

        n1 = np.arange(1, n)
        n2 = n - n1
        sum1 = y.cumsum()[:-1]
        mean1 = sum1 / n1
        mean2 = (y.sum() - sum1) / n2
        y_minus_mean_sq_sum1 = ((y[:-1] - mean1)**2).cumsum()
        y_minus_mean_sq_sum2 = ((y[1:] - mean2)[::-1]**2).cumsum()[::-1]

        if len(split_indices) != len(y)-1:
            # we are *not* splitting between all data points -> indexing necessary
            split_indices_minus_1 = split_indices - 1

            n1 = n1[split_indices_minus_1]
            n2 = n2[split_indices_minus_1]
            mean1 = mean1[split_indices_minus_1]
            mean2 = mean2[split_indices_minus_1]
            y_minus_mean_sq_sum1 = y_minus_mean_sq_sum1[split_indices_minus_1]
            y_minus_mean_sq_sum2 = y_minus_mean_sq_sum2[split_indices_minus_1]

        mu1, kappa1, alpha1, beta1 = self._compute_posterior_internal(n1, mean1, y_minus_mean_sq_sum1)
        mu2, kappa2, alpha2, beta2 = self._compute_posterior_internal(n2, mean2, y_minus_mean_sq_sum2)

        log_p_prior = np.log(self.partition_prior**(1+self.level) / n_splits)
        log_p_data1 = self._compute_log_p_data(alpha1, beta1, kappa1, n1)
        log_p_data2 = self._compute_log_p_data(alpha2, beta2, kappa2, n2)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        if delta == 0:
            return self.prior

        n = len(y)
        mean = y.mean()
        y_minus_mean_sq_sum = ((y - mean)**2).sum()

        return self._compute_posterior_internal(n, mean, y_minus_mean_sq_sum, delta)

    def _compute_posterior_internal(self, n, mean, y_minus_mean_sq_sum, delta=1):
        mu, kappa, alpha, beta = self.prior

        n_delta = n*delta
        kappa_post = kappa + n_delta
        mu_post = (kappa*mu + n_delta*mean) / kappa_post
        alpha_post = alpha + 0.5*n_delta
        beta_post = beta + 0.5 * delta * y_minus_mean_sq_sum + kappa * n_delta * (mean - mu)**2 / (2 * (kappa + n))

        return mu_post, kappa_post, alpha_post, beta_post

    def compute_posterior_mean(self):
        return self.posterior[0]  # mu is the posterior mean

    def predict_leaf(self):
        return self.compute_posterior_mean()

    def _compute_log_p_data(self, alpha1, beta1, kappa1, n1):
        mu, kappa, alpha, beta = self.prior

        return (gammaln(alpha1) - gammaln(alpha)
                + alpha * np.log(beta) - alpha1 * np.log(beta1)
                + 0.5 * np.log(kappa / kappa1)
                - 0.5 * n1 * np.log(2 * np.pi))
