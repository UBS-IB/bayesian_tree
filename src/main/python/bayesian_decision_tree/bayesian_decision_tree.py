from abc import ABC, abstractmethod

import numpy as np
from scipy.special import betaln, gammaln


class Node(ABC):
    """A node class with either no children, in which case it's a leaf node, or exactly two children."""
    def __init__(self, name, partition_prior, prior, posterior, level, child_type):
        self.name = name
        self.partition_prior = partition_prior
        self.prior = prior
        self.posterior = posterior
        self.level = level
        self.child_type = child_type

        # to be set later
        self.split_dimension = -1
        self.split_index = -1
        self.split_value = None
        self.child1 = None
        self.child2 = None

    def train(self, data, target, delta):
        #print('Training level {} with {:10} data points'.format(self.level, len(target)))  # xx
        """Trains this node with the given feature data matrix and the given target vector."""
        n_dim = data.shape[1]

        # compute data likelihood of not splitting and remember it as the best option so far
        log_p_data_post_best = self.compute_log_p_data_post_no_split(target)

        # compute data likelihoods of all possible splits along all data dimensions
        best_split_index = -1    # index of best split
        best_split_dimension = -1  # dimension of best split
        for dim in range(n_dim):
            data_dim = data[:, dim]
            sort_indices = np.argsort(data_dim)
            data_dim_sorted = data_dim[sort_indices]
            split_indices = 1 + np.where(np.diff(data_dim_sorted) != 0)[0]  # we can only split between *different* data points
            if len(split_indices) == 0:
                # no split possible along this dimension
                continue

            target_sorted = target[sort_indices]

            # compute data likelihoods of all possible splits along this dimension and find split with highest data likelihood
            log_p_data_post_split = self.compute_log_p_data_post_split(split_indices, target_sorted)
            i_max = log_p_data_post_split.argmax()
            if log_p_data_post_split[i_max] > log_p_data_post_best:
                # remember new best split
                log_p_data_post_best = log_p_data_post_split[i_max]
                best_split_index = split_indices[i_max]  # data index of best split
                best_split_dimension = dim

        # did we find a split that has a higher likelihood than the no-split likelihood?
        if best_split_index > 0:
            # split data and target to recursively train children
            sort_indices = np.argsort(data[:, best_split_dimension])
            data_sorted = data[sort_indices]
            target_sorted = target[sort_indices]
            data1 = data_sorted[:best_split_index]
            data2 = data_sorted[best_split_index:]
            target1 = target_sorted[:best_split_index]
            target2 = target_sorted[best_split_index:]

            # compute posteriors of children and priors for further splitting
            posterior1 = self.compute_posterior(target1)
            posterior2 = self.compute_posterior(target2)
            prior_child1 = self.compute_posterior(target1, delta)
            prior_child2 = self.compute_posterior(target1, delta)

            # store split info, create children and continue training them if there's data left to split
            self.split_dimension = best_split_dimension
            self.split_index = best_split_index
            self.split_value = 0.5 * (data_sorted[best_split_index-1, best_split_dimension] + data_sorted[best_split_index, best_split_dimension])

            self.child1 = self.child_type(self.name + '-child1', self.partition_prior, prior_child1, posterior1, self.level+1)
            self.child2 = self.child_type(self.name + '-child2', self.partition_prior, prior_child2, posterior2, self.level+1)

            if len(data1) > 1:
                self.child1.train(data1, target1, delta)
            if len(data2) > 1:
                self.child2.train(data2, target2, delta)

    def predict(self, data_eval):
        if self.child1 is not None:
            # query children and then re-assemble
            indices1 = np.where(data_eval[:, self.split_dimension] < self.split_value)[0]
            indices2 = np.where(data_eval[:, self.split_dimension] >= self.split_value)[0]
            predictions1 = self.child1.predict(data_eval[indices1])
            predictions2 = self.child2.predict(data_eval[indices2])

            predictions = np.zeros(len(data_eval))
            predictions[indices1] = predictions1
            predictions[indices2] = predictions2

            return predictions
        else:
            # no children -> predict mean
            return self.predict_leaf() * np.ones(len(data_eval))

    @abstractmethod
    def compute_log_p_data_post_no_split(self, target):
        pass

    @abstractmethod
    def compute_log_p_data_post_split(self, split_indices, target_sorted):
        pass

    @abstractmethod
    def compute_posterior(self, target, delta):
        pass

    @abstractmethod
    def compute_mean(self):
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
    def __init__(self, name, partition_prior, prior, posterior, level):
        super().__init__(name, partition_prior, prior, posterior, level, BinaryClassificationNode)

    def compute_log_p_data_post_no_split(self, target):
        alpha, beta = self.prior
        alpha_post, beta_post = self.compute_posterior(target)

        betaln_prior = betaln(alpha, beta)
        log_p_prior = np.log(1-self.partition_prior**(1+self.level))
        log_p_data = betaln(alpha_post, beta_post) - betaln_prior

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, split_indices, target_sorted):
        n = len(target_sorted)
        n_splits = len(split_indices)

        n1 = split_indices
        n2 = n - n1
        k1 = target_sorted.cumsum()[split_indices-1]
        k2 = target_sorted.sum() - k1

        alpha, beta = self.prior

        betaln_prior = betaln(alpha, beta)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / n_splits)
        log_p_data1 = self._compute_log_p_data(n1, k1, betaln_prior)
        log_p_data2 = self._compute_log_p_data(n2, k2, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, target, delta=1):
        alpha, beta = self.prior
        if delta == 0:
            return alpha, beta

        n = len(target)
        k = target.sum()
        alpha_post = alpha + delta*k
        beta_post = beta + delta*(n-k)
        return alpha_post, beta_post

    def compute_mean(self):
        # Beta prior: mean = alpha/(alpha+beta)
        alpha, beta = self.posterior
        return alpha / (alpha + beta)

    def predict_leaf(self):
        return 0 if self.compute_mean() < 0.5 else 1

    def _compute_log_p_data(self, n, k, betaln_prior):
        alpha, beta = self.prior
        return betaln(alpha+k, beta+(n-k)) - betaln_prior


class RegressionNode(Node):
    """
    Concrete node implementation for regression using a Normal-Gamma(mu, kappa, alpha, beta) prior
    for unknown mean and unknown variance.
    """
    def __init__(self, name, partition_prior, prior, posterior, level):
        super().__init__(name, partition_prior, prior, posterior, level, RegressionNode)

    def compute_log_p_data_post_no_split(self, target):
        n = len(target)
        mean = target.mean()

        target_minus_mean_sq_sum = ((target - mean)**2).sum()
        mu_post, kappa_post, alpha_post, beta_post = self._compute_posterior_internal(n, mean, target_minus_mean_sq_sum)
        log_p_prior = np.log(1 - self.partition_prior**(1 + self.level))
        log_p_data = self._compute_log_p_data(alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, split_indices, target_sorted):
        n = len(target_sorted)
        n_splits = len(split_indices)

        n1_full = np.arange(1, n)
        n2_full = n - n1_full
        sum1_full = target_sorted.cumsum()[:-1]
        mean1_full = sum1_full / n1_full
        mean2_full = (target_sorted.sum() - sum1_full) / n2_full

        if len(split_indices) == len(target_sorted)-1:
            # we evaluate splits between *all* data points -> no costly indexing with split_indices necessary
            n1 = n1_full
            n2 = n2_full
            mean1 = mean1_full
            mean2 = mean2_full

            target_minus_mean_sq_sum1 = ((target_sorted[:-1] - mean1_full)**2).cumsum()
            target_minus_mean_sq_sum2 = ((target_sorted[1:] - mean2_full)[::-1]**2).cumsum()[::-1]
        else:
            split_indices_minus_1 = split_indices - 1

            n1 = n1_full[split_indices_minus_1]
            n2 = n2_full[split_indices_minus_1]
            mean1 = mean1_full[split_indices_minus_1]
            mean2 = mean2_full[split_indices_minus_1]

            target_minus_mean_sq_sum1 = ((target_sorted[:-1] - mean1_full)**2).cumsum()[split_indices_minus_1]
            target_minus_mean_sq_sum2 = ((target_sorted[1:] - mean2_full)[::-1]**2).cumsum()[::-1][split_indices_minus_1]

        mu1, kappa1, alpha1, beta1 = self._compute_posterior_internal(n1, mean1, target_minus_mean_sq_sum1)
        mu2, kappa2, alpha2, beta2 = self._compute_posterior_internal(n2, mean2, target_minus_mean_sq_sum2)

        log_p_prior = np.log(self.partition_prior**(1+self.level) / n_splits)
        log_p_data1 = self._compute_log_p_data(alpha1, beta1, kappa1, n1)
        log_p_data2 = self._compute_log_p_data(alpha2, beta2, kappa2, n2)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, target, delta=1):
        if delta == 0:
            return self.prior

        n = len(target)
        mean = target.mean()
        target_minus_mean_sq_sum = ((target - mean)**2).sum()

        return self._compute_posterior_internal(n, mean, target_minus_mean_sq_sum, delta)

    def _compute_posterior_internal(self, n, mean, target_minus_mean_sq_sum, delta=1):
        mu, kappa, alpha, beta = self.prior

        n_delta = n*delta
        kappa_post = kappa + n_delta
        mu_post = (kappa*mu + n_delta*mean) / kappa_post
        alpha_post = alpha + 0.5*n_delta
        beta_post = beta + 0.5*delta*target_minus_mean_sq_sum + kappa*n_delta*(mean-mu)**2/(2*(kappa+n))

        return mu_post, kappa_post, alpha_post, beta_post

    def compute_mean(self):
        return self.posterior[0]  # mu is the posterior mean

    def predict_leaf(self):
        return self.compute_mean()

    def _compute_log_p_data(self, alpha1, beta1, kappa1, n1):
        mu, kappa, alpha, beta = self.prior

        return (gammaln(alpha1) - gammaln(alpha)
                + alpha * np.log(beta) - alpha1 * np.log(beta1)
                + 0.5 * np.log(kappa / kappa1)
                - 0.5 * n1 * np.log(2 * np.pi))
