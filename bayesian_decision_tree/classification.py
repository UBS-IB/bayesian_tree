"""
This module declares the Bayesian classification tree algorithms:
- BinaryClassificationNode
- MultiClassificationNode
"""
import numpy as np
from scipy.special import betaln

from bayesian_decision_tree.base import Node
from bayesian_decision_tree.utils import multivariate_betaln


class BinaryClassificationNode(Node):
    """
    Bayesian binary classification tree. Uses a beta prior. This is a
    """

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, BinaryClassificationNode, False)
        assert len(self.prior) == 2,\
            'Expected a Beta(alpha, beta) prior, i.e., a sequence with two entries, but got {}'.format(prior)
        assert len(self.posterior) == 2,\
            'Expected a Beta(alpha, beta) posterior, i.e., a sequence with two entries, but got {}'.format(posterior)

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

    def compute_log_p_data_post_split(self, y, split_indices, n_dim):
        n = len(y)
        n_splits = len(split_indices)

        n1 = split_indices
        n2 = n - n1
        k1 = y.cumsum()[split_indices - 1]
        k2 = y.sum() - k1

        alpha, beta = self.prior

        betaln_prior = betaln(alpha, beta)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / (n_splits * n_dim))
        log_p_data1 = self._compute_log_p_data(n1, k1, betaln_prior)
        log_p_data2 = self._compute_log_p_data(n2, k2, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        alpha, beta = self.prior
        if delta == 0:
            return alpha, beta

        # see https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
        n = len(y)
        k = y.sum()
        alpha_post = alpha + delta*(n-k)
        beta_post = beta + delta*k

        return alpha_post, beta_post

    def compute_posterior_mean(self):
        alpha, beta = self.posterior
        p_alpha = alpha / (alpha + beta)
        return np.array([p_alpha, 1-p_alpha])

    def _predict_leaf(self):
        # predict class
        return np.argmax(self.posterior)

    def _compute_log_p_data(self, n, k, betaln_prior):
        alpha, beta = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
        # which can be expressed as a fraction of beta functions
        return betaln(alpha+(n-k), beta+k) - betaln_prior


class MultiClassificationNode(Node):
    """
    Bayesian multi-class classification tree. Uses a Dirichlet prior."""

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, MultiClassificationNode, False)
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

    def compute_log_p_data_post_split(self, y, split_indices, n_dim):
        n_splits = len(split_indices)

        alphas = self.prior
        n_classes = len(alphas)
        k1 = np.array(n_classes * [None])
        k2 = np.array(n_classes * [None])
        for i in range(n_classes):
            k1_plus_sum = (y == i).cumsum()
            total = k1_plus_sum[-1]
            k1[i] = k1_plus_sum[split_indices-1]
            k2[i] = total - k1[i]

        betaln_prior = multivariate_betaln(alphas)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / (n_splits * n_dim))
        log_p_data1 = self._compute_log_p_data(k1, betaln_prior)
        log_p_data2 = self._compute_log_p_data(k2, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        alphas = self.prior
        if delta == 0:
            return alphas

        # see https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
        y_reshaped = np.broadcast_to(y, (len(alphas), len(y)))
        classes = np.arange(len(alphas)).reshape(-1, 1)
        k = np.sum(y_reshaped == classes, axis=1)
        alphas_post = alphas + delta*k

        return alphas_post

    def compute_posterior_mean(self):
        alphas = self.posterior
        return alphas / np.sum(alphas)

    def _predict_leaf(self):
        # predict class
        return np.argmax(self.posterior)

    def _compute_log_p_data(self, k, betaln_prior):
        alphas = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
        # which can be expressed as a fraction of beta functions
        return multivariate_betaln(alphas+k) - betaln_prior
