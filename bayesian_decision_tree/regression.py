"""This module declares the Bayesian Tree regression algorithms:
* RegressionNode
"""
import numpy as np
from scipy.special import gammaln
from bayesian_decision_tree.base import Node


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

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equations (86) - (89)
        n_delta = n*delta
        kappa_post = kappa + n_delta
        mu_post = (kappa*mu + n_delta*mean) / kappa_post
        alpha_post = alpha + 0.5*n_delta
        beta_post = beta + 0.5*delta*y_minus_mean_sq_sum + 0.5*kappa*n_delta*(mean-mu)**2 / (kappa+n)

        return mu_post, kappa_post, alpha_post, beta_post

    def compute_posterior_mean(self):
        return self.posterior[0]  # mu is the posterior mean

    def predict_leaf(self):
        return self.compute_posterior_mean()

    def _compute_log_p_data(self, alpha_new, beta_new, kappa_new, n_new):
        mu, kappa, alpha, beta = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equation (95)
        return (gammaln(alpha_new) - gammaln(alpha)
                + alpha*np.log(beta) - alpha_new*np.log(beta_new)
                + 0.5*np.log(kappa/kappa_new)
                - 0.5*n_new*np.log(2*np.pi))