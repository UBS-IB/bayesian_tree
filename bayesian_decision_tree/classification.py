"""
This module declares the Bayesian classification tree algorithms:

* ClassificationNode
"""
from abc import ABC

import numpy as np

from bayesian_decision_tree.base_hyperplane import BaseHyperplaneNode
from bayesian_decision_tree.base_perpendicular import BasePerpendicularNode
from bayesian_decision_tree.utils import multivariate_betaln


class BaseClassificationNode(ABC):
    """
    Bayesian multi-class classification tree. Uses a Dirichlet prior (a multivariate
    generalization of the Beta prior for more than 2 variables)

    Parameters
    ----------
    partition_prior : float, must be > 0.0 and < 1.0, typical value: 0.9
        The prior probability of splitting a node's data into two children.

        Small values tend to reduce the tree depth, leading to less expressiveness
        but also to less overfitting.

        Large values tend to increase the tree depth and thus lead to the tree
        better fitting the data, which can lead to overfitting.

    prior : array_like, shape = [number of classes]
        The hyperparameters [alpha_0, alpha_1, ..., alpha_{N-1}] of the Dirichlet
        conjugate prior, see [1] and [2]. All alpha_i must be positive, where
        alpha_i represents the number of prior pseudo-observations of class i.

        Small values for alpha_i represent a weak prior which leads to the
        training data dominating the posterior. This can lead to overfitting.

        Large values for alpha_i represent a strong prior and thus put less weight
        on the data. This can be used for regularization.

    level : DO NOT SET, ONLY USED BY SUBCLASSES

    See also
    --------
    demo_classification.py
    BinaryClassificationNode
    RegressionNode

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical/multinomial

    .. [2] https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions

    Examples
    --------
    See `demo_classification.py`.
    """

    # def __init__(self, partition_prior, prior, child_type, level=0):
    #     super().__init__(partition_prior, prior, child_type, False, level)

    def check_target(self, y):
        self._check_classification_target(y)

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
        if alphas is None:
            return np.nan * np.ones(self.prior.shape)

        return alphas / np.sum(alphas)

    def _compute_log_p_data(self, k, betaln_prior):
        alphas = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
        # which can be expressed as a fraction of beta functions
        return multivariate_betaln(alphas+k) - betaln_prior

    def _predict_leaf(self):
        # predict class
        return np.argmax(self.posterior)


class PerpendicularClassificationNode(BasePerpendicularNode, BaseClassificationNode):
    def __init__(self, partition_prior, prior, level=0):
        super().__init__(partition_prior, prior, PerpendicularClassificationNode, False, level)


class HyperplaneClassificationNode(BaseHyperplaneNode, BaseClassificationNode):
    def __init__(self, partition_prior, prior, optimizer, n_mc, use_polar, level=0):
        super().__init__(partition_prior, prior, HyperplaneClassificationNode, False, optimizer, n_mc, use_polar, level)
