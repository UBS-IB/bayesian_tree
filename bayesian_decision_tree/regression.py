"""
This module declares the Bayesian regression tree models:
* PerpendicularRegressionTree
* HyperplaneRegressionTree
"""
import numpy as np
from abc import ABC
from scipy.special import gammaln
from sklearn.base import RegressorMixin

from bayesian_decision_tree.base import BaseTree
from bayesian_decision_tree.base_hyperplane import BaseHyperplaneTree
from bayesian_decision_tree.base_perpendicular import BasePerpendicularTree


class BaseRegressionTree(BaseTree, ABC, RegressorMixin):
    """
    Abstract base class of all Bayesian regression trees (perpendicular and hyperplane). Performs
    medium-level fitting and prediction tasks and outsources the low-level work to subclasses.
    """

    def __init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level=0):
        BaseTree.__init__(self, partition_prior, prior, delta, prune, child_type, True, split_precision, level)

    def _check_target(self, y):
        if y.ndim != 1:
            raise ValueError('y should have 1 dimension but has {}'.format(y.ndim))

    def _compute_log_p_data_no_split(self, y, prior):
        y_sum = y.sum()
        y_squared_sum = (y ** 2).sum()
        n = len(y)
        mu_post, kappa_post, alpha_post, beta_post = self._compute_posterior_internal(prior, n, y_sum, y_squared_sum)
        log_p_prior = np.log(1 - self.partition_prior**(1 + self.level))
        log_p_data = self._compute_log_p_data(prior, alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def _compute_log_p_data_split(self, y, prior, n_dim, split_indices):
        n = len(y)
        n1 = np.arange(1, n)
        n2 = n - n1
        y_sum1 = y.cumsum()[:-1]
        y_sum2 = y.sum() - y_sum1
        y_squared_sum1 = (y[:-1] ** 2).cumsum()
        y_squared_sum2 = (y ** 2).sum() - y_squared_sum1

        if len(split_indices) != len(y)-1:
            # we are *not* splitting between all data points -> indexing necessary
            split_indices_minus_1 = split_indices - 1

            n1 = n1[split_indices_minus_1]
            n2 = n2[split_indices_minus_1]
            y_sum1 = y_sum1[split_indices_minus_1]
            y_sum2 = y_sum2[split_indices_minus_1]
            y_squared_sum1 = y_squared_sum1[split_indices_minus_1]
            y_squared_sum2 = y_squared_sum2[split_indices_minus_1]

        mu1, kappa1, alpha1, beta1 = self._compute_posterior_internal(prior, n1, y_sum1, y_squared_sum1)
        mu2, kappa2, alpha2, beta2 = self._compute_posterior_internal(prior, n2, y_sum2, y_squared_sum2)

        n_splits = len(split_indices)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / (n_splits * n_dim))

        log_p_data1 = self._compute_log_p_data(prior, alpha1, beta1, kappa1, n1)
        log_p_data2 = self._compute_log_p_data(prior, alpha2, beta2, kappa2, n2)

        return log_p_prior + log_p_data1 + log_p_data2

    def _get_prior(self, n_data, n_dim):
        if self.prior is not None:
            return self.prior
        else:
            # TODO: use actual data to compute mu and tau
            prior_pseudo_observation_count = max(1, n_data//100)
            mu = 0
            tau = 1
            kappa = prior_pseudo_observation_count
            alpha = prior_pseudo_observation_count/2
            beta = alpha/tau
            return np.array([mu, kappa, alpha, beta])

    def _compute_posterior(self, y, prior, delta=1):
        if delta == 0:
            return prior

        n = len(y)
        y_sum = y.sum()
        y_squared_sum = (y ** 2).sum()

        return self._compute_posterior_internal(prior, n, y_sum, y_squared_sum, delta)

    def _compute_posterior_internal(self, prior, n, y_sum, y_squared_sum, delta=1):
        mu, kappa, alpha, beta = prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equations (86) - (89)
        n_delta = n*delta
        kappa_post = kappa + n_delta
        mu_post = (kappa * mu + n_delta * y_sum / n) / kappa_post
        alpha_post = alpha + 0.5*n_delta
        beta_post = beta + 0.5 * delta * (y_squared_sum - y_sum ** 2 / n) + 0.5 * kappa * n_delta * (
                    y_sum / n - mu) ** 2 / (kappa + n)

        return mu_post, kappa_post, alpha_post, beta_post

    def _compute_posterior_mean(self):
        return self.posterior_[0]  # mu is the posterior mean

    def _compute_log_p_data(self, prior, alpha_new, beta_new, kappa_new, n_new):
        mu, kappa, alpha, beta = prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equation (95)
        return (gammaln(alpha_new) - gammaln(alpha)
                + alpha*np.log(beta) - alpha_new*np.log(beta_new)
                + 0.5*np.log(kappa/kappa_new)
                - 0.5*n_new*np.log(2*np.pi))

    def _predict_leaf(self):
        # predict posterior mean
        return self._compute_posterior_mean()

    def _get_raw_leaf_data_internal(self):
        # prior and posterior raw data
        return np.array([self.prior, self.posterior_])


class PerpendicularRegressionTree(BasePerpendicularTree, BaseRegressionTree):
    """
    Bayesian regression tree using axes-normal splits ("perpendicular").
    Uses a Normal-gamma(mu, kappa, alpha, beta) prior assuming unknown mean and unknown variance.

    Parameters
    ----------
    partition_prior : float, must be > 0.0 and < 1.0, typical value: 0.9
        The prior probability of splitting a node's data into two children.

        Small values tend to reduce the tree depth, leading to less expressiveness
        but also to less overfitting.

        Large values tend to increase the tree depth and thus lead to the tree
        better fitting the data, which can lead to overfitting.

    prior : array_like, shape = [4]
        The prior hyperparameters [mu, kappa, alpha, beta] of the Normal-gamma
        distribution (see also [1], [2], [3]):

        - mu:    prior pseudo-observation sample mean
        - kappa: prior pseudo-observation count used to compute mu
        - alpha: (prior pseudo-observation count used to compute sample variance)/2
        - beta:  alpha * (prior pseudo-observation sample variance)

        It is usually easier to compute these hyperparameters off more intuitive
        base quantities, see examples section.

    delta : float, default=0.0
        Determines the strengthening of the prior as the tree grows deeper,
        see [1]. Must be a value between 0.0 and 1.0.

    split_precision : float, default=0.0
        Determines the minimum distance between two contiguous points to consider a split. If the distance is below
        this threshold, the points are considered to overlap along this direction.

    level : DO NOT SET, ONLY USED BY SUBCLASSES

    See also
    --------
    demo_regression_perpendicular.py
    PerpendicularClassificationTree
    HyperplaneRegressionTree

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    .. [2] https://en.wikipedia.org/wiki/Normal-gamma_distribution#Interpretation_of_parameters

    .. [3] https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions

    Examples
    --------
    It is usually convenient to compute the prior hyperparameters as follows:

    >>> # prior mean; set to the mean of the target
    >>> mu = ...
    >>>
    >>> # prior standard deviation; set to about 0.1 times the standard deviation of the target
    >>> sd_prior = ...
    >>>
    >>> # the number of prior pseudo-observations; set to roughly 1 - 10 % of the number of training samples
    >>> prior_pseudo_observations = ...
    >>>
    >>> # now compute the prior
    >>> kappa = prior_pseudo_observations
    >>> alpha = prior_pseudo_observations/2
    >>> beta = alpha*sd_prior**2
    >>> prior = [mu, kappa, alpha, beta]

    See `demo_regression_perpendicular.py`.
    """

    def __init__(self, partition_prior=0.99, prior=None, delta=0, prune=False, split_precision=0.0, level=0):
        child_type = PerpendicularRegressionTree
        BasePerpendicularTree.__init__(self, partition_prior, prior, delta, prune, child_type, True, split_precision, level)
        BaseRegressionTree.__init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level)


class HyperplaneRegressionTree(BaseHyperplaneTree, BaseRegressionTree):
    """
    Bayesian regression tree using arbitrarily-oriented hyperplane splits.
    Uses a Normal-gamma(mu, kappa, alpha, beta) prior assuming unknown mean and unknown variance.

    Parameters
    ----------
    partition_prior : float, must be > 0.0 and < 1.0, typical value: 0.9
        The prior probability of splitting a node's data into two children.

        Small values tend to reduce the tree depth, leading to less expressiveness
        but also to less overfitting.

        Large values tend to increase the tree depth and thus lead to the tree
        better fitting the data, which can lead to overfitting.

    prior : array_like, shape = [4]
        The prior hyperparameters [mu, kappa, alpha, beta] of the Normal-gamma
        distribution (see also [1], [2], [3]):

        - mu:    prior pseudo-observation sample mean
        - kappa: prior pseudo-observation count used to compute mu
        - alpha: (prior pseudo-observation count used to compute sample variance)/2
        - beta:  alpha * (prior pseudo-observation sample variance)

        It is usually easier to compute these hyperparameters off more intuitive
        base quantities, see examples section.

    delta : float, default=0.0
        Determines the strengthening of the prior as the tree grows deeper,
        see [1]. Must be a value between 0.0 and 1.0.

    optimizer : object
        A global optimization algorithm object that performs optimal hyperparameter
        orientation search. The available options are (in the order in which you should
        try them):
        - ScipyOptimizer: A wrapper around scipy global optimizers. See usages for examples.
        - SimulatedAnnealingOptimizer: Experimental, but works well with n_scan=20, n_keep=10, spread_factor=0.95
        - RandomHyperplaneOptimizer: Experimental, mediocre performance
        - RandomTwoPointOptimizer: Experimental, mediocre performance
        - GradientDescentOptimizer: Experimental, mediocre performance

    split_precision : float, default=0.0
        Determines the minimum distance between two contiguous points to consider a split. If the distance is below
        this threshold, the points are considered to overlap along this direction.

    level : DO NOT SET, ONLY USED BY SUBCLASSES

    See also
    --------
    demo_regression_hyperplane.py
    HyperplaneClassificationTree
    PerpendicularRegressionTree

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    .. [2] https://en.wikipedia.org/wiki/Normal-gamma_distribution#Interpretation_of_parameters

    .. [3] https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions

    Examples
    --------
    It is usually convenient to compute the prior hyperparameters in the same manner as for
    the perpendicular case, see PerpendicularRegressionTree.

    See `demo_regression_hyperplane.py`.
    """

    def __init__(self, partition_prior=0.99, prior=None, delta=0, prune=False, optimizer=None, split_precision=0.0, level=0):
        child_type = HyperplaneRegressionTree
        BaseHyperplaneTree.__init__(self, partition_prior, prior, delta, prune, child_type, True, optimizer, split_precision, level)
        BaseRegressionTree.__init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level)
