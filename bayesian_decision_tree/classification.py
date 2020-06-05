"""
This module declares the Bayesian classification tree models:
* PerpendicularClassificationTree
* HyperplaneClassificationTree
"""
import numpy as np
from abc import ABC
from sklearn.base import ClassifierMixin

from bayesian_decision_tree.base import BaseTree
from bayesian_decision_tree.base_hyperplane import BaseHyperplaneTree
from bayesian_decision_tree.base_perpendicular import BasePerpendicularTree
from bayesian_decision_tree.utils import multivariate_betaln


class BaseClassificationTree(BaseTree, ABC, ClassifierMixin):
    """
    Abstract base class of all Bayesian classification trees (perpendicular and hyperplane). Performs
    medium-level fitting and prediction tasks and outsources the low-level work to subclasses.
    """

    def __init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level=0):
        BaseTree.__init__(self, partition_prior, prior, delta, prune, child_type, False, split_precision, level)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix or pandas.DataFrame, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """

        # input transformation and checks
        X, _ = self._normalize_data_and_feature_names(X)
        self._ensure_is_fitted(X)

        return self._predict(X, indices=np.arange(X.shape[0]), predict_class=False)

    def _check_target(self, y):
        if y.ndim != 1:
            raise ValueError('y should have 1 dimension but has {}'.format(y.ndim))

        n_classes = len(self.prior)
        if not np.all(np.unique(y) == np.arange(0, n_classes)):
            raise ValueError('Expected target values 0..{} but found {}..{}'.format(n_classes - 1, y.min(), y.max()))

    def _get_prior(self, n_data, n_dim):
        if self.prior is not None:
            return self.prior
        else:
            prior_pseudo_observation_count = max(1, n_data//100)
            return prior_pseudo_observation_count * np.ones(n_dim)

    def _compute_log_p_data_no_split(self, y, prior):
        posterior = self._compute_posterior(y, prior)

        log_p_prior = np.log(1-self.partition_prior**(1+self.level))
        log_p_data = multivariate_betaln(posterior) - multivariate_betaln(prior)

        return log_p_prior + log_p_data

    def _compute_log_p_data_split(self, y, prior, n_dim, split_indices):
        n_classes = len(prior)
        k1 = np.empty(n_classes, dtype=object)
        k2 = np.empty(n_classes, dtype=object)
        for i in range(n_classes):
            k1_and_total = (y == i).cumsum()
            total = k1_and_total[-1]
            k1[i] = k1_and_total[split_indices-1]
            k2[i] = total - k1[i]

        n_splits = len(split_indices)
        log_p_prior = np.log(self.partition_prior**(1+self.level) / (n_splits * n_dim))

        betaln_prior = multivariate_betaln(prior)
        log_p_data1 = self._compute_log_p_data(k1, prior, betaln_prior)
        log_p_data2 = self._compute_log_p_data(k2, prior, betaln_prior)

        return log_p_prior + log_p_data1 + log_p_data2

    def _compute_posterior(self, y, prior, delta=1):
        if delta == 0:
            return prior

        # see https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
        y_reshaped = np.broadcast_to(y, (len(prior), len(y)))
        classes = np.arange(len(prior)).reshape(-1, 1)
        k = np.sum(y_reshaped == classes, axis=1)
        posterior = prior + delta*k

        return posterior

    def _compute_posterior_mean(self):
        return self.posterior_ / np.sum(self.posterior_)

    def _compute_log_p_data(self, k, prior, betaln_prior):
        # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
        # which can be expressed as a fraction of beta functions
        return multivariate_betaln(prior+k) - betaln_prior

    def _predict_leaf(self):
        # predict class
        return np.argmax(self.posterior_)

    def _get_raw_leaf_data_internal(self):
        # prior and posterior raw data
        return np.array([self.prior, self.posterior_])


class PerpendicularClassificationTree(BasePerpendicularTree, BaseClassificationTree):
    """
    Bayesian binary or multiclass classification tree. Uses a Dirichlet prior (a
    multivariate generalization of the Beta prior for more than 2 variables).

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

    delta : float, default=0.0
        Determines the strengthening of the prior as the tree grows deeper,
        see [1]. Must be a value between 0.0 and 1.0.

    prune : boolean, default=False
        Prunes the tree after fitting if `True` by removing all splits that don't add information,
        i.e., where the predictions of both children are identical. It's usually sensible to set
        this to `True` in the classification case if you're only interested in class predictions
        (`predict(X)`), but it makes sense to set it to `False` if you're looking for class
        probabilities (`predict_proba(X)`). It can safely be set to 'True' in the regression case
        because it will only merge children if their predictions are identical.

    split_precision : float, default=0.0
        Determines the minimum distance between two contiguous points to consider a split. If the distance is below
        this threshold, the points are considered to overlap along this direction.

    level : DO NOT SET, ONLY USED BY SUBCLASSES

    See also
    --------
    demo_classification_perpendicular.py
    PerpendicularRegressionTree
    HyperplaneClassificationTree

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical/multinomial

    .. [2] https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions

    Examples
    --------
    See `demo_classification_perpendicular.py`.
    """

    def __init__(self, partition_prior=0.99, prior=None, delta=0, prune=False, split_precision=0.0, level=0):
        child_type = PerpendicularClassificationTree
        BasePerpendicularTree.__init__(self, partition_prior, prior, delta, prune, child_type, False, split_precision, level)
        BaseClassificationTree.__init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level)


class HyperplaneClassificationTree(BaseHyperplaneTree, BaseClassificationTree):
    """
    Bayesian binary or multiclass classification tree using arbitrarily-oriented
    hyperplane splits. Uses a Dirichlet prior (a multivariate generalization
    of the Beta prior for more than 2 variables).

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

    delta : float, default=0.0
        Determines the strengthening of the prior as the tree grows deeper,
        see [1]. Must be a value between 0.0 and 1.0.

    prune : boolean, default=False
        Prunes the tree after fitting if `True` by removing all splits that don't add information,
        i.e., where the predictions of both children are identical. It's usually sensible to set
        this to `True` in the classification case if you're only interested in class predictions
        (`predict(X)`), but it makes sense to set it to `False` if you're looking for class
        probabilities (`predict_proba(X)`). It can safely be set to 'True' in the regression case
        because it will only merge children if their predictions are identical.

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
    demo_classification_hyperplane.py
    HyperplaneRegressionTree
    PerpendicularClassificationTree

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical/multinomial

    .. [2] https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions

    Examples
    --------
    See `demo_classification_perpendicular.py`.
    """

    def __init__(self, partition_prior=0.99, prior=None, delta=None, prune=False, optimizer=None, split_precision=0.0, level=0):
        child_type = HyperplaneClassificationTree
        BaseHyperplaneTree.__init__(self, partition_prior, prior, delta, prune, child_type, False, optimizer, split_precision, level)
        BaseClassificationTree.__init__(self, partition_prior, prior, delta, prune, child_type, split_precision, level)
