from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln


class Node(ABC):
    """A node class with either no children, in which case it's a leaf node, or exactly two children."""

    def __init__(self, name, partition_prior, prior, posterior, level, child_type, is_regression):
        self.name = name
        self.partition_prior = partition_prior
        self.prior = np.array(prior)
        self.posterior = np.array(posterior) if posterior is not None else np.array(prior)
        self.level = level
        self.child_type = child_type
        self.is_regression = is_regression

        # to be set later
        self.split_dimension = -1
        self.split_index = -1
        self.split_value = None
        self.split_feature_name = None
        self.child1 = None
        self.child2 = None

    def fit(self, X, y, delta=0, verbose=False, feature_names=None):
        """Trains this node with the given feature data matrix (or pandas DataFrame) X and the given target vector y."""

        if verbose:
            print('Training level {} with {:10} data points'.format(self.level, len(y)))

        if feature_names is None and type(X) is pd.DataFrame:
            feature_names = X.columns
            X = X.values

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
            log_p_data_post_split = self.compute_log_p_data_post_split(y_sorted, split_indices, n_dim)
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
            self.split_feature_name = feature_names[best_split_dimension] if feature_names is not None else None
            self.split_value = 0.5 * (X_sorted[best_split_index-1, best_split_dimension] + X_sorted[best_split_index, best_split_dimension])

            self.child1 = self.child_type(self.name + '-child1', self.partition_prior, prior_child1, posterior1, self.level+1)
            self.child2 = self.child_type(self.name + '-child2', self.partition_prior, prior_child2, posterior2, self.level+1)

            if len(X1) > 1:
                self.child1.fit(X1, y1, delta, verbose, feature_names)
            if len(X2) > 1:
                self.child2.fit(X2, y2, delta, verbose, feature_names)

    def predict(self, X):
        return self._predict(X, predict_class=True)

    def predict_proba(self, X):
        return self._predict(X, predict_class=False)

    def _predict(self, X, predict_class):
        if self.is_regression and not predict_class:
            # probability prediction for regressions makes no sense
            raise ValueError('Cannot predict probabilities for regression trees')

        if not self.is_leaf():
            # query children and then re-assemble

            # convert X to a correctly shaped numpy array
            if np.isscalar(X):
                X = np.array(X).reshape(-1, 1)
            elif type(X) is list:
                X = np.array(X).reshape(-1, 1)
            elif len(X.shape) == 1:
                X = X.reshape(-1, 1)

            # query both children, let them predict their side, and then re-assemble
            indices1 = np.where(X[:, self.split_dimension] < self.split_value)[0]
            indices2 = np.where(X[:, self.split_dimension] >= self.split_value)[0]
            predictions_merged = None
            if len(indices1) > 0:
                predictions1 = self.child1._predict(X[indices1], predict_class)
                predictions_merged = self._create_predictions_merged(X, predict_class, predictions1)
                predictions_merged[indices1] = predictions1

            if len(indices2) > 0:
                predictions2 = self.child2._predict(X[indices2], predict_class)
                if predictions_merged is None:
                    predictions_merged = self._create_predictions_merged(X, predict_class, predictions2)
                predictions_merged[indices2] = predictions2

            return predictions_merged
        else:
            # no children -> predict leaf
            return self._predict_leaf() if predict_class else self.compute_posterior_mean().reshape(1, -1)

    @staticmethod
    def _create_predictions_merged(X, predict_class, predictions_child):
        # class predictions: 1D array
        # probability predictions: 2D array
        return np.zeros(len(X)) if predict_class else np.zeros((len(X), predictions_child.shape[1]))

    def depth_and_leaves(self):
        return self._update_depth_and_leaves(0, 0)

    def _update_depth_and_leaves(self, depth, leaves):
        if self.is_leaf():
            return max(depth, self.level), leaves+1
        else:
            if self.child1 is not None:
                depth, leaves = self.child1._update_depth_and_leaves(depth, leaves)

            if self.child2 is not None:
                depth, leaves = self.child2._update_depth_and_leaves(depth, leaves)

        return depth, leaves

    def is_leaf(self):
        return self.split_index == -1

    @abstractmethod
    def check_target(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_no_split(self, y):
        pass

    @abstractmethod
    def compute_log_p_data_post_split(self, y, split_indices, n_dim):
        pass

    @abstractmethod
    def compute_posterior(self, y, delta=1):
        pass

    @abstractmethod
    def compute_posterior_mean(self):
        pass

    @abstractmethod
    def _predict_leaf(self):
        pass

    def __str__(self):
        return self._str([], self.split_value, None)

    def __str__(self):
        return self._str([], self.split_value, '\u2523', '\u2517', '\u2503', '\u2265', None)

    def _str(self, anchor, parent_split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, is_first_child):
        anchor_str = ''.join(' ' + a for a in anchor)
        s = ''
        if is_first_child is not None:
            s += anchor_str + ' {}{}: '.format('<' if is_first_child else GEQ, parent_split_value)

        if self.is_leaf():
            s += 'y={}'.format(self._predict_leaf())
            if not self.is_regression:
                s += ', p(y)={}'.format(self.predict_proba(parent_split_value)[0])
        else:
            split_feature_name = self.split_feature_name if self.split_feature_name is not None else 'x{}'.format(self.split_dimension)
            s += '{}={}'.format(split_feature_name, self.split_value)

            s += '\n'
            anchor_child1 = [VERT_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_first_child else '  '), VERT_RIGHT])
            s += self.child1._str(anchor_child1, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, True)

            s += '\n'
            anchor_child2 = [DOWN_RIGHT] if len(anchor) == 0 else (anchor[:-1] + [(BAR if is_first_child else '  '), DOWN_RIGHT])
            s += self.child2._str(anchor_child2, self.split_value, VERT_RIGHT, DOWN_RIGHT, BAR, GEQ, False)
        return s

    def __repr__(self):
        return self.__str__()


class BinaryClassificationNode(Node):
    """Concrete node implementation for binary classification using a Beta(alpha, beta) prior."""

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


def multivariate_betaln(alphas):
    if len(alphas) == 2:
        return betaln(alphas[0], alphas[1])
    else:
        # see https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function
        return np.sum([gammaln(alpha) for alpha in alphas], axis=0) - gammaln(alphas.sum())


class MultiClassificationNode(Node):
    """Concrete node implementation for multi-class classification using a Dirichlet prior."""

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


class RegressionNode(Node):
    """
    Concrete node implementation for regression using a Normal-Gamma(mu, kappa, alpha, beta) prior
    for unknown mean and unknown variance.
    """

    def __init__(self, name, partition_prior, prior, posterior=None, level=0):
        super().__init__(name, partition_prior, prior, posterior, level, RegressionNode, True)

    def check_target(self, y):
        pass

    def compute_log_p_data_post_no_split(self, y):
        n = len(y)
        mean = y.mean()

        y_minus_mean_sq_sum = ((y - mean)**2).sum()
        mu_post, kappa_post, alpha_post, beta_post = self._compute_posterior(n, mean, y_minus_mean_sq_sum)
        log_p_prior = np.log(1 - self.partition_prior**(1 + self.level))
        log_p_data = self._compute_log_p_data(alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def compute_log_p_data_post_split(self, y, split_indices, n_dim):
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

        mu1, kappa1, alpha1, beta1 = self._compute_posterior(n1, mean1, y_minus_mean_sq_sum1)
        mu2, kappa2, alpha2, beta2 = self._compute_posterior(n2, mean2, y_minus_mean_sq_sum2)

        log_p_prior = np.log(self.partition_prior**(1+self.level) / (n_splits * n_dim))
        log_p_data1 = self._compute_log_p_data(alpha1, beta1, kappa1, n1)
        log_p_data2 = self._compute_log_p_data(alpha2, beta2, kappa2, n2)

        return log_p_prior + log_p_data1 + log_p_data2

    def compute_posterior(self, y, delta=1):
        if delta == 0:
            return self.prior

        n = len(y)
        mean = y.mean()
        y_minus_mean_sq_sum = ((y - mean)**2).sum()

        return self._compute_posterior(n, mean, y_minus_mean_sq_sum, delta)

    def _compute_posterior(self, n, mean, y_minus_mean_sq_sum, delta=1):
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

    def _predict_leaf(self):
        # predict posterior mean
        return self.compute_posterior_mean()

    def _compute_log_p_data(self, alpha_new, beta_new, kappa_new, n_new):
        mu, kappa, alpha, beta = self.prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equation (95)
        return (gammaln(alpha_new) - gammaln(alpha)
                + alpha*np.log(beta) - alpha_new*np.log(beta_new)
                + 0.5*np.log(kappa/kappa_new)
                - 0.5*n_new*np.log(2*np.pi))
