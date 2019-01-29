import numpy as np
from scipy.special import betaln, gammaln


def multivariate_betaln(alphas):
    if len(alphas) == 2:
        return betaln(alphas[0], alphas[1])
    else:
        # see https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function
        return np.sum([gammaln(alpha) for alpha in alphas], axis=0) - gammaln(alphas.sum())