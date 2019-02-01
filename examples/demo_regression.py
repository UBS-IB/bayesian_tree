import numpy as np
from sklearn.metrics import mean_squared_error

from bayesian_decision_tree.regression import RegressionNode
from examples.helper import plot_1d, plot_2d

# demo script for regression
if __name__ == '__main__':
    # training/test data
    X_train = np.linspace(0, 10, 100).reshape(-1, 1)
    y_train = 100 * np.sin(np.linspace(0, 10, 100)).reshape(-1, 1)
    X_test = X_train
    y_test = y_train

    # regression: Normal-Gamma prior, see https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
    mu = y_train.mean()
    sd_prior = y_train.std() / 10
    prior_obs = 1
    kappa = prior_obs
    alpha = prior_obs/2
    var_prior = sd_prior**2
    tau_prior = 1/var_prior
    beta = alpha/tau_prior

    prior = (mu, kappa, alpha, beta)

    # Bayesian decision tree parameters
    partition_prior = 0.9
    delta = 0

    root = RegressionNode(partition_prior, prior)

    # train
    root.fit(X_train, y_train, delta)
    print(root)
    print()
    print('Tree depth and number of leaves:', root.depth_and_leaves())

    # compute RMSE
    rmse_train = np.sqrt(mean_squared_error(root.predict(X_train), y_train))
    rmse_test = np.sqrt(mean_squared_error(root.predict(X_test), y_test))
    info_train = 'RMSE train: {:.4f}'.format(rmse_train)
    info_test = 'RMSE test:  {:.4f}'.format(rmse_test)
    print(info_train)
    print(info_test)

    # plot if 1D or 2D
    dimensions = X_train.shape[1]
    if dimensions == 1:
        plot_1d(root, X_train, y_train, info_train, X_test, y_test, info_test)
    elif dimensions == 2:
        plot_2d(root, X_train, y_train, info_train, X_test, y_test, info_test)
