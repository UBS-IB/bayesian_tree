from bayesian_decision_tree import *
from demo_helper import *
from sklearn.metrics import mean_squared_error


# demo script for regression
if __name__ == '__main__':
    proxies = {
        'http': 'SET_HTTP_PROXY',
        'https': 'SET_HTTPS_PROXY'
    }

    color0 = 'b'
    color1 = 'r'

    # regression: Normal-Gamma prior, see https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
    mu = 0  # probably better to choose the target mean
    sd_prior = 0.1
    prior_obs = 1
    kappa = prior_obs
    alpha = prior_obs/2
    var_prior = sd_prior**2
    tau_prior = 1/var_prior
    beta = alpha/tau_prior

    prior = (mu, kappa, alpha, beta)

    # Bayesian tree parameters
    partition_prior = 0.9
    delta = 0

    root = RegressionNode('root', partition_prior, prior, prior, 0)

    # training/test data
    train = np.hstack((
        np.linspace(0, 10, 100).reshape(-1, 1),
        np.sin(np.linspace(0, 10, 100)).reshape(-1, 1)
    ))
    test = train

    # train
    data = train[:, :-1]
    target = train[:, -1]
    root.train(data, target, delta)
    print(root)
    print()

    # compute RMSE
    rmse_train = np.sqrt(mean_squared_error(root.predict(train[:, :-1]), train[:, -1]))
    rmse_test = np.sqrt(mean_squared_error(root.predict(test[:, :-1]), test[:, -1]))
    info_train = 'RMSE train: {:.2f}'.format(rmse_train)
    info_test = 'RMSE test:  {:.2f}'.format(rmse_test)
    print(info_train)
    print(info_test)

    # plot if 1D or 2D
    dimensions = train.shape[1]-1
    if dimensions == 1:
        plot_1d(root, train, info_train, test, info_test)
    elif dimensions == 2:
        plot_2d(root, train, info_train, test, info_test, color0, color1)
