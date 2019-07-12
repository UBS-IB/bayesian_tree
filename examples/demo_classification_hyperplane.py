import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from sklearn.metrics import accuracy_score

from bayesian_decision_tree.classification import PerpendicularClassificationTree, HyperplaneClassificationTree
from bayesian_decision_tree.hyperplane_optimization import SimulatedAnnealingOptimizer, ScipyOptimizer
from examples import helper

# demo script for classification (binary or multiclass) using arbitrarily oriented hyperplanes
if __name__ == '__main__':
    # proxies (in case you're running this behind a firewall)
    args = helper.parse_args()
    proxies = {
        'http': args.http_proxy,
        'https': args.https_proxy
    }

    # data set: uncomment one of the following sections

    # artificial 4-class data somewhat similar to the Ripley data
    n_train = 500
    n_test = 2000
    x0 = [1, 3, 2, 4]
    x1 = [1, 1, 3, 3]
    sd = 0.7
    X_train = np.zeros((n_train, 2))
    y_train = np.zeros((n_train, 1))
    X_test = np.zeros((n_test, 2))
    y_test = np.zeros((n_test, 1))
    np.random.seed(666)
    for i in range(4):
        X_train[i * n_train//4:(i + 1) * n_train//4, 0] = np.random.normal(x0[i], sd, n_train//4)
        X_train[i * n_train//4:(i + 1) * n_train//4, 1] = np.random.normal(x1[i], sd, n_train//4)
        y_train[i * n_train//4:(i + 1) * n_train//4] = i

        X_test[i * n_test//4:(i + 1) * n_test//4, 0] = np.random.normal(x0[i], sd, n_test//4)
        X_test[i * n_test//4:(i + 1) * n_test//4, 1] = np.random.normal(x1[i], sd, n_test//4)
        y_test[i * n_test//4:(i + 1) * n_test//4] = i
    train = np.hstack((X_train, y_train))
    test = np.hstack((X_test, y_test))

    # or, alternatively, load a UCI dataset
    # train, test = helper.load_ripley(proxies)

    n_dim = len(np.unique(train[:, -1]))

    if train is test:
        # perform a 50:50 train:test split if no test data is given
        train = train[0::2]
        test = test[1::2]

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    # prior
    prior_pseudo_observations = 5
    prior = prior_pseudo_observations * np.ones(n_dim)

    # Bayesian decision tree parameters
    partition_prior = 0.9
    delta = 0

    # model
    model = HyperplaneClassificationTree(partition_prior, prior, optimizer=SimulatedAnnealingOptimizer(10, 10, 0.9, 666))

    # train
    model.fit(X_train, y_train, delta, prune=True)
    print(model)
    print()
    print('Tree depth and number of leaves: {}, {}'.format(model.get_depth(), model.get_n_leaves()))
    print('Feature importance:', model.feature_importance())

    # compute accuracy
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    info_train = 'Train accuracy: {:.4f} %'.format(100 * accuracy_train)
    info_test = 'Test accuracy:  {:.4f} %'.format(100 * accuracy_test)
    print(info_train)
    print(info_test)

    # plot if 2D
    dimensions = X_train.shape[1]
    if dimensions == 2:
        helper.plot_2d_hyperplane(model, X_train, y_train, info_train, X_test, y_test, info_test)
