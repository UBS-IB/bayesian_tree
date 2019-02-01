import numpy as np

from bayesian_decision_tree.classification import MultiClassificationNode
from examples.helper import plot_1d, plot_2d

# demo script for multi-class classification
if __name__ == '__main__':
    # binary classification: Beta prior
    prior = (1, 1, 1, 1)

    # Bayesian decision tree parameters
    partition_prior = 0.9
    delta = 0

    root = MultiClassificationNode(partition_prior, prior)

    # training/test data: artificial 4-class data somewhat similar to the Ripley data
    n_train = 500
    n_test = 2000
    x0 = [1, 3, 2, 4]
    x1 = [1, 1, 3, 3]
    sd = 0.7
    X_train = np.zeros((n_train, 2))
    y_train = np.zeros(n_train)
    X_test = np.zeros((n_test, 2))
    y_test = np.zeros(n_test)
    np.random.seed(666)
    for i in range(4):
        X_train[i * n_train//4:(i + 1) * n_train//4, 0] = np.random.normal(x0[i], sd, n_train//4)
        X_train[i * n_train//4:(i + 1) * n_train//4, 1] = np.random.normal(x1[i], sd, n_train//4)
        y_train[i * n_train//4:(i + 1) * n_train//4] = i

        X_test[i * n_test//4:(i + 1) * n_test//4, 0] = np.random.normal(x0[i], sd, n_test//4)
        X_test[i * n_test//4:(i + 1) * n_test//4, 1] = np.random.normal(x1[i], sd, n_test//4)
        y_test[i * n_test//4:(i + 1) * n_test//4] = i

    # train
    root.fit(X_train, y_train, delta)
    print(root)
    print()
    print('Tree depth and number of leaves:', root.depth_and_leaves())

    # compute accuracy
    prediction_train = root.predict(X_train)
    prediction_test = root.predict(X_test)
    accuracy_train = (prediction_train == y_train).mean()
    accuracy_test = (prediction_test == y_test).mean()
    info_train = 'Train accuracy: {:.4f} %'.format(100 * accuracy_train)
    info_test = 'Test accuracy:  {:.4f} %'.format(100 * accuracy_test)
    print(info_train)
    print(info_test)

    # plot if 1D or 2D
    dimensions = X_train.shape[1]
    if dimensions == 1:
        plot_1d(root, X_train, y_train, info_train, X_test, y_test, info_test)
    elif dimensions == 2:
        plot_2d(root, X_train, y_train, info_train, X_test, y_test, info_test)
