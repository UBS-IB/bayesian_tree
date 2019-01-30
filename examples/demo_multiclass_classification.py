import numpy as np
from bayesian_decision_tree.classification import MultiClassificationNode
from examples.helper import plot_1d, plot_2d


# demo script for multi-class classification
if __name__ == '__main__':
    # binary classification: Beta prior
    prior = (1, 1, 1, 1)

    # Bayesian tree parameters
    partition_prior = 0.9
    delta = 0

    root = MultiClassificationNode('root', partition_prior, prior)

    # training/test data: artificial 4-class data somewhat similar to the Ripley data
    n_train = 500
    n_test = 2000
    x1 = [1, 3, 2, 4]
    x2 = [1, 1, 3, 3]
    sd = 0.7
    train = np.zeros((n_train, 3))
    test = np.zeros((n_test, 3))
    np.random.seed(666)
    for i in range(4):
        train[i * n_train // 4:(i + 1) * n_train // 4, 0] = np.random.normal(x1[i], sd, n_train // 4)
        train[i * n_train // 4:(i + 1) * n_train // 4, 1] = np.random.normal(x2[i], sd, n_train // 4)
        train[i * n_train // 4:(i + 1) * n_train // 4, 2] = i

        test[i * n_test // 4:(i + 1) * n_test // 4, 0] = np.random.normal(x1[i], sd, n_test // 4)
        test[i * n_test // 4:(i + 1) * n_test // 4, 1] = np.random.normal(x2[i], sd, n_test // 4)
        test[i * n_test // 4:(i + 1) * n_test // 4, 2] = i

    # train
    X = train[:, :-1]
    y = train[:, -1]
    root.fit(X, y, delta)
    print(root)
    print()
    print('Tree depth and number of leaves:', root.depth_and_leaves())

    # compute accuracy
    prediction_train = root.predict(train[:, :-1])
    prediction_test = root.predict(test[:, :-1])
    accuracy_train = (prediction_train == train[:, -1]).mean()
    accuracy_test = (prediction_test == test[:, -1]).mean()
    info_train = 'Train accuracy: {:.4f} %'.format(100 * accuracy_train)
    info_test = 'Test accuracy:  {:.4f} %'.format(100 * accuracy_test)
    print(info_train)
    print(info_test)

    # plot if 1D or 2D
    dimensions = train.shape[1] - 1
    if dimensions == 1:
        plot_1d(root, train, info_train, test, info_test)
    elif dimensions == 2:
        plot_2d(root, train, info_train, test, info_test)
