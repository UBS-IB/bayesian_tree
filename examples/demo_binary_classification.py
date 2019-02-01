import argparse

from bayesian_decision_tree.classification import BinaryClassificationNode
from examples.helper import load_ripley, plot_1d, plot_2d


def parse_args():
    """Parse input arguments from the command line
    :return: the result from the ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Run demo of binary classification")

    parser.add_argument('--http_proxy', action='store',
                        required=False, help='HTTP Proxy', default=None
                        )
    parser.add_argument('--https_proxy', action='store',
                        required=False, help='HTTPS Proxy', default=None
                        )
    return parser.parse_args()


# demo script for binary classification
if __name__ == '__main__':

    args = parse_args()

    proxies = {
        'http': args.http_proxy,
        'https': args.https_proxy
    }
    print(proxies)
    # binary classification: Beta prior
    prior = (1, 1)

    # Bayesian decision tree parameters
    partition_prior = 0.9
    delta = 0

    root = BinaryClassificationNode(partition_prior, prior)

    # training/test data
    train, test = load_ripley(proxies)

    # train
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
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
