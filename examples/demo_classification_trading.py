import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, inv, eig
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

from bayesian_decision_tree.classification import PerpendicularClassificationTree


def get_covariance(sigma: float, delta: float, theta: np.ndarray) -> np.ndarray:
    theta_p = theta + theta.T
    return (sigma ** 2.0) * inv(theta_p) * (np.eye(theta.shape[0]) - expm(-theta_p * delta))


def sample_gaussian(n: int, covariance: np.ndarray) -> np.ndarray:
    d, v = eig(covariance)
    a = np.dot(v, np.diag(np.sqrt(np.real(d))))
    g = np.random.normal(0.0, 1.0, (a.shape[0], n))
    return np.dot(a, g)


def sample_mean_reversion(n: int, x0: np.ndarray, mu: np.ndarray, sigma: float, delta: float,
                          theta: np.ndarray) -> np.ndarray:
    if not positive_eigenvalues(theta):
        raise AssertionError("Input theta does not have all positive eigenvalues")
    covariance = get_covariance(sigma, delta, theta)
    if not positive_eigenvalues(covariance):
        raise AssertionError("Covariance does not have all positive eigenvalues")
    gaussian_matrix = sample_gaussian(n, covariance)
    sample_paths = np.ndarray(gaussian_matrix.shape)
    sample_paths[:, [0]] = x0
    exp_theta = expm(-theta * delta)
    for i in range(1, sample_paths.shape[1]):
        prev = sample_paths[:, [i - 1]]
        sample_paths[:, [i]] = mu + np.dot(exp_theta, (prev - mu)) + gaussian_matrix[:, [i - 1]]
    return sample_paths


def positive_eigenvalues(theta: np.ndarray) -> bool:
    d, v = eig(theta)
    return np.all(np.real(d) > 0.0)


# demo script for classification (binary or multiclass) using classic, axis-normal splits
if __name__ == '__main__':
    np.random.seed(0)
    default_font_size = 16
    model_type = 'tree'  # it can be 'tree' or 'nn'
    plt.rc('axes', titlesize=default_font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=default_font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=default_font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=default_font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=default_font_size)  # legend fontsize
    plt.rc('figure', titlesize=default_font_size)  # fontsize of the figure title
    n = 10_000
    n += 1  # used for the deltas
    mu = np.array([[100.0], [110.0], [105.0]])
    theta = np.array([[2.0, -0.5, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.1]])
    dt = 0.1
    sigma = 1.0
    d = mu.shape[0]
    paths = sample_mean_reversion(n, mu, mu, sigma, dt, theta)
    x = paths.T
    plt.plot(x)
    plt.hlines(mu, 0, n, linestyles=d * ['--'], zorder=100)
    plt.title('Stock prices')
    plt.legend(['Stock A', 'Stock B', 'Stock C'])
    ax = plt.gca()
    ax.set_xlim([0, n])
    ax.set_ylim([90, 120])
    plt.savefig('trading_example_prices.png')
    plt.show()

    # artificial 4-class data somewhat similar to the Ripley data
    y_diff = np.diff(x, axis=0)
    x = x[:-1, :]
    y = np.dot((np.sign(y_diff) + 1) / 2, np.reshape(2.0 ** np.arange(d), (d, 1))).astype(int)
    n_train = int(x.shape[0] * 0.8)
    X_train = x[:n_train, :]
    y_train = y[:n_train, :]
    X_test = x[n_train:, :]
    y_test = y[n_train:, :]
    y_diff_test = y_diff[n_train:, :]
    n_classes = len(np.unique(y))

    # prior
    prior_strength = 1
    prior = prior_strength * np.array(n_classes * [1.0]) / n_classes

    # model
    if model_type is 'tree':
        model = PerpendicularClassificationTree(
            partition_prior=0.9,
            prior=prior,
            delta=0,
            prune=False)
    elif model_type is 'nn':
        model = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            random_state=0)
    else:
        raise AssertionError('Model not included ' + model_type)

    # train
    model.fit(X_train, y_train)
    print(model)
    print()

    # compute accuracy
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    positions = (2 * (y_pred_test.reshape((y_pred_test.shape[0], 1)) // 2.0 ** np.arange(d).astype(int) % 2) - 1)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    info_train = 'Train accuracy: {:.4f} %'.format(100 * accuracy_train)
    info_test = 'Test accuracy:  {:.4f} %'.format(100 * accuracy_test)
    print(info_train)
    print(info_test)

    pnl = np.cumsum(positions * y_diff_test, axis=0)
    plt.plot(pnl)
    plt.hlines(0, 0, pnl.shape[0])
    ax = plt.gca()
    ax.set_xlim([0, pnl.shape[0]])
    ax.set_ylim(np.array([-30, 200]))
    plt.grid(True)
    plt.title('Test period PnL')
    plt.legend(['Stock A', 'Stock B', 'Stock C'])
    plt.savefig('trading_example_pnl_' + model_type + '.png')
    plt.show()

    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=[''.join(
                                     np.core.defchararray.add(['-' if x < 0 else '+' for x in (2 * row - 1)],
                                                              ['A', 'B', 'C'])) for row in
                                     np.reshape(np.arange(2 ** d), (2 ** d, 1)) // 2.0 ** np.arange(
                                         d).astype(int) % 2],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title('Test period confusion matrix')
    plt.xticks(rotation=90)
    plt.savefig('trading_example_confusion_matrix_' + model_type + '.png', bbox_inches='tight')
    plt.show()
