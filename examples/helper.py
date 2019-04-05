"""
A collection of publicly available data sets to test classification algorithms on and some helper
functions for plotting.
"""
import argparse

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import patches
from matplotlib.pyplot import cm
from sklearn.preprocessing import LabelBinarizer


def parse_args():
    """Parse input arguments from the command line
    :return: the result from the ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Run demo of binary classification")

    parser.add_argument(
        '--http_proxy',
        action='store',
        required=False,
        help='HTTP Proxy',
        default=None)

    parser.add_argument(
        '--https_proxy',
        action='store', required=False,
        help='HTTPS Proxy',
        default=None)

    return parser.parse_args()


def one_hot_encode(data, columns):
    columns = sorted(set(columns))[::-1]

    def ensure_matrix(x):
        return x if x.ndim == 2 else np.array(x).reshape(-1, 1)

    for c in columns:
        one_hot = LabelBinarizer().fit_transform(data[:, c])
        data = np.hstack((
            ensure_matrix(data[:, :c]),
            ensure_matrix(one_hot),
            ensure_matrix(data[:, c+1:])
        ))

    return data


def load_credit(proxies):
    content = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
        proxies=proxies).content
    df = pd.read_excel(io.BytesIO(content))
    train = df.iloc[1:, 1:].values.astype(np.float64)
    train = one_hot_encode(train, [2, 3])  # one-hot encode categorical features
    test = train
    return train, test


def load_dermatology(proxies):
    # Dermatology
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data', proxies=proxies).text
    text = text.replace('?', '-1')
    lines = text.split('\n')
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    age = train[:, 33]
    has_age = age != -1.0
    train = np.hstack((
        train[:, :33],
        (has_age * 1).reshape(-1, 1),    # age yes/no feature
        (age * has_age).reshape(-1, 1),  # set age to 0 if missing
        train[:, 34].reshape(-1, 1)-1,
    ))
    # train[:, -1] -= 1
    test = train
    return train, test


def load_diabetic(proxies):
    # Diabetic Retinopathy
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff', proxies=proxies).text
    text = text[text.index('@data'):]
    lines = text.split('\n')[1:]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    test = train
    return train, test


def load_eeg(proxies):
    # load EEG eye data
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff', proxies=proxies).text
    text = text[text.index('@DATA'):]
    lines = text.split('\n')[1:]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    test = train
    return train, test


def load_gamma(proxies):
    # load MAGIC Gamma telescope data
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', proxies=proxies).text
    text = text.replace('g', '0').replace('h', '1')
    lines = text.split('\n')
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    test = train
    return train, test


def load_glass(proxies):
    # load glass identificaion data
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', proxies=proxies).text
    lines = text.split('\n')
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    train = train[:, 1:]  # ignore ID row
    train[:, -1] -= 1  # convert 1..7 to 0..6
    train[np.where(train[:, -1] >= 4)[0], -1] -= 1  # skip missing class
    test = train
    return train, test


def load_haberman(proxies):
    # load Haberman's dataset
    text = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
        proxies=proxies).text
    lines = text.split('\n')
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    train[:, -1] -= 1
    test = train
    return train, test


def load_heart(proxies):
    text = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', proxies=proxies).text
    lines = text.split('\n')
    train = np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines)-1)])
    train = one_hot_encode(train, [2, 6, 12])  # one-hot encode categorical features
    train[:, -1] -= 1
    test = train
    return train, test


def load_ripley(proxies):
    # load Ripley's synthetic dataset
    def parse_ripley(text):
        lines = text.split('\n')[1:]
        return np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines)-1)])
    train = parse_ripley(requests.get('https://www.stats.ox.ac.uk/pub/PRNN/synth.tr', proxies=proxies).text)
    test = parse_ripley(requests.get('https://www.stats.ox.ac.uk/pub/PRNN/synth.te', proxies=proxies).text)
    return train, test


def load_seismic(proxies):
    # load seismic bumps dataset
    text = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff', proxies=proxies).text
    text = text[text.index('@data'):]
    text = text.replace('a', '0').replace('b', '1').replace('c', '2').replace('d', '3')
    text = text.replace('N', '0').replace('W', '1')
    lines = text.split('\n')[1:]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    test = train
    return train, test


def plot_1d(root, X_train, y_train, info_train, X_test, y_test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)
    plt.subplot(211)
    plt.plot(X_train[:, 0], y_train, 'o-')
    plt.title(info_train)
    draw_node_1d(root, bounds=(X_train[:, 0].min(), X_train[:, 0].max()))
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()

    plt.subplot(212)
    plt.plot(X_test[:, 0], y_test, 'o-')
    draw_node_1d(root, bounds=(X_test[:, 0].min(), X_test[:, 0].max()))
    plt.title(info_test)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()

    plt.show()


def plot_2d(root, X_train, y_train, info_train, X_test, y_test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)

    n_classes = int(y_train.max()) + 1
    colormap = cm.gist_rainbow

    def plot(X, y, info):
        for i in range(n_classes):
            class_i = y == i
            plt.plot(X[np.where(class_i)[0], 0],
                     X[np.where(class_i)[0], 1],
                     'o',
                     ms=4,
                     c=colormap(i/n_classes),
                     label='Class {}'.format(i))

            bounds = ((X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max()))
            draw_node_2d(root, bounds, colormap, n_classes)
        plt.title(info)
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()

    plt.subplot(211)
    plot(X_train, y_train, info_train)

    plt.subplot(212)
    plot(X_test, y_test, info_test)

    plt.show()


def draw_node_2d(node, bounds, colormap, n_classes):
    if node.is_leaf():
        x = bounds[0][0]
        y = bounds[1][0]
        w = bounds[0][1] - x
        h = bounds[1][1] - y

        mean = node.compute_posterior_mean()
        if not node.is_regression:
            mean = (np.arange(len(mean)) * mean).sum()

        plt.gca().add_patch(patches.Rectangle((x, y), w, h, color=colormap(mean/n_classes), alpha=0.1, linewidth=0))
    else:
        draw_node_2d(node.child1, compute_child_bounds_2d(bounds, node, True), colormap, n_classes)
        draw_node_2d(node.child2, compute_child_bounds_2d(bounds, node, False), colormap, n_classes)


def compute_child_bounds_2d(bounds, parent, lower):
    b = bounds[parent.split_dimension]
    b = (b[0], min(b[1], parent.split_value)) if lower else (max(b[0], parent.split_value), b[1])
    return (b, bounds[1]) if parent.split_dimension == 0 else (bounds[0], b)


def compute_child_bounds_1d(bounds, parent, lower):
    b = bounds
    b = (b[0], min(b[1], parent.split_value)) if lower else (max(b[0], parent.split_value), b[1])
    return b


def draw_node_1d(node, bounds):
    if node.is_leaf():
        x0 = bounds[0]
        x1 = bounds[1]

        mean = node.compute_posterior_mean()
        # alpha = np.abs(mean-0.5)
        # alpha = max(0.1, alpha)  # make sure very faint colors become visibly colored
        # color = color0 if mean < 0.5 else color1
        plt.plot([x0, x1], [mean, mean], 'r')
    else:
        draw_node_1d(node.child1, compute_child_bounds_1d(bounds, node, True))
        draw_node_1d(node.child2, compute_child_bounds_1d(bounds, node, False))
