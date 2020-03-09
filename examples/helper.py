"""
A collection of publicly available data sets to test classification models on,
 plus some helper functions for plotting.
"""
import argparse
import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import patches
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
    lines = text.split('\n')
    lines = [line for line in lines if '?' not in line]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    train[:, -1] -= 1
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


def load_seeds(proxies):
    # load wheat seeds dataset
    def parse_ripley(text):
        lines = text.split('\n')
        return np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines)-1)])
    train = parse_ripley(requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', proxies=proxies).text)
    train[:, -1] -= 1
    test = train
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


def plot_1d_perpendicular(root, X_train, y_train, info_train, X_test, y_test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)
    plt.subplot(211)
    plt.plot(X_train[:, 0], y_train, 'o-')
    plt.title(info_train)
    draw_node_1d_perpendicular(root, bounds=(X_train[:, 0].min(), X_train[:, 0].max()))
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.gca().set_aspect(1)

    plt.subplot(212)
    plt.plot(X_test[:, 0], y_test, 'o-')
    draw_node_1d_perpendicular(root, bounds=(X_test[:, 0].min(), X_test[:, 0].max()))
    plt.title(info_test)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.gca().set_aspect(1)

    plt.show()


def plot_2d_perpendicular(root, X_train, y_train, info_train, X_test, y_test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)

    n_classes = int(y_train.max()) + 1
    colormap = plt.get_cmap('gist_rainbow')

    def plot(X, y, info):
        for i in range(n_classes)[::-1]:
            class_i = y == i
            plt.plot(X[np.where(class_i)[0], 0],
                     X[np.where(class_i)[0], 1],
                     'o',
                     ms=4,
                     c=colormap(i/n_classes),
                     label='Class {}'.format(i),
                     alpha=0.5)

            bounds = ((X[:, 0].min(), X[:, 0].max()), (X[:, 1].min(), X[:, 1].max()))
            draw_node_2d_perpendicular(root, bounds, colormap, n_classes)
        plt.title(info)
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()

    plt.subplot(211)
    plot(X_train, y_train, info_train)
    plt.gca().set_aspect(1)

    plt.subplot(212)
    plot(X_test, y_test, info_test)
    plt.gca().set_aspect(1)

    plt.show()


def draw_node_2d_perpendicular(node, bounds, colormap, n_classes):
    if node.is_leaf():
        x = bounds[0][0]
        y = bounds[1][0]
        w = bounds[0][1] - x
        h = bounds[1][1] - y

        mean = node._compute_posterior_mean()
        if not node.is_regression:
            mean = (np.arange(len(mean)) * mean).sum()

        plt.gca().add_patch(patches.Rectangle((x, y), w, h, color=colormap(mean/n_classes), alpha=0.1, linewidth=0))
    else:
        draw_node_2d_perpendicular(node.child1_, compute_child_bounds_2d_perpendicular(bounds, node, True), colormap, n_classes)
        draw_node_2d_perpendicular(node.child2_, compute_child_bounds_2d_perpendicular(bounds, node, False), colormap, n_classes)


def compute_child_bounds_2d_perpendicular(bounds, parent, lower):
    b = bounds[parent.split_dimension_]
    b = (b[0], min(b[1], parent.split_value_)) if lower else (max(b[0], parent.split_value_), b[1])
    return (b, bounds[1]) if parent.split_dimension_ == 0 else (bounds[0], b)


def compute_child_bounds_1d_perpendicular(bounds, parent, lower):
    b = bounds
    b = (b[0], min(b[1], parent.split_value)) if lower else (max(b[0], parent.split_value), b[1])
    return b


def draw_node_1d_perpendicular(node, bounds):
    if node.is_leaf():
        x0 = bounds[0]
        x1 = bounds[1]

        mean = node._compute_posterior_mean()
        # alpha = np.abs(mean-0.5)
        # alpha = max(0.1, alpha)  # make sure very faint colors become visibly colored
        # color = color0 if mean < 0.5 else color1
        plt.plot([x0, x1], [mean, mean], 'r')
    else:
        draw_node_1d_perpendicular(node.child1, compute_child_bounds_1d_perpendicular(bounds, node, True))
        draw_node_1d_perpendicular(node.child2, compute_child_bounds_1d_perpendicular(bounds, node, False))


class Line:
    def __init__(self, p0, p1):
        if p0[0] > p1[0]:
            p1, p0 = p0, p1

        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)

    def intersect(self, other):
        da = self.p1-self.p0
        ma = da[1]/da[0]

        db = other.p1-other.p0
        mb = db[1]/db[0]

        x0a = self.p0[0]
        x1a = self.p1[0]
        x0b = other.p0[0]
        x1b = other.p1[0]
        y0a = self.p0[1]
        y0b = other.p0[1]

        x = (y0a-y0b + mb*x0b-ma*x0a) / (mb-ma)
        y = y0a + ma*(x-x0a)

        if x0a <= x <= x1a and x0b <= x <= x1b:
            return np.array([x, y])
        else:
            return None

    def plot(self, *args, **kwargs):
        plt.plot([self.p0[0], self.p1[0]], [self.p0[1], self.p1[1]], *args, **kwargs)

    def __str__(self):
        return f'{self.p0} -> {self.p1}'


@dataclass
class Parent:
    line: Line
    origin: np.ndarray
    normal: np.ndarray
    side: str


# plots the root node split and all child nodes recursively
def plot_root(root, X, y, title, cmap):
    plt.title(title)

    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b.', ms=3)
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r.', ms=3)

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    top = Line([x_min, y_max], [x_max, y_max])
    bottom = Line([x_min, y_min], [x_max, y_min])

    def plot_node(node, node_vs_color={}, level=0, parents=[], side=None):
        if node.best_hyperplane_origin_ is None:
            return

        # pick an arbitrary origin and get the normal
        origin = node.best_hyperplane_origin_
        normal = node.best_hyperplane_normal_

        # construct line segment
        m = -normal[0]/normal[1]
        y0 = origin[1] + m*(x_min-origin[0])
        y1 = origin[1] + m*(x_max-origin[0])

        # raw line without intersections
        line = Line([x_min, y0], [x_max, y1])

        # intersect with parents
        for parent in parents:
            p = line.intersect(parent.line)
            if p is not None:
                # determine side of line to keep
                activation0 = np.dot(line.p0 - parent.origin, parent.normal)

                if (parent.side == 'L' and activation0 > 0) or (parent.side == 'R' and activation0 < 0):
                    line = Line(line.p0, p)
                else:
                    line = Line(p, line.p1)

        # intersect with top/bottom
        p = line.intersect(top)
        if p is not None:
            if y0 > y_max:
                line = Line(p, line.p1)
            else:
                line = Line(line.p0, p)

        p = line.intersect(bottom)
        if p is not None:
            if y0 < y_min:
                line = Line(p, line.p1)
            else:
                line = Line(line.p0, p)

        # generate line name
        if side is not None:
            side_name = ' - '.join(f'{parents[i].side}{level-len(parents)+i+1}' for i in range(len(parents)))
        else:
            side_name = ''

        side_name = 'Root' if len(side_name) == 0 else 'Root - ' + side_name

        # make sure node colors don't change
        if id(node) not in node_vs_color:
            color = cmap(len(node_vs_color))
            node_vs_color[id(node)] = color
        else:
            color = node_vs_color[id(node)]

        # compute line width as a function of the stiffness
        stiffness = np.linalg.norm(normal)
        lw = 2  # 100/stiffness

        line.plot(color=color, label=side_name, lw=lw, alpha=0.7)

        if node.child1_:
            plot_node(node.child1_, node_vs_color, level+1, parents=parents + [Parent(line, origin, normal, 'L')], side='L')

        if node.child2_:
            plot_node(node.child2_, node_vs_color, level+1, parents=parents + [Parent(line, origin, normal, 'R')], side='R')

    plot_node(root)


def plot_2d_hyperplane(root, X_train, y_train, info_train, X_test, y_test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)

    n_classes = int(y_train.max()) + 1
    colormap = plt.get_cmap('gist_rainbow')

    x_min = min(X_train[:, 0].min(), X_test[:, 0].min())
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max())
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min())
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max())

    def plot(X, y, info):
        for i in range(n_classes):
            class_i = y == i
            plt.plot(X[np.where(class_i)[0], 0],
                     X[np.where(class_i)[0], 1],
                     'o',
                     ms=4,
                     c=colormap(i/n_classes),
                     label='Class {}'.format(i))

        plot_root(root, X, y, info, plt.get_cmap('tab20'))

        plt.title(info)
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()

    plt.subplot(211)
    plot(X_train, y_train, info_train)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.gca().set_aspect(1)

    plt.subplot(212)
    plot(X_test, y_test, info_test)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.gca().set_aspect(1)

    plt.show()
