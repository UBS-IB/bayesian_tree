import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import patches


def load_ripley(proxies):
    # load Ripley's synthetic dataset
    def parse_ripley(text):
        lines = text.split('\n')[1:]
        return np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines)-1)])
    train = parse_ripley(requests.get('http://www.stats.ox.ac.uk/pub/PRNN/synth.tr', proxies=proxies).text)
    test = parse_ripley(requests.get('http://www.stats.ox.ac.uk/pub/PRNN/synth.te', proxies=proxies).text)
    return train, test


def load_haberman(proxies):
    # load Haberman's dataset
    def parse_haberman(text):
        lines = text.split('\n')
        data = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
        data[:, -1] -= 1
        return data
    train = parse_haberman(requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', proxies=proxies).text)
    test = train
    return train, test


def load_seismic(proxies):
    # load seismic bumps dataset
    def parse_seismic(text):
        text = text[text.index('@data'):]
        text = text.replace('a', '0').replace('b', '1').replace('c', '2').replace('d', '3')
        text = text.replace('N', '0').replace('W', '1')
        lines = text.split('\n')[1:]
        return np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    train = parse_seismic(requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff', proxies=proxies).text)
    test = train
    return train, test


def load_gamma(proxies):
    # load Gamma data
    def parse_gamma(text):
        text = text.replace('g', '0').replace('h', '1')
        lines = text.split('\n')
        return np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines)-1)])
    train = parse_gamma(requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', proxies=proxies).text)
    test = train
    return train, test


def plot_1d(root, train, info_train, test, info_test):
    plt.figure(figsize=[10, 16], dpi=75)
    plt.subplot(211)
    plt.plot(train[:, 0], train[:, 1])
    plt.title(info_train)
    draw_node_1d(root, bounds=(train[:, 0].min(), train[:, 0].max()))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    plt.subplot(212)
    plt.plot(test[:, 0], test[:, 1])
    draw_node_1d(root, bounds=(test[:, 0].min(), test[:, 0].max()))
    plt.title(info_test)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    plt.show()


def plot_2d(root, train, info_train, test, info_test, color0, color1):
    plt.figure(figsize=[10, 16], dpi=75)

    plt.subplot(211)
    class0 = train[:, -1] == 0
    plt.plot(train[np.where(class0)[0], 0], train[np.where(class0)[0], 1], 'o', ms=4, c=color0, label='Class 0')
    plt.plot(train[np.where(1-class0)[0], 0], train[np.where(1-class0)[0], 1], 'o', ms=4, c=color1, label='Class 1')
    draw_node_2d(root, ((train[:, 0].min(), train[:, 0].max()), (train[:, 1].min(), train[:, 1].max())), color0, color1)
    plt.title(info_train)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    plt.subplot(212)
    class0 = test[:, -1] == 0
    plt.plot(test[np.where(class0)[0], 0], test[np.where(class0)[0], 1], 'o', ms=4, c=color0, label='Class 0')
    plt.plot(test[np.where(1-class0)[0], 0], test[np.where(1-class0)[0], 1], 'o', ms=4, c=color1, label='Class 1')
    draw_node_2d(root, ((test[:, 0].min(), test[:, 0].max()), (test[:, 1].min(), test[:, 1].max())), color0, color1)
    plt.title(info_test)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    plt.show()


def draw_node_2d(node, bounds, color0, color1):
    if node.child1 is not None:
        draw_node_2d(node.child1, compute_child_bounds_2d(bounds, node, True), color0, color1)
        draw_node_2d(node.child2, compute_child_bounds_2d(bounds, node, False), color0, color1)
    else:
        x = bounds[0][0]
        y = bounds[1][0]
        w = bounds[0][1] - x
        h = bounds[1][1] - y

        mean = node.compute_mean()
        alpha = np.abs(mean-0.5)
        alpha = max(0.1, alpha)  # make sure very faint colors become visibly colored
        color = color0 if mean < 0.5 else color1
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, color=color, alpha=alpha, linewidth=0))

def compute_child_bounds_2d(bounds, parent, lower):
    b = bounds[parent.split_dimension]
    b = (b[0], min(b[1], parent.split_value)) if lower else (max(b[0], parent.split_value), b[1])
    return (b, bounds[1]) if parent.split_dimension == 0 else (bounds[0], b)

def compute_child_bounds_1d(bounds, parent, lower):
    b = bounds
    b = (b[0], min(b[1], parent.split_value)) if lower else (max(b[0], parent.split_value), b[1])
    return b

def draw_node_1d(node, bounds):
    if node.child1 is not None:
        draw_node_1d(node.child1, compute_child_bounds_1d(bounds, node, True))
        draw_node_1d(node.child2, compute_child_bounds_1d(bounds, node, False))
    else:
        x1 = bounds[0]
        x2 = bounds[1]

        mean = node.compute_mean()
        # alpha = np.abs(mean-0.5)
        # alpha = max(0.1, alpha)  # make sure very faint colors become visibly colored
        # color = color0 if mean < 0.5 else color1
        plt.plot([x1, x2], [mean, mean], 'r')
