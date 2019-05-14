# A Bayesian Decision Tree Algorithm
This is an implementation of the paper: [A Bayesian Decision Tree Algorithm](https://arxiv.org/abs/1901.03214) by Nuti et al.

## Feature Support

This package implements:
* Classification (binary and multi-class)
* Regression

## Installation

To install `bayesian-decision-tree` simply:
```
git clone https://github.com/UBS-IB/bayesian_tree
cd bayesian_tree
pip install -e .
```

## Usage

We include some examples for various uses in the `examples` directory. However, very simply, you can do __one__ of :

```
from bayesian_decision_tree.regression import PerpendicularRegressionNode
from bayesian_decision_tree.regression import HyperplaneRegressionNode
from bayesian_decision_tree.classification import PerpendicularClassificationNode
from bayesian_decision_tree.classification import HyperplaneClassificationNode
```
followed by instantiating the node with appropriate parameters and then calling `fit(X, y)` on it.

## TODO
- Add parallelization option (dask)
