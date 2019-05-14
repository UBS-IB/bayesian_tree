# A Bayesian Decision Tree Algorithm
This is an implementation of the paper: [A Bayesian Decision Tree Algorithm](https://arxiv.org/abs/1901.03214) by Nuti et al.

## Feature Support

This package implements:
* Classification (binary and multiclass)
* Regression
* Both models are available in two versions respectively:
  * **Perpendicular Trees**:
    The classic decision/regression tree structure with splis along a single
    feature dimension, analogous to e.g. the sklearn
    [decision](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    and
    [regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    trees.
    
    The models are called
    [PerpendicularClassificationNode](bayesian_decision_tree/classification.py)
    and
    [PerpendicularRegressionNode](bayesian_decision_tree/classification.py)
     
  * **Hyperplane Trees**:
    Decision/regression trees using _arbitrarily oriented hyperplanes_. These models
    are more flexible than perpendicular trees as they cover a much larger search
    space.
    
    All else equal, hyperplane trees typically lead to shallower trees with fewer
    leaf nodes compared to their perpendicular counterparts because they can employ
    more than just a single feature dimension per split. This can lead to less
    overfitting and better generalization performance, but no such guarantees exist
    because hyperplane trees are still being constructed in a greedy and therefore
    non-optimal manner.
    
    Note that hyperplane trees take much longer to train and can only be trained
    stochastically using global optimizers due to the exponentially large search
    space.
    
    The models are called
    [HyperplaneClassificationNode](bayesian_decision_tree/classification.py)
    and
    [HyperplaneRegressionNode](bayesian_decision_tree/classification.py).

## Installation

To install `bayesian-decision-tree` simply:
```
git clone https://github.com/UBS-IB/bayesian_tree
cd bayesian_tree
pip install -e .
```

## Usage

We include some examples for various uses in the [examples](examples) directory.
The models are fully compatible with sklearn models, so you can use them for e.g.
cross-validation or performance evaluation.

## TODO
- Add parallelization option (dask)
