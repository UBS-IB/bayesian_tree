# A Bayesian Decision Tree Algorithm
This is an implementation of the paper: [A Bayesian Decision Tree Algorithm](https://arxiv.org/abs/1901.03214) by Nuti et al.

## Feature Support

This package implements:
* Classification (binary and multiclass)
* Regression
* Both models are available in two versions respectively:
  * **Perpendicular Trees**:
    The classic decision/regression tree structure with splits along a single
    feature dimension (i.e., _perpendicular_ to a feature dimension axis),
    analogous to e.g. the scikit-learn
    [decision](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    and
    [regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    trees.
    
    The models are called
    [`PerpendicularClassificationTree`](bayesian_decision_tree/classification.py)
    and
    [`PerpendicularRegressionTree`](bayesian_decision_tree/regression.py).
     
  * **Hyperplane Trees**:
    Decision/regression trees using _arbitrarily-oriented hyperplanes_. These models
    are more flexible than perpendicular trees as they cover a much larger search
    space to naturally make use of correlations between features.
    
    All else equal, hyperplane trees typically lead to shallower trees with fewer
    leaf nodes compared to their perpendicular counterparts because they can employ
    more than just a single feature dimension per split. This can lead to less
    overfitting and better generalization performance, but no such guarantees exist
    because hyperplane trees are still being constructed in a greedy manner.
    
    Note that hyperplane trees take much longer to train and need to be trained
    stochastically using global optimizers due to the exponentially large search
    space.
    
    The models are called
    [`HyperplaneClassificationTree`](bayesian_decision_tree/classification.py)
    and
    [`HyperplaneRegressionTree`](bayesian_decision_tree/regression.py).

## Installation

To install you can either use _conda_ or _pip_:

#### Conda
```
git clone https://github.com/UBS-IB/bayesian_tree
cd bayesian_tree
conda build conda.recipe
conda install --use-local bayesian_decision_tree
```

#### PIP
```
git clone https://github.com/UBS-IB/bayesian_tree
cd bayesian_tree
pip install -e .
```

## Usage

We include some examples for various uses in the [examples](examples) directory.
The models are fully compatible with scikit-learn, so you can use them for e.g.
cross-validation or performance evaluation using scikit-learn functions.

## TODO
- Add parallelization option (dask)
