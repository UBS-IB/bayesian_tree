import pandas as pd
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.sparse import csc_matrix, csr_matrix

from bayesian_decision_tree.classification import PerpendicularClassificationTree, HyperplaneClassificationTree
from bayesian_decision_tree.hyperplane_optimization import ScipyOptimizer, RandomTwoPointOptimizer
from bayesian_decision_tree.hyperplane_optimization import SimulatedAnnealingOptimizer, RandomHyperplaneOptimizer
from bayesian_decision_tree.regression import PerpendicularRegressionTree, HyperplaneRegressionTree

# possible data matrix types/transforms that need to work for fit()
data_matrix_transforms = [
    lambda X: X,
    lambda X: csc_matrix(X),
    lambda X: csr_matrix(X),
    lambda X: pd.DataFrame(data=X),
    lambda X: pd.DataFrame(data=X).to_sparse()
]


# classification tree models in all flavours
def create_classification_trees(prior, partition_prior):
    return [
        PerpendicularClassificationTree(partition_prior, prior),
        HyperplaneClassificationTree(partition_prior, prior),
        HyperplaneClassificationTree(partition_prior, prior, optimizer=ScipyOptimizer(DifferentialEvolutionSolver, 666)),
        HyperplaneClassificationTree(partition_prior, prior, optimizer=RandomTwoPointOptimizer(100, 666)),
        HyperplaneClassificationTree(partition_prior, prior, optimizer=RandomHyperplaneOptimizer(100, 666)),
        HyperplaneClassificationTree(partition_prior, prior, optimizer=SimulatedAnnealingOptimizer(10, 10, 0.9, 666)),
    ]


# regression tree models in all flavours
def create_regression_trees(prior, partition_prior):
    return [
        PerpendicularRegressionTree(partition_prior, prior),
        HyperplaneRegressionTree(partition_prior, prior),
        HyperplaneRegressionTree(partition_prior, prior, optimizer=ScipyOptimizer(DifferentialEvolutionSolver, 666)),
        HyperplaneRegressionTree(partition_prior, prior, optimizer=RandomHyperplaneOptimizer(100, 666)),
        HyperplaneRegressionTree(partition_prior, prior, optimizer=SimulatedAnnealingOptimizer(10, 10, 0.9, 666)),
    ]
