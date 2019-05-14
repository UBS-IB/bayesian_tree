import pandas as pd
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver


from scipy.sparse import csc_matrix, csr_matrix

from bayesian_decision_tree.classification import PerpendicularClassificationNode, HyperplaneClassificationNode
from bayesian_decision_tree.hyperplane_optimization import ScipyOptimizer, RandomTwoPointOptimizer, MyOptimizer, RandomHyperplaneOptimizer
from bayesian_decision_tree.regression import PerpendicularRegressionNode, HyperplaneRegressionNode

# possible data matrix types/transforms that need to work for fit()
data_matrix_transforms = [
    lambda X: X,
    lambda X: csc_matrix(X),
    lambda X: csr_matrix(X),
    lambda X: pd.DataFrame(data=X),
    lambda X: pd.DataFrame(data=X).to_sparse()
]


# classification tree models in all flavours
def create_classification_models(prior, partition_prior):
    return [
        PerpendicularClassificationNode(partition_prior, prior),
        HyperplaneClassificationNode(partition_prior, prior),
        HyperplaneClassificationNode(partition_prior, prior, optimizer=ScipyOptimizer(DifferentialEvolutionSolver, 666)),
        HyperplaneClassificationNode(partition_prior, prior, optimizer=RandomTwoPointOptimizer(100, 666)),
        HyperplaneClassificationNode(partition_prior, prior, optimizer=RandomHyperplaneOptimizer(100, 666)),
        HyperplaneClassificationNode(partition_prior, prior, optimizer=MyOptimizer(10, 10, 0.9, 666)),
    ]


# regression tree models in all flavours
def create_regression_models(prior, partition_prior):
    return [
        PerpendicularRegressionNode(partition_prior, prior),
        HyperplaneRegressionNode(partition_prior, prior),
        HyperplaneRegressionNode(partition_prior, prior, optimizer=ScipyOptimizer(DifferentialEvolutionSolver, 666)),
        HyperplaneRegressionNode(partition_prior, prior, optimizer=RandomHyperplaneOptimizer(100, 666)),
        HyperplaneRegressionNode(partition_prior, prior, optimizer=MyOptimizer(10, 10, 0.9, 666)),
    ]
