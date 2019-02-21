import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

# possible data matrix types/transforms that need to work for fit()
data_matrix_transforms = [
    lambda X: X,
    lambda X: csc_matrix(X),
    lambda X: csr_matrix(X),
    lambda X: pd.DataFrame(data=X),
    lambda X: pd.DataFrame(data=X).to_sparse()
]
