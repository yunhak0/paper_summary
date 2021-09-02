from tqdm import tqdm
import numpy as np
from scipy import sparse

def get_index(rating: sparse.csr_matrix) -> dict:
    """Get Indexes of Rating Matrix

    Args:
        rating (sparse.csr_matrix): rating matrix with csr_matrix format

    Returns:
        dict: keys are user, item, data_idx
        user is index or id of user.
        item is index or id of item.
        data_idx is index of csr_matrix.data
    """    
    dict_idx = {'user': [], 'item':[], 'data_idx': []}
    for u, (begin, end) in tqdm(enumerate(zip(rating.indptr,
                                              rating.indptr[1:])),
                                total=len(rating.indptr)-1,
                                desc='indexing'):
        dict_idx['user'].append(u)
        dict_idx['item'].append(list(rating.indices[begin:end]))
        dict_idx['data_idx'].append(list(range(begin, end)))
    return dict_idx

# https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python
def sparse_corrcoef(x, y=None):
    """Calculate Correlation Coefficient of Compressed Sparse Row Matrix

    Args:
        x (scipy.sparse.csr.csr_matrix): Input Matrix 1
        y (scipy.sparse.csr.csr_matrix, optional): An additional set of variables and observations.
        y has the same shape as x.. Defaults to None.

    Returns:
        ndarray: The correlation coefficient matrix of the variables.
    """
    if y is not None:
        x = sparse.vstack((x, y), format='csr')

    x = x.astype(np.float64)
    n = x.shape[1]

    # Compute the covariance matrix
    rowsum = x.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (x.dot(x.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    coeffs = sparse.csr_matrix(coeffs)

    return coeffs
