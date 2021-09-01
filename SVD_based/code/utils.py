from tqdm import tqdm
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
                                total=len(rating.indptr)-1):
        dict_idx['user'].append(u)
        dict_idx['item'].append(list(rating.indices[begin:end]))
        dict_idx['data_idx'].append(list(range(begin, end)))
    return dict_idx
