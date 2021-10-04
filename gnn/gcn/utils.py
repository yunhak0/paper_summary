import numpy as np
from scipy import io
from scipy import sparse

import networkx as nx
import torch

def load_matfile(file, variable_name='network', to_graph=True):
    """Load *.mat file as networkx.Graph object
    Args:
        file (str): path of the '.mat' file
        variable_name (str, optional): key name (object) of mat file. Defaults to 'network'.
    Returns:
        networkx.Graph: Graph object of networkx
    """    
    network = io.loadmat(file)[variable_name]

    if to_graph:
        network = nx.Graph(network)

    return network

# https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :]
                    for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mat):
    rowsum = np.array(mat.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat

def scipy_sparse_to_torch_sparse(sparse_mat):
    sparse_mat = sparse_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mat.data)
    shape = torch.Size(sparse_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(path='./gnn/data/cora_data', data='cora'):
    if path[-1:] == '/':
        content_file = '{}{}.content'.format(path, data)
        cites_file = '{}{}.cites'.format(path, data)
    else:
        content_file = '{}/{}.content'.format(path, data)
        cites_file = '{}/{}.cites'.format(path, data)

    idx_feature_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    features = sparse.csr_matrix(idx_feature_labels[:, 1:-1],
                                 dtype=np.float32)
    labels = encode_onehot(idx_feature_labels[:, -1])

    # build graph
    idx = np.array(idx_feature_labels[:, 0],
                   dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(cites_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj_mat = sparse.coo_matrix((np.ones(edges.shape[0]),
                                (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj_mat = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat)

    features = normalize(features)
    adj_mat = normalize(adj_mat + sparse.eye(adj_mat.shape[0]))

    # to Tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj_mat = scipy_sparse_to_torch_sparse(adj_mat)

    return adj_mat, features, labels

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)
