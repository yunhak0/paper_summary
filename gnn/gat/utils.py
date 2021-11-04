import numpy as np
import scipy.sparse as sp
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset
import json
import os
import zipfile
import networkx as nx
from networkx.readwrite import json_graph

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(dataset="cora", path="./gnn/data/cora_data/", 
              adj_normalize=False, features_normalize=False):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if adj_normalize:
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    if features_normalize:
        features = normalize_features(features)
    

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_model_state(args, model):
    state = {
        'n_epochs': args.n_epochs,
        'patience': args.patience,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'n_heads_final_layer': args.n_heads_final_layer,
        'n_hidden': args.n_hidden,
        'leaky_relu_alpha': args.leaky_relu_alpha,
        'dropout': args.dropout,
        'add_skip_connection': args.add_skip_connection,

        'state_dict': model.state_dict()
    }

    return state

# For Inductive data - PPI
def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data

def load_ppi_data(args):
    dataset_name = args.dataset_name.lower()
    PPI_PATH = './gnn/data/ppi_data'
    PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'

    if dataset_name == 'ppi':  # Protein-Protein Interaction dataset

        # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed (lazy execution ^^)
        if not os.path.exists(PPI_PATH):  # download the first time this is ran
            os.makedirs(PPI_PATH)

            # Step 1: Download the ppi.zip (contains the PPI dataset)
            zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
            download_url_to_file(PPI_URL, zip_tmp_path)

            # Step 2: Unzip it
            with zipfile.ZipFile(zip_tmp_path) as zf:
                zf.extractall(path=PPI_PATH)
            print(f'Unzipping to: {PPI_PATH} finished.')

            # Step3: Remove the temporary resource file
            os.remove(zip_tmp_path)
            print(f'Removing tmp file {zip_tmp_path}.')

        # Collect train/val/test graphs here
        edge_index_list = []
        node_features_list = []
        node_labels_list = []

        # Dynamically determine how many graphs we have per split (avoid using constants when possible)
        num_graphs_per_split_cumulative = [0]

        # Small optimization "trick" since we only need test in the playground.py
        splits = ['test'] if args.ppi_load_test_only else ['train', 'valid', 'test']

        for split in splits:
            # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
            # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
            # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
            # Note: node features are already preprocessed
            node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

            # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
            # SHAPE = (NS, 121)
            node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

            # Graph topology stored in a special nodes-links NetworkX format
            nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
            # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
            # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
            # because of the edge index and the way GAT implementation #3 works
            collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
            # For each node in the above collection, ids specify to which graph the node belongs to
            graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
            num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

            # Split the collection of graphs into separate PPI graphs
            for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
                mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
                graph_node_ids = np.asarray(mask).nonzero()[0]
                graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
                print(f'Loading {split} graph {graph_id} to CPU. '
                      f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

                # shape = (2, E) - where E is the number of edges in the graph
                # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
                # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
                edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
                edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
                edge_index_list.append(edge_index)
                # shape = (N, 50) - where N is the number of nodes in the graph
                node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
                # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
                node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))

        #
        # Prepare graph data loaders
        #

        # Optimization, do a shortcut in case we only need the test data loader
        if args.ppi_load_test_only:
            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=args.batch_size,
                shuffle=False
            )
            return data_loader_test
        else:

            data_loader_train = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=args.batch_size,
                shuffle=True
            )

            data_loader_val = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                batch_size=args.batch_size,
                shuffle=False  # no need to shuffle the validation and test graphs
            )

            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                batch_size=args.batch_size,
                shuffle=False
            )

            return data_loader_train, data_loader_val, data_loader_test
    else:
        raise Exception(f'{dataset_name} not yet supported.')

class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).
    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it
    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]

def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.
    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.
    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index
