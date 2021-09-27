import os
import numpy as np
import networkx as nx
from multiprocessing import cpu_count
from tqdm import tqdm

from gensim.models import Word2Vec
from sklearn.multiclass import OneVsRestClassifier


class Graph:
    def __init__(self, graph, p, q, dim, window_size,
                 n_epochs=10, directed=False):
        self.G = graph
        self.p = p
        self.q = q
        self.dim = dim
        self.window_size = window_size
        self.n_epochs = n_epochs
        self.directed = directed

    def node2vec_walk(self, start_node, walk_length):
        
        walk = [start_node]

        while len(walk) < walk_length:
            cur_node = walk[-1]
            cur_nbrs = sorted(self.G.neighbors(cur_node))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(self.alias_nodes[cur_node][0],
                                                    self.alias_nodes[cur_node][1])])
                else:
                    prv_node = walk[-2]
                    nxt_node = cur_nbrs[alias_draw(self.alias_edges[(prv_node, cur_node)][0],
                                                   self.alias_edges[(prv_node, cur_node)][1])]
                    walk.append(nxt_node)
            else:
                break
        return walk

    def run_walk(self, num_walks, walk_length):
        walks = []
        nodes = list(self.G.nodes())
        for _ in tqdm(range(num_walks)):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(start_node=node,
                                                walk_length=walk_length))
        return walks

    def search_bias(self, edge_from, edge_to):
        unnormalized_probs = []
        for edge_to_nbr in sorted(self.G.neighbors(edge_to)):
            if edge_to_nbr == edge_from:
                unnormalized_probs.append(self.G[edge_to][edge_to_nbr]['weight']/self.p)
            elif self.G.has_edge(edge_to_nbr, edge_from):
                unnormalized_probs.append(self.G[edge_to][edge_to_nbr]['weight'])
            else:
                unnormalized_probs.append(self.G[edge_to][edge_to_nbr]['weight']/self.q)
        normalized_constant = sum(unnormalized_probs)
        normalized_probs = [float(p)/normalized_constant for p in unnormalized_probs]

        return alias_setup(normalized_probs) 

    def preprocess_modified_weights(self):
        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr]['weight'] for nbr in sorted(self.G.neighbors(node))]
            normalized_constant = sum(unnormalized_probs)
            normalized_probs = [float(p)/normalized_constant for p in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        
        alias_edges = {}
        if self.directed:
            for edge in self.G.edges():
                alias_edges[edge] = self.search_bias(edge[0], edge[1])
        else:
            for edge in self.G.edges():
                alias_edges[edge] = self.search_bias(edge[0], edge[1]) 
                alias_edges[(edge[1], edge[0])] = self.search_bias(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def fit(self, walks):
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=self.dim, window=self.window_size,
                         min_count=0, sg=1, workers=cpu_count(),
                         iter=self.n_epochs)
        return model


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels
