import numpy as np
import networkx as nx
import torch
import torch.nn as nn

class deepwalk():
    def __init__(self,
                 graph,
                 window_size=10,
                 embedding_size=128,
                 walk_per_vertex=80,
                 walk_length=40,
                 learning_rate=0.025,

                 random_state=None):
        """[summary]

        Args:
            graph (networkx.Graph): undirected Graph object
            window_size (int, optional): window size (w) in skipgram. Defaults to 10.
            embedding_size (int, optional): embedding size (d). Defaults to 128.
            walk_per_vertex (int, optional): walk per vertex (𝛾). Defaults to 80.
            walk_length (int, optional): walk length (t) in random walk.
            Defaults to 40.
            learning_rate (float, optional): learning rate (𝛼) in optimization.
            Defaults to 0.025.
            random_state ([type], optional): Random State instance or None.
            Defaults to None.
        """
        self.graph = graph
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_per_vertex = walk_per_vertex
        self.walk_length = walk_length
        self.learning_rate = learning_rate
        self.gpu_idx = np.random.randint(0, torch.cuda.device_count())
        self.device = torch.device(f'cuda:{self.gpu_idx}' if torch.cuda.is_available() else 'cpu')

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Initialize
        self.mat_v_represent = nn.Embedding(len(self.graph), embedding_size)

    def random_walk(self):
        return self

    def skip_gram(self):
        return self

    def hierarchical_softmax(self):
        return self

    def sgd(self):
        return self
