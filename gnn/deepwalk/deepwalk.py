import numpy as np
import torch
import torch.nn as nn

class deepwalk(nn.Module):
    def __init__(self,
                 graph,
                 embedding_size=128):
        """[summary]

        Args:
            graph (networkx.Graph): The undirected graph object (G)
            embedding_size (int, optional): embedding size (d). Defaults to 128.
            
        """
        super(deepwalk, self).__init__()
        self.graph = graph
        self.embedding_size = embedding_size

        GPU_IDX = np.random.randint(0, torch.cuda.device_count())
        self.device = torch.device(f'cuda:{GPU_IDX}'
                                   if torch.cuda.is_available() else 'cpu')

        # Initialize
        self.word_represent = nn.Embedding(len(self.graph.nodes), self.embedding_size)
        self.node_represent = nn.Embedding(len(self.graph.nodes), self.embedding_size)

    def skip_gram(self, coded_walk, target, context):
        syn0 = self.word_represent(target)
        syn1 = self.node_represent(context)

        loss = torch.zeros(1, requires_grad=True, dtype=torch.float)

        path_to_target = coded_walk[target]
        root = self.root
        for i in path_to_target:
            if i == '0':
                loss = loss + torch.log(torch.sigmoid(torch.dot(syn0, syn1)))
                root = root.left
            else:
                loss = loss + torch.log(torch.sigmoid(-1*torch.dot(syn0, syn1)))
                root = root.right
        loss = -1 * loss
        return loss

    def forward(self, coded_walk, target, context):
        loss = self.skip_gram(coded_walk, target, context)
        return loss
