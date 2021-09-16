import numpy as np
from torch import nn
import torch
from deepwalk import Graph, Tree

class skip_gram(nn.Module):
    def __init__(self,
                 embedding_size=128,
                 learning_rate=0.025):
        """[summary]

        Args:
            embedding_size (int, optional): embedding size (d). Defaults to 128.
            learning_rate (float, optional): learning rate (𝛼) in optimization.
            Defaults to 0.025.
        """
        super(skip_gram, self).__init__()
        super(Graph.Graph, self).__init__()
        super(Tree.Tree, self).__init__()
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

        GPU_IDX = np.random.randint(0, torch.cuda.device_count())
        self.device = torch.device(f'cuda:{GPU_IDX}'
                                   if torch.cuda.is_available() else 'cpu')

        # Initialize
        self.word_represent = nn.Embedding(len(self.graph.nodes), self.embedding_size)
        self.node_represent = nn.Embedding(len(self.graph.nodes), self.embedding_size)

    def forward(self, target, context):
        syn0 = self.word_represent(target)
        syn1 = self.node_represent(context)

        loss = torch.zeros(1, requires_grad=True, dtype=torch.float)

        path_to_target = self.codes[target]
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
