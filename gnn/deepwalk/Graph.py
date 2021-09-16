import numpy as np
from torch.utils.data import Dataset 
import torch

class Graph(Dataset):
    def __init__(self,
                 graph,
                 window_size=10,
                 walk_length=40,
                 random_state=None):
        """[summary]

        Args:
            graph (networkx.Graph): undirected Graph object
            window_size (int, optional): window size (w) in skipgram. Defaults to 10.
            walk_length (int, optional): walk length (t) in random walk.
            Defaults to 40.
            random_state ([type], optional): Random State instance or None.
            Defaults to None.
        """
        self.graph = graph
        self.window_size = window_size
        self.walk_length = walk_length
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        
        self.nodes = list(graph.nodes())

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        walk = random_walk(graph=self.graph,
                           node=self.nodes[idx],
                           walk_length=self.walk_length)

        target = []
        context = []

        # Algorithm2: SkipGram 1-2 lines
        for j, v_j in enumerate(walk):
            left_window = walk[max(0, j-self.walk_length):j]
            right_window = walk[j:min(j+self.walk_length+1, len(walk))]
            u = left_window + right_window

            target.extend([v_j] * len(u))
            context.extend(u)

        return {'target': torch.LongTensor(target),
                'context': torch.LongTensor(context)}


def random_walk(graph, init_node, walk_length):
    """Random Walk

    Args:
        init_node (int): Node index of the Graph

    Returns:
        list: Random Walk of the Graph
    """
    # Starting Node
    walk = [init_node]
    node = init_node

    for _ in range(walk_length):
        # Get Neighbors
        neighbors = list(graph.neighbors(node))
        # Select the next node from neighbors
        node = np.random.choice(neighbors)
        # Update walk
        walk.append(node)
    
    return walk
