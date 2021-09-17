import numpy as np
from tqdm import tqdm

class random_walk:
    def __init__(self,
                 graph,
                 # walks_per_vertex=80,
                 walk_length=40):
        self.graph = graph
        # self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length

    def run(self, init_node):
        walk = [init_node]
        node = init_node

        for _ in range(self.walk_length):
            # Get Neighbors
            neighbors = list(self.graph.neighbors(node))
            # Select the next node from neighbors
            node = np.random.choice(neighbors)
            # Update walk
            walk.append(node)
        
        return [str(node) for node in walk]
