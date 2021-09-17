import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from sklearn.multiclass import OneVsRestClassifier

import utils
import random_walk
import skip_gram

class deepwalk:
    def __init__(self,
                 graph,
                 walks_per_vertex=80,
                 walk_length=40,
                 embedding_size=128,
                 window_size=10,
                 random_state=None):
        self.graph = graph
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.window_size = window_size

        if random_state is not None:
            np.random.seed(random_state)
        
        self.nodes = list(self.graph.nodes())

    def _random_walk(self, n_walks):
        walks = []
        for _ in range(n_walks):
            np.random.shuffle(self.nodes)
            for node in self.nodes:
                rand_walk = random_walk.random_walk(
                    graph=self.graph,
                    # walks_per_vertex=self.walks_per_vertex,
                    walk_length=self.walk_length
                )
                walks.append(rand_walk.run(init_node=node))
        return walks

    def fit(self):
        with utils.tqdm_joblib(tqdm(desc='Deep Walking', total=cpu_count())) as progress_bar:
            walks = Parallel(n_jobs=cpu_count(), verbose=False)(
                delayed(self._random_walk)(n)
                for n in utils.partition_num(self.walks_per_vertex, cpu_count())
            )
        walks = list(itertools.chain(*walks))
        model = skip_gram.skip_gram(graph=self.graph,
                                    walks=walks,
                                    embedding_size=self.embedding_size,
                                    window_size=self.window_size).fit()
        return model

# https://github.com/phanein/deepwalk/blob/master/example_graphs/scoring.py
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
