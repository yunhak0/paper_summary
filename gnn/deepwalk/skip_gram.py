from gensim.models import Word2Vec
from multiprocessing import cpu_count

class skip_gram:
    def __init__(self,
                 graph,
                 walks,
                 embedding_size=128,
                 window_size=10):
        self.graph = graph
        self.walks = walks
        self.embedding_size = embedding_size
        self.window_size = window_size

    def fit(self):
        model = Word2Vec(self.walks,
                         size=self.embedding_size,
                         window=self.window_size,
                         min_count=0,
                         sg=1,  # Skip Gram
                         hs=1,  # Hierarchical Softmax
                         workers=cpu_count())
        return model
