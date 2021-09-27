from scipy import io
import contextlib
import joblib

import networkx as nx

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

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]

# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
