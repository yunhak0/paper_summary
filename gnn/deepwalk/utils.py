from scipy import io
import networkx as nx

def load_matfile(file, variable_name='network'):
    """Load *.mat file as networkx.Graph object

    Args:
        file (str): path of the '.mat' file
        variable_name (str, optional): key name (object) of mat file. Defaults to 'network'.

    Returns:
        networkx.Graph: Graph object of networkx
    """    
    network = io.loadmat(file)[variable_name]
    G = nx.Graph(network)
    return G
