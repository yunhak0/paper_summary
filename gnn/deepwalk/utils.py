from scipy import io
import networkx as nx
import torch

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

# https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283/2
def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")