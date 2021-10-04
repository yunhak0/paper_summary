import math
import torch

from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

class GCNLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdev, stdev)
        if self.bias is not None:
            self.bias.data.uniform_(-stdev, stdev)

    def forward(self, x, adj_mat):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj_mat, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(Module):
    def __init__(self, n_features, n_hidden, n_class, dropout):
        super(GCN, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.dropout = dropout

        self.gcn_layer1 = GCNLayer(n_features, n_hidden)
        self.gcn_layer2 = GCNLayer(n_hidden, n_class)

    def forward(self, x, adj_mat):
        x = F.relu(self.gcn_layer1(x, adj_mat))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_layer2(x, adj_mat)
        return F.log_softmax(x, dim=1)
