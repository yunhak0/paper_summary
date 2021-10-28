import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

class GATLayer(Module):
    def __init__(self, in_feature, out_feature, n_attention,
                 concat=True,
                 neg_input_slope=0.02,
                 dropout=0.06):
        super(GATLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.n_attention = n_attention
        self.concat = concat
        self.neg_input_slope = neg_input_slope
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        self.attention_weight = Parameter(torch.FloatTensor(2 * out_feature, 1))
        self.reset_parameters()

        self.leakyrelu = nn.LeakyReLU(self.neg_input_slope)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.attention_weight)

    def forward(self, x, adj):
        Wh = torch.matmul(x, self.weight)
        e_ij = self._attention_coef(Wh)

        # Select Neighbors
        zeros = -9e15 * torch.ones(e_ij)
        attention = torch.where(adj > 0, e_ij, zeros)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh) # N x F'

        if self.concat:
            h_prime = F.elu(h_prime)
        
        return h_prime
    
    def _attention_coef(self, Wh):
        Wh_i = torch.matmul(Wh, self.attention_weight[:self.out_feature, :])
        Wh_j = torch.matmul(Wh, self.attention_weight[self.out_feature:, :])
        e_ij = self.leakyrelu(Wh_i + Wh_j.T)
        return e_ij  # N x N

class GAT(Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout, negative_slope, n_head):
        super(GAT, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.n_head = n_head

        self.attentions = [GATLayer(n_feature, n_hidden, n_head, concat=True,
                                    neg_input_slope=negative_slope, dropout=dropout)
                           for _ in range(n_head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.final_attention = GATLayer(n_hidden * n_head, n_class, concat=False,
                                        neg_input_slope=negative_slope, dropout=dropout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attention(x, adj) for attention in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.final_attention(x, adj))
        x = F.log_softmax(x, dim=1)
        return x
