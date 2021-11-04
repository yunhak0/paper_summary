import numpy as np

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class GAT(nn.Module):
    def __init__(self,
                 n_layers,
                 n_features,
                 n_hiddens,
                 n_class,
                 n_attn_heads,
                 leaky_relu_alpha=0.2,
                 dropout=0.6,
                 activation=nn.ELU(),
                 skip_connection=False,
                 concat=True):
        super().__init__()
        assert len(n_attn_heads) == 2,\
            "The 'n_attn_heads' should be the list that has 2 elements: [building block layers, final layer]."
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_class = n_class
        self.n_attn_heads = n_attn_heads
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation
        self.skip_connection = skip_connection
        self.concat = concat

        # Building Block Layer
        layers = []
        layers.append(GATLayer(n_F_in=n_features, n_F_out=n_hiddens,
                               n_heads=n_attn_heads[0],
                               leaky_relu_alpha=leaky_relu_alpha,
                               dropout=dropout,
                               activation=activation,
                               skip_connection=skip_connection,
                               concat=concat))
        middle_layers = [GATLayer(n_F_in=n_hiddens * n_attn_heads[0],
                                  n_F_out=n_hiddens,
                                  n_heads=n_attn_heads[0],
                                  leaky_relu_alpha=leaky_relu_alpha,
                                  dropout=dropout,
                                  activation=activation,
                                  skip_connection=skip_connection,
                                  concat=concat)
                         for _ in range(1, n_layers - 1)]
        final_layer = GATLayer(n_F_in=n_hiddens * n_attn_heads[0],
                               n_F_out=n_class,
                               n_heads=n_attn_heads[1],
                               leaky_relu_alpha=leaky_relu_alpha,
                               dropout=dropout,
                               activation=None,
                               skip_connection=skip_connection,
                               concat=False)
        if middle_layers != []:
            layers.extend(middle_layers)
        layers.append(final_layer)

        self.gat_fin = nn.Sequential(*layers)

    def forward(self, data):
        return self.gat_fin(data)

class GATAttn(nn.Module):
    def __init__(self, n_F_in, n_F_out, n_heads,
                 leaky_relu_alpha=0.2,
                 dropout=0.6,
                 activation=nn.ELU()):
        super().__init__()
        self.n_F_in = n_F_in
        self.n_F_out = n_F_out
        self.n_heads = n_heads
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation

        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)

        # Weight matrix
        # K is number of attention heads
        # N is number of nodes
        # F is number of input features
        # F' is number of output features
        # W := [K, F, F']
        self.W = Parameter(torch.FloatTensor(n_heads, n_F_in, n_F_out))

        # Attention mechanism
        # a := [K, 1, F']
        self.a_source = Parameter(torch.FloatTensor(n_heads, 1, n_F_out))
        self.a_target = Parameter(torch.FloatTensor(n_heads, 1, n_F_out))

        # Initialized parameters
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.a_source, gain=1.414)
        nn.init.xavier_normal_(self.a_target, gain=1.414)

    def forward(self, data):
        # x := [N, F]
        # adj := [N, N]
        x, adj = data

        if self.dropout != 0.0:
            x = F.dropout(x, self.dropout, training=self.training)

        # Wh := x (prod) self.W = [N, F] (prod) [K, F, F'] = [K, N, F']
        Wh = torch.matmul(x, self.W)

        # Attention Coefficients [K, N, N]
        e_ij = self._get_attn_coef(Wh, adj)

        # Attention
        attention = F.softmax(e_ij, dim=2)
        if self.dropout != 0.0:
            attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted features
        if self.dropout != 0.0:
            Wh = F.dropout(Wh, self.dropout, training=self.training)

        # Embedding
        # h' := attention (prod) Wh = [K, N, N] (prod) [K, N, F'] = [K, N, F']
        h_prime_init = torch.matmul(attention, Wh)

        return (h_prime_init, adj)

    def _get_attn_coef(self, Wh, adj):
        # Wh_i or Wh_j := a (Hadamard prod) Wh = [K, 1, F'] (h-prod) [K, N, F']
        # = [K, N, F'] -> sum(dim=2) -> [K, N, 1]
        Wh_i = (self.a_source * Wh).sum(dim=2, keepdim=True)
        Wh_j = (self.a_target * Wh).sum(dim=2, keepdim=True)

        # e_ij := Wh_i + Wh_j.transpose() = [K, N, 1] + [K, 1, N] = [K, N, N]
        e_ij = self.leaky_relu(Wh_i + Wh_j.view(Wh_j.shape[0], Wh_j.shape[2], Wh_j.shape[1]))

        # Mask attention coefficients [K, N, N]
        zeros = -9e15 * torch.ones_like(e_ij)
        e_ij = torch.where(adj > 0, e_ij, zeros)

        return e_ij

class GATLayer(nn.Module):
    def __init__(self, n_F_in, n_F_out, n_heads,
                 leaky_relu_alpha=0.2,
                 dropout=0.6,
                 activation=nn.ELU(),
                 skip_connection=False,
                 concat=True):
        super().__init__()
        self.n_F_in = n_F_in
        self.n_F_out = n_F_out
        self.n_heads = n_heads
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation
        self.skip_connection = skip_connection
        self.concat = concat

        if skip_connection:
            self.skip_proj = Parameter(torch.FloatTensor(n_heads, n_F_in, n_F_out))
            nn.init.xavier_normal_(self.skip_proj, gain=1.414)
        else:
            self.register_parameter('skip_proj', None)

        self.multi_head_attns = GATAttn(n_F_in, n_F_out, n_heads,
                                        leaky_relu_alpha,
                                        dropout,
                                        activation)

    def forward(self, data):
        x, adj = data

        # h_prime [K, N, F_out]
        h_prime = self.multi_head_attns(data)[0]

        if self.skip_connection:
            if h_prime.size()[-1] == x.size()[-1]:
                h_prime += x
            else:
                # x [K, N, F_in] (prod) skip_proj [K, F_in, F_out] = [K, N, F_out]
                h_prime += torch.matmul(x, self.skip_proj)

        if self.concat:
            h_prime = h_prime.transpose(0, 1).reshape(-1, self.n_heads * self.n_F_out)
        else:
            h_prime = h_prime.mean(dim=0)

        # Apply nonlinearity activation function
        if self.activation is not None:
            h_prime = self.activation(h_prime)

        return (h_prime, adj)
