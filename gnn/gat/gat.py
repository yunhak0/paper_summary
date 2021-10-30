import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module


# Inspired by 
# https://github.com/PetarV-/GAT
# https://github.com/Diego999/pyGAT
# https://github.com/gordicaleksa/pytorch-GAT

class GATAttn(Module):
    def __init__(self, n_F_in, n_F_out, n_heads,
                 leaky_relu_alpha=0.2,
                 dropout=0.6,
                 activation=nn.ELU(),
                 skip_connection=False):
        super().__init__()
        self.n_F_in = n_F_in
        self.n_F_out = n_F_out
        self.n_heads = n_heads
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation
        self.skip_connection = skip_connection

        self.leaky_relu = nn.LeakyReLU(self.leaky_relu_alpha)

        # Weight matrix: W
        self.W = Parameter(torch.FloatTensor(n_F_in, n_F_out))
        # Attention mechanism: a
        self.a = Parameter(torch.FloatTensor(2 * n_F_out, 1))
        # Skip Connection
        if skip_connection:
            self.skip_proj = nn.Linear(n_F_in, n_F_out * n_heads, bias=False)
        else:
            self.register_parameter('skip_proj', None)
        
        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def forward(self, x, adj):
        if self.dropout != 0.0:
            x = F.dropout(x, self.dropout, training=self.training)
        Wh = torch.mm(x, self.W)

        # Attention
        e_ij = self._get_attention_coef(Wh, adj)
        attention = F.softmax(e_ij, dim=1)
        if self.dropout != 0.0:
            attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted Feature
        if self.dropout != 0.0:
            Wh = F.dropout(Wh, self.dropout, training=self.training)

        h_prime_init = torch.matmul(attention, Wh)

        # Skip Connection
        if self.skip_connection:
            if h_prime_init.shape[-1] == x.shape[-1]:
                h_prime_init += x
            else:
                h_prime_init += self.skip_proj(x)

        # Apply nonlinearity activation function
        if self.activation is not None:
            h_prime_init = self.activation(h_prime_init)

        return h_prime_init

    def _get_attention_coef(self, Wh, adj):
        Wh_i = torch.matmul(Wh, self.a[:self.n_F_out, :])
        Wh_j = torch.matmul(Wh, self.a[self.n_F_out:, :])
        e_ij = self.leaky_relu(Wh_i + Wh_j.T)

        # Masked attention coef
        zeros = -9e15 * torch.ones_like(e_ij)
        e_ij = torch.where(adj > 0, e_ij, zeros)

        return e_ij


class GATLayer(GATAttn):
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
        self.leaky_relu_alphay = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation
        self.skip_connection = skip_connection
        self.concat = concat

        # Multi-heads Attention
        self.attentions = [GATAttn(n_F_in, n_F_out, n_heads,
                                   leaky_relu_alpha, dropout, activation,
                                   skip_connection)
                           for _ in range(n_heads)]

        for i, attn in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attn)

    def forward(self, x, adj):
        if self.concat:
            h_prime = torch.cat([attn_i(x, adj) for attn_i in self.attentions], dim=1)
        else:
            h_prime = [attn_i(x, adj) for attn_i in self.attentions]
            h_prime = h_prime.sum() / self.n_heads

        return h_prime


class GAT(GATLayer):
    def __init__(self, n_layers, n_feature, n_hidden, n_class,
                 n_heads,
                 leaky_relu_alpha=0.2,
                 dropout=0.6,
                 activation=nn.ELU(),
                 skip_connection=False,
                 concat=True):
        super().__init__()
        assert len(n_heads) == 2, "The 'n_heads' is the list that has 2 elements: [building block layer, final layer]."
        self.n_layers = n_layers
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_heads = n_heads
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout
        self.activation = activation
        self.skip_connection = skip_connection
        self.concat = concat

        # Building Block Layers
        layers = [GATLayer(n_F_in=n_feature, n_F_out=n_hidden,
                           n_heads=n_heads[0],
                           leaky_relu_alpha=leaky_relu_alpha,
                           dropout=dropout,
                           activation=activation,
                           skip_connection=skip_connection,
                           concat=concat)
                  for _ in range(n_layers - 1)]

        # Final Layer
        final_layer = GATLayer(n_F_in=n_hidden * n_heads[0],
                               n_F_out=n_class,
                               n_heads=n_heads[1],
                               leaky_relu_alpha=leaky_relu_alpha,
                               dropout=dropout,
                               activation=None,
                               skip_connection=skip_connection,
                               concat=False)
        layers.append(final_layer)

        self.gat = nn.Sequential(*layers)

    def forward(self, x, adj):
        return self.gat(x, adj)
