import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class gcn_layer(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)

    def forward(self, x, adj):
        support = torch.mm(x, self.W)
        outs = torch.mm(adj, support)
        outs = self.activation(outs)
        return outs


class gae(nn.Module):
    def __init__(self, in_features, n_hidden, dim_z):
        super().__init__()
        self.gcn_layer1 = gcn_layer(in_features, n_hidden)
        self.gcn_layer2 = gcn_layer(n_hidden, dim_z,
                                    activation=lambda x: x)

    def forward(self, x, adj):
        # Encoder
        hidden = self.gcn_layer1(x, adj)
        z = self.mu = self.gcn_layer2(hidden, adj)
        # Decoder
        in_prod = torch.matmul(z, z.t())
        adj_hat = torch.sigmoid(in_prod)
        return adj_hat

    def loss(self, preds, labels, pos_weight, norm):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        return cost


class vgae(nn.Module):
    def __init__(self, in_features, n_hidden, dim_z):
        super().__init__()
        self.in_features = in_features
        self.n_hidden = n_hidden
        self.dim_z = dim_z
        self.gcn_layer1 = gcn_layer(in_features, n_hidden)
        self.gcn_mu = gcn_layer(n_hidden, dim_z,
                                activation=lambda x: x)
        self.gcn_log_sigma = gcn_layer(n_hidden, dim_z,
                                       activation=lambda x: x)
    def forward(self, x, adj):
        # Encoder
        hidden = self.gcn_layer1(x, adj)
        self.mu = self.gcn_mu(hidden, adj)
        self.log_sigma = self.gcn_log_sigma(hidden, adj)

        # Reparameterization trick
        if self.training:
            std = torch.exp(self.log_sigma)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(self.mu)
        else:
            z = self.mu

        # Decoder
        in_prod = torch.matmul(z, z.t())
        adj_hat = torch.sigmoid(in_prod)

        return adj_hat

    def loss(self, preds, labels, n_nodes, pos_weight, norm):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        kl_divergence = (0.5 / n_nodes) * torch.mean(torch.sum(1 + 2 * self.log_sigma - self.mu**2 - self.log_sigma.exp()**2, dim=1))
        cost -= kl_divergence
        return cost
