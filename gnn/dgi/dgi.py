import torch
import torch.nn as nn

# https://github.com/PetarV-/DGI/blob/master/layers
class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.w = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj_mat):
        fts = self.w(x)
        output = torch.unsqueeze(torch.spmm(adj_mat, torch.squeeze(fts, 0)), 0)
        if self.bias is not None:
            output += self.bias
        return self.act(output)

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        return self.sigmoid(torch.mean(h, dim=1))

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.w = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h):
        s_x = torch.unsqueeze(s, 1).expand_as(h)
        s_x = torch.squeeze(self.w(h, s_x), 2)

        return s_x

class DGI(nn.Module):
    def __init__(self, in_features, n_h):
        super(DGI, self).__init__()

        self.gcn = GCN(in_features, n_h)
        self.readout = Readout()
        self.disc = Discriminator(n_h)

    def forward(self, pos, neg, adj_mat):
        h = self.gcn(pos, adj_mat)
        h_tilde = self.gcn(neg, adj_mat)

        s = self.readout(h)

        D_pos = self.disc(s, h)
        D_neg = self.disc(s, h_tilde)

        res = torch.cat((D_pos, D_neg), dim=1)

        return res

    def embed(self, features, adj_mat):
        h = self.gcn(features, adj_mat)
        s = self.readout(h)

        return h.detach(), s.detach()
