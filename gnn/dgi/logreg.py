import torch
import torch.nn as nn

class logreg(nn.Module):
    def __init__(self, in_features, n_class):
        super(logreg, self).__init__()
        self.w = nn.Linear(in_features, n_class)

        for m in self.modules():
            self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features):
        res = self.w(features)
        return res
