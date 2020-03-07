import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim,  name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim

        if out_state_dim == None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.fc2 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )
        self.name = name

    def forward(self, input, adj):
        state_in = self.fc1(input)

        ##torch.bmm(p1, p2)
        # p1: the first batch of matrices to be multiplied, p2 is the second
        # eg: p1:(c, w1, h1), p2:(c, w2, h2) -> bmm(p1, p2): (c, w1, h2)
        forward_input = self.fc2(torch.bmm(adj, input))

        return state_in + forward_input



    def __repr__(self):
        return self.__class__.__name__ + ' (' +  self.name + ')'