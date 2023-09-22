from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


class ModulatedGraphConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        adj,
        bias=True,
        num_experts=0,
        reg_type="add",
        symmetric=True,
    ):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conditional = num_experts > 0
        self.symmetric = symmetric
        self.reg_type = reg_type

        self.W = nn.Parameter(
            torch.zeros(size=(2, in_features, out_features), dtype=torch.float)
        )
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(
            torch.ones(size=(adj.size(0), out_features), dtype=torch.float)
        )

        self.adj = adj

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float)
            )
            stdv = 1.0 / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

        if self.conditional:
            self._routing_fn = _routing(in_features, num_experts, 0.2)
            self.experts = nn.Parameter(
                adj.unsqueeze(0).repeat(num_experts, 1, 1)
            )
            nn.init.xavier_uniform_(self.experts)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        if self.conditional:
            adj = self.adj[None, :].to(input.device)

            c = F.avg_pool2d(input, [self.adj.size(1), 1])
            r_w = self._routing_fn(c)
            cond_e = torch.sum(r_w[:, :, None, None] * self.experts, 1)
            if self.reg_type == "add":
                adj = adj + cond_e
            elif self.reg_type == "mul":
                adj = adj * cond_e
            elif self.reg_type == "no_skeleton":
                adj = cond_e
            else:
                assert False, "Invalid regulazation type"

            # symmetry modulation
            if self.symmetric:
                adj = (adj.transpose(1, 2) + adj) / 2
        else:
            # add modulation
            adj = self.adj.to(input.device) + self.adj2.to(input.device)

            # symmetry modulation
            adj = (adj.T + adj) / 2

        # mix modulation
        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(
            adj * (1 - E), self.M * h1
        )
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
