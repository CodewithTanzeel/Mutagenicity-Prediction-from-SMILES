"""
Graph Isomorphism Network (GIN), from "How Powerful are Graph Neural
Networks?" (Xu, Hu, Leskovec, Jegelka, ICLR 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src, index, num_nodes):
    out = src.new_zeros((num_nodes, src.shape[1]))
    out.index_add_(0, index, src)
    return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        assert num_layers >= 1
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.bns = nn.ModuleList(
            nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers - 1)
        )

    def forward(self, x):
        for i, linear in enumerate(self.linears[:-1]):
            x = F.relu(self.bns[i](linear(x)))
        return self.linears[-1](x)


class GINConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, learn_eps=True):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=2)
        init_eps = 0.0
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(init_eps))
        else:
            self.register_buffer("eps", torch.tensor(init_eps))

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        neighbor_sum = scatter_sum(x[src], dst, num_nodes=x.shape[0])
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)


def readout_sum(x, batch_vec, num_graphs):
    return scatter_sum(x, batch_vec, num_graphs)


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=5,
                 dropout=0.5, learn_eps=True):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(dim, hidden_dim, hidden_dim, learn_eps))
            dim = hidden_dim

        self.pred_heads = nn.ModuleList()
        self.pred_heads.append(nn.Linear(in_dim, num_classes))       # layer 0 (raw features)
        for _ in range(num_layers - 1):
            self.pred_heads.append(nn.Linear(hidden_dim, num_classes))

        self.dropout = dropout

    def node_embeddings(self, x, edge_index):
        layer_reps = [x]
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            layer_reps.append(h)
        return layer_reps

    def forward(self, x, edge_index, batch_vec, num_graphs):
        layer_reps = self.node_embeddings(x, edge_index)

        logits = 0.0
        for h_k, head in zip(layer_reps, self.pred_heads):
            pooled = readout_sum(h_k, batch_vec, num_graphs)     # [num_graphs, dim_k]
            logits = logits + F.dropout(head(pooled), p=self.dropout, training=self.training)
        return logits

    def graph_embedding(self, x, edge_index, batch_vec, num_graphs):
        layer_reps = self.node_embeddings(x, edge_index)
        return readout_sum(layer_reps[-1], batch_vec, num_graphs)
