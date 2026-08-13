"""
Dataset loading for MUTAG, in the exact text format released alongside the
official GIN paper code (https://github.com/weihua916/powerful-gnns).

File format of dataset/MUTAG/MUTAG.txt
---------------------------------------
line 0:            N                              <- number of graphs
for each graph:
    line:          n_nodes  graph_label
    for each of the n_nodes lines:
        line:      tag  degree  neighbor_1 neighbor_2 ... neighbor_degree

`tag` is a discrete node label (roughly: atom type, for MUTAG). We turn the
set of tags seen across the whole dataset into a one-hot vector -- that
one-hot vector is the initial node feature h_v^(0) fed into the GIN.

We do NOT use any graph library. A graph is just:
    - edge_index : LongTensor [2, num_edges]   (both directions, i.e. undirected)
    - tags       : LongTensor [num_nodes]       (raw integer tag per node)
    - label      : int                          (graph class)
"""

import torch


class GraphSample:
    __slots__ = ["edge_index", "tags", "label", "num_nodes", "x"]

    def __init__(self, edge_index, tags, label):
        self.edge_index = edge_index      # [2, E] long tensor, both directions
        self.tags = tags                  # [N] long tensor of raw tag ids
        self.label = label                # python int
        self.num_nodes = tags.shape[0]


def load_mutag(path="dataset/MUTAG/MUTAG.txt"):
    """Parses the raw txt file into a list of GraphSample objects, plus the
    number of distinct node tags and number of classes seen."""

    with open(path, "r") as f:
        lines = f.read().strip().split("\n")

    ptr = 0
    num_graphs = int(lines[ptr]); ptr += 1

    graphs = []
    all_tags = set()
    all_labels = set()

    for _ in range(num_graphs):
        n_nodes, g_label_raw = map(int, lines[ptr].split())
        ptr += 1
        all_labels.add(g_label_raw)

        src, dst, tags = [], [], []
        for v in range(n_nodes):
            row = list(map(int, lines[ptr].split()))
            ptr += 1
            tag, deg = row[0], row[1]
            neighbors = row[2:2 + deg]
            tags.append(tag)
            all_tags.add(tag)
            for u in neighbors:
                # store both directions explicitly -- this is exactly what
                # "undirected message passing" means at the tensor level
                src.append(v); dst.append(u)
                src.append(u); dst.append(v)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        tags_t = torch.tensor(tags, dtype=torch.long)
        graphs.append(GraphSample(edge_index, tags_t, g_label_raw))

    # raw graph labels aren't guaranteed to be contiguous 0..C-1 (MUTAG
    # ships them as {0, 2}), so remap to contiguous class ids.
    label_list = sorted(all_labels)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    for g in graphs:
        g.label = label_to_idx[g.label]

    return graphs, sorted(all_tags), label_list


def build_node_features(graphs, tag_list):
    """Convert integer tags into one-hot feature vectors, in place.
    Returns feature dimension."""
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    feat_dim = len(tag_list)

    for g in graphs:
        idx = torch.tensor([tag_to_idx[int(t)] for t in g.tags], dtype=torch.long)
        x = torch.zeros(g.num_nodes, feat_dim)
        x[torch.arange(g.num_nodes), idx] = 1.0
        g.x = x  # [N, feat_dim]

    return feat_dim


def load_fold_indices(fold, path_dir="dataset/MUTAG/10fold_idx"):
    """The official repo ships the exact 10-fold train/test split used to
    produce the paper's numbers, so results are reproducible."""
    train_idx = list(map(int, open(f"{path_dir}/train_idx-{fold}.txt").read().split()))
    test_idx = list(map(int, open(f"{path_dir}/test_idx-{fold}.txt").read().split()))
    return train_idx, test_idx
