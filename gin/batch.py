"""
Mini-batching graphs.

A GNN mini-batch is built by taking several small graphs and gluing them
into ONE big graph whose adjacency is block-diagonal (no edges between
graphs). This lets every layer be written as if there were a single graph,
while a `batch_vec` tensor remembers which node belongs to which original
graph -- used only at readout time to pool per-graph.

Example with 2 graphs of sizes 3 and 2:
    graph A nodes: 0,1,2      graph B nodes: 0,1
    after batching: A -> 0,1,2   B -> 3,4   (B's node ids shifted by +3)
    batch_vec = [0,0,0,1,1]
"""

import torch


class Batch:
    __slots__ = ["x", "edge_index", "batch_vec", "num_graphs", "labels"]


def collate(graphs):
    """graphs: list[GraphSample] (with .x already set by build_node_features)"""
    xs, edge_indices, batch_vec, labels = [], [], [], []
    node_offset = 0

    for g_idx, g in enumerate(graphs):
        xs.append(g.x)
        edge_indices.append(g.edge_index + node_offset)  # shift ids into batch space
        batch_vec.append(torch.full((g.num_nodes,), g_idx, dtype=torch.long))
        labels.append(g.label)
        node_offset += g.num_nodes

    b = Batch()
    b.x = torch.cat(xs, dim=0)                        # [total_N, feat_dim]
    b.edge_index = torch.cat(edge_indices, dim=1)      # [2, total_E]
    b.batch_vec = torch.cat(batch_vec, dim=0)          # [total_N]
    b.num_graphs = len(graphs)
    b.labels = torch.tensor(labels, dtype=torch.long)
    return b
