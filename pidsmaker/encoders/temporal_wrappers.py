"""Encoder wrappers used by network IDSs (Euler, Argus, LARES).

- `NodeIdEmbedding` concatenates a learned embedding of each node's identity to its features,
  equivalent to the one-hot node ID features of Euler, Argus and LARES followed by a linear
  layer. With `all_nodes`, all the nodes of the dataset are embedded in every graph, as Euler
  and Argus do (nodes absent from the graph are isolated and only get their ID embedding):
  every node then has an embedding trained at each graph, e.g. by uniform negative edges.
  Otherwise, only the nodes of each graph are embedded, and nodes unseen during training keep
  an untrained embedding.
- `SnapshotGRU` runs a GRU over the successive embeddings of each node, one step per graph
  (i.e. per time window or batch), as Euler and Argus do over snapshots. The hidden state of
  each node is kept across graphs and detached after each graph (truncated backpropagation
  through time, whereas Euler backpropagates through all the snapshots of the training set).
  The embedding of a node is its new hidden state projected by a linear layer ("detector" mode
  of Euler).

When the encoder embeds other nodes than those of the batch (`all_nodes`), its output carries
the IDs of the embedded nodes (`n_id`) and the edges of the batch in this node space
(`edge_index`), used by the model to gather the embeddings of edges and by objectives.
"""

import torch
import torch.nn as nn


class NodeIdEmbedding(nn.Module):
    def __init__(self, encoder, num_nodes, dim, all_nodes=False):
        super().__init__()
        self.encoder = encoder
        self.embedding = nn.Embedding(num_nodes, dim)
        self.all_nodes = all_nodes

    def forward(self, x, original_n_id, edge_index, **kwargs):
        if not self.all_nodes:
            x = torch.cat([x, self.embedding(original_n_id.long())], dim=-1)
            return self.encoder(x=x, original_n_id=original_n_id, edge_index=edge_index, **kwargs)

        # All the nodes of the dataset, in the ID space of the dataset
        n_id = original_n_id.long()
        num_nodes = self.embedding.num_embeddings
        x_all = x.new_zeros((num_nodes, x.size(1)))
        x_all[n_id] = x
        x_all = torch.cat([x_all, self.embedding.weight], dim=-1)
        all_n_id = torch.arange(num_nodes, device=x.device)
        global_edge_index = n_id[edge_index]
        res = dict(
            self.encoder(x=x_all, original_n_id=all_n_id, edge_index=global_edge_index, **kwargs)
        )
        res["n_id"] = all_n_id
        res["edge_index"] = global_edge_index
        return res

    def reset_state(self):
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()


class SnapshotGRU(nn.Module):
    def __init__(self, encoder, num_nodes, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.register_buffer("state", torch.zeros(num_nodes, hidden_dim))

    def forward(self, original_n_id, **kwargs):
        res = dict(self.encoder(original_n_id=original_n_id, **kwargs))
        # Nodes embedded by the encoder (all the nodes of the dataset with `all_nodes`)
        n_id = res.get("n_id", original_n_id).long()
        h_new = self.gru(self.dropout(torch.tanh(res["h"])), self.state[n_id])
        self.state[n_id] = h_new.detach()
        res["h"] = self.out(h_new)
        return res

    def reset_state(self):
        self.state.zero_()
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
