---
title: Encoders
description: GNN and non-GNN encoders available in PIDSMaker, and their arguments.
---

# Encoders

Encoders compute the node (and edge) embeddings that are passed to the [decoders](decoders.md), whose outputs feed the [objectives](objectives.md) that compute the loss.
They can be GNNs that leverage the graph structure (`tgn`, `graph_attention`, `sage`...), but also a single linear layer like in Velox (`none`) or a custom MLP (`custom_mlp`).
Several encoders can be chained with `training.encoder.used_methods`, e.g. `tgn,graph_attention`.

`hetero_graph_transformer` is a Heterogeneous Graph Transformer (HGT): each node type and edge type gets its own attention parameters. It is not available on OpTC.

Encoders of the [network systems](../guide/network.md):

`gcn`
:   Graph convolutional layers whose edges are weighted by `batch.edge_weight`, set by the `merge_edges` transformation (Euler).

`argus`
:   GCN layers followed by an edge-conditioned convolution (NNConv) on the edge features, if any (Argus).

`lares`
:   A learned embedding of each host's identity, then a single sum-aggregation layer over the messages of the neighbors and the features of their edges (LARES). `flow` sets the direction of aggregation, and `use_node_features` adds the node features to the ID embedding.

`egraphsage`
:   GraphSAGE layers whose messages include the edge features, aggregated in both directions with `undirected` (E-GraphSAGE, Anomal-E).

`node_id_embedding` and `snapshot_gru` wrap another encoder, like `tgn` (e.g. `gcn,node_id_embedding,snapshot_gru`), and can't be combined with `tgn`. `node_id_embedding` concatenates a learned embedding of each node's identity to its input features; with `all_nodes`, all the nodes of the dataset are embedded in every graph (edge-level objectives only). `snapshot_gru` runs a GRU over the successive embeddings of each node, one step per graph (Euler, Argus).

## Arguments

--8<-- "scripts/args/args_encoders.md"
