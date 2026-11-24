---
title: Encoders
description: GNN and non-GNN encoders available in PIDSMaker, and their arguments.
---

# Encoders

Encoders compute the node (and edge) embeddings that are passed to the [decoders](decoders.md), whose outputs feed the [objectives](objectives.md) that compute the loss.
They can be GNNs that leverage the graph structure (`tgn`, `graph_attention`, `sage`...), but also a single linear layer like in Velox (`none`) or a custom MLP (`custom_mlp`).
Several encoders can be chained with `training.encoder.used_methods`, e.g. `tgn,graph_attention`.

The main encoders, grouped by what they model, with the systems that use them:

| Role | Encoders | Used by |
|---|---|---|
| Message passing within each graph | `sage`, `gat`, `graph_attention`, `gin`, `gcn`, `sum_aggregation`, `rcaid_gat`, `magic_gat` | Flash and ThreaTrace (`sage`), Kairos, Orthrus and Jbeil (`graph_attention`), Euler (`gcn`), NodLink, R-Caid, MAGIC |
| Edge types and attributes in messages | `rgcn`, `rgcn_per_type`, `argus`, `egraphsage`, `lares`, `gin` (GINE, with edge features) | OCR-APT (`rgcn_per_type`), Argus, E-GraphSAGE and Anomal-E (`egraphsage`), LARES |
| Identity of each node | `node_id_embedding`, `lares` | Euler, Argus, LARES |
| Memory across time windows | `tgn`, `snapshot_gru` | Kairos, Orthrus and Jbeil (`tgn`), Euler and Argus (`snapshot_gru`) |
| No message passing | `none` (linear layer), `custom_mlp` | Velox |
| Heterogeneous graphs | `hetero_graph_transformer` | |

Encoders that learn the identity of nodes are meant for nodes without informative features, such as the hosts of network datasets; encoders that read edge attributes need a dataset that has some, or the edge type in their place (`batching.edge_features=edge_type`). Any encoder can otherwise be used on provenance and network datasets.

`hetero_graph_transformer` is a Heterogeneous Graph Transformer (HGT): each node type and edge type gets its own attention parameters. It is not available on OpTC.

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
