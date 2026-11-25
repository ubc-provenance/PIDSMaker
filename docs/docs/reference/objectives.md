---
title: Objectives
description: Self-supervised and supervised training objectives available in PIDSMaker, the systems that use them, and their arguments.
---

# Objectives

An objective consists of a loss function and a decoder. Node-level objectives compute a loss for every node in a time-window graph, whereas edge-level ones compute a loss for every edge. This makes node-level objectives usually faster but less powerful than edge-level objectives to capture pair-wise information.

| Level | Objective | Learns to | Used by |
|---|---|---|---|
| Edge | `predict_edge_type` | Predict the type of each edge | Velox, Orthrus, Kairos |
| Edge | `predict_edge_contrastive` | Tell real edges from negative ones (link prediction) | Euler, Argus, Jbeil, E-GraphSAGE, ShadeWatcher |
| Edge | `autoencode_edge_embeddings` | Reconstruct the embedding of each edge | LARES |
| Edge | `deep_graph_infomax` | Tell edges from those of a corrupted graph | Anomal-E |
| Edge | `reconstruct_edge_embeddings` | Reconstruct edge embeddings | |
| Edge | `predict_edge_supervised` | Separate benign edges from given attack edges | [Supervised fine-tuning](../features/supervised.md) |
| Node | `predict_node_type` | Predict the type of each node | Flash, R-Caid, ThreaTrace |
| Node | `reconstruct_node_features`, `reconstruct_node_embeddings` | Reconstruct node features or embeddings | NodLink |
| Node | `reconstruct_masked_features`, `predict_masked_struct` | Reconstruct masked features and structure | MAGIC |
| Node | `one_class` | Map benign nodes into a hypersphere per node type (Deep SVDD) | OCR-APT |

Objectives are not tied to a family of datasets, except `predict_node_type`, which needs at least two node types: network datasets only have hosts (see [Running across families](../guide/systems.md#running-across-families) for the combinations that are tested). On network datasets, `predict_edge_type` predicts the `operation` of events (e.g. `LogOn`), or the attribute set as edge type with `construction.edge_type` (e.g. `logon_type`).

`predict_edge_contrastive`
:   Link prediction: the decoder scores the edges of the graph against negative edges. `loss` is `bce` (default), `ap`, the surrogate average precision loss of Argus, or `bpr`, the pairwise ranking loss of ShadeWatcher between each negative edge and the positive edge it was drawn for. `neg_sampling` draws the negatives: `shuffle_dst` (default) pairs each source with the destination of another edge, `uniform` and `uniform_dst` draw uniform nodes (`neg_ratio` negatives per edge), and `tgn` uses destinations embedded by the TGN (requires `batching.intra_graph_batching.tgn_last_neighbor.negative_destinations`). With the `inner_product` decoder, `inner_product.negate` scores edges by the negated inner product, as ShadeWatcher, whose inner product is the probability that two entities would not interact.

`autoencode_edge_embeddings`
:   The edge autoencoder of LARES: edge embeddings combine the embeddings of the two nodes with the edge features (gated skip connection, `use_edge_feats`), and edges are scored by their reconstruction error.

`deep_graph_infomax`
:   Deep Graph Infomax on edge embeddings, as in Anomal-E: edges are scored by the reconstruction error of their embedding by a PCA (`pca_components`) fitted on the training edges.

`autoencode_edge_embeddings` and `deep_graph_infomax` have no decoder (`decoder: none`).

`predict_edge_type` also accepts `loss: AMS`, an additive margin softmax configured in its `AMS` section.

`predict_edge_supervised` is the only supervised objective: it learns from attack edges that you provide. See [Supervised Fine-Tuning](../features/supervised.md).

## Arguments

--8<-- "scripts/args/args_objectives.md"
