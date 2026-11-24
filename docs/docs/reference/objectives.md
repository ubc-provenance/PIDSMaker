---
title: Objectives
description: Self-supervised and supervised training objectives available in PIDSMaker, and their arguments.
---

# Objectives

An objective consists of a loss function and a decoder. Node-level objectives compute a loss for every node in a time-window graph, whereas edge-level ones compute a loss for every edge. This makes node-level objectives usually faster but less powerful than edge-level objectives to capture pair-wise information.

Objectives of the [network systems](../guide/network.md):

`predict_edge_contrastive`
:   Link prediction: the decoder scores the edges of the graph against negative edges. `loss` is `bce` (default) or `ap`, the surrogate average precision loss of Argus. `neg_sampling` draws the negatives: `shuffle_dst` (default) pairs each source with the destination of another edge, `uniform` and `uniform_dst` draw uniform nodes (`neg_ratio` negatives per edge), and `tgn` uses destinations embedded by the TGN (requires `batching.intra_graph_batching.tgn_last_neighbor.negative_destinations`).

`autoencode_edge_embeddings`
:   The edge autoencoder of LARES: edge embeddings combine the embeddings of the two nodes with the edge features (gated skip connection, `use_edge_feats`), and edges are scored by their reconstruction error.

`deep_graph_infomax`
:   Deep Graph Infomax on edge embeddings, as in Anomal-E: edges are scored by the reconstruction error of their embedding by a PCA (`pca_components`) fitted on the training edges.

`autoencode_edge_embeddings` and `deep_graph_infomax` have no decoder (`decoder: none`).

`predict_edge_type` also accepts `loss: AMS`, an additive margin softmax configured in its `AMS` section.

`predict_edge_supervised` is the only supervised objective: it learns from attack edges that you provide. See [Supervised Fine-Tuning](../features/supervised.md).

## Arguments

--8<-- "scripts/args/args_objectives.md"
