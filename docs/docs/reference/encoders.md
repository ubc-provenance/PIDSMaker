---
title: Encoders
description: GNN and non-GNN encoders available in PIDSMaker, and their arguments.
---

# Encoders

Encoders compute the node (and edge) embeddings that are passed to the [decoders](decoders.md), whose outputs feed the [objectives](objectives.md) that compute the loss.
They can be GNNs that leverage the graph structure (`tgn`, `graph_attention`, `sage`...), but also a single linear layer like in Velox (`none`) or a custom MLP (`custom_mlp`).
Several encoders can be chained with `training.encoder.used_methods`, e.g. `tgn,graph_attention`.

`hetero_graph_transformer` is a Heterogeneous Graph Transformer (HGT): each node type and edge type gets its own attention parameters. It is not available on OpTC.

## Arguments

--8<-- "scripts/args/args_encoders.md"
