---
title: Decoders
description: Decoders that map embeddings to the output shape of each objective, and their arguments.
---

# Decoders

Decoders take as input node and/or edge embeddings and pass them through another neural network in such a way that the last layer has a shape that fits the downstream objective. For example, a `predict_edge_type` objective requires the final shape to be the number of edge types, whereas a `reconstruct_node_features` objective needs a shape that matches the input features given to the encoder.
Decoders are usually much simpler than encoders, and can be customized via `edge_mlp` for edge-level tasks like `predict_edge_type` or via `node_mlp` for node-level tasks like `reconstruct_node_features`.

The `autoencode_edge_embeddings` and `deep_graph_infomax` [objectives](objectives.md) don't use a decoder (`decoder: none`): they have their own networks.

## Arguments

--8<-- "scripts/args/args_decoders.md"
