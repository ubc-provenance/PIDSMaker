# Configuration

This page summarizes all available arguments for editing YML config files and CLI arguments.

## Text embedding methods

Used to transform textual attributes of entities (e.g. file paths, process command lines, socket IP addresses and ports) into a vector.
Some methods like `word2vec` and `doc2vec` learn this vector from the text corpus, while others like `hierarchical_hashing` compute the vector in a deterministic way. 

Other methods like `only_type` and `only_ones` simply skip this embedding step and assign either a one-hot encoded type or ones to each entity. Those methods thus do not require any specific argument.
In all methods, the resulting vectors are used as node features during the `gnn_training` task.

--8<-- "scripts/args/args_featurizations.md"

## Encoders

Those are neural network encoders. They can be GNNs (`tgn`, `graph_attention`, etc.) in the case where the graph structure is leveraged, but can also be a simple linear layer like the one used in Velox (`none`) or a more complex custom MLP (`custom_mlp`).
The job of encoders is to compute the node and edge embeddings given the next step to the decoder and objective to compute the loss.

--8<-- "scripts/args/args_encoders.md"

## Decoders

Decoders take as input node and/or edge embeddings and pass them though another neural network in such a way that the last layer has a shape that fits the downstream objective. For example, a `predict_edge_type` objective requires the final shape to be the number of edge types, whereas a `reconstruct_node_features` objective needs a shape that matches the input features given to the encoder.
Decoders are usually much simpler than encoders, and can be customed via `edge_mlp` for edge-level tasks like `predict_edge_type` or via `node_mlp` for node-level tasks like `reconstruct_node_features`.

--8<-- "scripts/args/args_decoders.md"

## Objectives

An objective simply consists in a loss function and a decoder. Node-level objectives compute a loss for every node in a time-window graph, whereas edge-level ones compute loss for all edges. This makes node-level objectives usually faster but less powerful than edge-level objectives to capture pair-wise information.

--8<-- "scripts/args/args_objectives.md"

## Tasks

Tasks are steps composing the pipeline, starting from graph construction (`build_graphs`) to detection (`evaluation`) or optionally triage (`tracing`).
Each task takes as input the output from the previous task and write its output to the disk so that the next task can use it. This process enables "checkpointing" across the pipeline and avoids the duplication of compute. More information on tasks and the pipeline [here](pipeline.md).

### Preprocessing

--8<-- "scripts/args/args_preprocessing.md"

### Featurization

--8<-- "scripts/args/args_featurization.md"

### Detection

--8<-- "scripts/args/args_detection.md"

### Triage

--8<-- "scripts/args/args_triage.md"

