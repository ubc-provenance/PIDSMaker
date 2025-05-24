Those are neural network encoders. They can be GNNs (`tgn`, `graph_attention`, etc.) in the case where the graph structure is leveraged, but can also be a simple linear layer like the one used in Velox (`none`) or a more complex custom MLP (`custom_mlp`).
The job of encoders is to compute the node and edge embeddings given the next step to the decoder and objective to compute the loss.

--8<-- "scripts/args/args_encoders.md"
