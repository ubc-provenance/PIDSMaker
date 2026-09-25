An objective simply consists in a loss function and a decoder. Node-level objectives compute a loss for every node in a time-window graph, whereas edge-level ones compute loss for all edges. This makes node-level objectives usually faster but less powerful than edge-level objectives to capture pair-wise information.

`predict_edge_supervised` is the only supervised objective: it learns from attack edges that you provide. See [Supervised Fine-Tuning](../features/supervised.md).

## Arguments

--8<-- "scripts/args/args_objectives.md"
