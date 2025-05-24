An objective simply consists in a loss function and a decoder. Node-level objectives compute a loss for every node in a time-window graph, whereas edge-level ones compute loss for all edges. This makes node-level objectives usually faster but less powerful than edge-level objectives to capture pair-wise information.

## Arguments

--8<-- "scripts/args/args_objectives.md"
