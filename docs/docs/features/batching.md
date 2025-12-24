## Batching

Batching refers to grouping edges, nodes, or graphs into a temporal graph provided as input to the model.

We provide three batching strategies that can be configured via dedicated [batching arguments](../config/tasks.md/#detection).

**Global Batching**: takes as input a large flattened graph comprising all events in the dataset and partitions it into equal-size graphs based on number of edges, minutes, or similar.

**Intra-graph Batching**: applies similar batching as global batching but within each built graph.

**Inter-graph Batching**: groups multiple graphs into a single batch. This batch is a large graph where all graphs are stacked together without any overlap, following the mini-batching strategy from [PyG](https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html).

![Batching](../img/batching.jpg)

!!! note
    All three batching strategies apply sequentially and can be used together.

### Choosing the Right Batch Size

The batch size plays a key role in determining the trade-off between memory usage, speed, and learning effectiveness. Selecting an appropriate batch size requires careful consideration of the graph’s scale, the temporal dynamics, and the available hardware resources.

**Large Batches**

- ✅ Enhance training speed through better GPU parallelization  
- ✅ Capture events over longer time periods  
- ❌ Increase GPU memory usage
- ❌ May cause high node in-degree, leading to over-squashing (loss of neighbor information)

**Small Batches**

- ✅ Reduce GPU memory consumption  
- ✅ Enable fine-grained neighborhood aggregation  
- ❌ Extend training time  
- ❌ Risk missing graph patterns spanning longer time ranges if temporal features are not captured

### TGN Last Neighbor Sampling

In the [Temporal Graph Network (TGN)](https://arxiv.org/abs/2006.10637) architecture, the objective is to predict edges within a graph batch at time $t$ based on the last neighbors of each node seen in batches happening prior to $t$. Setting `tgn_last_neighbor` to the argument `graph_preprocessing.intra_graph_batching.used_methods` enables to pre-compute the TGN graph for each preprocessed graph in the dataset. Specifically, it does not replace the graph directly but adds `tgn_*` attributes to it, which can be used by the downstream encoder.

To use the TGN architecture, you **should** also add `tgn` to `gnn_training.encoder.used_methods` as it enables to properly handle TGN attributes.
The `TGNEncoder` accepts as argument an `encoder`, defined in the config, which will be applied to the pre-computed TGN graph.

Examples of TGN config can be found in `kairos.yml` and `orthrus.yml`.

### Neighbor Sampling

!!! info
    Not implemented yet.
