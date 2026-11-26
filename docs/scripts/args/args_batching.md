<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">save_on_disk</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">node_features</span>: <span class="value">str (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">edges_distribution_scaling</span>: <span class="value">str (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">edge_features</span>: <span class="value">str (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">compact_node_ids</span>: <span class="value">bool (6)</span></li>
    <li class='no-bullet'><span class="key-leaf">fix_buggy_graph_reindexer</span>: <span class="value">bool (7)</span></li>
    <li class='bullet'><span class="key">global_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">global_batching_batch_size</span>: <span class="value">int (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">global_batching_batch_size_inference</span>: <span class="value">int (10)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">intra_graph_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (11)</span></li>
        <li class='bullet'><span class="key">edges</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">intra_graph_batch_size</span>: <span class="value">int (12)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">tgn_last_neighbor</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">tgn_neighbor_size</span>: <span class="value">int (13)</span></li>
            <li class='no-bullet'><span class="key-leaf">tgn_neighbor_n_hop</span>: <span class="value">int (14)</span></li>
            <li class='no-bullet'><span class="key-leaf">fix_buggy_orthrus_TGN</span>: <span class="value">bool (15)</span></li>
            <li class='no-bullet'><span class="key-leaf">fix_tgn_neighbor_loader</span>: <span class="value">bool (16)</span></li>
            <li class='no-bullet'><span class="key-leaf">directed</span>: <span class="value">bool (17)</span></li>
            <li class='no-bullet'><span class="key-leaf">insert_neighbors_before</span>: <span class="value">bool (18)</span></li>
            <li class='no-bullet'><span class="key-leaf">reset_state_every_day</span>: <span class="value">bool (19)</span></li>
            <li class='no-bullet'><span class="key-leaf">negative_destinations</span>: <span class="value">bool (20)</span></li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">inter_graph_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (21)</span></li>
        <li class='no-bullet'><span class="key-leaf">inter_graph_batch_size</span>: <span class="value">int (22)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Whether to store the graphs on disk upon building the graphs.             Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.<br>
2. Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector,                                     `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones                                     with length `node_type`, `edges_distribution` counts emitted and received edges.<br><br><b>Available options (multi selection)</b>:<br>`node_type`<br>`node_emb`<br>`only_ones`<br>`edges_distribution`
3. Scaling of the `edges_distribution` counts: `max` divides them by the largest count of                 the time window (default), `none` keeps the raw counts (as ThreaTrace), `log1p` applies log(1 + x).<br><br><b>Available options (one selection)</b>:<br>`max`<br>`none`<br>`log1p`
4. Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet`                                 considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector                                 used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN,                                 `edge_attrs` are the edge attributes of the dataset's schema besides the edge type (one-hot categorical and                                 log-scaled numeric attributes, e.g. ports or logon types), `none` uses no features.<br><br><b>Available options (multi selection)</b>:<br>`edge_type`<br>`edge_type_triplet`<br>`msg`<br>`time_encoding`<br>`edge_attrs`<br>`none`
5. Whether the GNN should be trained on all datasets in `multi_dataset`.<br>
6. Renumbers the nodes of the graphs from 0 to (number of nodes in the graphs - 1), keeping                                 their order, instead of using the node IDs of the database, which also count the nodes                                 absent from all graphs (86% of the 44.5M IDs of TRACE_E3). Tensors indexed by node (TGN                                 memory, neighbor loader, node ID embeddings, `all_nodes` embeddings, ...) are sized                                 accordingly. Node IDs are restored in the outputs. Results are unchanged, except for                                 methods drawing random nodes among all IDs (e.g. uniform negatives of Euler and Argus).<br>
7. A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d)                                                 slightly modify node features. Setting this to true fixes the bug.<br>
8. Flattens the time window-based graphs into a single large                             temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges,                             the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types,                             `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.<br><br><b>Available options (one selection)</b>:<br>`edges`<br>`minutes`<br>`unique_edge_types`<br>`none`
9. Controls the value associated with `global_batching.used_method` (training+inference).<br>
10. Controls the value associated with `global_batching.used_method` (inference only).<br>
11. Breaks each previously computed graph into even smaller graphs.                                 `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500`                                 creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph                                 based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.                                `none` does not alter any graph.<br><br><b>Available options (multi selection)</b>:<br>`edges`<br>`tgn_last_neighbor`<br>`none`
12. Controls the value associated with `global_batching.used_method`.<br>
13. Number of last neighbors to store for each node.<br>
14. If greater than one, will also gather the last neighbors of neighbors.<br>
15. A bug has been in the first version of the framework, where the features of last neighbors not appearing                                             in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.<br>
16. We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This                                                 is an unofficial fix.<br>
17. The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.<br>
18. Whether to insert the edges of the current graph before loading last neighbors.<br>
19. Empties the last neighbors and resets the TGN memory at the first batch of each day                         (date of its first event), as Kairos, which trains and tests on one graph per day.<br>
20. Draws a negative destination for each edge, uniformly among the destinations                         of the training graphs (training) or of all graphs (validation, test), and adds                         it to the nodes of the TGN graph to be embedded with memory and neighbors, as                         TGN and Jbeil (used by `predict_edge_contrastive.neg_sampling=tgn`).<br>
21. Batches multiple graphs into a single large one for parallel training.                                 Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.<br><br><b>Available options (one selection)</b>:<br>`graph_batching`<br>`none`
22. Controls the value associated with `inter_graph_batching.used_method`.<br>
