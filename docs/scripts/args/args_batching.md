<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">save_on_disk</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">node_features</span>: <span class="value">str (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">edge_features</span>: <span class="value">str (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">fix_buggy_graph_reindexer</span>: <span class="value">bool (5)</span></li>
    <li class='bullet'><span class="key">global_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">global_batching_batch_size</span>: <span class="value">int (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">global_batching_batch_size_inference</span>: <span class="value">int (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">intra_graph_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (9)</span></li>
        <li class='bullet'><span class="key">edges</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">intra_graph_batch_size</span>: <span class="value">int (10)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">tgn_last_neighbor</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">tgn_neighbor_size</span>: <span class="value">int (11)</span></li>
            <li class='no-bullet'><span class="key-leaf">tgn_neighbor_n_hop</span>: <span class="value">int (12)</span></li>
            <li class='no-bullet'><span class="key-leaf">fix_buggy_orthrus_TGN</span>: <span class="value">bool (13)</span></li>
            <li class='no-bullet'><span class="key-leaf">fix_tgn_neighbor_loader</span>: <span class="value">bool (14)</span></li>
            <li class='no-bullet'><span class="key-leaf">directed</span>: <span class="value">bool (15)</span></li>
            <li class='no-bullet'><span class="key-leaf">insert_neighbors_before</span>: <span class="value">bool (16)</span></li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">inter_graph_batching</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">inter_graph_batch_size</span>: <span class="value">int (18)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Whether to store the graphs on disk upon building the graphs.             Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.<br>
2. Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector,                                     `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones                                     with length `node_type`, `edges_distribution` counts emitted and received edges.<br><br><b>Available options (multi selection)</b>:<br>`node_type`<br>`node_emb`<br>`only_ones`<br>`edges_distribution`
3. Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet`                                 considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector                                 used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN, `none` uses no features.<br><br><b>Available options (multi selection)</b>:<br>`edge_type`<br>`edge_type_triplet`<br>`msg`<br>`time_encoding`<br>`none`
4. Whether the GNN should be trained on all datasets in `multi_dataset`.<br>
5. A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d)                                                 slightly modify node features. Setting this to true fixes the bug.<br>
6. Flattens the time window-based graphs into a single large                             temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges,                             the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types,                             `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.<br><br><b>Available options (one selection)</b>:<br>`edges`<br>`minutes`<br>`unique_edge_types`<br>`none`
7. Controls the value associated with `global_batching.used_method` (training+inference).<br>
8. Controls the value associated with `global_batching.used_method` (inference only).<br>
9. Breaks each previously computed graph into even smaller graphs.                                 `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500`                                 creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph                                 based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.                                `none` does not alter any graph.<br><br><b>Available options (multi selection)</b>:<br>`edges`<br>`tgn_last_neighbor`<br>`none`
10. Controls the value associated with `global_batching.used_method`.<br>
11. Number of last neighbors to store for each node.<br>
12. If greater than one, will also gather the last neighbors of neighbors.<br>
13. A bug has been in the first version of the framework, where the features of last neighbors not appearing                                             in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.<br>
14. We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This                                                 is an unofficial fix.<br>
15. The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.<br>
16. Whether to insert the edges of the current graph before loading last neighbors.<br>
17. Batches multiple graphs into a single large one for parallel training.                                 Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.<br><br><b>Available options (one selection)</b>:<br>`graph_batching`<br>`none`
18. Controls the value associated with `inter_graph_batching.used_method`.<br>
