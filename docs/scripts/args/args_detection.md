<div class="annotate">

<ul>
    <li class='bullet'><span class="key">graph_preprocessing</span>
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
    </li>
    <li class='bullet'><span class="key">gnn_training</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">use_seed</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">deterministic</span>: <span class="value">bool (19)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">patience</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">weight_decay</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">node_hid_dim</span>: <span class="value">int (20)</span></li>
        <li class='no-bullet'><span class="key-leaf">node_out_dim</span>: <span class="value">int (21)</span></li>
        <li class='no-bullet'><span class="key-leaf">grad_accumulation</span>: <span class="value">int (22)</span></li>
        <li class='no-bullet'><span class="key-leaf">inference_device</span>: <span class="value">str (23)</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (24)</span></li>
        <li class='bullet'><span class="key">encoder</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
            <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (25)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">decoder</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (26)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_few_shot</span>: <span class="value">bool (27)</span></li>
            <li class='bullet'><span class="key">few_shot</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">include_attacks_in_ssl_training</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">freeze_encoder</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">num_epochs_few_shot</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">patience_few_shot</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">lr_few_shot</span>: <span class="value">float</span></li>
                <li class='no-bullet'><span class="key-leaf">weight_decay_few_shot</span>: <span class="value">float</span></li>
                <li class='bullet'><span class="key">decoder</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str</span></li>
                </ul>
                </li>
            </ul>
            </li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">viz_malicious_nodes</span>: <span class="value">bool (28)</span></li>
        <li class='no-bullet'><span class="key-leaf">ground_truth_version</span>: <span class="value">str (29)</span></li>
        <li class='no-bullet'><span class="key-leaf">best_model_selection</span>: <span class="value">str (30)</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str</span></li>
        <li class='bullet'><span class="key">node_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (31)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool (32)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool (33)</span></li>
            <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int (34)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">tw_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (35)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">node_tw_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (36)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">queue_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (37)</span></li>
            <li class='no-bullet'><span class="key-leaf">queue_threshold</span>: <span class="value">int</span></li>
            <li class='bullet'><span class="key">kairos_idf_queue</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">include_test_set_in_IDF</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">provnet_lof_queue</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">queue_arg</span>: <span class="value">str</span></li>
            </ul>
            </li>
        </ul>
        </li>
        <li class='bullet'><span class="key">edge_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">malicious_edge_selection</span>: <span class="value">str (38)</span></li>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (39)</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. Whether to store the graphs on disk upon building the graphs.                 Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.<br>
2. Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector,                                      `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones                                       with length `node_type`, `edges_distribution` counts emitted and received edges.<br><br><b>Available options (multi selection)</b>:<br>`node_type`<br>`node_emb`<br>`only_ones`<br>`edges_distribution`
3. Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet`                                     considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector                                     used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN, `none` uses no features.<br><br><b>Available options (multi selection)</b>:<br>`edge_type`<br>`edge_type_triplet`<br>`msg`<br>`time_encoding`<br>`none`
4. Whether the GNN should be trained on all datasets in `multi_dataset`.<br>
5. A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d)                                                     slightly modify node features. Setting this to true fixes the bug.<br>
6. Flattens the time window-based graphs into a single large                                 temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges,                                 the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types,                                 `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.<br><br><b>Available options (one selection)</b>:<br>`edges`<br>`minutes`<br>`unique_edge_types`<br>`none`
7. Controls the value associated with `global_batching.used_method` (training+inference).<br>
8. Controls the value associated with `global_batching.used_method` (inference only).<br>
9. Breaks each previously computed graph into even smaller graphs.                                     `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500`                                     creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph                                     based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.                                    `none` does not alter any graph.<br><br><b>Available options (multi selection)</b>:<br>`edges`<br>`tgn_last_neighbor`<br>`none`
10. Controls the value associated with `global_batching.used_method`.<br>
11. Number of last neighbors to store for each node.<br>
12. If greater than one, will also gather the last neighbors of neighbors.<br>
13. A bug has been in the first version of the framework, where the features of last neighbors not appearing                                                 in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.<br>
14. We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This                                                     is an unofficial fix.<br>
15. The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.<br>
16. Whether to insert the edges of the current graph before loading last neighbors.<br>
17. Batches multiple graphs into a single large one for parallel training.                                     Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.<br><br><b>Available options (one selection)</b>:<br>`graph_batching`<br>`none`
18. Controls the value associated with `inter_graph_batching.used_method`.<br>
19. Whether to force PyTorch to use deterministic algorithms.<br>
20. Number of neurons in the middle layers of the encoder.<br>
21. Number of neurons in the last layer of the encoder.<br>
22. Number of epochs to gather gradients before backprop.<br>
23. Device used during testing.<br><br><b>Available options (one selection)</b>:<br>`cpu`<br>`cuda`
24. Which training pipeline use.<br><br><b>Available options (one selection)</b>:<br>`default`
25. First part of the neural network. Usually GNN encoders to capture complex patterns.<br><br><b>Available options (multi selection)</b>:<br>`tgn`<br>`graph_attention`<br>`sage`<br>`gat`<br>`gin`<br>`sum_aggregation`<br>`rcaid_gat`<br>`magic_gat`<br>`glstm`<br>`custom_mlp`<br>`none`
26. Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)<br><br><b>Available options (multi selection)</b>:<br>`predict_edge_type`<br>`predict_node_type`<br>`predict_masked_struct`<br>`detect_edge_few_shot`<br>`predict_edge_contrastive`<br>`reconstruct_node_features`<br>`reconstruct_node_embeddings`<br>`reconstruct_edge_embeddings`<br>`reconstruct_masked_features`
27. Old feature: need some work to update it.<br>
28. Whether to generate images of malicious nodes' neighborhoods (not stable).<br>
29. <br><b>Available options (one selection)</b>:<br>`orthrus`
30. Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination`                                         selects the model that does the best separation between top-score TPs and top-score FPs.<br><br><b>Available options (one selection)</b>:<br>`best_adp`<br>`best_discrimination`
31. Method to calculate the threshold value used to detect anomalies.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
32. Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).<br>
33. Whether to cluster nodes after thresholding as done in Orthrus<br>
34. Number of top-score nodes selected before clustering.<br>
35. Time-window detection. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
36. Node-level detection where a same node in multiple time windows is                         considered as multiple unique nodes. More realistic evaluation for near real-time detection. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
37. Queue-level detection as in Kairos. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`kairos_idf_queue`<br>`provnet_lof_queue`
38. The ground truth only contains node-level labels.                     This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination                     node is malicious. `both` labels an edge as malicious if both end nodes are malicious.<br><br><b>Available options (one selection)</b>:<br>`src_node`<br>`dst_node`<br>`both_nodes`
39. <br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
