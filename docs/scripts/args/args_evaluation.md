<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">viz_malicious_nodes</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">ground_truth_version</span>: <span class="value">str (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">best_model_selection</span>: <span class="value">str (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str</span></li>
    <li class='bullet'><span class="key">node_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int (7)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">tw_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">node_tw_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">queue_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (10)</span></li>
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
        <li class='no-bullet'><span class="key-leaf">malicious_edge_selection</span>: <span class="value">str (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (12)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Whether to generate images of malicious nodes' neighborhoods (not stable).<br>
2. <br><b>Available options (one selection)</b>:<br>`orthrus`<br>`reapr`
3. Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination`                                     selects the model that does the best separation between top-score TPs and top-score FPs.<br><br><b>Available options (one selection)</b>:<br>`best_adp`<br>`best_discrimination`
4. Method to calculate the threshold value used to detect anomalies.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
5. Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).<br>
6. Whether to cluster nodes after thresholding as done in Orthrus<br>
7. Number of top-score nodes selected before clustering.<br>
8. Time-window detection. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
9. Node-level detection where a same node in multiple time windows is                     considered as multiple unique nodes. More realistic evaluation for near real-time detection. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
10. Queue-level detection as in Kairos. The code is broken and needs work to be updated.<br><br><b>Available options (one selection)</b>:<br>`kairos_idf_queue`<br>`provnet_lof_queue`
11. The ground truth only contains node-level labels.                 This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination                 node is malicious. `both` labels an edge as malicious if both end nodes are malicious.<br><br><b>Available options (one selection)</b>:<br>`src_node`<br>`dst_node`<br>`both_nodes`
12. <br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`
