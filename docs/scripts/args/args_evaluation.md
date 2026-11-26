<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">viz_malicious_nodes</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">ground_truth_version</span>: <span class="value">str (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">evaluate_selected_epoch_only</span>: <span class="value">bool (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">save_figures</span>: <span class="value">bool (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">best_model_selection</span>: <span class="value">str (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str</span></li>
    <li class='bullet'><span class="key">node_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (6)</span></li>
        <li class='bullet'><span class="key">max_val_loss</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">float (7)</span></li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int (10)</span></li>
        <li class='no-bullet'><span class="key-leaf">ocrapt_contamination</span>: <span class="value">float (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">ocrapt_min_contamination</span>: <span class="value">float (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">magic_knn_k</span>: <span class="value">int (13)</span></li>
        <li class='no-bullet'><span class="key-leaf">magic_max_reference_size</span>: <span class="value">int (14)</span></li>
        <li class='no-bullet'><span class="key-leaf">magic_embedding</span>: <span class="value">str (15)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">tw_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (16)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">node_tw_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">queue_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (18)</span></li>
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
    <li class='bullet'><span class="key">host_two_stage_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">int (19)</span></li>
        <li class='no-bullet'><span class="key-leaf">beta</span>: <span class="value">int (20)</span></li>
        <li class='no-bullet'><span class="key-leaf">root_attack_nodes</span>: <span class="value">str (21)</span></li>
        <li class='no-bullet'><span class="key-leaf">test_end</span>: <span class="value">str (22)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">edge_evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">malicious_edge_selection</span>: <span class="value">str (23)</span></li>
        <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (24)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Whether to generate images of malicious nodes' neighborhoods (not stable).<br>
2. <br><b>Available options (one selection)</b>:<br>`orthrus`<br>`reapr`<br>`threatrace`
3. Only evaluates the epoch selected by a label-free `best_model_selection` (`last_epoch`,                 `best_val_score`), instead of evaluating every epoch on the test set (faster on large                 test sets, without per-epoch test metrics).<br>
4. Whether to save the figures of the evaluation (scores, ADP and discrimination                 curves) for each evaluated epoch. Metrics are computed either way. Figures with                 one point per edge or node are slow to draw on large test sets.<br>
5. Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination`                                     selects the model that does the best separation between top-score TPs and top-score FPs. These two use test labels.                                     `best_val_score` selects the model with the best score on validation data computed by the objective                                     (link prediction AUC + AP of validation edges for `predict_edge_contrastive`), as Euler's early stopping.                                     `last_epoch` selects the last evaluated epoch (no labels used). `best_mcc` selects the best MCC on the test set (as LARES' code).<br><br><b>Available options (one selection)</b>:<br>`best_adp`<br>`best_discrimination`<br>`best_ap@10`<br>`best_val_score`<br>`best_mcc`<br>`last_epoch`
6. Method to calculate the threshold value used to detect anomalies.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`percentile`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`<br>`fixed_zero`<br>`ocrapt`
7. Weights the margin applied to the threshold value.<br>
8. Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).<br>
9. Whether to cluster nodes after thresholding as done in Orthrus<br>
10. Number of top-score nodes selected before clustering.<br>
11. For threshold_method=ocrapt: max per-type contamination (top fraction flagged), clamped from that type's own val malicious fraction.<br>
12. For threshold_method=ocrapt: min per-type contamination floor.<br>
13. For threshold_method=magic: number of nearest benign training nodes whose                     mean distance scores a test node (default 10, as in the MAGIC paper).<br>
14. For threshold_method=magic: the k-NN reference set is a random sample of at most                     this many benign training node embeddings (0: all of them).<br>
15. For threshold_method=magic: node embedding scored by the k-NN detector, the output                     of the last encoder layer (`last`, MAGIC's code) or the concatenation of all its                     layers (`concat`, MAGIC's paper).<br><br><b>Available options (one selection)</b>:<br>`last`<br>`concat`
16. Time-window detection: each test time window is scored by reducing the losses of its edges, and classified as malicious if its score exceeds the threshold.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`percentile`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`<br>`fixed_zero`<br>`ocrapt`
17. Node-level detection where a same node in multiple time windows is                     considered as multiple unique nodes. More realistic evaluation for near real-time detection.<br><br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`percentile`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`<br>`fixed_zero`<br>`ocrapt`
18. Queue-level detection as in Kairos: anomalous time windows sharing rare (high-IDF) nodes are grouped into queues, and the time windows of queues whose anomaly score exceeds `queue_threshold` are flagged. `provnet_lof_queue` is not supported anymore.<br><br><b>Available options (one selection)</b>:<br>`kairos_idf_queue`<br>`provnet_lof_queue`
19. Number of top host scores clustered by K-means in the first stage (LARES: 3).<br>
20. Number of top outgoing edges predicted malicious per selected host (LARES: 350).<br>
21. Comma-separated index IDs of the root sources of the attacks: only their malicious                     edges are positives, as LARES. `dataset`: the `root_attack_nodes` of the dataset (9660                     on LANL, 201,402,501 on OpTC flows). Empty: all malicious edges.<br>
22. End of the evaluated test period (`%Y-%m-%d %H:%M:%S` in the dataset's timezone).                     Empty: the whole test period.<br>
23. The ground truth only contains node-level labels.                 This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination                 node is malicious. `both` labels an edge as malicious if both end nodes are malicious.<br><br><b>Available options (one selection)</b>:<br>`src_node`<br>`dst_node`<br>`both_nodes`
24. <br><b>Available options (one selection)</b>:<br>`max_val_loss`<br>`mean_val_loss`<br>`percentile`<br>`threatrace`<br>`magic`<br>`flash`<br>`nodlink`<br>`fixed_zero`<br>`ocrapt`
