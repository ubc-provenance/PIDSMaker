<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">seed</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">deterministic</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">num_epochs</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">patience</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float</span></li>
    <li class='no-bullet'><span class="key-leaf">weight_decay</span>: <span class="value">float</span></li>
    <li class='no-bullet'><span class="key-leaf">node_hid_dim</span>: <span class="value">int (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">node_out_dim</span>: <span class="value">int (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">grad_accumulation</span>: <span class="value">int (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">stable_optim</span>: <span class="value">bool (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">inference_device</span>: <span class="value">str (6)</span></li>
    <li class='no-bullet'><span class="key-leaf">fuse_duplicate_edges_training</span>: <span class="value">bool (7)</span></li>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (8)</span></li>
    <li class='bullet'><span class="key">threatrace</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">ratio_threshold</span>: <span class="value">float (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_submodels</span>: <span class="value">int (10)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_retrain_epochs</span>: <span class="value">int (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">retrain_loss_threshold</span>: <span class="value">float (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_rounds_without_progress</span>: <span class="value">int (13)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">ocrapt_early_stop</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">enabled</span>: <span class="value">bool (14)</span></li>
        <li class='no-bullet'><span class="key-leaf">patience</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">min_delta</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">max_delta</span>: <span class="value">float</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">encoder</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (15)</span></li>
        <li class='no-bullet'><span class="key-leaf">x_is_tuple</span>: <span class="value">bool (16)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">decoder</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_few_shot</span>: <span class="value">bool (18)</span></li>
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

</div>

1. Whether to force PyTorch to use deterministic algorithms.<br>
2. Number of neurons in the middle layers of the encoder.<br>
3. Number of neurons in the last layer of the encoder.<br>
4. Number of epochs to gather gradients before backprop.<br>
5. Use AdamW + warmup cosine schedule + gradient clipping for stable training.<br>
6. Device used during testing.<br><br><b>Available options (one selection)</b>:<br>`cpu`<br>`cuda`
7. During training only, keeps one edge per (source, destination) pair in each batch, so that repeated events between the same entities count once in the loss.<br>
8. Which training pipeline use.             `threatrace` trains ThreaTrace's successive submodels (see `training.threatrace`).<br><br><b>Available options (one selection)</b>:<br>`default`<br>`pretrained`<br>`threatrace`
9. ThreaTrace's `thre`: a node is explained by a submodel when its type is predicted                     with a top-1 / top-2 probability ratio of at least this value. Anomalous nodes are                     those no submodel explains (used by `threshold_method: threatrace`).<br>
10. Maximum number of submodels trained by `used_method: threatrace`.<br>
11. Maximum number of epochs to train each submodel after the first one.<br>
12. Training of a submodel after the first one stops once its mean loss on the nodes                     not yet explained is below this value.<br>
13. Training stops after this number of consecutive rounds where the submodel                     explains no new training node.<br>
14. Off by default, no-op unless set.<br>
15. First part of the neural network. Usually GNN encoders to capture complex patterns.<br><br><b>Available options (multi selection)</b>:<br>`tgn`<br>`graph_attention`<br>`sage`<br>`gcn`<br>`node_id_embedding`<br>`snapshot_gru`<br>`textrcnn`<br>`lares`<br>`egraphsage`<br>`argus`<br>`shadewatcher`<br>`gat`<br>`gin`<br>`rgcn`<br>`rgcn_per_type`<br>`sum_aggregation`<br>`rcaid_gat`<br>`magic_gat`<br>`glstm`<br>`custom_mlp`<br>`none`<br>`hetero_graph_transformer`
16. Whether to consider nodes differently when being source or destination.<br>
17. Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)<br><br><b>Available options (multi selection)</b>:<br>`predict_edge_supervised`<br>`predict_edge_type`<br>`predict_node_type`<br>`predict_process_tag`<br>`predict_masked_struct`<br>`detect_edge_few_shot`<br>`predict_edge_contrastive`<br>`deep_graph_infomax`<br>`predict_snapshot_embedding`<br>`reconstruct_node_features`<br>`reconstruct_node_embeddings`<br>`autoencode_edge_embeddings`<br>`reconstruct_edge_embeddings`<br>`reconstruct_masked_features`<br>`one_class`
18. Old feature: need some work to update it.<br>
