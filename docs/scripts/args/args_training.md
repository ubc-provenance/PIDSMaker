<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">use_seed</span>: <span class="value">bool</span></li>
    <li class='no-bullet'><span class="key-leaf">deterministic</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">num_epochs</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">patience</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float</span></li>
    <li class='no-bullet'><span class="key-leaf">weight_decay</span>: <span class="value">float</span></li>
    <li class='no-bullet'><span class="key-leaf">node_hid_dim</span>: <span class="value">int (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">node_out_dim</span>: <span class="value">int (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">grad_accumulation</span>: <span class="value">int (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">inference_device</span>: <span class="value">str (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (6)</span></li>
    <li class='bullet'><span class="key">encoder</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">x_is_tuple</span>: <span class="value">bool (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">decoder</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_few_shot</span>: <span class="value">bool (10)</span></li>
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
5. Device used during testing.<br><br><b>Available options (one selection)</b>:<br>`cpu`<br>`cuda`
6. Which training pipeline use.<br><br><b>Available options (one selection)</b>:<br>`default`
7. First part of the neural network. Usually GNN encoders to capture complex patterns.<br><br><b>Available options (multi selection)</b>:<br>`tgn`<br>`graph_attention`<br>`sage`<br>`gat`<br>`gin`<br>`sum_aggregation`<br>`rcaid_gat`<br>`magic_gat`<br>`glstm`<br>`custom_mlp`<br>`none`
8. Whether to consider nodes differently when being source or destination.<br>
9. Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)<br><br><b>Available options (multi selection)</b>:<br>`predict_edge_type`<br>`predict_node_type`<br>`predict_masked_struct`<br>`detect_edge_few_shot`<br>`predict_edge_contrastive`<br>`reconstruct_node_features`<br>`reconstruct_node_embeddings`<br>`reconstruct_edge_embeddings`<br>`reconstruct_masked_features`
10. Old feature: need some work to update it.<br>
