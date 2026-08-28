<div class="annotate">

<ul>
    <li class='bullet'><span class="key">predict_edge_type</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_triplet_types</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_node_type</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_masked_struct</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">detect_edge_few_shot</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (5)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_edge_contrastive</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (6)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_node_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_node_embeddings</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (10)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_edge_embeddings</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (12)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_masked_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (13)</span></li>
        <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (14)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">one_class</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (15)</span></li>
        <li class='no-bullet'><span class="key-leaf">beta</span>: <span class="value">float (16)</span></li>
        <li class='no-bullet'><span class="key-leaf">eps</span>: <span class="value">float (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">warmup</span>: <span class="value">int (18)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
2. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
3. <br><b>Available options (one selection)</b>:<br>`cross_entropy`<br>`BCE`
4. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
5. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
6. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
7. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
8. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
9. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
10. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
11. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
12. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
13. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
14. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
15. Decoder applied to embeddings before the hypersphere; use 'none' (identity).<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
16. Soft-boundary fraction; radius is the (1-beta) distance quantile.<br>
17. Center slack keeping |c| away from zero.<br>
18. Kept for parity; a no-op (c/r update every train step).<br>
