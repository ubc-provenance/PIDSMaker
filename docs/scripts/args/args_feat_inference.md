<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">to_remove</span>: <span class="value">bool</span></li>
    <li class='no-bullet'><span class="key-leaf">continue_pretrain</span>: <span class="value">bool (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">continue_pretrain_epochs</span>: <span class="value">int (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">continue_pretrain_lr_factor</span>: <span class="value">float (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">edge_batch_size</span>: <span class="value">int (4)</span></li>
    <li class='bullet'><span class="key">rename_attack</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">enabled</span>: <span class="value">bool (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">entities</span>: <span class="value">str (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">target_node_type</span>: <span class="value">str (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">top_k</span>: <span class="value">int (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">target_label</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">seed</span>: <span class="value">int (10)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Continue pretraining on the target dataset before inference. Default: False.<br>
2. Number of epochs for continue-pretraining on target dataset. Default: 3.<br>
3. LR multiplier relative to original peak LR (e.g., 0.1 = 10x lower). Default: 0.1.<br>
4. Number of edges to process in each batch during feat_inference. Default: 5000.<br>
5. If True, after embeddings are computed, replace the embedding of selected test-set nodes with a benign target embedding. Simulates an inference-time renaming/mimicry attack.<br>
6. Comma-separated BASE NAMES of attack entities (e.g. 'main, XIM, sendmail'). Each is matched two ways against test-set nodes of `target_node_type`: (a) exact label equality, and (b) labels ending with '/<base>' (path-style).<br>
7. Node type the attack targets (only nodes of this type are rewritten).<br><br><b>Available options (one selection)</b>:<br>`subject`<br>`file`<br>`netflow`
8. If >0, automatically build a benign-target pool from the top-K most frequent BARE labels (no '/') of `target_node_type` seen in the training split, and assign one target per attack entity. Path-suffix matches preserve the original prefix (e.g. '/tmp/main' -> '/tmp/<assigned-target>'). If 0, use the manual `target_label` for every entity (e.g. for an 'unseen-by-pretraining' target).<br>
9. (Manual mode only, used when top_k=0) Bare base name to replace every attack entity. Path-suffix matches still preserve the prefix (e.g. '/tmp/main' -> '/tmp/<target_label>').<br>
10. Seed used to shuffle the auto-picked benign pool before assigning one target per entity. Vary it for sensitivity analysis over which benign label each malicious entity is mimicking.<br>
