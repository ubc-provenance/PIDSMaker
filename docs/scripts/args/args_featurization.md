<div class="annotate">

<ul>
    <li class='bullet'><span class="key">feat_training</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">seed</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">training_split</span>: <span class="value">str (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">pretrain_datasets</span>: <span class="value">str (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (6)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">feat_inference</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">to_remove</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">continue_pretrain</span>: <span class="value">bool (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">continue_pretrain_epochs</span>: <span class="value">int (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">continue_pretrain_lr_factor</span>: <span class="value">float (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">edge_batch_size</span>: <span class="value">int (10)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.<br>
2. Epochs to train the embedding method. Arg not used by some methods.<br>
3. The partition of data used to train the featurization method.<br><br><b>Available options (one selection)</b>:<br>`train`<br>`all`
4. Whether the featurization method should be trained on all datasets in `multi_dataset`.<br>
5. Comma-separated list of dataset names for multi-dataset pretraining (used by word2vec, doc2vec, etc.).<br>
6. Algorithms used to create node and edge features.<br><br><b>Available options (one selection)</b>:<br>`word2vec`<br>`doc2vec`<br>`fasttext`<br>`alacarte`<br>`temporal_rw`<br>`flash`<br>`hierarchical_hashing`<br>`magic`<br>`only_type`<br>`only_ones`<br>`spider`
7. Continue pretraining on the target dataset before inference. Default: False.<br>
8. Number of epochs for continue-pretraining on target dataset. Default: 3.<br>
9. LR multiplier relative to original peak LR (e.g., 0.1 = 10x lower). Default: 0.1.<br>
10. Number of edges to process in each batch during feat_inference. Default: 5000.<br>
