<div class="annotate">

<ul>
    <li class='bullet'><span class="key">featurization</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_seed</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">training_split</span>: <span class="value">str (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (5)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">feat_inference</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">to_remove</span>: <span class="value">bool</span></li>
    </ul>
    </li>
</ul>

</div>

1. Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.<br>
2. Epochs to train the embedding method. Arg not used by some methods.<br>
3. The partition of data used to train the featurization method.<br><br><b>Available options (one selection)</b>:<br>`train`<br>`all`
4. Whether the featurization method should be trained on all datasets in `multi_dataset`.<br>
5. Algorithms used to create node and edge features.<br><br><b>Available options (one selection)</b>:<br>`word2vec`<br>`doc2vec`<br>`fasttext`<br>`alacarte`<br>`temporal_rw`<br>`flash`<br>`hierarchical_hashing`<br>`magic`<br>`only_type`<br>`only_ones`
