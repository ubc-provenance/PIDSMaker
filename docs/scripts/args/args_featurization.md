<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">seed</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">training_split</span>: <span class="value">str (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">pretrain_datasets</span>: <span class="value">str (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (6)</span></li>
    <li class='bullet'><span class="key">tokenization</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (7)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.<br>
2. Epochs to train the embedding method. Arg not used by some methods.<br>
3. The partition of data used to train the featurization method.<br><br><b>Available options (one selection)</b>:<br>`train`<br>`all`
4. Whether the featurization method should be trained on all datasets in `multi_dataset`.<br>
5. Comma-separated list of dataset names for multi-dataset pretraining (used by word2vec, doc2vec, etc.).<br>
6. Algorithms used to create node and edge features.<br><br><b>Available options (one selection)</b>:<br>`word2vec`<br>`doc2vec`<br>`graph2vec`<br>`fasttext`<br>`alacarte`<br>`temporal_rw`<br>`flash`<br>`hierarchical_hashing`<br>`magic`<br>`only_type`<br>`only_ones`<br>`ocrapt_features`<br>`pretrained`
7. How node labels are split into tokens before featurization. `nltk` splits words with NLTK,                     `hierarchical` keeps each level of paths and IPs (Kairos), `whole_label` uses the whole label as a single token (DeepWalk),                     `bpe` is the provenance-specific BPE of SPIDER, `huggingface` uses the tokenizer of a HuggingFace language model,                     `none` is for featurization methods that do not tokenize labels.<br><br><b>Available options (one selection)</b>:<br>`none`<br>`nltk`<br>`hierarchical`<br>`whole_label`<br>`bpe`<br>`huggingface`
