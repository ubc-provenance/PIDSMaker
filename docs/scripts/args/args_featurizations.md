<div class="annotate">

<ul>
    <li class='bullet'><span class="key">word2vec</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">window_size</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">use_skip_gram</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">num_workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">compute_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">negative</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">decline_rate</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">doc2vec</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">include_neighbors</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">float</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">fasttext</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">window_size</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">negative</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">num_workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">use_pretrained_fb_model</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">alacarte</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">walk_length</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">num_walks</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">context_window_size</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">use_skip_gram</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">num_workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">compute_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">add_paths</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">temporal_rw</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">walk_length</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">num_walks</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">trw_workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">time_weight</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">half_life</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">window_size</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">use_skip_gram</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">wv_workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">compute_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">negative</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">decline_rate</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">flash</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">workers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">hierarchical_hashing</span>
    
    
    </li>
    <li class='bullet'><span class="key">magic</span>
    
    
    </li>
    <li class='bullet'><span class="key">only_type</span>
    
    
    </li>
    <li class='bullet'><span class="key">only_ones</span>
    
    
    </li>
    <li class='bullet'><span class="key">spider</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">model_size</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">model_type</span>: <span class="value">str (2)</span></li>
        <li class='bullet'><span class="key">modernbert</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">global_attn_every_n_layers</span>: <span class="value">int (3)</span></li>
            <li class='no-bullet'><span class="key-leaf">local_attention_window</span>: <span class="value">int (4)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">ropebert</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (5)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">llama</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (6)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">deepwalk</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">window</span>: <span class="value">int (7)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (8)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int (9)</span></li>
            <li class='no-bullet'><span class="key-leaf">workers</span>: <span class="value">int (10)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">node2vec</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">p</span>: <span class="value">float (11)</span></li>
            <li class='no-bullet'><span class="key-leaf">q</span>: <span class="value">float (12)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">t5</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">corpus_mode</span>: <span class="value">str (13)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (14)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (15)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_walks</span>: <span class="value">int (16)</span></li>
            <li class='no-bullet'><span class="key-leaf">shuffle_neighbors</span>: <span class="value">bool (17)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (18)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gnn_distill</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (19)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (20)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (21)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (22)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (23)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (24)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (25)</span></li>
            <li class='no-bullet'><span class="key-leaf">loss_weight</span>: <span class="value">float (26)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_loss_weight</span>: <span class="value">float (27)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_lr</span>: <span class="value">float (28)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (29)</span></li>
            <li class='no-bullet'><span class="key-leaf">edge_projection</span>: <span class="value">bool (30)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (31)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">behavior_cluster</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (32)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_weight</span>: <span class="value">float (33)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_weight</span>: <span class="value">float (34)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (35)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (36)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (37)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_contrastive_head</span>: <span class="value">bool (38)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (39)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (40)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (41)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_mode</span>: <span class="value">str (42)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_target</span>: <span class="value">str (43)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (44)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">spider</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (45)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (46)</span></li>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (47)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (48)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (49)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (50)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (51)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (52)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (53)</span></li>
            <li class='no-bullet'><span class="key-leaf">supcon_weight</span>: <span class="value">float (54)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_weight</span>: <span class="value">float (55)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (56)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (57)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (58)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (59)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (60)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (61)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_loss</span>: <span class="value">str (62)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_loss</span>: <span class="value">str (63)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_data</span>: <span class="value">str (64)</span></li>
            <li class='no-bullet'><span class="key-leaf">student_only_mode</span>: <span class="value">str (65)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gpt2_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (66)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">llama3_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (67)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">opt_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (68)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">logbert</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">hvm_weight</span>: <span class="value">float (69)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">graphmae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (70)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (71)</span></li>
            <li class='no-bullet'><span class="key-leaf">decoder_num_layers</span>: <span class="value">int (72)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float (73)</span></li>
            <li class='no-bullet'><span class="key-leaf">replace_rate</span>: <span class="value">float (74)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (75)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (76)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (77)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (78)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (79)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (80)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (81)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (82)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (83)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (84)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">dgi</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (85)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (86)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (87)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (88)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (89)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (90)</span></li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">pretrain_tokens</span>: <span class="value">int (91)</span></li>
        <li class='no-bullet'><span class="key-leaf">warmup_tokens</span>: <span class="value">int (92)</span></li>
        <li class='no-bullet'><span class="key-leaf">batch_size</span>: <span class="value">int (93)</span></li>
        <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (94)</span></li>
        <li class='no-bullet'><span class="key-leaf">mask_rate_fixed</span>: <span class="value">float (95)</span></li>
        <li class='no-bullet'><span class="key-leaf">mask_rate_min</span>: <span class="value">float (96)</span></li>
        <li class='no-bullet'><span class="key-leaf">scheduler</span>: <span class="value">str (97)</span></li>
        <li class='no-bullet'><span class="key-leaf">mask_edge_type</span>: <span class="value">bool (98)</span></li>
        <li class='no-bullet'><span class="key-leaf">pretrain_datasets</span>: <span class="value">str (99)</span></li>
        <li class='bullet'><span class="key">tokenizer</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">bpe_vocab_size</span>: <span class="value">int (100)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (101)</span></li>
            <li class='no-bullet'><span class="key-leaf">mode</span>: <span class="value">str (102)</span></li>
            <li class='no-bullet'><span class="key-leaf">normalize_netflow_ips</span>: <span class="value">bool (103)</span></li>
            <li class='no-bullet'><span class="key-leaf">canonicalize_neighbors</span>: <span class="value">bool (104)</span></li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">spider_path</span>: <span class="value">str (105)</span></li>
        <li class='no-bullet'><span class="key-leaf">graph_context_mode</span>: <span class="value">str (106)</span></li>
        <li class='bullet'><span class="key">corpus</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">walk_length</span>: <span class="value">int (107)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_walks</span>: <span class="value">int (108)</span></li>
            <li class='no-bullet'><span class="key-leaf">time_weight</span>: <span class="value">str (109)</span></li>
            <li class='no-bullet'><span class="key-leaf">half_life</span>: <span class="value">float (110)</span></li>
            <li class='no-bullet'><span class="key-leaf">random_walk_start</span>: <span class="value">bool (111)</span></li>
            <li class='no-bullet'><span class="key-leaf">diversity_weight</span>: <span class="value">float (112)</span></li>
            <li class='no-bullet'><span class="key-leaf">lsh_dedup</span>: <span class="value">bool (113)</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. BERT model size: 'tiny', 'small', 'medium', or 'base'.<br>
2. Model architecture type: 'bert' (traditional BERT), 'roberta' (RoBERTa with improved LM head), 'modernbert' (RoPE, RMSNorm, GeGLU, alternating attention), 'ropebert' (RoPE, RMSNorm, GeGLU, full global attention), 'llama' (decoder-only causal LM), 't5' (encoder-decoder for entity pretraining), 'deepwalk' (Word2Vec Skip-gram on random walks over node labels), 'node2vec' (Word2Vec with biased 2nd-order walks via p/q), 'graphmae' (masked graph autoencoder on temporal neighborhoods), 'gae' (graph autoencoder with adjacency reconstruction on temporal neighborhoods), 'dgi' (Deep Graph Infomax: mutual information maximization on temporal neighborhoods), 'gnn_distill' (T5 encoder trained via GNN neighborhood distillation), 'behavior_cluster' (T5 encoder trained via behavioral signature prediction + contrastive learning), 'spider' (class-supervised GNN teacher with T5 student distillation), 'logbert' (BERT with MLM + hypersphere volume minimization loss from LogBERT), 'gpt2_pretrained' (HuggingFace GPT-2, fine-tuned on walks with causal LM), 'llama3_pretrained' (HuggingFace Llama 3.2, fine-tuned on walks with causal LM), or 'opt_pretrained' (HuggingFace OPT, fine-tuned on walks with causal LM).<br><br><b>Available options (one selection)</b>:<br>`bert`<br>`roberta`<br>`modernbert`<br>`ropebert`<br>`llama`<br>`t5`<br>`deepwalk`<br>`node2vec`<br>`graphmae`<br>`gae`<br>`dgi`<br>`gnn_distill`<br>`behavior_cluster`<br>`spider`<br>`logbert`<br>`gpt2_pretrained`<br>`llama3_pretrained`<br>`opt_pretrained`
3. Use global attention every N layers. Other layers use local sliding window attention. Default: 3.<br>
4. Sliding window size for local attention layers. Default: 128 tokens.<br>
5. Base frequency for rotary position embeddings. Default: 10000.0.<br>
6. Base frequency for rotary position embeddings. Default: 10000.0.<br>
7. Skip-gram context window size. Default: 5.<br>
8. Number of Word2Vec training epochs. Default: 10.<br>
9. Minimum label frequency to include in vocabulary. Default: 1.<br>
10. Number of parallel workers for Word2Vec training. Default: 4.<br>
11. Return parameter. Higher p = less likely to revisit previous node. Default: 1.0.<br>
12. In-out parameter. q > 1 = BFS-like (local); q < 1 = DFS-like (explore). Default: 1.0.<br>
13. Corpus composition — 'walks' (temporal random walks only, default), 'neighborhoods' (1-hop temporal neighborhoods only), 'both' (walks + neighborhoods), or 'parallel_edges' (one training example per edge, no cross-neighbor dependencies).<br><br><b>Available options (one selection)</b>:<br>`walks`<br>`neighborhoods`<br>`both`<br>`parallel_edges`
14. Minimum number of neighbors per neighborhood sample. Default: 3.<br>
15. Maximum number of neighbors per neighborhood sample. Default: 15.<br>
16. Number of neighborhood samples per direction per node. Default: 3.<br>
17. Randomize neighbor order in neighborhood samples instead of temporal order. Prevents the model from memorizing spurious temporal orderings. Default: False.<br>
18. Filter out noisy/uninformative edges from the corpus (shared libraries, system pseudo-files, common configs, linker cache, locale data, /dev pseudo-devices, /proc, /sys). Removes hub entities that connect everything and prevent meaningful clustering. Default: False.<br>
19. GNN edge embedding dimension. Default: 128.<br>
20. GNN hidden dimension. Default: 128.<br>
21. Number of GNN encoder layers. Default: 1.<br>
22. Number of attention heads in GNN layers. Default: 4.<br>
23. Min direct neighbors to sample per entity. Default: 5.<br>
24. Max direct neighbors to sample per entity. Default: 20.<br>
25. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
26. Weight for distillation loss (T5 encoder → GNN alignment). Default: 0.5.<br>
27. Weight for GNN masked reconstruction loss. Default: 1.0.<br>
28. Learning rate for GNN teacher parameters. Default: 1e-3.<br>
29. EMA momentum for teacher update (0.999 = slow-moving). Default: 0.999.<br>
30. Apply per-edge-type linear projection to source nodes before GNN message passing. Default: False.<br>
31. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from GNN neighborhoods. Default: False.<br>
32. Contrastive projection head output dimensionality. Default: 128.<br>
33. Weight for multi-label BCE loss. Default: 1.0.<br>
34. Weight for contrastive (NT-Xent) loss. Default: 0.5.<br>
35. Temperature for InfoNCE cosine similarity scaling. Default: 0.07.<br>
36. Skip entities with fewer behavior labels in their signature. Default: 2.<br>
37. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from signature extraction. Default: True.<br>
38. Use contrastive projection head for inference embeddings (proj_dim) instead of raw encoder (emb_dim). Default: False.<br>
39. Floor: oversample small classes to this minimum per epoch. 0 = no floor. Default: 0.<br>
40. Cap entities per signature class per epoch. Rotates across epochs. 0 = no cap. Default: 0.<br>
41. K in P×K batch sampling: number of entities per class per batch. Guarantees every entity has at least K-1 positives for contrastive learning. Default: 4.<br>
42. Classification head mode: 'multilabel' = BCE over behavior labels, 'class' = cross-entropy over entity classes. Default: multilabel.<br>
43. Contrastive positive definition: 'signature' = entities with identical behavior signatures are positives, 'entity_class' = entities with the same coarse functional class are positives. Default: signature.<br>
44. Remove entity type prefix tokens ([PROC], [FILE], [SOCK]) from tokenized sequences. Forces the model to learn identity from behavior rather than type. Default: False.<br>
45. GNN edge embedding dimension. Default: 256.<br>
46. GNN encoder hidden dimension. Default: 256.<br>
47. Contrastive projection dimension (teacher). Default: 256.<br>
48. Number of attention heads in GNN TransformerConv. Default: 4.<br>
49. Min direct neighbors to sample per entity. Default: 5.<br>
50. Max direct neighbors to sample per entity. Default: 20.<br>
51. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
52. Remove shared libs, /dev, /proc, /sys from GNN neighborhoods. Default: True.<br>
53. EMA momentum for teacher T5 update. Default: 0.999.<br>
54. Weight for supervised contrastive loss on GNN teacher. Default: 1.0.<br>
55. Weight for distillation loss (student → GNN targets). Default: 0.5.<br>
56. SupCon temperature. Default: 0.07.<br>
57. Skip entities with fewer behavior labels (for entity class assignment). Default: 2.<br>
58. Floor: oversample small classes to this minimum per epoch. Default: 100.<br>
59. Cap entities per class per epoch. Default: 2000.<br>
60. K in P×K batching. Default: 4.<br>
61. Remove entity type prefix tokens from inputs. Default: False.<br>
62. Distillation loss: sce or mse. Default: sce.<br><br><b>Available options (one selection)</b>:<br>`sce`<br>`mse`
63. Teacher loss: contrastive (SupCon) or bce (cross-entropy). Default: contrastive.<br><br><b>Available options (one selection)</b>:<br>`contrastive`<br>`bce`
64. Teacher data modalities (comma-separated): signature, gnn_emb, or both. Default: signature,gnn_emb.<br>
65. Student-only ablation: none, student_signature, or student_class. Default: none.<br><br><b>Available options (one selection)</b>:<br>`none`<br>`student_signature`<br>`student_class`
66. Max sequence length for HF tokenizer. Default: 512.<br>
67. Max sequence length for HF tokenizer. Default: 512.<br>
68. Max sequence length for HF tokenizer. Default: 512.<br>
69. Weight for hypersphere volume minimization loss relative to MLM loss. Default: 0.1.<br>
70. Number of GAT encoder layers. Default: 2.<br>
71. Number of attention heads in GAT layers. Default: 4.<br>
72. Number of GAT decoder layers. Default: 1.<br>
73. Fraction of nodes to mask during pretraining. Default: 0.5.<br>
74. Fraction of masked nodes replaced with random tokens (rest get [MASK]). Default: 0.1.<br>
75. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
76. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
77. Number of training epochs. Default: 100.<br>
78. Learning rate. Default: 0.001.<br>
79. Number of GAT encoder layers. Default: 2.<br>
80. Number of attention heads in GAT layers. Default: 4.<br>
81. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
82. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
83. Number of training epochs. Default: 100.<br>
84. Learning rate. Default: 0.001.<br>
85. Number of GAT encoder layers. Default: 2.<br>
86. Number of attention heads in GAT layers. Default: 4.<br>
87. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
88. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
89. Number of training epochs. Default: 100.<br>
90. Learning rate. Default: 0.001.<br>
91. Total number of tokens to process during pretraining (controls training duration).<br>
92. Number of tokens for learning rate warmup at the start of pretraining.<br>
93. Number of walks per batch during pretraining.<br>
94. Peak learning rate for pretraining.<br>
95. Initial fixed mask rate (fraction of nodes masked). Decays to mask_rate_min.<br>
96. Minimum mask rate after decay.<br>
97. Learning rate scheduler type for pretraining.<br><br><b>Available options (one selection)</b>:<br>`cosine`<br>`linear`
98. Enable structured masking of edge types during pretraining and edge-type scoring at inference.<br>
99. Comma-separated list of dataset names to use for multi-dataset pretraining.<br>
100. Target BPE vocabulary size for the tokenizer.<br>
101. Maximum token sequence length after tokenization. Walks exceeding this are truncated.<br>
102. Tokenizer mode: 'domain_bpe' uses domain-specific pre-tokenization followed by BPE; 'bpe_only' skips domain rules and applies BPE directly on raw words.<br><br><b>Available options (one selection)</b>:<br>`domain_bpe`<br>`bpe_only`
103. Replace IP addresses in netflow entities with category tokens ([PRIVATE_IP], [PUBLIC_IP], [LOCALHOST_IP]) instead of keeping individual octets.<br>
104. Enable multi-level neighbor canonicalization. When True, decoder targets keep only structural/categorical tokens ([] special tokens + OS-agnostic category tokens like [CAT_WEBSERVER], [FCAT_LOG_WEB]) while encoder input gets full detail + category tokens prepended. During continue_pretrain, neighbors keep all tokens with category tokens prepended.<br>
105. Path to a pretrained SPIDER model folder. May contain any subset of: corpus.pt (sampler states + indexid2msg), tokenizer.pt, pretrain_*.pt (model checkpoint), behavior_vocab.txt. Present artifacts are loaded; missing ones are computed from scratch. Works with any model_type.<br>
106. Graph context scope for walk sampling at both pretraining and inference time. 'window' = each time-window snapshot independently; 'day' = merge all snapshots from the same calendar day; 'all' = merge all snapshots in the relevant split (train split at pretrain time; val or test split at inference time — no training data leaks into inference context).<br><br><b>Available options (one selection)</b>:<br>`window`<br>`day`<br>`all`
107. Number of nodes per random walk during pretraining.<br>
108. Number of random walks to sample per node per epoch during pretraining.<br>
109. Temporal weighting for neighbor selection: 'uniform', 'exponential', or 'linear'.<br>
110. Half-life (in seconds) for exponential time weighting of neighbor selection.<br>
111. Randomize the temporal entry point for each walk. When True, the first hop picks a random edge instead of always starting from the earliest/latest timestamp.<br>
112. Edge-type diversity bias for random walks during pretraining. 0 = no bias (default), higher values increasingly favor edges whose type is underrepresented in the current walk. Only affects pretraining; finetuning/inference use natural distribution.<br>
113. Enable structural-group near-duplicate removal on the tokenized walk corpus before pretraining. Groups walks by their structural token sequence (bracket-enclosed and EVENT_* tokens), then removes walks whose content tokens differ by at most 2 from an already-kept walk. Report written to walk_dedup_report.txt in the model directory.<br>
