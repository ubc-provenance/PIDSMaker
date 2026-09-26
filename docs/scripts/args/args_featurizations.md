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
    <li class='bullet'><span class="key">ocrapt_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">use_lifespan</span>: <span class="value">bool (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_cumulative_active_time</span>: <span class="value">bool (2)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">pretrained</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">model_size</span>: <span class="value">str (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">model_type</span>: <span class="value">str (4)</span></li>
        <li class='bullet'><span class="key">deepwalk</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">window</span>: <span class="value">int (5)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (6)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int (7)</span></li>
            <li class='no-bullet'><span class="key-leaf">workers</span>: <span class="value">int (8)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">node2vec</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">p</span>: <span class="value">float (9)</span></li>
            <li class='no-bullet'><span class="key-leaf">q</span>: <span class="value">float (10)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gnn_distill</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (11)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (12)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (13)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (14)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (15)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (16)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (17)</span></li>
            <li class='no-bullet'><span class="key-leaf">loss_weight</span>: <span class="value">float (18)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_loss_weight</span>: <span class="value">float (19)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_lr</span>: <span class="value">float (20)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (21)</span></li>
            <li class='no-bullet'><span class="key-leaf">edge_projection</span>: <span class="value">bool (22)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (23)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">behavior_cluster</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (24)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_weight</span>: <span class="value">float (25)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_weight</span>: <span class="value">float (26)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (27)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (28)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (29)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_contrastive_head</span>: <span class="value">bool (30)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (31)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (32)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (33)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_mode</span>: <span class="value">str (34)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_target</span>: <span class="value">str (35)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (36)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">spider</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (37)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (38)</span></li>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (39)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (40)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (41)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (42)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (43)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (44)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (45)</span></li>
            <li class='no-bullet'><span class="key-leaf">supcon_weight</span>: <span class="value">float (46)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_weight</span>: <span class="value">float (47)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (48)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (49)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (50)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (51)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (52)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (53)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_loss</span>: <span class="value">str (54)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_loss</span>: <span class="value">str (55)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_data</span>: <span class="value">str (56)</span></li>
            <li class='no-bullet'><span class="key-leaf">student_only_mode</span>: <span class="value">str (57)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gpt2_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (58)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">llama3_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (59)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">opt_pretrained</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (60)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">graphmae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (61)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (62)</span></li>
            <li class='no-bullet'><span class="key-leaf">decoder_num_layers</span>: <span class="value">int (63)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float (64)</span></li>
            <li class='no-bullet'><span class="key-leaf">replace_rate</span>: <span class="value">float (65)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (66)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (67)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (68)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (69)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (70)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (71)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (72)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (73)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (74)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (75)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">dgi</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (76)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (77)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (78)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (79)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (80)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (81)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">training</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">pretrain_tokens</span>: <span class="value">int (82)</span></li>
            <li class='no-bullet'><span class="key-leaf">warmup_tokens</span>: <span class="value">int (83)</span></li>
            <li class='no-bullet'><span class="key-leaf">batch_size</span>: <span class="value">int (84)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (85)</span></li>
            <li class='no-bullet'><span class="key-leaf">scheduler</span>: <span class="value">str (86)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">mlm</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">mask_rate_fixed</span>: <span class="value">float (87)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_rate_min</span>: <span class="value">float (88)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_edge_type</span>: <span class="value">bool (89)</span></li>
            <li class='bullet'><span class="key">modernbert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">global_attn_every_n_layers</span>: <span class="value">int (90)</span></li>
                <li class='no-bullet'><span class="key-leaf">local_attention_window</span>: <span class="value">int (91)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">ropebert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (92)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">llama</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (93)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">logbert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">hvm_weight</span>: <span class="value">float (94)</span></li>
            </ul>
            </li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">pretrain_datasets</span>: <span class="value">str (95)</span></li>
        <li class='bullet'><span class="key">tokenizer</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">bpe_vocab_size</span>: <span class="value">int (96)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (97)</span></li>
            <li class='no-bullet'><span class="key-leaf">mode</span>: <span class="value">str (98)</span></li>
            <li class='no-bullet'><span class="key-leaf">normalize_netflow_ips</span>: <span class="value">bool (99)</span></li>
            <li class='no-bullet'><span class="key-leaf">canonicalize_neighbors</span>: <span class="value">bool (100)</span></li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">weights_path</span>: <span class="value">str (101)</span></li>
        <li class='no-bullet'><span class="key-leaf">graph_context_mode</span>: <span class="value">str (102)</span></li>
        <li class='bullet'><span class="key">walks</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">walk_length</span>: <span class="value">int (103)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_walks</span>: <span class="value">int (104)</span></li>
            <li class='no-bullet'><span class="key-leaf">time_weight</span>: <span class="value">str (105)</span></li>
            <li class='no-bullet'><span class="key-leaf">half_life</span>: <span class="value">float (106)</span></li>
            <li class='no-bullet'><span class="key-leaf">random_walk_start</span>: <span class="value">bool (107)</span></li>
            <li class='no-bullet'><span class="key-leaf">diversity_weight</span>: <span class="value">float (108)</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. Off by default, hurts generalization (paper Appendix E).<br>
2. Off by default, same reason as use_lifespan.<br>
3. SPIDER encoder size preset (selects hidden / layers / heads / FFN). Standard presets: 'tiny' (H=128), 'mini' (H=256), 'med' (H=512), 'baseline' (H=768). HF pretrained model_types use their own size keys (e.g. 'small' / 'medium' / 'large' / 'xl' for gpt2_pretrained, '1b' / '3b' for llama3_pretrained).<br>
4. SPIDER pretraining objective + architecture. See config/pretrained/README.md for full descriptions. Families:
  MLM walk-based     : bert, roberta, modernbert, ropebert, llama, logbert
  Walk embedding     : deepwalk, node2vec
  GNN self-supervised: graphmae, gae, dgi
  GNN token-budget   : gnn_distill, behavior_cluster, spider
  HF pretrained      : gpt2_pretrained, llama3_pretrained, opt_pretrained<br><br><b>Available options (one selection)</b>:<br>`bert`<br>`roberta`<br>`modernbert`<br>`ropebert`<br>`llama`<br>`logbert`<br>`deepwalk`<br>`node2vec`<br>`graphmae`<br>`gae`<br>`dgi`<br>`gnn_distill`<br>`behavior_cluster`<br>`spider`<br>`gpt2_pretrained`<br>`llama3_pretrained`<br>`opt_pretrained`
5. Skip-gram context window size. Default: 5.<br>
6. Number of Word2Vec training epochs. Default: 10.<br>
7. Minimum label frequency to include in vocabulary. Default: 1.<br>
8. Number of parallel workers for Word2Vec training. Default: 4.<br>
9. Return parameter. Higher p = less likely to revisit previous node. Default: 1.0.<br>
10. In-out parameter. q > 1 = BFS-like (local); q < 1 = DFS-like (explore). Default: 1.0.<br>
11. GNN edge embedding dimension. Default: 128.<br>
12. GNN hidden dimension. Default: 128.<br>
13. Number of GNN encoder layers. Default: 1.<br>
14. Number of attention heads in GNN layers. Default: 4.<br>
15. Min direct neighbors to sample per entity. Default: 5.<br>
16. Max direct neighbors to sample per entity. Default: 20.<br>
17. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
18. Weight for distillation loss (T5 encoder → GNN alignment). Default: 0.5.<br>
19. Weight for GNN masked reconstruction loss. Default: 1.0.<br>
20. Learning rate for GNN teacher parameters. Default: 1e-3.<br>
21. EMA momentum for teacher update (0.999 = slow-moving). Default: 0.999.<br>
22. Apply per-edge-type linear projection to source nodes before GNN message passing. Default: False.<br>
23. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from GNN neighborhoods. Default: False.<br>
24. Contrastive projection head output dimensionality. Default: 128.<br>
25. Weight for multi-label BCE loss. Default: 1.0.<br>
26. Weight for contrastive (NT-Xent) loss. Default: 0.5.<br>
27. Temperature for InfoNCE cosine similarity scaling. Default: 0.07.<br>
28. Skip entities with fewer behavior labels in their signature. Default: 2.<br>
29. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from signature extraction. Default: True.<br>
30. Use contrastive projection head for inference embeddings (proj_dim) instead of raw encoder (emb_dim). Default: False.<br>
31. Floor: oversample small classes to this minimum per epoch. 0 = no floor. Default: 0.<br>
32. Cap entities per signature class per epoch. Rotates across epochs. 0 = no cap. Default: 0.<br>
33. K in P×K batch sampling: number of entities per class per batch. Guarantees every entity has at least K-1 positives for contrastive learning. Default: 4.<br>
34. Classification head mode: 'multilabel' = BCE over behavior labels, 'class' = cross-entropy over entity classes. Default: multilabel.<br>
35. Contrastive positive definition: 'signature' = entities with identical behavior signatures are positives, 'entity_class' = entities with the same coarse functional class are positives. Default: signature.<br>
36. Remove entity type prefix tokens ([PROC], [FILE], [SOCK]) from tokenized sequences. Forces the model to learn identity from behavior rather than type. Default: False.<br>
37. GNN edge embedding dimension. Default: 256.<br>
38. GNN encoder hidden dimension. Default: 256.<br>
39. Contrastive projection dimension (teacher). Default: 256.<br>
40. Number of attention heads in GNN TransformerConv. Default: 4.<br>
41. Min direct neighbors to sample per entity. Default: 5.<br>
42. Max direct neighbors to sample per entity. Default: 20.<br>
43. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
44. Remove shared libs, /dev, /proc, /sys from GNN neighborhoods. Default: True.<br>
45. EMA momentum for teacher T5 update. Default: 0.999.<br>
46. Weight for supervised contrastive loss on GNN teacher. Default: 1.0.<br>
47. Weight for distillation loss (student → GNN targets). Default: 0.5.<br>
48. SupCon temperature. Default: 0.07.<br>
49. Skip entities with fewer behavior labels (for entity class assignment). Default: 2.<br>
50. Floor: oversample small classes to this minimum per epoch. Default: 100.<br>
51. Cap entities per class per epoch. Default: 2000.<br>
52. K in P×K batching. Default: 4.<br>
53. Remove entity type prefix tokens from inputs. Default: False.<br>
54. Distillation loss: sce or mse. Default: sce.<br><br><b>Available options (one selection)</b>:<br>`sce`<br>`mse`
55. Teacher loss: contrastive (SupCon) or bce (cross-entropy). Default: contrastive.<br><br><b>Available options (one selection)</b>:<br>`contrastive`<br>`bce`
56. Teacher data modalities (comma-separated): signature, gnn_emb, or both. Default: signature,gnn_emb.<br>
57. Student-only ablation: none, student_signature, or student_class. Default: none.<br><br><b>Available options (one selection)</b>:<br>`none`<br>`student_signature`<br>`student_class`
58. Max sequence length for HF tokenizer. Default: 512.<br>
59. Max sequence length for HF tokenizer. Default: 512.<br>
60. Max sequence length for HF tokenizer. Default: 512.<br>
61. Number of GAT encoder layers. Default: 2.<br>
62. Number of attention heads in GAT layers. Default: 4.<br>
63. Number of GAT decoder layers. Default: 1.<br>
64. Fraction of nodes to mask during pretraining. Default: 0.5.<br>
65. Fraction of masked nodes replaced with random tokens (rest get [MASK]). Default: 0.1.<br>
66. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
67. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
68. Number of training epochs. Default: 100.<br>
69. Learning rate. Default: 0.001.<br>
70. Number of GAT encoder layers. Default: 2.<br>
71. Number of attention heads in GAT layers. Default: 4.<br>
72. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
73. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
74. Number of training epochs. Default: 100.<br>
75. Learning rate. Default: 0.001.<br>
76. Number of GAT encoder layers. Default: 2.<br>
77. Number of attention heads in GAT layers. Default: 4.<br>
78. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
79. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
80. Number of training epochs. Default: 100.<br>
81. Learning rate. Default: 0.001.<br>
82. Total number of tokens to process during pretraining (controls training duration).<br>
83. Number of tokens for learning rate warmup at the start of pretraining.<br>
84. Mini-batch size during pretraining. Interpretation depends on model_type: walks per batch (MLM), entities per batch (GNN token-budget), labels per batch (HF pretrained), neighborhoods per batch (GNN-SSL).<br>
85. Peak learning rate for pretraining (after warmup).<br>
86. Learning rate scheduler shape after warmup. Only consumed by MLM and HF pretrained; GNN token-budget hardcodes cosine.<br><br><b>Available options (one selection)</b>:<br>`cosine`<br>`linear`
87. Initial fixed mask rate (fraction of nodes masked). Decays to mask_rate_min.<br>
88. Minimum mask rate after decay.<br>
89. Enable structured masking of edge types during pretraining and edge-type scoring at inference.<br>
90. Use global attention every N layers. Other layers use local sliding window attention. Default: 3.<br>
91. Sliding window size for local attention layers. Default: 128 tokens.<br>
92. Base frequency for rotary position embeddings. Default: 10000.0.<br>
93. Base frequency for rotary position embeddings. Default: 10000.0.<br>
94. Weight for hypersphere volume minimization loss relative to MLM loss. Default: 0.1.<br>
95. Comma-separated list of dataset names to use for multi-dataset pretraining.<br>
96. Target BPE vocabulary size for the tokenizer.<br>
97. Maximum token sequence length after tokenization. Walks exceeding this are truncated.<br>
98. Tokenizer mode: 'domain_bpe' uses domain-specific pre-tokenization followed by BPE; 'bpe_only' skips domain rules and applies BPE directly on raw words.<br><br><b>Available options (one selection)</b>:<br>`domain_bpe`<br>`bpe_only`
99. Replace IP addresses in netflow entities with category tokens ([PRIVATE_IP], [PUBLIC_IP], [LOCALHOST_IP]) instead of keeping individual octets.<br>
100. Enable multi-level neighbor canonicalization. When True, decoder targets keep only structural/categorical tokens ([] special tokens + OS-agnostic category tokens like [CAT_WEBSERVER], [FCAT_LOG_WEB]) while encoder input gets full detail + category tokens prepended. During continue_pretrain, neighbors keep all tokens with category tokens prepended.<br>
101. Path to a folder of pretrained weights (e.g. the SPIDER weights). May contain any subset of: corpus.pt (sampler states + indexid2msg), tokenizer.pt, pretrain_*.pt (model checkpoint), behavior_vocab.txt. Present artifacts are loaded; missing ones are computed from scratch. Works with any model_type.<br>
102. Graph context scope for walk sampling at both pretraining and inference time. 'window' = each time-window snapshot independently; 'day' = merge all snapshots from the same calendar day; 'all' = merge all snapshots in the relevant split (train split at pretrain time; val or test split at inference time — no training data leaks into inference context).<br><br><b>Available options (one selection)</b>:<br>`window`<br>`day`<br>`all`
103. Number of nodes per random walk during pretraining.<br>
104. Number of random walks to sample per node per epoch during pretraining.<br>
105. Temporal weighting for neighbor selection: 'uniform', 'exponential', or 'linear'.<br>
106. Half-life (in seconds) for exponential time weighting of neighbor selection.<br>
107. Randomize the temporal entry point for each walk. When True, the first hop picks a random edge instead of always starting from the earliest/latest timestamp.<br>
108. Edge-type diversity bias for random walks during pretraining. 0 = no bias (default), higher values increasingly favor edges whose type is underrepresented in the current walk. Only affects pretraining; finetuning/inference use natural distribution.<br>
