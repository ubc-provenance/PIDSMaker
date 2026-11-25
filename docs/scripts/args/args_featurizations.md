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
    <li class='bullet'><span class="key">graph2vec</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">wl_iterations</span>: <span class="value">int (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">negative</span>: <span class="value">int (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">alpha</span>: <span class="value">float (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">infer_epochs</span>: <span class="value">int (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_workers</span>: <span class="value">int (6)</span></li>
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
        <li class='no-bullet'><span class="key-leaf">use_lifespan</span>: <span class="value">bool (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_cumulative_active_time</span>: <span class="value">bool (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">pretrained</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">model_size</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">model_type</span>: <span class="value">str (10)</span></li>
        <li class='bullet'><span class="key">deepwalk</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">window</span>: <span class="value">int (11)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (12)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_count</span>: <span class="value">int (13)</span></li>
            <li class='no-bullet'><span class="key-leaf">workers</span>: <span class="value">int (14)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">node2vec</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">p</span>: <span class="value">float (15)</span></li>
            <li class='no-bullet'><span class="key-leaf">q</span>: <span class="value">float (16)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gnn_distill</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (17)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (18)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (19)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (20)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (21)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (22)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (23)</span></li>
            <li class='no-bullet'><span class="key-leaf">loss_weight</span>: <span class="value">float (24)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_loss_weight</span>: <span class="value">float (25)</span></li>
            <li class='no-bullet'><span class="key-leaf">gnn_lr</span>: <span class="value">float (26)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (27)</span></li>
            <li class='no-bullet'><span class="key-leaf">edge_projection</span>: <span class="value">bool (28)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (29)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">behavior_cluster</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (30)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_weight</span>: <span class="value">float (31)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_weight</span>: <span class="value">float (32)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (33)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (34)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (35)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_contrastive_head</span>: <span class="value">bool (36)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (37)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (38)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (39)</span></li>
            <li class='no-bullet'><span class="key-leaf">bce_mode</span>: <span class="value">str (40)</span></li>
            <li class='no-bullet'><span class="key-leaf">contrastive_target</span>: <span class="value">str (41)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (42)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">spider</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">emb_dim</span>: <span class="value">int (43)</span></li>
            <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (44)</span></li>
            <li class='no-bullet'><span class="key-leaf">proj_dim</span>: <span class="value">int (45)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (46)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_min</span>: <span class="value">int (47)</span></li>
            <li class='no-bullet'><span class="key-leaf">n_neighbors_max</span>: <span class="value">int (48)</span></li>
            <li class='no-bullet'><span class="key-leaf">diverse_neighbors</span>: <span class="value">bool (49)</span></li>
            <li class='no-bullet'><span class="key-leaf">filter_noisy_edges</span>: <span class="value">bool (50)</span></li>
            <li class='no-bullet'><span class="key-leaf">ema_momentum</span>: <span class="value">float (51)</span></li>
            <li class='no-bullet'><span class="key-leaf">supcon_weight</span>: <span class="value">float (52)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_weight</span>: <span class="value">float (53)</span></li>
            <li class='no-bullet'><span class="key-leaf">temperature</span>: <span class="value">float (54)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_signature_size</span>: <span class="value">int (55)</span></li>
            <li class='no-bullet'><span class="key-leaf">min_entities_per_class</span>: <span class="value">int (56)</span></li>
            <li class='no-bullet'><span class="key-leaf">max_entities_per_class</span>: <span class="value">int (57)</span></li>
            <li class='no-bullet'><span class="key-leaf">samples_per_class</span>: <span class="value">int (58)</span></li>
            <li class='no-bullet'><span class="key-leaf">strip_entity_type</span>: <span class="value">bool (59)</span></li>
            <li class='no-bullet'><span class="key-leaf">distill_loss</span>: <span class="value">str (60)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_loss</span>: <span class="value">str (61)</span></li>
            <li class='no-bullet'><span class="key-leaf">teacher_data</span>: <span class="value">str (62)</span></li>
            <li class='no-bullet'><span class="key-leaf">student_only_mode</span>: <span class="value">str (63)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">graphmae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (64)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (65)</span></li>
            <li class='no-bullet'><span class="key-leaf">decoder_num_layers</span>: <span class="value">int (66)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float (67)</span></li>
            <li class='no-bullet'><span class="key-leaf">replace_rate</span>: <span class="value">float (68)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (69)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (70)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (71)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (72)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">gae</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (73)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (74)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (75)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (76)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (77)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (78)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">dgi</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (79)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (80)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_min</span>: <span class="value">int (81)</span></li>
            <li class='no-bullet'><span class="key-leaf">neighborhood_max</span>: <span class="value">int (82)</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int (83)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (84)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">training</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">pretrain_tokens</span>: <span class="value">int (85)</span></li>
            <li class='no-bullet'><span class="key-leaf">warmup_tokens</span>: <span class="value">int (86)</span></li>
            <li class='no-bullet'><span class="key-leaf">batch_size</span>: <span class="value">int (87)</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float (88)</span></li>
            <li class='no-bullet'><span class="key-leaf">scheduler</span>: <span class="value">str (89)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">mlm</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">mask_rate_fixed</span>: <span class="value">float (90)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_rate_min</span>: <span class="value">float (91)</span></li>
            <li class='no-bullet'><span class="key-leaf">mask_edge_type</span>: <span class="value">bool (92)</span></li>
            <li class='bullet'><span class="key">modernbert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">global_attn_every_n_layers</span>: <span class="value">int (93)</span></li>
                <li class='no-bullet'><span class="key-leaf">local_attention_window</span>: <span class="value">int (94)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">ropebert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (95)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">llama</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">rope_theta</span>: <span class="value">float (96)</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">logbert</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">hvm_weight</span>: <span class="value">float (97)</span></li>
            </ul>
            </li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">pretrain_datasets</span>: <span class="value">str (98)</span></li>
        <li class='no-bullet'><span class="key-leaf">weights_path</span>: <span class="value">str (99)</span></li>
        <li class='no-bullet'><span class="key-leaf">graph_context_mode</span>: <span class="value">str (100)</span></li>
        <li class='bullet'><span class="key">walks</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">walk_length</span>: <span class="value">int (101)</span></li>
            <li class='no-bullet'><span class="key-leaf">num_walks</span>: <span class="value">int (102)</span></li>
            <li class='no-bullet'><span class="key-leaf">time_weight</span>: <span class="value">str (103)</span></li>
            <li class='no-bullet'><span class="key-leaf">half_life</span>: <span class="value">float (104)</span></li>
            <li class='no-bullet'><span class="key-leaf">random_walk_start</span>: <span class="value">bool (105)</span></li>
            <li class='no-bullet'><span class="key-leaf">diversity_weight</span>: <span class="value">float (106)</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. Depth D of the Weisfeiler-Lehman rooted subgraphs (RSGs) of each node, built from                 node and edge types (ProGrapher: 3 on small graphs, 4 on large ones).<br>
2. Number of negative RSGs sampled per RSG (PV-DBOW).<br>
3. RSGs seen fewer times in the training snapshots are ignored.<br>
4. Initial learning rate of doc2vec.<br>
5. Epochs to infer the embedding of each snapshot (all splits) from the trained RSG vectors.<br>
6. Processes extracting RSGs and inferring embeddings, and threads of doc2vec                 training (training is only deterministic with 1).<br>
7. Off by default, hurts generalization (paper Appendix E).<br>
8. Off by default, same reason as use_lifespan.<br>
9. SPIDER encoder size preset (selects hidden / layers / heads / FFN). Standard presets: 'tiny' (H=128), 'mini' (H=256), 'med' (H=512), 'baseline' (H=768). HF pretrained model_types use their own size keys (e.g. 'small' / 'medium' / 'large' / 'xl' for gpt2_pretrained, '1b' / '3b' for llama3_pretrained).<br>
10. SPIDER pretraining objective + architecture. See config/pretrained/README.md for full descriptions. Families:
  MLM walk-based     : bert, roberta, modernbert, ropebert, llama, logbert
  Walk embedding     : deepwalk, node2vec
  GNN self-supervised: graphmae, gae, dgi
  GNN token-budget   : gnn_distill, behavior_cluster, spider
  HF pretrained      : gpt2_pretrained, llama3_pretrained, opt_pretrained<br><br><b>Available options (one selection)</b>:<br>`bert`<br>`roberta`<br>`modernbert`<br>`ropebert`<br>`llama`<br>`logbert`<br>`deepwalk`<br>`node2vec`<br>`graphmae`<br>`gae`<br>`dgi`<br>`gnn_distill`<br>`behavior_cluster`<br>`spider`<br>`gpt2_pretrained`<br>`llama3_pretrained`<br>`opt_pretrained`
11. Skip-gram context window size. Default: 5.<br>
12. Number of Word2Vec training epochs. Default: 10.<br>
13. Minimum label frequency to include in vocabulary. Default: 1.<br>
14. Number of parallel workers for Word2Vec training. Default: 4.<br>
15. Return parameter. Higher p = less likely to revisit previous node. Default: 1.0.<br>
16. In-out parameter. q > 1 = BFS-like (local); q < 1 = DFS-like (explore). Default: 1.0.<br>
17. GNN edge embedding dimension. Default: 128.<br>
18. GNN hidden dimension. Default: 128.<br>
19. Number of GNN encoder layers. Default: 1.<br>
20. Number of attention heads in GNN layers. Default: 4.<br>
21. Min direct neighbors to sample per entity. Default: 5.<br>
22. Max direct neighbors to sample per entity. Default: 20.<br>
23. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
24. Weight for distillation loss (T5 encoder → GNN alignment). Default: 0.5.<br>
25. Weight for GNN masked reconstruction loss. Default: 1.0.<br>
26. Learning rate for GNN teacher parameters. Default: 1e-3.<br>
27. EMA momentum for teacher update (0.999 = slow-moving). Default: 0.999.<br>
28. Apply per-edge-type linear projection to source nodes before GNN message passing. Default: False.<br>
29. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from GNN neighborhoods. Default: False.<br>
30. Contrastive projection head output dimensionality. Default: 128.<br>
31. Weight for multi-label BCE loss. Default: 1.0.<br>
32. Weight for contrastive (NT-Xent) loss. Default: 0.5.<br>
33. Temperature for InfoNCE cosine similarity scaling. Default: 0.07.<br>
34. Skip entities with fewer behavior labels in their signature. Default: 2.<br>
35. Remove shared libs, /dev, /proc, /sys, linker cache, common configs from signature extraction. Default: True.<br>
36. Use contrastive projection head for inference embeddings (proj_dim) instead of raw encoder (emb_dim). Default: False.<br>
37. Floor: oversample small classes to this minimum per epoch. 0 = no floor. Default: 0.<br>
38. Cap entities per signature class per epoch. Rotates across epochs. 0 = no cap. Default: 0.<br>
39. K in P×K batch sampling: number of entities per class per batch. Guarantees every entity has at least K-1 positives for contrastive learning. Default: 4.<br>
40. Classification head mode: 'multilabel' = BCE over behavior labels, 'class' = cross-entropy over entity classes. Default: multilabel.<br>
41. Contrastive positive definition: 'signature' = entities with identical behavior signatures are positives, 'entity_class' = entities with the same coarse functional class are positives. Default: signature.<br>
42. Remove entity type prefix tokens ([PROC], [FILE], [SOCK]) from tokenized sequences. Forces the model to learn identity from behavior rather than type. Default: False.<br>
43. GNN edge embedding dimension. Default: 256.<br>
44. GNN encoder hidden dimension. Default: 256.<br>
45. Contrastive projection dimension (teacher). Default: 256.<br>
46. Number of attention heads in GNN TransformerConv. Default: 4.<br>
47. Min direct neighbors to sample per entity. Default: 5.<br>
48. Max direct neighbors to sample per entity. Default: 20.<br>
49. Dedup neighbors by (edge_type, node_type, label) and prioritize edge type diversity. Default: True.<br>
50. Remove shared libs, /dev, /proc, /sys from GNN neighborhoods. Default: True.<br>
51. EMA momentum for teacher T5 update. Default: 0.999.<br>
52. Weight for supervised contrastive loss on GNN teacher. Default: 1.0.<br>
53. Weight for distillation loss (student → GNN targets). Default: 0.5.<br>
54. SupCon temperature. Default: 0.07.<br>
55. Skip entities with fewer behavior labels (for entity class assignment). Default: 2.<br>
56. Floor: oversample small classes to this minimum per epoch. Default: 100.<br>
57. Cap entities per class per epoch. Default: 2000.<br>
58. K in P×K batching. Default: 4.<br>
59. Remove entity type prefix tokens from inputs. Default: False.<br>
60. Distillation loss: sce or mse. Default: sce.<br><br><b>Available options (one selection)</b>:<br>`sce`<br>`mse`
61. Teacher loss: contrastive (SupCon) or bce (cross-entropy). Default: contrastive.<br><br><b>Available options (one selection)</b>:<br>`contrastive`<br>`bce`
62. Teacher data modalities (comma-separated): signature, gnn_emb, or both. Default: signature,gnn_emb.<br>
63. Student-only ablation: none, student_signature, or student_class. Default: none.<br><br><b>Available options (one selection)</b>:<br>`none`<br>`student_signature`<br>`student_class`
64. Number of GAT encoder layers. Default: 2.<br>
65. Number of attention heads in GAT layers. Default: 4.<br>
66. Number of GAT decoder layers. Default: 1.<br>
67. Fraction of nodes to mask during pretraining. Default: 0.5.<br>
68. Fraction of masked nodes replaced with random tokens (rest get [MASK]). Default: 0.1.<br>
69. Minimum number of neighbors in temporal neighborhood. Default: 5.<br>
70. Maximum number of neighbors in temporal neighborhood. Default: 20.<br>
71. Number of training epochs. Default: 100.<br>
72. Learning rate. Default: 0.001.<br>
73. Number of GAT encoder layers. Default: 2.<br>
74. Number of attention heads in GAT layers. Default: 4.<br>
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
85. Total number of tokens to process during pretraining (controls training duration).<br>
86. Number of tokens for learning rate warmup at the start of pretraining.<br>
87. Mini-batch size during pretraining. Interpretation depends on model_type: walks per batch (MLM), entities per batch (GNN token-budget), labels per batch (HF pretrained), neighborhoods per batch (GNN-SSL).<br>
88. Peak learning rate for pretraining (after warmup).<br>
89. Learning rate scheduler shape after warmup. Only consumed by MLM and HF pretrained; GNN token-budget hardcodes cosine.<br><br><b>Available options (one selection)</b>:<br>`cosine`<br>`linear`
90. Initial fixed mask rate (fraction of nodes masked). Decays to mask_rate_min.<br>
91. Minimum mask rate after decay.<br>
92. Enable structured masking of edge types during pretraining and edge-type scoring at inference.<br>
93. Use global attention every N layers. Other layers use local sliding window attention. Default: 3.<br>
94. Sliding window size for local attention layers. Default: 128 tokens.<br>
95. Base frequency for rotary position embeddings. Default: 10000.0.<br>
96. Base frequency for rotary position embeddings. Default: 10000.0.<br>
97. Weight for hypersphere volume minimization loss relative to MLM loss. Default: 0.1.<br>
98. Comma-separated list of dataset names to use for multi-dataset pretraining.<br>
99. Path to a folder of pretrained weights (e.g. the SPIDER weights). May contain any subset of: corpus.pt (sampler states + indexid2msg), tokenizer.pt (BPE vocabulary), pretrain_*.pt (model checkpoint), behavior_vocab.txt. Present artifacts are loaded; missing ones are computed from scratch. Works with any model_type.<br>
100. Graph context scope for walk sampling at both pretraining and inference time. 'window' = each time-window snapshot independently; 'day' = merge all snapshots from the same calendar day; 'all' = merge all snapshots in the relevant split (train split at pretrain time; val or test split at inference time — no training data leaks into inference context).<br><br><b>Available options (one selection)</b>:<br>`window`<br>`day`<br>`all`
101. Number of nodes per random walk during pretraining.<br>
102. Number of random walks to sample per node per epoch during pretraining.<br>
103. Temporal weighting for neighbor selection: 'uniform', 'exponential', or 'linear'.<br>
104. Half-life (in seconds) for exponential time weighting of neighbor selection.<br>
105. Randomize the temporal entry point for each walk. When True, the first hop picks a random edge instead of always starting from the earliest/latest timestamp.<br>
106. Edge-type diversity bias for random walks during pretraining. 0 = no bias (default), higher values increasingly favor edges whose type is underrepresented in the current walk. Only affects pretraining; finetuning/inference use natural distribution.<br>
