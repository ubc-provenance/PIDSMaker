<div class="annotate">

<ul>
    <li class='bullet'><span class="key">predict_edge_supervised</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">mode</span>: <span class="value">str (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">top_n_attacks</span>: <span class="value">int (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">attack_patterns</span>: <span class="value">list (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_edges_per_pattern</span>: <span class="value">int (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">attack_edges_path</span>: <span class="value">str (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">pos_weight</span>: <span class="value">float (7)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_edge_type</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_triplet_types</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">edge_type_in_msg</span>: <span class="value">bool (10)</span></li>
        <li class='bullet'><span class="key">AMS</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">version</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">margin</span>: <span class="value">float</span></li>
            <li class='no-bullet'><span class="key-leaf">scale</span>: <span class="value">int</span></li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_node_type</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">flash_snapshots</span>: <span class="value">bool (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">flash_drop_conf</span>: <span class="value">float (13)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_process_tag</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (14)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">detection</span>: <span class="value">str (15)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_masked_struct</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (16)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">max_samples</span>: <span class="value">int (18)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">detect_edge_few_shot</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (19)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_edge_contrastive</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (20)</span></li>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (21)</span></li>
        <li class='no-bullet'><span class="key-leaf">ap_margin</span>: <span class="value">float (22)</span></li>
        <li class='no-bullet'><span class="key-leaf">neg_sampling</span>: <span class="value">str (23)</span></li>
        <li class='no-bullet'><span class="key-leaf">neg_ratio</span>: <span class="value">float (24)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">deep_graph_infomax</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (25)</span></li>
        <li class='no-bullet'><span class="key-leaf">pca_components</span>: <span class="value">int (26)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">predict_snapshot_embedding</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (27)</span></li>
        <li class='no-bullet'><span class="key-leaf">top_k</span>: <span class="value">int (28)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_node_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (29)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (30)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_node_embeddings</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (31)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (32)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">autoencode_edge_embeddings</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (33)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_edge_feats</span>: <span class="value">bool (34)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_edge_embeddings</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (35)</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (36)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">reconstruct_masked_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (37)</span></li>
        <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (38)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">one_class</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (39)</span></li>
        <li class='no-bullet'><span class="key-leaf">beta</span>: <span class="value">float (40)</span></li>
        <li class='no-bullet'><span class="key-leaf">eps</span>: <span class="value">float (41)</span></li>
        <li class='no-bullet'><span class="key-leaf">warmup</span>: <span class="value">int (42)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
2. 'scores': pick top-N from an edge_scores pkl; 'patterns': match hand-crafted TTP patterns; 'synthetic': use pre-computed embeddings for explicit attack edge definitions.<br><br><b>Available options (one selection)</b>:<br>`scores`<br>`patterns`<br>`synthetic`
3. (scores mode) Number of top-loss edges to use as attack examples.<br>
4. (patterns mode) List of TTP pattern dicts.<br>
5. (patterns mode) Max edges collected per pattern.<br>
6. (synthetic mode) Path to YAML file with explicit (src_type, src_label, edge_type, dst_type, dst_label) attack edge definitions.<br>
7. BCE pos_weight for the attack class (on top of 1:1 oversampling).<br>
8. Loss used to predict the edge type. AMS: additive margin softmax (see `AMS`).<br><br><b>Available options (one selection)</b>:<br>`cross_entropy`<br>`BCE`<br>`AMS`
9. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
10. Keeps the edge type in the edge messages `msg` (TGN memory and `msg` edge                 features), as in Kairos. Messages of an edge are only used after its type is                 predicted. By default, the edge type is removed from the messages.<br>
11. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
12. With the `flash` threshold method: trains each epoch on the training nodes still misclassified, and flags a node only if no snapshot of the epochs so far predicts its type correctly with a confidence above the threshold, as in FLASH.<br>
13. With `flash_snapshots`: training nodes predicted with this confidence or more are removed after an epoch, as the correctly predicted ones (0.9 in FLASH).<br>
14. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
15. Score of the processes at inference. `loss`: cross-entropy of their tag (tags                 unseen in training have their own class, never predicted). `kmeans`: distance of                 their embedding to the median of the nearest K-Means cluster of training                 processes (as in the paper).<br><br><b>Available options (one selection)</b>:<br>`loss`<br>`kmeans`
16. <br><b>Available options (one selection)</b>:<br>`cross_entropy`<br>`BCE`
17. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
18. Number of existing edges sampled per batch as positives, with as many negative                 node pairs, capped by the number of nodes (MAGIC: 10000). 0 uses all the edges.<br>
19. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
20. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
21. `bce`: binary cross-entropy of positive vs. random negative edges (default),                 `ap`: surrogate average precision loss of Argus, `bpr`: pairwise BPR loss of                 ShadeWatcher, softplus(s_neg - s_pos) between each negative edge and its positive edge.<br><br><b>Available options (one selection)</b>:<br>`bce`<br>`ap`<br>`bpr`
22. Margin of the squared hinge of the `ap` loss (default: 0.8).<br>
23. Negative edges: `shuffle_dst` pairs each source with the destination of another                 edge of the graph (negatives keep the degree of destinations), `uniform` draws                 sources and destinations uniformly among the embedded nodes (as Euler and Argus),                 `uniform_dst` pairs the source of each edge with uniform destinations among the                 embedded nodes, `tgn` pairs it with a uniform destination of the dataset embedded                 by the TGN (as TGN and Jbeil, requires `tgn_last_neighbor.negative_destinations`).<br><br><b>Available options (one selection)</b>:<br>`shuffle_dst`<br>`uniform`<br>`uniform_dst`<br>`tgn`
24. Number of negatives per positive edge with `uniform` sampling.<br>
25. Unused: the discriminator of DGI is bilinear.<br><br><b>Available options (one selection)</b>:<br>`none`
26. Number of principal components of the PCA fitted on the embeddings of training                 edges. Edges are scored by the reconstruction error of their embedding.<br>
27. Unused: the encoder predicts the embedding.<br><br><b>Available options (one selection)</b>:<br>`none`
28. Number of top-ranked rooted subgraphs (key indicators) whose nodes get a score:                 other nodes get 0. 0 scores all nodes by the rank of their best rooted subgraph.<br>
29. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
30. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
31. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
32. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
33. Unused: the objective has its own autoencoder.<br><br><b>Available options (one selection)</b>:<br>`none`
34. Adds the edge features (`batching.edge_features`) to the edge embedding through a                 gated skip connection, as LARES.<br>
35. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
36. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
37. <br><b>Available options (one selection)</b>:<br>`SCE`<br>`MSE`<br>`MSE_sum`<br>`MAE`<br>`none`
38. Decoder used before computing loss.<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
39. Decoder applied to embeddings before the hypersphere; use 'none' (identity).<br><br><b>Available options (one selection)</b>:<br>`edge_mlp`<br>`node_mlp`<br>`magic_gat`<br>`nodlink`<br>`inner_product`<br>`none`
40. Soft-boundary fraction; radius is the (1-beta) distance quantile.<br>
41. Center slack keeping |c| away from zero.<br>
42. Kept for parity; a no-op (c/r update every train step).<br>
