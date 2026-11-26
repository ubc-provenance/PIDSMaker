<div class="annotate">

<ul>
    <li class='bullet'><span class="key">tgn</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">tgn_memory_dim</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">tgn_time_dim</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">use_node_feats_in_gnn</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_memory</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">use_time_order_encoding</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">project_src_dst</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">mode</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">use_residual_norm</span>: <span class="value">bool (1)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">graph_attention</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">concat</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">final_activation</span>: <span class="value">bool (2)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">sage</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">gcn</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (3)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">node_id_embedding</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">dim</span>: <span class="value">int (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">all_nodes</span>: <span class="value">bool (5)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">snapshot_gru</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float (7)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">textrcnn</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">sequence_length</span>: <span class="value">int (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">rnn_type</span>: <span class="value">str (10)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">lares</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_node_features</span>: <span class="value">bool (12)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">egraphsage</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (13)</span></li>
        <li class='no-bullet'><span class="key-leaf">undirected</span>: <span class="value">bool (14)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">argus</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (15)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">shadewatcher</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">hidden_dims</span>: <span class="value">str (16)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_dim</span>: <span class="value">int (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_neg_ratio</span>: <span class="value">int (18)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_margin</span>: <span class="value">float (19)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">gat</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">concat</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">gin</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">rgcn</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">rgcn_per_type</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">sum_aggregation</span>
    
    
    </li>
    <li class='bullet'><span class="key">rcaid_gat</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (20)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (21)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">magic_gat</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">negative_slope</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">alpha_l</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">glstm</span>
    
    
    </li>
    <li class='bullet'><span class="key">custom_mlp</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">none</span>
    
    
    </li>
    <li class='bullet'><span class="key">hetero_graph_transformer</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str (22)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (23)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (24)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Adds the projected input node features to the output of the GNN wrapped by TGN, followed by dropout and LayerNorm.<br>
2. Applies `activation` to the output of the last layer too, as in Kairos.<br>
3. Number of GCN layers. Edges are weighted by `batch.edge_weight` (see the `merge_edges` transformation).<br>
4. Size of the learned embedding of each node's identity, concatenated to its input features (as the one-hot node IDs of Euler, Argus and LARES).<br>
5. Embeds all the nodes of the dataset in every graph, as Euler and Argus (nodes absent                 from the graph only get their ID embedding). Combined with `snapshot_gru`, the state of                 every node is updated at every graph, and with uniform negative sampling, negative                 edges reach all nodes. Only for edge-level objectives.<br>
6. Size of the GRU state, projected to `node_out_dim` (0: `node_out_dim`).<br>
7. Dropout applied to the input of the GRU.<br>
8. Number of previous snapshots (graphs) whose graph2vec embeddings form the input                 sequence (ProGrapher's L). Shorter sequences are used when fewer snapshots were seen.<br>
9. Number of layers of the bidirectional RNN (hidden size: `node_hid_dim`).<br>
10. Bidirectional RNN computing the left and right contexts of each snapshot:                 `rnn` is the ReLU recurrence of the TextRCNN equations, `lstm` and `gru` are gated.<br><br><b>Available options (one selection)</b>:<br>`lstm`<br>`gru`<br>`rnn`
11. `target_to_source` (LARES): hosts aggregate the messages of the destinations of                 their outgoing edges; `source_to_target`: of the sources of their incoming edges.<br><br><b>Available options (one selection)</b>:<br>`target_to_source`<br>`source_to_target`
12. Adds a projection of the node features to the node ID embedding (LARES only uses                 node IDs).<br>
13. Number of E-GraphSAGE layers, whose messages include the edge features.<br>
14. Aggregates each edge in both directions, as E-GraphSAGE and Anomal-E (default: True).<br>
15. Number of layers: GCN layers followed by a last edge-conditioned convolution (NNConv) if edge features are used.<br>
16. Comma-separated output sizes of the GNN layers propagating the TransR entity                 embeddings (of size `node_hid_dim`) over the knowledge graph (ShadeWatcher: `32,16`).                 The output concatenates the entity embeddings and the output of each layer,                 projected to `node_out_dim` if their sizes differ.<br>
17. Size of the relation space of TransR (relation embeddings and projections).<br>
18. Number of corrupted triples per triple of the TransR loss.<br>
19. Margin between the distances of valid and corrupted triples in the TransR loss.<br>
20. Number of layers (local + root attention, then MLP update).<br>
21. Number of attention heads (averaged) of each attention.<br>
22. Unused: the input projections always use ReLU.<br>
23. Number of attention heads of each Heterogeneous Graph Transformer (HGT) layer.<br>
24. Number of HGT layers. Each node and edge type gets its own parameters; not available on OpTC.<br>
