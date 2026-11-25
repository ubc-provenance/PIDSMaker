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
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (2)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">node_id_embedding</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">dim</span>: <span class="value">int (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">all_nodes</span>: <span class="value">bool (4)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">snapshot_gru</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">hidden_dim</span>: <span class="value">int (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float (6)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">textrcnn</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">sequence_length</span>: <span class="value">int (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">rnn_type</span>: <span class="value">str (9)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">lares</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str (10)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_node_features</span>: <span class="value">bool (11)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">egraphsage</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">undirected</span>: <span class="value">bool (13)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">argus</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (14)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">shadewatcher</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">hidden_dims</span>: <span class="value">str (15)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_dim</span>: <span class="value">int (16)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_neg_ratio</span>: <span class="value">int (17)</span></li>
        <li class='no-bullet'><span class="key-leaf">kg_margin</span>: <span class="value">float (18)</span></li>
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
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str (19)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (20)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (21)</span></li>
    </ul>
    </li>
</ul>

</div>

1. Adds the projected input node features to the output of the GNN wrapped by TGN, followed by dropout and LayerNorm.<br>
2. Number of GCN layers. Edges are weighted by `batch.edge_weight` (see the `merge_edges` transformation).<br>
3. Size of the learned embedding of each node's identity, concatenated to its input features (as the one-hot node IDs of Euler, Argus and LARES).<br>
4. Embeds all the nodes of the dataset in every graph, as Euler and Argus (nodes absent                 from the graph only get their ID embedding). Combined with `snapshot_gru`, the state of                 every node is updated at every graph, and with uniform negative sampling, negative                 edges reach all nodes. Only for edge-level objectives.<br>
5. Size of the GRU state, projected to `node_out_dim` (0: `node_out_dim`).<br>
6. Dropout applied to the input of the GRU.<br>
7. Number of previous snapshots (graphs) whose graph2vec embeddings form the input                 sequence (ProGrapher's L). Shorter sequences are used when fewer snapshots were seen.<br>
8. Number of layers of the bidirectional RNN (hidden size: `node_hid_dim`).<br>
9. Bidirectional RNN computing the left and right contexts of each snapshot:                 `rnn` is the ReLU recurrence of the TextRCNN equations, `lstm` and `gru` are gated.<br><br><b>Available options (one selection)</b>:<br>`lstm`<br>`gru`<br>`rnn`
10. `target_to_source` (LARES): hosts aggregate the messages of the destinations of                 their outgoing edges; `source_to_target`: of the sources of their incoming edges.<br><br><b>Available options (one selection)</b>:<br>`target_to_source`<br>`source_to_target`
11. Adds a projection of the node features to the node ID embedding (LARES only uses                 node IDs).<br>
12. Number of E-GraphSAGE layers, whose messages include the edge features.<br>
13. Aggregates each edge in both directions, as E-GraphSAGE and Anomal-E (default: True).<br>
14. Number of layers: GCN layers followed by a last edge-conditioned convolution (NNConv) if edge features are used.<br>
15. Comma-separated output sizes of the GNN layers propagating the TransR entity                 embeddings (of size `node_hid_dim`) over the knowledge graph (ShadeWatcher: `32,16`).                 The output concatenates the entity embeddings and the output of each layer,                 projected to `node_out_dim` if their sizes differ.<br>
16. Size of the relation space of TransR (relation embeddings and projections).<br>
17. Number of corrupted triples per triple of the TransR loss.<br>
18. Margin between the distances of valid and corrupted triples in the TransR loss.<br>
19. Unused: the input projections always use ReLU.<br>
20. Number of attention heads of each Heterogeneous Graph Transformer (HGT) layer.<br>
21. Number of HGT layers. Each node and edge type gets its own parameters; not available on OpTC.<br>
