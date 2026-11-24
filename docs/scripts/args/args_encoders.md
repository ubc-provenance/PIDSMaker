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
    <li class='bullet'><span class="key">lares</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_node_features</span>: <span class="value">bool (8)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">egraphsage</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (9)</span></li>
        <li class='no-bullet'><span class="key-leaf">undirected</span>: <span class="value">bool (10)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">argus</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (11)</span></li>
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
        <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int (13)</span></li>
        <li class='no-bullet'><span class="key-leaf">num_layers</span>: <span class="value">int (14)</span></li>
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
7. `target_to_source` (LARES): hosts aggregate the messages of the destinations of                 their outgoing edges; `source_to_target`: of the sources of their incoming edges.<br><br><b>Available options (one selection)</b>:<br>`target_to_source`<br>`source_to_target`
8. Adds a projection of the node features to the node ID embedding (LARES only uses                 node IDs).<br>
9. Number of E-GraphSAGE layers, whose messages include the edge features.<br>
10. Aggregates each edge in both directions, as E-GraphSAGE and Anomal-E (default: True).<br>
11. Number of layers: GCN layers followed by a last edge-conditioned convolution (NNConv) if edge features are used.<br>
12. Unused: the input projections always use ReLU.<br>
13. Number of attention heads of each Heterogeneous Graph Transformer (HGT) layer.<br>
14. Number of HGT layers. Each node and edge type gets its own parameters; not available on OpTC.<br>
