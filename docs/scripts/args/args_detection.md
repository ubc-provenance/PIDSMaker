<div class="annotate">

<ul>
    <li class='bullet'><span class="key">graph_preprocessing</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">node_features</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">edge_features</span>: <span class="value">str (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">multi_dataset_training</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">fix_buggy_graph_reindexer</span>: <span class="value">bool</span></li>
        <li class='bullet'><span class="key">global_batching</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (3)</span></li>
            <li class='no-bullet'><span class="key-leaf">global_batching_batch_size</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">global_batching_batch_size_inference</span>: <span class="value">int</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">intra_graph_batching</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (4)</span></li>
            <li class='bullet'><span class="key">edges</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">intra_graph_batch_size</span>: <span class="value">int</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">tgn_last_neighbor</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">tgn_neighbor_size</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">tgn_neighbor_n_hop</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">fix_buggy_orthrus_TGN</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">fix_tgn_neighbor_loader</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">directed</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">insert_neighbors_before</span>: <span class="value">bool</span></li>
            </ul>
            </li>
        </ul>
        </li>
        <li class='bullet'><span class="key">inter_graph_batching</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (5)</span></li>
            <li class='no-bullet'><span class="key-leaf">inter_graph_batch_size</span>: <span class="value">int</span></li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">gnn_training</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">use_seed</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">num_epochs</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">patience</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">weight_decay</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">node_hid_dim</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">node_out_dim</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">grad_accumulation</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">inference_device</span>: <span class="value">str</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (6)</span></li>
        <li class='bullet'><span class="key">flash</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">in_channel</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">out_channel</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">lr</span>: <span class="value">float</span></li>
            <li class='no-bullet'><span class="key-leaf">weight_decay</span>: <span class="value">float</span></li>
            <li class='no-bullet'><span class="key-leaf">epochs</span>: <span class="value">int</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">encoder</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
            <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (7)</span></li>
            <li class='bullet'><span class="key">tgn</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">tgn_memory_dim</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">tgn_time_dim</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">use_node_feats_in_gnn</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">use_memory</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">use_time_order_encoding</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">project_src_dst</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">graph_attention</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
                <li class='no-bullet'><span class="key-leaf">num_heads</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">concat</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">flow</span>: <span class="value">str</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">sage</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">activation</span>: <span class="value">str</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">GLSTM</span>
            <ul>
            </ul>
            </li>
            <li class='bullet'><span class="key">rcaid_gat</span>
            <ul>
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
            <li class='bullet'><span class="key">GIN</span>
            <ul>
            </ul>
            </li>
            <li class='bullet'><span class="key">sum_aggregation</span>
            <ul>
            </ul>
            </li>
            <li class='bullet'><span class="key">custom_mlp</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">none</span>
            <ul>
            </ul>
            </li>
        </ul>
        </li>
        <li class='bullet'><span class="key">decoder</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (8)</span></li>
            <li class='bullet'><span class="key">reconstruct_node_features</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (9)</span></li>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (10)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
            </ul>
            </li>
            <li class='bullet'><span class="key">reconstruct_node_embeddings</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (11)</span></li>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (12)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
            </ul>
            </li>
            <li class='bullet'><span class="key">reconstruct_edge_embeddings</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (13)</span></li>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (14)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
            </ul>
            </li>
            <li class='bullet'><span class="key">reconstruct_masked_features</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (15)</span></li>
                <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float</span></li>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (16)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
            </ul>
            </li>
            <li class='bullet'><span class="key">predict_edge_type</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (17)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
                <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">use_triplet_types</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">predict_node_type</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (18)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
                <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">predict_masked_struct</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (19)</span></li>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (20)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
                <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">detect_edge_few_shot</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (21)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
            </ul>
            </li>
            <li class='bullet'><span class="key">predict_edge_contrastive</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (22)</span></li>
                <li class='bullet'><span class="key">edge_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                    <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                </ul>
                </li>
                <li class='bullet'><span class="key">node_mlp</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                <li class='bullet'><span class="key">nodlink</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">none</span>
                <ul>
                </ul>
                </li>
                <li class='bullet'><span class="key">inner_product</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
                </ul>
                </li>
            </ul>
            </li>
            <li class='no-bullet'><span class="key-leaf">use_few_shot</span>: <span class="value">bool</span></li>
            <li class='bullet'><span class="key">few_shot</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">include_attacks_in_ssl_training</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">freeze_encoder</span>: <span class="value">bool</span></li>
                <li class='no-bullet'><span class="key-leaf">num_epochs_few_shot</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">patience_few_shot</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">lr_few_shot</span>: <span class="value">float</span></li>
                <li class='no-bullet'><span class="key-leaf">weight_decay_few_shot</span>: <span class="value">float</span></li>
                <li class='bullet'><span class="key">decoder</span>
                <ul>
                    <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str</span></li>
                    <li class='bullet'><span class="key">reconstruct_node_features</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (23)</span></li>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (24)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">reconstruct_node_embeddings</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (25)</span></li>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (26)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">reconstruct_edge_embeddings</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (27)</span></li>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (28)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">reconstruct_masked_features</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (29)</span></li>
                        <li class='no-bullet'><span class="key-leaf">mask_rate</span>: <span class="value">float</span></li>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (30)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">predict_edge_type</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (31)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
                        <li class='no-bullet'><span class="key-leaf">use_triplet_types</span>: <span class="value">bool</span></li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">predict_node_type</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (32)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">predict_masked_struct</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">loss</span>: <span class="value">str (33)</span></li>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (34)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='no-bullet'><span class="key-leaf">balanced_loss</span>: <span class="value">bool</span></li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">detect_edge_few_shot</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (35)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class='bullet'><span class="key">predict_edge_contrastive</span>
                    <ul>
                        <li class='no-bullet'><span class="key-leaf">decoder</span>: <span class="value">str (36)</span></li>
                        <li class='bullet'><span class="key">edge_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
                            <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int</span></li>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">node_mlp</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str</span></li>
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
                        <li class='bullet'><span class="key">nodlink</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">none</span>
                        <ul>
                        </ul>
                        </li>
                        <li class='bullet'><span class="key">inner_product</span>
                        <ul>
                            <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
                        </ul>
                        </li>
                    </ul>
                    </li>
                </ul>
                </li>
            </ul>
            </li>
        </ul>
        </li>
    </ul>
    </li>
    <li class='bullet'><span class="key">evaluation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">viz_malicious_nodes</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">ground_truth_version</span>: <span class="value">str (37)</span></li>
        <li class='no-bullet'><span class="key-leaf">best_model_selection</span>: <span class="value">str (38)</span></li>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str</span></li>
        <li class='bullet'><span class="key">node_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (39)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">tw_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (40)</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">node_tw_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (41)</span></li>
            <li class='no-bullet'><span class="key-leaf">use_dst_node_loss</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">use_kmeans</span>: <span class="value">bool</span></li>
            <li class='no-bullet'><span class="key-leaf">kmeans_top_K</span>: <span class="value">int</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">queue_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">queue_threshold</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (42)</span></li>
            <li class='bullet'><span class="key">kairos_idf_queue</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">include_test_set_in_IDF</span>: <span class="value">bool</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">provnet_lof_queue</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">queue_arg</span>: <span class="value">str</span></li>
            </ul>
            </li>
        </ul>
        </li>
        <li class='bullet'><span class="key">edge_evaluation</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">malicious_edge_selection</span>: <span class="value">str (43)</span></li>
            <li class='no-bullet'><span class="key-leaf">threshold_method</span>: <span class="value">str (44)</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. <b>Available options (multi selection)</b>:<br><br>node_type<br>node_emb<br>only_ones<br>edges_distribution
2. <b>Available options (multi selection)</b>:<br><br>edge_type<br>edge_type_triplet<br>msg<br>time_encoding<br>none
3. <b>Available options (one selection)</b>:<br><br>edges<br>minutes<br>unique_edge_types<br>none
4. <b>Available options (one selection)</b>:<br><br>edges<br>tgn_last_neighbor<br>none
5. <b>Available options (one selection)</b>:<br><br>graph_batching<br>none
6. <b>Available options (one selection)</b>:<br><br>orthrus<br>provd
7. <b>Available options (multi selection)</b>:<br><br>tgn<br>graph_attention<br>sage<br>GLSTM<br>rcaid_gat<br>magic_gat<br>GIN<br>sum_aggregation<br>custom_mlp<br>none
8. <b>Available options (multi selection)</b>:<br><br>reconstruct_node_features<br>reconstruct_node_embeddings<br>reconstruct_edge_embeddings<br>reconstruct_masked_features<br>predict_edge_type<br>predict_node_type<br>predict_masked_struct<br>detect_edge_few_shot<br>predict_edge_contrastive
9. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
10. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
11. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
12. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
13. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
14. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
15. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
16. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
17. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
18. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
19. <b>Available options (one selection)</b>:<br><br>cross_entropy<br>BCE
20. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
21. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
22. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
23. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
24. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
25. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
26. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
27. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
28. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
29. <b>Available options (one selection)</b>:<br><br>SCE<br>MSE<br>MSE_sum<br>MAE<br>none
30. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
31. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
32. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
33. <b>Available options (one selection)</b>:<br><br>cross_entropy<br>BCE
34. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
35. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
36. <b>Available options (one selection)</b>:<br><br>edge_mlp<br>node_mlp<br>magic_gat<br>nodlink<br>none
37. <b>Available options (one selection)</b>:<br><br>orthrus
38. <b>Available options (one selection)</b>:<br><br>best_adp
39. <b>Available options (one selection)</b>:<br><br>max_val_loss<br>mean_val_loss<br>threatrace<br>magic<br>flash<br>nodlink
40. <b>Available options (one selection)</b>:<br><br>max_val_loss<br>mean_val_loss<br>threatrace<br>magic<br>flash<br>nodlink
41. <b>Available options (one selection)</b>:<br><br>max_val_loss<br>mean_val_loss<br>threatrace<br>magic<br>flash<br>nodlink
42. <b>Available options (one selection)</b>:<br><br>kairos_idf_queue<br>provnet_lof_queue
43. <b>Available options (one selection)</b>:<br><br>src_node<br>dst_node<br>both_nodes
44. <b>Available options (one selection)</b>:<br><br>max_val_loss<br>mean_val_loss<br>threatrace<br>magic<br>flash<br>nodlink
