<div class="annotate">

<ul>
    <li class='bullet'><span class="key">build_graphs</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">use_all_files</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">mimicry_edge_num</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">time_window_size</span>: <span class="value">float</span></li>
        <li class='no-bullet'><span class="key-leaf">use_hashed_label</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">fuse_edge</span>: <span class="value">bool</span></li>
        <li class='bullet'><span class="key">node_label_features</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">subject</span>: <span class="value">str (2)</span></li>
            <li class='no-bullet'><span class="key-leaf">file</span>: <span class="value">str (3)</span></li>
            <li class='no-bullet'><span class="key-leaf">netflow</span>: <span class="value">str (4)</span></li>
        </ul>
        </li>
        <li class='no-bullet'><span class="key-leaf">multi_dataset</span>: <span class="value">str (5)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">transformation</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (6)</span></li>
        <li class='bullet'><span class="key">rcaid_pseudo_graph</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">use_pruning</span>: <span class="value">bool</span></li>
        </ul>
        </li>
        <li class='bullet'><span class="key">synthetic_attack_naive</span>
        <ul>
            <li class='no-bullet'><span class="key-leaf">num_attacks</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">num_malicious_process</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">num_unauthorized_file_access</span>: <span class="value">int</span></li>
            <li class='no-bullet'><span class="key-leaf">process_selection_method</span>: <span class="value">str</span></li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. <b>Available options (one selection)</b>:<br><br>orthrus<br>magic
2. <b>Available options (multi selection)</b>:<br><br>type<br>path<br>cmd_line
3. <b>Available options (multi selection)</b>:<br><br>type<br>path
4. <b>Available options (multi selection)</b>:<br><br>type<br>remote_ip<br>remote_port
5. <b>Available options (one selection)</b>:<br><br>THEIA_E5<br>THEIA_E3<br>CADETS_E5<br>CADETS_E3<br>CLEARSCOPE_E5<br>CLEARSCOPE_E3<br>optc_h201<br>optc_h501<br>optc_h051<br>none
6. <b>Available options (multi selection)</b>:<br><br>undirected<br>dag<br>rcaid_pseudo_graph<br>none<br>synthetic_attack_naive
