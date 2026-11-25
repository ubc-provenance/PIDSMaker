<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (1)</span></li>
    <li class='bullet'><span class="key">rcaid_pseudo_graph</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">use_pruning</span>: <span class="value">bool (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">prune_threshold</span>: <span class="value">float (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">root_node_types</span>: <span class="value">str (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">keep_original_edges</span>: <span class="value">bool (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">merge_same_label_roots</span>: <span class="value">bool (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_roots_per_node</span>: <span class="value">int (7)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">merge_edges</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">per_edge_type</span>: <span class="value">bool (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">aggregate_attributes</span>: <span class="value">bool (9)</span></li>
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

</div>

1. Applies transformations to graphs after their construction. Multiple transformations can be applied sequentially. Example: `used_methods=undirected,dag`<br><br><b>Available options (multi selection)</b>:<br>`undirected`<br>`dag`<br>`rcaid_pseudo_graph`<br>`merge_edges`<br>`none`<br>`synthetic_attack_naive`
2. Whether roots reaching more than `prune_threshold` of the nodes of the graph                     get no pseudo edges (R-Caid's pruning of highly-connected roots).<br>
3. Fraction of the nodes of the graph above which the descendants of a root are                     pruned (0.5 for DARPA TC in the paper).<br>
4. Comma-separated node types that can be roots (e.g. `file,netflow`: files and                     sockets, as in the paper), or `all`.<br>
5. Whether the transformed graph keeps the edges of the provenance graph next to                     the pseudo edges (root -> descendant), which have no edge type. If False, only                     the pseudo edges are kept.<br>
6. Whether the pseudo edges of the roots with the same type and label (thus the                     same features) towards a same node are merged into one, from the first such root.<br>
7. Maximum number of pseudo-roots of a node (0: no limit): its roots with the                     fewest descendants. Bounds the pseudo edges by this number times the number of                     nodes, e.g. when a process reads thousands of files that become roots of all                     its descendants.<br>
8. Whether repeated edges are merged per edge type (one edge per source,                     destination and type) or regardless of their type (one edge per source and                     destination, as in Euler). Merged edges have a `count` and a `weight` attribute.<br>
9. Whether the edge attributes of a merged edge (`batching.edge_features=edge_attrs`)                     aggregate all its events, as the edge features of Euler, Argus and LARES: the number                     of events, the number of events with each value of each categorical attribute                     (e.g. successful and failed logons) and the number, mean and std of the values of                     numeric attributes, each standardized over the edges of the graph as                     sigmoid(floor(x / std)). Otherwise, the attributes of the first event are kept.<br>
