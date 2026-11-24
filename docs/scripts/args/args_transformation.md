<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (1)</span></li>
    <li class='bullet'><span class="key">rcaid_pseudo_graph</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">use_pruning</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">merge_edges</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">per_edge_type</span>: <span class="value">bool (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">aggregate_attributes</span>: <span class="value">bool (3)</span></li>
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
2. Whether repeated edges are merged per edge type (one edge per source,                     destination and type) or regardless of their type (one edge per source and                     destination, as in Euler). Merged edges have a `count` and a `weight` attribute.<br>
3. Whether the edge attributes of a merged edge (`batching.edge_features=edge_attrs`)                     aggregate all its events, as the edge features of Euler, Argus and LARES: the number                     of events, the number of events with each value of each categorical attribute                     (e.g. successful and failed logons) and the number, mean and std of the values of                     numeric attributes, each standardized over the edges of the graph as                     sigmoid(floor(x / std)). Otherwise, the attributes of the first event are kept.<br>
