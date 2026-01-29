<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (1)</span></li>
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

</div>

1. Applies transformations to graphs after their construction. Multiple transformations can be applied sequentially. Example: `used_methods=undirected,dag`<br><br><b>Available options (multi selection)</b>:<br>`undirected`<br>`dag`<br>`rcaid_pseudo_graph`<br>`none`<br>`synthetic_attack_naive`
