<div class="annotate">

<ul>
    <li class='bullet'><span class="key">training_loop</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">run_evaluation</span>: <span class="value">str (1)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">experiment</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (2)</span></li>
        <li class='bullet'><span class="key">uncertainty</span>
        <ul>
            <li class='bullet'><span class="key">hyperparameter</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">hyperparameters</span>: <span class="value">str (3)</span></li>
                <li class='no-bullet'><span class="key-leaf">iterations</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">delta</span>: <span class="value">float</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">mc_dropout</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">iterations</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">dropout</span>: <span class="value">float</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">deep_ensemble</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">method</span>: <span class="value">str</span></li>
                <li class='no-bullet'><span class="key-leaf">iterations</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">restart_from</span>: <span class="value">str</span></li>
            </ul>
            </li>
            <li class='bullet'><span class="key">bagged_ensemble</span>
            <ul>
                <li class='no-bullet'><span class="key-leaf">iterations</span>: <span class="value">int</span></li>
                <li class='no-bullet'><span class="key-leaf">min_num_days</span>: <span class="value">int</span></li>
            </ul>
            </li>
        </ul>
        </li>
    </ul>
    </li>
</ul>

</div>

1. <br><b>Available options (one selection)</b>:<br>`each_epoch`<br>`best_epoch`
2. <br><b>Available options (one selection)</b>:<br>`uncertainty`<br>`none`
3. <br><b>Available options (multi selection)</b>:<br>`lr`<br>`num_epochs`<br>`text_h_dim`<br>`gnn_h_dim`
