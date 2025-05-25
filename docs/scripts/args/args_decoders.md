<div class="annotate">

<ul>
    <li class='bullet'><span class="key">edge_mlp</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">architecture_str</span>: <span class="value">str (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">src_dst_projection_coef</span>: <span class="value">int (2)</span></li>
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
    
    
    </li>
    <li class='bullet'><span class="key">inner_product</span>
    
    
    </li>
    <li class='bullet'><span class="key">none</span>
    
    
    </li>
</ul>

</div>

1. A string describing a simple neural network. Example: if the encoder's output has shape `node_out_dim=128`                                 setting `architecture_str=linear(2) | relu | linear(0.5)` creates this MLP: nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128).                                 Precisely, in linear(x), x is the multiplier of input neurons.<br>
2. <br><b>Available options (one selection)</b>:<br>`M`<br>`u`<br>`l`<br>`t`<br>`i`<br>`p`<br>`l`<br>`i`<br>`e`<br>`r`<br>` `<br>`o`<br>`f`<br>` `<br>`i`<br>`n`<br>`p`<br>`u`<br>`t`<br>` `<br>`n`<br>`e`<br>`u`<br>`r`<br>`o`<br>`n`<br>`s`<br>` `<br>`t`<br>`o`<br>` `<br>`p`<br>`r`<br>`o`<br>`j`<br>`e`<br>`c`<br>`t`<br>` `<br>`s`<br>`r`<br>`c`<br>` `<br>`a`<br>`n`<br>`d`<br>` `<br>`d`<br>`s`<br>`t`<br>` `<br>`n`<br>`o`<br>`d`<br>`e`<br>`s`<br>`.`
