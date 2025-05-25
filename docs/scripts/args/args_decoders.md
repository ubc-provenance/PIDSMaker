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

1. A string describing a simple neural network. Example: if the encoder's output has shape `node_out_dim=128`                                 setting `architecture_str=linear(2) | relu | linear(0.5)` creates this MLP: nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, y).                                 Precisely, in linear(x), x is the multiplier of input neurons. The final layer `nn.Linear(128, y)` is added automatically such that `y` is the                                 output size matching the downstream objective (e.g. edge type prediction involves predicting 10 edge types, so the output of the decoder should be 10).<br>
2. Multiplier of input neurons to project src and dst nodes.<br>
