<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (1)</span></li>
    <li class='bullet'><span class="key">depimpact</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">score_method</span>: <span class="value">str (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">workers</span>: <span class="value">int</span></li>
        <li class='no-bullet'><span class="key-leaf">visualize</span>: <span class="value">bool</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">ocrapt</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">num_hops</span>: <span class="value">int (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">top_k</span>: <span class="value">int (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">min_nodes</span>: <span class="value">int (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_edges</span>: <span class="value">int (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">abnormality_level</span>: <span class="value">str (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">correlate_anomalous_once</span>: <span class="value">bool</span></li>
        <li class='no-bullet'><span class="key-leaf">remove_duplicated_subgraph</span>: <span class="value">bool</span></li>
    </ul>
    </li>
</ul>

</div>

1. Post-processing step to reconstruct attack paths or reduce false positives. `depimpact` is used in Orthrus; `ocrapt_subgraph` is OCR-APT's anomalous-subgraph stage.<br><br><b>Available options (one selection)</b>:<br>`depimpact`<br>`ocrapt_subgraph`
2. <br><b>Available options (one selection)</b>:<br>`component`<br>`shortest_path`<br>`1-hop`<br>`2-hop`<br>`3-hop`
3. <br><b>Available options (one selection)</b>:<br>`degree`<br>`recon_loss`<br>`degree_recon`
4. hops for correlating anomalies into subgraphs<br>
5. top-K seed nodes per node type (by Anomaly_score)<br>
6. minimum nodes per constructed subgraph<br>
7. subgraphs above this are Louvain-partitioned + edge-sampled<br>
8. least subgraph severity to keep (summed Prediction_probability)<br><br><b>Available options (one selection)</b>:<br>`Negligible`<br>`Minor`<br>`Moderate`<br>`Significant`<br>`Critical`
