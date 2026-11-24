<div class="annotate">

<ul>
    <li class='no-bullet'><span class="key-leaf">used_method</span>: <span class="value">str (1)</span></li>
    <li class='no-bullet'><span class="key-leaf">use_all_files</span>: <span class="value">bool</span></li>
    <li class='no-bullet'><span class="key-leaf">mimicry_edge_num</span>: <span class="value">int</span></li>
    <li class='no-bullet'><span class="key-leaf">time_window_size</span>: <span class="value">float (2)</span></li>
    <li class='no-bullet'><span class="key-leaf">use_hashed_label</span>: <span class="value">bool (3)</span></li>
    <li class='no-bullet'><span class="key-leaf">fuse_edge</span>: <span class="value">bool (4)</span></li>
    <li class='no-bullet'><span class="key-leaf">consistent_edge_types</span>: <span class="value">bool (5)</span></li>
    <li class='no-bullet'><span class="key-leaf">operations</span>: <span class="value">str (6)</span></li>
    <li class='no-bullet'><span class="key-leaf">drop_noise_events</span>: <span class="value">bool (7)</span></li>
    <li class='no-bullet'><span class="key-leaf">benign_training</span>: <span class="value">bool (8)</span></li>
    <li class='no-bullet'><span class="key-leaf">edge_type</span>: <span class="value">str (9)</span></li>
    <li class='bullet'><span class="key">node_label_features</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">subject</span>: <span class="value">str (10)</span></li>
        <li class='no-bullet'><span class="key-leaf">file</span>: <span class="value">str (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">netflow</span>: <span class="value">str (12)</span></li>
        <li class='no-bullet'><span class="key-leaf">host</span>: <span class="value">str (13)</span></li>
    </ul>
    </li>
    <li class='no-bullet'><span class="key-leaf">null_label_tokens</span>: <span class="value">bool (14)</span></li>
    <li class='no-bullet'><span class="key-leaf">multi_dataset</span>: <span class="value">str (15)</span></li>
    <li class='bullet'><span class="key">cleaning</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">used_methods</span>: <span class="value">str (16)</span></li>
    </ul>
    </li>
</ul>

</div>

1. The method to build time window graphs.<br><br><b>Available options (one selection)</b>:<br>`default`<br>`magic`
2. The size of each graph in minutes. The notation should always be float (e.g. 10.0). Supports sizes < 1.0.<br>
3. Whether to hash the textual features.<br>
4. Whether to fuse duplicate sequential edges into a single edge.<br>
5. Map OpTC edge types to DARPA TC equivalents for shared vocabulary across datasets.<br>
6. Events kept when building graphs, by value of their `operation` column:                 `default` keeps the default events of the dataset's schema (all events, except for                 OpTC network flows where only FLOW_START events are kept, as in LARES and Argus),                 `all` keeps all events, otherwise a comma-separated list of operations                 (e.g. `FLOW_START,FLOW_MESSAGE`).<br>
7. Drops the noise events of the dataset's schema when building graphs (OpTC                 network flows: NetBIOS, LLMNR and SSDP broadcasts and multicasts and ICMPv6 neighbor                 discovery, which link every host to most hosts in every time window). No effect on datasets                 without noise events. Default: True.<br>
8. Drops the events labelled malicious from the graphs of the train and val dates, so                 that models are trained and validated on benign events only (datasets with event                 labels whose train or val dates have attacks). No effect on other datasets and                 dates. Default: True.<br>
9. Edge attribute used as edge type by graphs and models (e.g. the type predicted by                 `predict_edge_type`). `operation` is the default type of events of the dataset                 (e.g. system calls in provenance datasets), other values are categorical edge                 attributes of the dataset's schema (e.g. `logon_type` in LANL).<br><br><b>Available options (one selection)</b>:<br>`auth_type`<br>`dst_port`<br>`dst_user_kind`<br>`logon_type`<br>`operation`<br>`protocol`<br>`src_port`<br>`src_user_kind`<br>`success`
10. Attributes of `subject` nodes used as node labels (concatenated): path, cmd_line. `type` adds the node type, and `auto` uses the type-specific default label, only made of non-null attributes.<br><br><b>Available options (multi selection)</b>:<br>`auto`<br>`type`<br>`path`<br>`cmd_line`
11. Attributes of `file` nodes used as node labels (concatenated): path. `type` adds the node type, and `auto` uses the type-specific default label, only made of non-null attributes.<br><br><b>Available options (multi selection)</b>:<br>`auto`<br>`type`<br>`path`
12. Attributes of `netflow` nodes used as node labels (concatenated): local_ip, local_port, remote_ip, remote_port. `type` adds the node type, and `auto` uses the type-specific default label, only made of non-null attributes.<br><br><b>Available options (multi selection)</b>:<br>`auto`<br>`type`<br>`local_ip`<br>`local_port`<br>`remote_ip`<br>`remote_port`
13. Attributes of `host` nodes used as node labels (concatenated): name. `type` adds the node type, and `auto` uses the type-specific default label, only made of non-null attributes.<br><br><b>Available options (multi selection)</b>:<br>`auto`<br>`type`<br>`name`
14. When enabled, null entity attributes produce explicit tokens ([NO_CMD], [NO_PATH], [NO_IP]) instead of empty strings.<br>
15. A comma-separated list of datasets on which training is performed. Evaluation is done only the primary dataset run in CLI.<br><br><b>Available options (one selection)</b>:<br>`THEIA_E5`<br>`THEIA_E3`<br>`CADETS_E5`<br>`CADETS_E3`<br>`CLEARSCOPE_E5`<br>`CLEARSCOPE_E3`<br>`optc_h201`<br>`optc_h501`<br>`optc_h051`<br>`TRACE_E5`<br>`TRACE_E3`<br>`FIVEDIRECTIONS_E5`<br>`FIVEDIRECTIONS_E3`<br>`PROVENANCE_BENIGN`<br>`ATLASV2_EDR`<br>`CARBANAKV2_EDR`<br>`LANL`<br>`OPTC_NET`<br>`none`
16. Cleaning methods applied to the raw data before building the graphs, or `none`.                     `remove_null_attributes`: null-like attributes (`None`, `NA`, `<UNNAMED fd:3>`, invalid IPs and ports) are removed from labels;                     `normalize_labels`: whitespaces, slashes, path prefixes and quotes are normalized and long labels truncated;                     `remove_duplicate_events`: events with the same source, destination, type and timestamp are kept once;                     `remove_self_loops`: events from a node to itself are removed;                     `remove_noisy_netflows`: netflows to multicast/broadcast/... IPs are removed with their events;                     `dedup_window_edges`: only the first event of each (source, destination, type) is kept in each time window.<br><br><b>Available options (multi selection)</b>:<br>`remove_null_attributes`<br>`normalize_labels`<br>`remove_duplicate_events`<br>`remove_self_loops`<br>`remove_noisy_netflows`<br>`dedup_window_edges`<br>`none`
