<div class="annotate">

<ul>
    <li class='bullet'><span class="key">remove_null_attributes</span>
    
    
    </li>
    <li class='bullet'><span class="key">normalize_labels</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">collapse_whitespace</span>: <span class="value">bool (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">collapse_slashes</span>: <span class="value">bool (2)</span></li>
        <li class='no-bullet'><span class="key-leaf">strip_path_prefixes</span>: <span class="value">bool (3)</span></li>
        <li class='no-bullet'><span class="key-leaf">strip_quotes</span>: <span class="value">bool (4)</span></li>
        <li class='no-bullet'><span class="key-leaf">lowercase</span>: <span class="value">bool (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_length</span>: <span class="value">int (6)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">remove_duplicate_events</span>
    
    
    </li>
    <li class='bullet'><span class="key">remove_self_loops</span>
    
    
    </li>
    <li class='bullet'><span class="key">remove_noisy_netflows</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">ip_categories</span>: <span class="value">str (7)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">dedup_window_edges</span>
    
    
    </li>
</ul>

</div>

1. Trims labels and replaces runs of whitespaces by a single space.<br>
2. Replaces repeated '/' (except in URLs) and repeated '\' by a single one.<br>
3. Removes Windows NT path prefixes such as '\\?\' and '//?/'.<br>
4. Removes single and double quotes, e.g. around Windows executable paths.<br>
5. Lowercases labels, e.g. for case-insensitive Windows paths.<br>
6. Labels longer than this number of characters are truncated. 0 disables truncation.<br>
7. Netflow nodes whose remote IP is in one of these categories are removed.                 `broadcast` is 255.255.255.255.<br><br><b>Available options (multi selection)</b>:<br>`multicast`<br>`broadcast`<br>`unspecified`<br>`loopback`<br>`link_local`
