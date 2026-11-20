<div class="annotate">

<ul>
    <li class='bullet'><span class="key">none</span>
    
    
    </li>
    <li class='bullet'><span class="key">nltk</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">type_aware</span>: <span class="value">bool (1)</span></li>
        <li class='no-bullet'><span class="key-leaf">substitutions</span>: <span class="value">str (2)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">hierarchical</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">substitutions</span>: <span class="value">str (3)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">whole_label</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">substitutions</span>: <span class="value">str (4)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">bpe</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">vocab_size</span>: <span class="value">int (5)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (6)</span></li>
        <li class='no-bullet'><span class="key-leaf">mode</span>: <span class="value">str (7)</span></li>
        <li class='no-bullet'><span class="key-leaf">substitutions</span>: <span class="value">str (8)</span></li>
        <li class='no-bullet'><span class="key-leaf">canonicalize_neighbors</span>: <span class="value">bool (9)</span></li>
    </ul>
    </li>
    <li class='bullet'><span class="key">huggingface</span>
    <ul>
        <li class='no-bullet'><span class="key-leaf">model_name</span>: <span class="value">str (10)</span></li>
        <li class='no-bullet'><span class="key-leaf">max_seq_len</span>: <span class="value">int (11)</span></li>
        <li class='no-bullet'><span class="key-leaf">substitutions</span>: <span class="value">str (12)</span></li>
    </ul>
    </li>
</ul>

</div>

1. NLTK word tokenization. If True, process and file labels are split on '/' and netflow labels on ':' and '.'                 (Orthrus, R-Caid, NodLink). If False, all labels are split on '/', ':' and '.' (FLASH).<br>
2. Variable values replaced by a placeholder token before tokenization, or `none`.             `hash`: hex strings of 7+ chars -> [HASH]; `guid`: GUIDs/UUIDs -> [GUID]; `random`: random-looking strings and Windows SIDs -> [RANDOM];             `version`: 1.2.3 -> [VER]; `timestamp`: words starting with a date -> [TIMESTAMP]; `env_var`: $HOME, %APPDATA% -> [ENV];             `number`: numbers above 255 -> [NUM]; `ip`: netflow IPs -> [PRIVATE_IP]/[PUBLIC_IP]/[LOCALHOST_IP];             `port`: netflow ports -> [PORT_HTTPS], [PORT_SSH], ... or [PORT_LOW]/[PORT_REG]/[PORT_EPH].<br><br><b>Available options (multi selection)</b>:<br>`hash`<br>`guid`<br>`random`<br>`version`<br>`timestamp`<br>`env_var`<br>`number`<br>`ip`<br>`port`<br>`none`
3. Variable values replaced by a placeholder token before tokenization, or `none`.             `hash`: hex strings of 7+ chars -> [HASH]; `guid`: GUIDs/UUIDs -> [GUID]; `random`: random-looking strings and Windows SIDs -> [RANDOM];             `version`: 1.2.3 -> [VER]; `timestamp`: words starting with a date -> [TIMESTAMP]; `env_var`: $HOME, %APPDATA% -> [ENV];             `number`: numbers above 255 -> [NUM]; `ip`: netflow IPs -> [PRIVATE_IP]/[PUBLIC_IP]/[LOCALHOST_IP];             `port`: netflow ports -> [PORT_HTTPS], [PORT_SSH], ... or [PORT_LOW]/[PORT_REG]/[PORT_EPH].<br><br><b>Available options (multi selection)</b>:<br>`hash`<br>`guid`<br>`random`<br>`version`<br>`timestamp`<br>`env_var`<br>`number`<br>`ip`<br>`port`<br>`none`
4. Variable values replaced by a placeholder token before tokenization, or `none`.             `hash`: hex strings of 7+ chars -> [HASH]; `guid`: GUIDs/UUIDs -> [GUID]; `random`: random-looking strings and Windows SIDs -> [RANDOM];             `version`: 1.2.3 -> [VER]; `timestamp`: words starting with a date -> [TIMESTAMP]; `env_var`: $HOME, %APPDATA% -> [ENV];             `number`: numbers above 255 -> [NUM]; `ip`: netflow IPs -> [PRIVATE_IP]/[PUBLIC_IP]/[LOCALHOST_IP];             `port`: netflow ports -> [PORT_HTTPS], [PORT_SSH], ... or [PORT_LOW]/[PORT_REG]/[PORT_EPH].<br><br><b>Available options (multi selection)</b>:<br>`hash`<br>`guid`<br>`random`<br>`version`<br>`timestamp`<br>`env_var`<br>`number`<br>`ip`<br>`port`<br>`none`
5. Target BPE vocabulary size.<br>
6. Maximum number of tokens per label (and per walk for the pretrained MLM models). Longer sequences are truncated.<br>
7. 'domain_bpe' splits labels with domain-specific rules before BPE; 'bpe_only' applies BPE directly on whitespace-separated words.<br><br><b>Available options (one selection)</b>:<br>`domain_bpe`<br>`bpe_only`
8. Variable values replaced by a placeholder token before tokenization, or `none`.             `hash`: hex strings of 7+ chars -> [HASH]; `guid`: GUIDs/UUIDs -> [GUID]; `random`: random-looking strings and Windows SIDs -> [RANDOM];             `version`: 1.2.3 -> [VER]; `timestamp`: words starting with a date -> [TIMESTAMP]; `env_var`: $HOME, %APPDATA% -> [ENV];             `number`: numbers above 255 -> [NUM]; `ip`: netflow IPs -> [PRIVATE_IP]/[PUBLIC_IP]/[LOCALHOST_IP];             `port`: netflow ports -> [PORT_HTTPS], [PORT_SSH], ... or [PORT_LOW]/[PORT_REG]/[PORT_EPH].<br><br><b>Available options (multi selection)</b>:<br>`hash`<br>`guid`<br>`random`<br>`version`<br>`timestamp`<br>`env_var`<br>`number`<br>`ip`<br>`port`<br>`none`
9. Enable multi-level neighbor canonicalization. When True, decoder targets keep only structural/categorical tokens ([] special tokens + OS-agnostic category tokens like [CAT_WEBSERVER], [FCAT_LOG_WEB]) while encoder input gets full detail + category tokens prepended. During continue_pretrain, neighbors keep all tokens with category tokens prepended.<br>
10. HuggingFace id of the tokenizer (e.g. openai-community/gpt2). 'auto' uses the tokenizer of the HuggingFace pretrained model set in `featurization.pretrained` (gpt2_pretrained, llama3_pretrained, opt_pretrained).<br>
11. Maximum number of tokens per label. Longer labels are truncated.<br>
12. Variable values replaced by a placeholder token before tokenization, or `none`.             `hash`: hex strings of 7+ chars -> [HASH]; `guid`: GUIDs/UUIDs -> [GUID]; `random`: random-looking strings and Windows SIDs -> [RANDOM];             `version`: 1.2.3 -> [VER]; `timestamp`: words starting with a date -> [TIMESTAMP]; `env_var`: $HOME, %APPDATA% -> [ENV];             `number`: numbers above 255 -> [NUM]; `ip`: netflow IPs -> [PRIVATE_IP]/[PUBLIC_IP]/[LOCALHOST_IP];             `port`: netflow ports -> [PORT_HTTPS], [PORT_SSH], ... or [PORT_LOW]/[PORT_REG]/[PORT_EPH].<br><br><b>Available options (multi selection)</b>:<br>`hash`<br>`guid`<br>`random`<br>`version`<br>`timestamp`<br>`env_var`<br>`number`<br>`ip`<br>`port`<br>`none`
