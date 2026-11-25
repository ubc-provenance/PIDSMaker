---
title: Featurization
description: Featurization and tokenization methods that turn entity labels into node features, and their arguments.
---

# Featurization

Featurization methods are used to transform textual attributes of entities (e.g. file paths, process command lines, socket IP addresses and ports) into a vector.
Some methods like `word2vec` and `doc2vec` learn this vector from the text corpus, while others like `hierarchical_hashing` compute the vector in a deterministic way. 

Other methods like `only_type` and `only_ones` simply skip this embedding step and assign either a one-hot encoded type or ones to each entity. Those methods thus do not require any specific argument.
In all methods, the resulting vectors are used as node features during the `training` task.

The `graph2vec` method (ProGrapher) embeds each time-window graph as a whole rather than its nodes: the Weisfeiler-Lehman rooted subgraphs of all nodes, built from node and edge types up to depth `wl_iterations`, are the words of a document embedded with doc2vec (PV-DBOW). The model is trained on the snapshots of `training_split`, then the embedding of every snapshot is inferred (`infer_epochs`) and stored in its graph, with the rooted subgraphs of its nodes; nodes only get their type as features. It is used by the `textrcnn` encoder and the `predict_snapshot_embedding` objective.

The `pretrained` method uses an encoder pretrained on provenance data (SPIDER, language models, self-supervised GNNs, …), selected with `pretrained.model_type`. See [Pretrained Encoders](../features/pretrained_encoders.md).

!!! info "Network datasets"
    The nodes of network datasets are hosts, labelled by their name or ID (e.g. `C1065` in LANL). Text methods run on them (Velox uses `word2vec` on LANL), but these labels carry little information: the network systems use `only_type` or `only_ones`, and learn the identity of each host in their encoder (`node_id_embedding`, `lares`). `fasttext` stops with an error when no token appears `min_count` times, which happens with unique host names.

    The attributes of events (logon types, ports, protocols, sizes...) are not featurized here: `feat_inference` encodes them as edge features (one-hot categorical and log-scaled numeric attributes), used with `batching.edge_features=edge_attrs` and selected with `feat_inference.edge_attributes`.

## Methods

--8<-- "scripts/args/args_featurizations.md"

## Tokenization

Methods that learn from text first split the label of each entity into tokens. The tokenization is set in `featurization.tokenization`, independently of the featurization method:
`used_method` selects the tokenizer and the sub-dict of the same name sets its arguments. Each system config sets its own tokenization, so it is reproduced exactly, and any featurization method can use the tokenization of another system.

| Tokenization | Splits labels into | Used by |
|---|---|---|
| `nltk` | words, with NLTK, after separating paths on `/` and IPs on `:` and `.` (`type_aware: True`), or every label on `/`, `:` and `.` (`type_aware: False`) | Orthrus, R-Caid, NodLink (`type_aware: True`), FLASH (`type_aware: False`) |
| `hierarchical` | each level of the path or IP: `/usr/bin/ls` → `/usr`, `/usr/bin`, `/usr/bin/ls` | Kairos |
| `whole_label` | a single token per entity (type and label) | DeepWalk, Node2Vec |
| `bpe` | domain-specific pieces (path components, flags, ports, IP categories, …) and BPE subwords learned on the training labels | SPIDER and the other pretrained encoders |
| `huggingface` | the subwords of a HuggingFace tokenizer (`model_name`) | GPT-2, Llama and OPT pretrained encoders |
| `none` | no tokens | methods that do not read labels: `only_type`, `only_ones`, `magic`, `ocrapt_features`, `alacarte`, `graph2vec` |

For example, Orthrus with the BPE tokenization of SPIDER:

```shell
python pidsmaker/main.py orthrus CADETS_E3 --featurization.tokenization.used_method=bpe
```

`doc2vec`, `temporal_rw`, `flash` and `hierarchical_hashing` accept every tokenization except `none`. `word2vec` and `fasttext` also exclude `whole_label`, since they learn from one sentence per label.
The BPE vocabulary is learned on the same labels as the featurization method and saved with it.
The pretrained encoders are built around one tokenization, which must match `pretrained.model_type`.

```yaml
featurization:
  tokenization:
    used_method: nltk
    nltk:
      type_aware: True
      substitutions: none
```

### Substitutions

Some values change between otherwise identical entities: hashes, UUIDs, random temporary file names, PIDs, ephemeral ports.
Each tokenization sets in `substitutions` which of them are replaced by a placeholder token before tokenizing, so that such entities share their tokens.
It is a comma-separated list, or `none`:

| Substitution | Replaces | Placeholder |
|---|---|---|
| `hash` | hexadecimal strings of 7+ characters (MD5, SHA, commit ids) | `[HASH]` |
| `guid` | GUIDs / UUIDs, also inside a word | `[GUID]` |
| `random` | random-looking strings (mixed case and digits) and Windows SIDs | `[RANDOM]` |
| `version` | version numbers (`1.2.3`, `v2.0-rc1`) | `[VER]` |
| `timestamp` | words starting with a date (`2019-05-08...`) | `[TIMESTAMP]` |
| `env_var` | environment variables (`$HOME`, `${PATH}`, `%APPDATA%`) | `[ENV]` |
| `number` | numbers above 255 (PIDs, inodes); smaller ones like IP octets are kept | `[NUM]` |
| `ip` | IP addresses of netflow entities | `[PRIVATE_IP]`, `[PUBLIC_IP]`, `[LOCALHOST_IP]` |
| `port` | ports of netflow entities | `[PORT_HTTPS]`, `[PORT_SSH]`, … or `[PORT_LOW]`, `[PORT_REG]`, `[PORT_EPH]` |

The original systems use `none`, except SPIDER and the other BPE-based pretrained encoders, which use all of them. For example, Orthrus with hashes, random strings and large numbers replaced:

```shell
python pidsmaker/main.py orthrus CADETS_E3 --featurization.tokenization.nltk.substitutions="hash, random, number"
```

The `bpe` tokenization applies them during its domain-specific pre-tokenization (`mode: domain_bpe`), where they are checked on each path component and argument.
The other tokenizations apply them to whole words of the label, separated by spaces and by `/ \ : = , &` (by spaces and `ip:port` in netflow labels), before tokenizing it; placeholders are kept as single tokens.

--8<-- "scripts/args/args_tokenizations.md"
