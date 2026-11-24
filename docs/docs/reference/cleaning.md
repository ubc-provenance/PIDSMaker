---
title: Cleaning
description: Cleaning methods applied to the raw events before graph construction, and their effect on detection.
---

# Cleaning

The databases of the datasets contain events and attributes that are artifacts of the collection rather than system behavior: null values stored as text, corrupted IPs, events logged twice, events of a process on itself, multicast name resolution traffic, etc.
They are cleaned during the `construction` task, before the graphs are built, with `construction.cleaning`.

The cleaning methods were designed for provenance datasets. On network datasets, the methods that filter events (`remove_duplicate_events`, `remove_self_loops`, `dedup_window_edges`) apply to the events between hosts: E-GraphSAGE and Anomal-E remove self-loops. `remove_noisy_netflows` only applies to the netflow nodes of provenance datasets; the broadcast and multicast flows of `OPTC_NET` are dropped by `construction.drop_noise_events` instead.

`used_methods` is a comma-separated list of methods, or `none`. The default `none` builds the same graphs as the original systems, and keeps reusing the graphs already built.

| Method | Effect | Removes nodes |
|---|---|---|
| `remove_null_attributes` | Null-like attribute values (`None`, `NA`, `null`, `unknown`, `<UNNAMED fd:3>`, IPs that do not parse, ports outside 1-65535) are removed from node labels instead of becoming tokens such as `None` | no |
| `normalize_labels` | Trims labels, collapses whitespaces and repeated slashes, removes Windows path prefixes (`\\?\`, `//?/`) and quotes, optionally lowercases, and truncates labels longer than `max_length` characters | no |
| `remove_duplicate_events` | Events with the same source, destination, type and timestamp are kept once | no |
| `remove_self_loops` | Events from a node to itself are removed | only nodes without other events |
| `remove_noisy_netflows` | Netflow nodes whose remote IP is in `ip_categories` (multicast and 255.255.255.255 by default) are removed with their events | yes |
| `dedup_window_edges` | Only the first event of each (source, destination, type) is kept in each time window graph | no |

For example, Velox with the methods that do not remove attack nodes on any dataset:

```shell
python pidsmaker/main.py velox CADETS_E3 \
  --construction.cleaning.used_methods=remove_null_attributes,normalize_labels,remove_duplicate_events,remove_self_loops
```

```yaml
construction:
  cleaning:
    used_methods: remove_null_attributes, normalize_labels, remove_duplicate_events, remove_self_loops
    normalize_labels:
      collapse_whitespace: True
      collapse_slashes: True
      strip_path_prefixes: True
      strip_quotes: True
      lowercase: False
      max_length: 512
    remove_noisy_netflows:
      ip_categories: multicast, broadcast
```

The order of the methods has no effect. The construction folder only depends on the arguments of the used methods.

## What is cleaned in each dataset

The methods come from a profiling of the databases and of the graphs built from them.

- **Null values stored as text.** Node attributes that are missing in the database are stored as `None`, so that 98% of the files of CADETS_E3 are labeled `file None`. Netflows of ClearScope E3 have the IP `0` or `<UNNAMED fd:N>`, those of ClearScope E5 `None`, those of Theia E3 `NA`, and those of Trace E3 hexadecimal characters such as `E` and `D` as remote IP. `remove_null_attributes` removes them from the labels.
- **Label formatting.** Command lines of Theia contain runs of spaces (`/bin/sh -c    cd /`), paths contain repeated slashes (`/var/lib/apt//lists/`), Windows paths are prefixed (`//?/C:/Program Files/...`) or quoted, and some command lines reach 40,000 characters (Trace E3). `normalize_labels` gives the same label to these variants.
- **Duplicated events.** 2% (ATLASv2) to 55% (CADETS E3) of the events have the same source, destination, type and timestamp as another event. `fuse_edge` already merges most of them, but not with `fuse_edge: False`.
  Event UUIDs cannot be used to find duplicates: CADETS reuses them for unrelated events.
- **Self-loops.** ClearScope logs reads of processes on themselves (17,000 edges in the graphs of ClearScope E3, 60,000 in E5) and OpTC logs `TERMINATE` and `OPEN` events of processes on themselves (93,000 edges in OpTC h201).
- **Multicast traffic.** In OpTC h201, 3.1 of the 3.4 million netflow nodes are multicast name resolution (LLMNR, SSDP) to `224.0.0.252`, `ff02::1:3` or `239.255.255.250`.
- **Repeated events in a time window.** After `fuse_edge`, 20% (CADETS E3) to 73% (ClearScope E5) of the edges of a graph repeat the source, destination and type of an earlier edge of the same graph, e.g. reads of `/dev/ashmem`. `dedup_window_edges` keeps one of them. Encoders that use the order and number of events (e.g. TGN) lose this information.

## Effect on detection

Cleaning changes the input of the systems, so its effect depends on the system. Best ADP of single runs (seed 0) on CADETS_E3:

| Cleaning | Velox | Velox with SPIDER |
|---|---|---|
| `none` | 0.62 | 1.0 |
| `remove_null_attributes` | **0.97** | 0.62 |
| `normalize_labels` | 0.62 | 1.0 |
| `remove_duplicate_events` | 0.34 | 0.91 |
| `remove_self_loops` | 0.63 | 1.0 |
| `remove_noisy_netflows` | 0.62 | 1.0 |
| `dedup_window_edges` | 0.68 | 0.70 |
| all methods | **1.0** | 0.77 |

And of Velox on other datasets:

| Dataset | `none` | all methods |
|---|---|---|
| THEIA_E3 | 0.79 | 0.80 |
| CLEARSCOPE_E3 | 0.01 | 0.25 |
| optc_h201 | 0.12 | 0.33 (7 attack nodes removed) |

- `remove_null_attributes` helps featurization methods learned on the dataset: all processes of CADETS have a null path, so with `none` the token `None` is in the label of every process (`subject None sleep`).
- Pretrained encoders such as SPIDER were pretrained on labels that were not cleaned: methods that change the labels or the events change their input, and lower their results.
- `normalize_labels`, `remove_self_loops` and `remove_noisy_netflows` have no effect on CADETS_E3, which has few such labels and events.

As detectors are unstable (see [Instability](../features/instability.md)), measure the effect of cleaning on several runs before using it.

## Attack nodes

Methods that remove nodes can remove attack nodes, which are then not evaluated. The construction task writes `cleaning_report.json` in its folder, with the number of events and nodes removed by each method and `ground_truth_nodes_lost`, the number of attack nodes of the ground truth that are not in the graphs anymore. The ground truth is only read to write this report: it has no effect on cleaning.

`remove_noisy_netflows` is the only method that removed attack nodes in our datasets: 7 multicast netflow nodes in OpTC h201.

--8<-- "scripts/args/args_cleanings.md"
