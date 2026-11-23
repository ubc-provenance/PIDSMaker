---
title: Systems
description: The PIDSs reimplemented in PIDSMaker, their papers, and the components each one uses at every stage of the pipeline.
---

# Systems

PIDSMaker reimplements state-of-the-art PIDSs as YAML configs in `config/`. Since every system is described with the same components, they can be compared side by side, and any component can be swapped into any other system.
{ .lead }

## Supported systems

| System | Config | Venue | Paper |
|---|---|---|---|
| **Velox** | `velox` | <span class="pm-chip">USENIX Security 2025</span> | [:lucide-file-text: Paper](https://tfjmp.org/publications/2025-usenixsec-2.pdf) |
| **Orthrus** | `orthrus`, `orthrus_non_snooped` | <span class="pm-chip">USENIX Security 2025</span> | [:lucide-file-text: Paper](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) |
| **OCR-APT** | `ocrapt` | <span class="pm-chip">ACM CCS 2025</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2510.15188) |
| **R-Caid** | `rcaid` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://gangw.web.illinois.edu/rcaid-sp24.pdf) |
| **Flash** | `flash` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://dartlab.org/assets/pdf/flash.pdf) |
| **Kairos** | `kairos` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2308.05034) |
| **MAGIC** | `magic` | <span class="pm-chip">USENIX Security 2024</span> | [:lucide-file-text: Paper](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf) |
| **NodLink** | `nodlink` | <span class="pm-chip">NDSS 2024</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2311.02331) |
| **ThreaTrace** | `threatrace` | <span class="pm-chip">IEEE TIFS 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2111.04333) |

Run any of them with its config name:

```shell
python pidsmaker/main.py kairos CADETS_E3
```

## Components at a glance

How each system is configured at the main stages of the [pipeline](pipeline.md). All systems are evaluated at the node level (`node_evaluation`) by default.

| System | Featurization | Encoder | Objective | Threshold |
|---|---|---|---|---|
| Velox | `word2vec` | `none` (linear layer) | `predict_edge_type` | `max_val_loss` |
| Orthrus | `word2vec` | `tgn`, `graph_attention` | `predict_edge_type` | `max_val_loss` + k-means |
| Kairos | `hierarchical_hashing` | `graph_attention`, `tgn` | `predict_edge_type` | `max_val_loss` |
| MAGIC | `only_type` | `magic_gat` | `reconstruct_masked_features`, `predict_masked_struct` | `magic` |
| Flash | `flash` | `sage` | `predict_node_type` | `flash` |
| R-Caid | `doc2vec` | `rcaid_gat` | `predict_node_type` | `max_val_loss` |
| NodLink | `fasttext` | `sum_aggregation` | `reconstruct_node_features` | `nodlink` |
| ThreaTrace | `only_type` | `sage` | `predict_node_type` | `threatrace` |
| OCR-APT | `ocrapt_features` | `rgcn_per_type` | `one_class` | `ocrapt` |

A few systems also use specific transformations or post-processing: R-Caid builds its pseudo-graph (`transformation.used_methods: rcaid_pseudo_graph`), NodLink makes graphs undirected (`undirected`), and OCR-APT extracts anomalous subgraphs in the triage stage (`triage.used_method: ocrapt_subgraph`).

The available methods for each column are listed in the reference: [Featurization](../reference/featurization.md), [Encoders](../reference/encoders.md), [Objectives](../reference/objectives.md) and [Tasks](../reference/tasks.md#stage-6-evaluation).

## Variants

`orthrus_non_snooped`
:   Orthrus without data snooping, for fair comparisons: its word2vec model is trained on the training split only (`featurization.training_split: train` instead of `all`), and the k-means clustering after thresholding is disabled. Velox inherits from this config.

`velox`
:   `orthrus_non_snooped` with the GNN replaced by a single linear layer (`training.encoder.used_methods: none`) and without TGN neighbor sampling (`batching.intra_graph_batching.used_methods: edges`). It is the deliberately simple system proposed in [Bilot et al.](https://tfjmp.org/publications/2025-usenixsec-2.pdf).

`pretrained_<system>`
:   The same systems, with node features computed by a [pretrained encoder](../features/pretrained_encoders.md) such as SPIDER: `pretrained_velox`, `pretrained_orthrus`, `pretrained_kairos`, `pretrained_magic`, `pretrained_flash`, `pretrained_nodlink` and `pretrained_threatrace`.

`cybergfm`
:   The [CyberGFM](https://arxiv.org/abs/2601.05988) baseline: BERT pretrained on random walks, then fine-tuned as a detector with link prediction.

!!! warning "Data snooping"
    Some original configs rely on test data: for example, `orthrus` trains its word2vec model on all splits, test included (`featurization.training_split: all`), and `kairos` does too. PIDSMaker keeps the original configs for reproducibility, but use the non-snooped variants when comparing systems. See [Bilot et al.](https://www.usenix.org/system/files/usenixsecurity25-bilot.pdf) for a detailed analysis.

## Results and reproducibility

The default configs contain the original hyperparameters of each system. The best hyperparameters found by grid search for the main systems are in [Tuned hyperparameters](tuned-systems.md).
PIDSs are [unstable](../features/instability.md): always evaluate them over several runs.

## Adding a system

Want to add your PIDS? Follow the [tutorial](tutorial.md) to implement it with existing and new components, then [open a pull request](../community/contributing.md).
