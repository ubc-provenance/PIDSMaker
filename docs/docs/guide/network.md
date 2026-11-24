---
title: Network intrusion detection
description: The network intrusion detection systems (Euler, Argus, Jbeil, LARES, E-GraphSAGE, Anomal-E) and network datasets (LANL, OpTC flows) of PIDSMaker, and how to run any system on network or provenance datasets.
---

# Network intrusion detection

Besides PIDSs, PIDSMaker runs **network intrusion detection systems (NIDSs)** that model the communications between the hosts of a network, on the **LANL** authentications and the network flows of **DARPA OpTC**. They use the same seven [pipeline](pipeline.md) stages as PIDSs, so any component can be swapped between both families, and any system runs on network or provenance datasets.
{ .lead }

## Network systems

| System | Config | Venue | Paper |
|---|---|---|---|
| **LARES** | `lares` | <span class="pm-chip">ACSAC 2026</span> | [:lucide-code: Code](https://github.com/TristanBilot/lares) |
| **Argus** | `argus` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://doi.org/10.1109/SP54263.2024.00012) |
| **Jbeil** | `jbeil` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://doi.org/10.1109/SP54263.2024.00009) |
| **Euler** | `euler` | <span class="pm-chip">NDSS 2022</span> | [:lucide-file-text: Paper](https://doi.org/10.14722/ndss.2022.24107) |
| **E-GraphSAGE** | `egraphsage` | <span class="pm-chip">IEEE/IFIP NOMS 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/abs/2103.16329) |
| **Anomal-E** | `anomal_e` | <span class="pm-chip">Knowledge-Based Systems 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/abs/2207.06819) |

```shell
python pidsmaker/main.py euler LANL
python pidsmaker/main.py lares OPTC_NET --tuned   # OpTC hyperparameters of LARES
```

All of them are self-supervised and trained on benign events only: edges (authentications or flows) that the model finds unlikely are anomalous. E-GraphSAGE, originally a supervised edge classifier, is trained here with self-supervised link prediction like the other systems.

### Components

| System | Transformation | Encoder | Objective | Evaluation | Epoch selection |
|---|---|---|---|---|---|
| Euler | `merge_edges` | `gcn`, `node_id_embedding`, `snapshot_gru` | `predict_edge_contrastive` (uniform negatives) | `edge_evaluation` | `best_val_score` |
| Argus | `merge_edges` (aggregated attributes) | `argus`, `node_id_embedding`, `snapshot_gru` | `predict_edge_contrastive` (`ap` loss) | `edge_evaluation` | `last_epoch` |
| Jbeil | | `graph_attention`, `tgn` | `predict_edge_contrastive` (`tgn` negatives) | `edge_evaluation` | `best_val_score` |
| LARES | `merge_edges` (aggregated attributes) | `lares` | `autoencode_edge_embeddings` | `host_two_stage_evaluation` | `best_mcc` |
| E-GraphSAGE | `merge_edges` | `egraphsage` | `predict_edge_contrastive` (uniform negatives) | `edge_evaluation` | `best_val_score` |
| Anomal-E | `merge_edges` | `egraphsage` | `deep_graph_infomax` (PCA scores) | `edge_evaluation` | `last_epoch` |

The header of each config in `config/` describes the system and its differences with the original implementation. The OpTC settings of LARES (30 min windows, α = 5, β = 30, ...) and Argus (6 min snapshots) are in `config/tuned_baselines/optc_net/` and loaded with `--tuned`; the defaults of the configs are those of LANL.

## Network datasets

| Dataset | Events | Hosts | Labels |
|---|---|--:|---|
| [`LANL`](datasets.md#lanl) | 49.3M NTLM authentications | 15,610 | Red team events |
| [`OPTC_NET`](datasets.md#optc_net) | 1.86B network flows | 625 | Malicious flows labelled by LARES |

Nodes are hosts and edges are authentications or flows, with their attributes (logon types, ports, protocols, sizes...). Malicious events are labelled in the database, so the ground truth is event-level. See [Datasets](datasets.md#network-datasets) for their description, and [Building datasets from scratch](../getting-started/from-scratch.md#network-datasets-lanl-and-optc-flows) to build their databases.

## Building blocks

The network systems added general-purpose components, available to every system:

| Component | Arguments | Description |
|---|---|---|
| Dataset schemas | `pidsmaker/config/schemas.py` | Node types, edge types and event attributes of each family of datasets, referenced by the `schema` key of a dataset. |
| Edge type | `construction.edge_type` | Edge attribute used as edge type (default `operation`), e.g. `logon_type` on LANL. |
| Event selection | `construction.operations`, `construction.drop_noise_events` | Events kept to build the graphs (OpTC flows: FLOW START only, without broadcast noise, by default). |
| Benign training | `construction.benign_training` | Drops the malicious events of the train and val dates (on by default). |
| Edge attributes | `batching.edge_features=edge_attrs`, `feat_inference.edge_attributes` | Event attributes as edge features: one-hot categorical and log-scaled numeric attributes. |
| Edge merging | `transformation.used_methods=merge_edges` | Merges the repeated edges of a time window into weighted edges, optionally aggregating their attributes (counts per value, mean and std). |
| Snapshot encoders | `node_id_embedding`, `snapshot_gru` | Learned node ID embeddings, and a GRU over the successive embeddings of each node across graphs. |
| Negative sampling | `predict_edge_contrastive.neg_sampling`, `neg_ratio` | `shuffle_dst` (default), `uniform`, `uniform_dst` or `tgn` negative edges. |
| Host-centered evaluation | `evaluation.used_method=host_two_stage_evaluation` | The two-stage evaluation of LARES. See [Evaluation](evaluation.md#host-centered-evaluation-lares). |
| Label-free epoch selection | `evaluation.best_model_selection=best_val_score` or `last_epoch` | Selects the epoch without test labels. See [Choosing the reported epoch](evaluation.md#choosing-the-reported-epoch). |

All arguments are listed in the [Reference](../reference/tasks.md).

## Any system on any dataset

PIDSs run on the network datasets, and NIDSs on the provenance datasets:

```shell
python pidsmaker/main.py orthrus LANL
python pidsmaker/main.py euler CLEARSCOPE_E3
```

- Network datasets have a single node type (hosts), so objectives that predict node types (`predict_node_type`, used by Flash, R-Caid and ThreaTrace) can't be used on them. Edge-level objectives can: for example, `--construction.edge_type=logon_type` makes Velox predict the logon type of LANL authentications.
- Provenance datasets have no event attributes: replace `edge_attrs` by the edge type to run Argus, LARES, E-GraphSAGE and Anomal-E on them, with `--batching.edge_features=edge_type`.

!!! warning "Comparing with published results"
    - The `max_val_loss` threshold yields almost no true positives on the network datasets: AUC and AP are the meaningful metrics.
    - On `OPTC_NET`, link prediction on validation edges is close to perfect, so label-free epoch selection picks a nearly arbitrary epoch: report results over several epochs and [runs](../features/instability.md).
    - Euler and Argus evaluate OpTC on another preprocessing and labelling (IP addresses as nodes, all the events of compromised hosts malicious), so their published OpTC results are not comparable with `OPTC_NET`.
