---
title: Systems
description: The provenance-based and network intrusion detection systems reimplemented in PIDSMaker, their papers, the components each one uses at every stage of the pipeline, and which datasets each one runs on.
---

# Systems

PIDSMaker reimplements state-of-the-art intrusion detection systems as YAML configs in `config/`: **provenance-based IDSs (PIDSs)**, which detect attacks in the system events of hosts, and **network IDSs (NIDSs)**, which detect them in the communications between hosts. All of them are described with the same components, so they can be compared side by side, any component can be swapped into any other system, and most systems run on [both kinds of datasets](#running-across-families).
{ .lead }

## Supported systems

| System | Data | Config | Venue | Paper |
|---|---|---|---|---|
| **Velox** | <span class="pm-family pm-family--prov">Provenance</span> | `velox` | <span class="pm-chip">USENIX Security 2025</span> | [:lucide-file-text: Paper](https://tfjmp.org/publications/2025-usenixsec-2.pdf) |
| **Orthrus** | <span class="pm-family pm-family--prov">Provenance</span> | `orthrus`, `orthrus_non_snooped` | <span class="pm-chip">USENIX Security 2025</span> | [:lucide-file-text: Paper](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) |
| **OCR-APT** | <span class="pm-family pm-family--prov">Provenance</span> | `ocrapt` | <span class="pm-chip">ACM CCS 2025</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2510.15188) |
| **R-Caid** | <span class="pm-family pm-family--prov">Provenance</span> | `rcaid` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://gangw.web.illinois.edu/rcaid-sp24.pdf) |
| **Flash** | <span class="pm-family pm-family--prov">Provenance</span> | `flash` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://dartlab.org/assets/pdf/flash.pdf) |
| **Kairos** | <span class="pm-family pm-family--prov">Provenance</span> | `kairos` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2308.05034) |
| **MAGIC** | <span class="pm-family pm-family--prov">Provenance</span> | `magic` | <span class="pm-chip">USENIX Security 2024</span> | [:lucide-file-text: Paper](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf) |
| **NodLink** | <span class="pm-family pm-family--prov">Provenance</span> | `nodlink` | <span class="pm-chip">NDSS 2024</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2311.02331) |
| **ProGrapher** | <span class="pm-family pm-family--prov">Provenance</span> | `prographer` | <span class="pm-chip">USENIX Security 2023</span> | [:lucide-file-text: Paper](https://www.usenix.org/system/files/usenixsecurity23-yang-fan.pdf) |
| **ThreaTrace** | <span class="pm-family pm-family--prov">Provenance</span> | `threatrace` | <span class="pm-chip">IEEE TIFS 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/pdf/2111.04333) |
| **ShadeWatcher** | <span class="pm-family pm-family--prov">Provenance</span> | `shadewatcher` | <span class="pm-chip">IEEE S&P 2022</span> | [:lucide-file-text: Paper](https://jun-zeng.github.io/file/shadewatcher_paper.pdf) |
| **LARES** | <span class="pm-family pm-family--net">Network</span> | `lares` | <span class="pm-chip">ACSAC 2026</span> | [:lucide-file-text: Paper](https://ora.ox.ac.uk/objects/uuid:07b7d2a5-91f8-4c37-a90c-4ca615b428a7/files/ss7526g13z) |
| **Argus** | <span class="pm-family pm-family--net">Network</span> | `argus` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://doi.org/10.1109/SP54263.2024.00012) |
| **Jbeil** | <span class="pm-family pm-family--net">Network</span> | `jbeil` | <span class="pm-chip">IEEE S&P 2024</span> | [:lucide-file-text: Paper](https://doi.org/10.1109/SP54263.2024.00009) |
| **Euler** | <span class="pm-family pm-family--net">Network</span> | `euler` | <span class="pm-chip">NDSS 2022</span> | [:lucide-file-text: Paper](https://doi.org/10.14722/ndss.2022.24107) |
| **E-GraphSAGE** | <span class="pm-family pm-family--net">Network</span> | `egraphsage` | <span class="pm-chip">IEEE/IFIP NOMS 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/abs/2103.16329) |
| **Anomal-E** | <span class="pm-family pm-family--net">Network</span> | `anomal_e` | <span class="pm-chip">Knowledge-Based Systems 2022</span> | [:lucide-file-text: Paper](https://arxiv.org/abs/2207.06819) |

Run any of them with its config name and a dataset:

<div class="pm-families" markdown>

=== "Provenance"

    ```shell
    python pidsmaker/main.py kairos CADETS_E3
    ```

=== "Network"

    ```shell
    python pidsmaker/main.py euler LANL
    python pidsmaker/main.py lares OPTC_NET --tuned   # OpTC hyperparameters of LARES
    ```

</div>

All systems are self-supervised and trained on benign data only. PIDSs flag the entities (processes, files, netflows) whose activity deviates from what they learned; NIDSs flag the authentications or flows that their model finds unlikely. E-GraphSAGE, originally a supervised edge classifier, is trained here with self-supervised link prediction like the other NIDSs. The header of each config describes the system and its differences with the original implementation.

## Components at a glance

How each system is configured at the main stages of the [pipeline](pipeline.md):

| System | Featurization | Encoder | Objective | Evaluation | Threshold |
|---|---|---|---|---|---|
| Velox | `word2vec` | `none` (linear layer) | `predict_edge_type` | `node_evaluation` | `max_val_loss` |
| Orthrus | `word2vec` | `tgn`, `graph_attention` | `predict_edge_type` | `node_evaluation` | `max_val_loss` + k-means |
| Kairos | `hierarchical_hashing` | `graph_attention`, `tgn` | `predict_edge_type` | `node_evaluation` | `max_val_loss` |
| MAGIC | `only_type` | `magic_gat` | `reconstruct_masked_features`, `predict_masked_struct` | `node_evaluation` | `magic` |
| Flash | `flash` | `sage` | `predict_node_type` | `node_evaluation` | `flash` |
| R-Caid | `doc2vec` | `rcaid_gat` | `predict_process_tag` | `node_evaluation` | `max_val_loss` |
| NodLink | `fasttext` | `sum_aggregation` | `reconstruct_node_features` | `node_evaluation` | `nodlink` |
| ProGrapher | `graph2vec` | `textrcnn` | `predict_snapshot_embedding` | `tw_evaluation` (or `node_evaluation`) | `max_val_loss` |
| ThreaTrace | `only_type` | `sage` | `predict_node_type` | `node_evaluation` | `threatrace` |
| OCR-APT | `ocrapt_features` | `rgcn_per_type` | `one_class` | `node_evaluation` | `ocrapt` |
| ShadeWatcher | `word2vec` | `shadewatcher` (TransR + attentive GNN) | `predict_edge_contrastive` (`bpr` loss) | `node_evaluation` | `max_val_loss` |
| Euler | `only_type` | `gcn`, `node_id_embedding`, `snapshot_gru` | `predict_edge_contrastive` (uniform negatives) | `edge_evaluation` | `max_val_loss` |
| Argus | `only_type` | `argus`, `node_id_embedding`, `snapshot_gru` | `predict_edge_contrastive` (`ap` loss) | `edge_evaluation` | `max_val_loss` |
| Jbeil | `only_type` | `graph_attention`, `tgn` | `predict_edge_contrastive` (`tgn` negatives) | `edge_evaluation` | `max_val_loss` |
| LARES | `only_type` | `lares` | `autoencode_edge_embeddings` | `host_two_stage_evaluation` | k-means on the top host scores |
| E-GraphSAGE | `only_ones` | `egraphsage` | `predict_edge_contrastive` (uniform negatives) | `edge_evaluation` | `max_val_loss` |
| Anomal-E | `only_ones` | `egraphsage` | `deep_graph_infomax` (PCA scores) | `edge_evaluation` | `max_val_loss` |

The two families mostly differ at both ends of the pipeline. Most PIDSs embed the textual labels of entities (file paths, command lines, IP addresses), and all are evaluated per node, against the attack nodes of the ground truth. Hosts have no such labels, only a name or an ID, so NIDSs learn an embedding of the identity of each host (`node_id_embedding`, `lares`) and rely on the attributes of events; they are evaluated per edge, against the malicious events labelled in the dataset.

Other stages:

- **Transformation.** R-Caid builds its pseudo-graph (`rcaid_pseudo_graph`: the provenance graph plus an edge from each file or socket root to each of its descendants, at most 16 roots per node) and NodLink makes graphs undirected (`undirected`). Euler, Argus, LARES, E-GraphSAGE and Anomal-E merge the repeated edges of each time window (`merge_edges`); Argus and LARES aggregate the attributes of the merged events (`merge_edges.aggregate_attributes`). ShadeWatcher merges them per edge type (`merge_edges.per_edge_type`), so that each (entity, event type, entity) triple of its knowledge graph is unique.
- **Edge features.** Velox, Orthrus and OCR-APT use the edge type; Kairos and Jbeil the TGN messages (`msg,time_encoding`), which hold the types of the past edges in Kairos (`predict_edge_type.edge_type_in_msg`); Argus, LARES, E-GraphSAGE and Anomal-E the event attributes (`edge_attrs`).
- **Epoch selection.** PIDSs report the epoch with the best ADP (`best_adp`, the default). Euler, Jbeil and E-GraphSAGE select it without labels on validation link prediction (`best_val_score`), Argus and Anomal-E keep the last epoch (`last_epoch`), and LARES selects the best MCC, as its code. See [Choosing the reported epoch](evaluation.md#choosing-the-reported-epoch).
- **Triage.** OCR-APT extracts anomalous subgraphs from its detections (`triage.used_method: ocrapt_subgraph`).
- **Knowledge graph.** ShadeWatcher sees each edge both as an interaction between two entities, scored by the inner product of their embeddings, and as a triple of a knowledge graph whose relation is the edge type. Its encoder adds the loss of TransR on these triples to the BPR loss of the interactions (`predict_edge_contrastive` with `loss: bpr`). Entities are embedded from their features (`word2vec` and type) rather than one embedding per entity, as test entities are mostly unseen during training: add `node_id_embedding` to the encoders to also learn the identity of each entity.
- **Snapshot-level detection (ProGrapher).** ProGrapher embeds each time window as a whole with graph2vec (Weisfeiler-Lehman rooted subgraphs of node and edge types, embedded with doc2vec), and a TextRCNN predicts the embedding of each window from the previous ones (`textrcnn.sequence_length`): windows far from their prediction are anomalous. Its node scores rank the entities of each window by its key indicators, the rooted subgraphs that explain the deviation the most, so it is evaluated per window as in the paper (`tw_evaluation`, the default), or per node to compare its key indicators with the other PIDSs (`--evaluation.used_method=node_evaluation`, optionally with `--training.decoder.predict_snapshot_embedding.top_k=4`). Its graph2vec model is trained on the train split only; the paper learns the embeddings of all snapshots, test included (`--featurization.training_split=all`). Its epoch is selected without labels, on the mean prediction error of the validation windows (`best_val_score`).

The available methods for each column are listed in the reference: [Featurization](../reference/featurization.md), [Encoders](../reference/encoders.md), [Objectives](../reference/objectives.md) and [Tasks](../reference/tasks.md#stage-6-evaluation).

## Running across families

Datasets are described by a [schema](datasets.md#adding-custom-datasets) (node types, edge types and event attributes) rather than by assumptions on a particular dataset, so a system written for one family runs on the other:

```shell
python pidsmaker/main.py orthrus LANL           # a PIDS on a network dataset
python pidsmaker/main.py euler CLEARSCOPE_E3    # a NIDS on a provenance dataset
```

Two differences between the families limit this:

| | On provenance datasets | On network datasets |
|---|---|---|
| **PIDSs** | All | Velox, Orthrus, Kairos, MAGIC, OCR-APT and ShadeWatcher. Flash and ThreaTrace predict node types (`predict_node_type`), which needs at least two node types, and R-Caid the executables of processes (`predict_process_tag`): network datasets only have hosts. NodLink is not tested on network datasets. |
| **NIDSs** | Euler and Jbeil as is. Argus, LARES, E-GraphSAGE and Anomal-E use event attributes, which provenance datasets don't have: replace them by the edge type with `--batching.edge_features=edge_type`. | All |

The combinations of this table are those run by the tests (`tests/test_network_datasets.py`). A few useful options when moving across families:

- `--construction.edge_type=logon_type` uses a categorical attribute of the events as edge type, e.g. the logon type of LANL authentications as the class predicted by Velox's `predict_edge_type`.
- `--batching.edge_features=edge_type,edge_attrs` gives the event attributes (ports, protocols, logon types...) to a PIDS encoder that reads edge features, such as Orthrus.
- Systems keep their evaluation: a PIDS on LANL is evaluated per host (`node_evaluation`, where the malicious hosts are the endpoints of malicious events), a NIDS on CLEARSCOPE_E3 per event (`edge_evaluation`, where the malicious events are those between attack nodes, with `malicious_edge_selection: both_nodes`). Switch with `evaluation.used_method`.

## Variants

`orthrus_non_snooped`
:   Orthrus without data snooping, for fair comparisons: its word2vec model is trained on the training split only (`featurization.training_split: train` instead of `all`), and the k-means clustering after thresholding is disabled. Velox inherits from this config.

`velox`
:   `orthrus_non_snooped` with the GNN replaced by a single linear layer (`training.encoder.used_methods: none`) and without TGN neighbor sampling (`batching.intra_graph_batching.used_methods: edges`). It is the deliberately simple system proposed in [Bilot et al.](https://tfjmp.org/publications/2025-usenixsec-2.pdf).

`pretrained_<system>`
:   The provenance systems, with node features computed by a [pretrained encoder](../features/pretrained_encoders.md) such as SPIDER: `pretrained_velox`, `pretrained_orthrus`, `pretrained_kairos`, `pretrained_magic`, `pretrained_flash`, `pretrained_nodlink` and `pretrained_threatrace`.

`cybergfm`
:   The [CyberGFM](https://arxiv.org/abs/2601.05988) baseline: BERT pretrained on random walks, then fine-tuned as a detector with link prediction.

`--tuned` on `OPTC_NET`
:   The defaults of the network configs are those of LANL. `--tuned` loads the OpTC settings of LARES (30 min windows, α = 5, β = 30) and Argus (6 min snapshots), from `config/tuned_baselines/optc_net/`.

!!! warning "Data snooping"
    Some original configs rely on test data: for example, `orthrus` trains its word2vec model on all splits, test included (`featurization.training_split: all`), and `kairos` does too. PIDSMaker keeps the original configs for reproducibility, but use the non-snooped variants when comparing systems. See [Bilot et al.](https://www.usenix.org/system/files/usenixsecurity25-bilot.pdf) for a detailed analysis.

## Results and reproducibility

The default configs contain the original hyperparameters of each system. To find better ones on a dataset, use [hyperparameter tuning](../features/tuning.md).
Systems trained with self-supervision are [unstable](../features/instability.md), PIDSs and NIDSs alike: always evaluate them over several runs.

!!! warning "Comparing NIDSs with published results"
    - The `max_val_loss` threshold yields almost no true positives on the network datasets: AUC and AP are the meaningful metrics.
    - On `OPTC_NET`, link prediction on validation edges is close to perfect, so label-free epoch selection picks a nearly arbitrary epoch: report results over several epochs and [runs](../features/instability.md).
    - Euler and Argus evaluate OpTC on another preprocessing and labelling (IP addresses as nodes, all the events of compromised hosts malicious), so their published OpTC results are not comparable with `OPTC_NET`.

## Adding a system

Want to add your system? Follow the [tutorial](tutorial.md) to implement it with existing and new components, then [open a pull request](../community/contributing.md).
