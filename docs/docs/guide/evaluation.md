---
title: Evaluation & labeling
description: How PIDSMaker labels and evaluates the detections of provenance and network systems at the node, edge, time-window, queue and host levels, computes thresholds and metrics such as ADP, and selects the reported epoch.
---

# Evaluation & labeling

The same detector can look excellent or useless depending on **what** is counted as a detection, **how** it is labeled, and **which threshold** is used. This page explains each of these choices in PIDSMaker, for provenance and network datasets, so that results can be compared fairly.
{ .lead }

## From losses to metrics

<ol class="pm-steps" markdown>
<li markdown>**Losses.** After training, the model computes a loss for every edge (or node, for node-level objectives) of the validation and test graphs. This is done at epoch 0 and then every two epochs, and saved per time window.</li>
<li markdown>**Scores.** The evaluation method groups the losses into the **samples** it classifies (nodes, edges, time windows...) and reduces them into one anomaly score per sample.</li>
<li markdown>**Threshold.** A threshold is computed from the **validation** losses, which contain benign activity only. Samples scoring above it are predicted malicious.</li>
<li markdown>**Labels.** Each sample gets a ground-truth label, derived from the list of attack nodes of provenance datasets, or from the labels of the events of [network datasets](#event-level-labels-network-datasets).</li>
<li markdown>**Metrics.** Predictions are compared with the labels, for each evaluated epoch, and the metrics of the best epoch are reported.</li>
</ol>

The evaluation method is set with `evaluation.used_method`, and its options in the sub-section of the same name. Each family has its usual granularity, which `evaluation.used_method` changes for any system:

<div class="pm-families" markdown>

| | Provenance systems | Network systems |
|---|---|---|
| **Default evaluation** | `node_evaluation`: is each process, file or netflow part of an attack? | `edge_evaluation`: is each authentication or flow malicious? `host_two_stage_evaluation` for LARES: which hosts are the sources of the attack, and which of their edges are malicious? |
| **Ground truth** | Attack nodes, in [ground-truth files](#ground-truth) | Malicious events, [labelled in the database](#event-level-labels-network-datasets) |
| **Reported epoch** | Best ADP (`best_adp`) | Chosen without labels (`best_val_score`, `last_epoch`), or best MCC for LARES |

</div>

See the [settings of each system](#settings-of-each-system).

## Evaluation granularities

The ground truth lists attack **nodes** (provenance datasets) or malicious **events** (network datasets). Each evaluation method turns it into labels for the unit it classifies:

<div class="pm-eval-grid">
<figure class="pm-eval-card"><svg viewBox="0 0 240 152" role="img"><rect class="win" x="8" y="22" width="68" height="86" rx="8"/><text class="lbl" x="42.0" y="16" text-anchor="middle">TW1</text><rect class="win" x="86" y="22" width="68" height="86" rx="8"/><text class="lbl" x="120.0" y="16" text-anchor="middle">TW2</text><rect class="win" x="164" y="22" width="68" height="86" rx="8"/><text class="lbl" x="198.0" y="16" text-anchor="middle">TW3</text><line class="e" x1="32.0" y1="48.0" x2="51.0" y2="48.0"/><line class="e" x1="27.3" y1="55.3" x2="38.3" y2="79.8"/><line class="e" x1="56.7" y1="55.3" x2="45.7" y2="79.8"/><circle class="n" cx="24" cy="48" r="7"/><circle class="n" cx="60" cy="48" r="7"/><circle class="n" cx="42" cy="88" r="7"/><line class="e" x1="110.0" y1="48.0" x2="129.0" y2="48.0"/><line class="e" x1="105.3" y1="55.3" x2="116.3" y2="79.8"/><line class="e" x1="134.7" y1="55.3" x2="123.7" y2="79.8"/><circle class="n" cx="102" cy="48" r="7"/><circle class="n" cx="138" cy="48" r="7"/><circle class="m" cx="120" cy="88" r="7"/><line class="e" x1="188.0" y1="48.0" x2="207.0" y2="48.0"/><line class="e" x1="183.3" y1="55.3" x2="194.3" y2="79.8"/><line class="e" x1="212.7" y1="55.3" x2="201.7" y2="79.8"/><circle class="n" cx="180" cy="48" r="7"/><circle class="n" cx="216" cy="48" r="7"/><circle class="m" cx="198" cy="88" r="7"/><path class="unit-m" style="fill:none" d="M120,97 C120,128 198,128 198,97"/><text class="note" x="120.0" y="146" text-anchor="middle">same node = 1 sample</text></svg><figcaption><strong>Node <code>node_evaluation</code></strong>One sample per node over the whole test set. Its score is the max loss of its edges in all windows. Malicious if the node is in the ground truth.</figcaption></figure>
<figure class="pm-eval-card"><svg viewBox="0 0 240 152" role="img"><rect class="win" x="8" y="22" width="68" height="86" rx="8"/><text class="lbl" x="42.0" y="16" text-anchor="middle">TW1</text><rect class="win" x="86" y="22" width="68" height="86" rx="8"/><text class="lbl" x="120.0" y="16" text-anchor="middle">TW2</text><rect class="win" x="164" y="22" width="68" height="86" rx="8"/><text class="lbl" x="198.0" y="16" text-anchor="middle">TW3</text><circle class="unit-m" cx="120" cy="88" r="12"/><circle class="unit-m" cx="198" cy="88" r="12"/><line class="e" x1="32.0" y1="48.0" x2="51.0" y2="48.0"/><line class="e" x1="27.3" y1="55.3" x2="38.3" y2="79.8"/><line class="e" x1="56.7" y1="55.3" x2="45.7" y2="79.8"/><circle class="n" cx="24" cy="48" r="7"/><circle class="n" cx="60" cy="48" r="7"/><circle class="n" cx="42" cy="88" r="7"/><line class="e" x1="110.0" y1="48.0" x2="129.0" y2="48.0"/><line class="e" x1="105.3" y1="55.3" x2="116.3" y2="79.8"/><line class="e" x1="134.7" y1="55.3" x2="123.7" y2="79.8"/><circle class="n" cx="102" cy="48" r="7"/><circle class="n" cx="138" cy="48" r="7"/><circle class="m" cx="120" cy="88" r="7"/><line class="e" x1="188.0" y1="48.0" x2="207.0" y2="48.0"/><line class="e" x1="183.3" y1="55.3" x2="194.3" y2="79.8"/><line class="e" x1="212.7" y1="55.3" x2="201.7" y2="79.8"/><circle class="n" cx="180" cy="48" r="7"/><circle class="n" cx="216" cy="48" r="7"/><circle class="m" cx="198" cy="88" r="7"/><text class="note" x="120.0" y="146" text-anchor="middle">same node = 2 samples</text></svg><figcaption><strong>Node × time window <code>node_tw_evaluation</code></strong>One sample per (node, window) pair: the same node in two windows counts twice. Malicious if the node is malicious in that window.</figcaption></figure>
<figure class="pm-eval-card"><svg viewBox="0 0 240 152" role="img"><rect class="win" x="8" y="22" width="68" height="86" rx="8"/><text class="lbl" x="42.0" y="16" text-anchor="middle">TW1</text><rect class="win" x="86" y="22" width="68" height="86" rx="8"/><text class="lbl" x="120.0" y="16" text-anchor="middle">TW2</text><rect class="win" x="164" y="22" width="68" height="86" rx="8"/><text class="lbl" x="198.0" y="16" text-anchor="middle">TW3</text><line class="e" x1="32.0" y1="48.0" x2="51.0" y2="48.0"/><line class="e" x1="27.3" y1="55.3" x2="38.3" y2="79.8"/><line class="e" x1="56.7" y1="55.3" x2="45.7" y2="79.8"/><circle class="n" cx="24" cy="48" r="7"/><circle class="n" cx="60" cy="48" r="7"/><circle class="n" cx="42" cy="88" r="7"/><line class="e" x1="110.0" y1="48.0" x2="129.0" y2="48.0"/><line class="em" x1="105.3" y1="55.3" x2="116.3" y2="79.8"/><line class="em" x1="134.7" y1="55.3" x2="123.7" y2="79.8"/><circle class="n" cx="102" cy="48" r="7"/><circle class="n" cx="138" cy="48" r="7"/><circle class="m" cx="120" cy="88" r="7"/><line class="e" x1="188.0" y1="48.0" x2="207.0" y2="48.0"/><line class="em" x1="183.3" y1="55.3" x2="194.3" y2="79.8"/><line class="em" x1="212.7" y1="55.3" x2="201.7" y2="79.8"/><circle class="n" cx="180" cy="48" r="7"/><circle class="n" cx="216" cy="48" r="7"/><circle class="m" cx="198" cy="88" r="7"/><text class="note" x="120.0" y="146" text-anchor="middle">each edge = 1 sample</text></svg><figcaption><strong>Edge <code>edge_evaluation</code></strong>One sample per event, scored by its own loss. Malicious if its source, destination or both are attack nodes, as set by <code>malicious_edge_selection</code>.</figcaption></figure>
<figure class="pm-eval-card"><svg viewBox="0 0 240 152" role="img"><rect class="win" x="8" y="22" width="68" height="86" rx="8"/><text class="lbl" x="42.0" y="16" text-anchor="middle">TW1</text><rect class="win" x="86" y="22" width="68" height="86" rx="8"/><text class="lbl" x="120.0" y="16" text-anchor="middle">TW2</text><rect class="win" x="164" y="22" width="68" height="86" rx="8"/><text class="lbl" x="198.0" y="16" text-anchor="middle">TW3</text><rect class="unit" x="5" y="19" width="74" height="92" rx="10"/><rect class="unit-m" x="83" y="19" width="74" height="92" rx="10"/><rect class="unit-m" x="161" y="19" width="74" height="92" rx="10"/><line class="e" x1="32.0" y1="48.0" x2="51.0" y2="48.0"/><line class="e" x1="27.3" y1="55.3" x2="38.3" y2="79.8"/><line class="e" x1="56.7" y1="55.3" x2="45.7" y2="79.8"/><circle class="n" cx="24" cy="48" r="7"/><circle class="n" cx="60" cy="48" r="7"/><circle class="n" cx="42" cy="88" r="7"/><line class="e" x1="110.0" y1="48.0" x2="129.0" y2="48.0"/><line class="e" x1="105.3" y1="55.3" x2="116.3" y2="79.8"/><line class="e" x1="134.7" y1="55.3" x2="123.7" y2="79.8"/><circle class="n" cx="102" cy="48" r="7"/><circle class="n" cx="138" cy="48" r="7"/><circle class="m" cx="120" cy="88" r="7"/><line class="e" x1="188.0" y1="48.0" x2="207.0" y2="48.0"/><line class="e" x1="183.3" y1="55.3" x2="194.3" y2="79.8"/><line class="e" x1="212.7" y1="55.3" x2="201.7" y2="79.8"/><circle class="n" cx="180" cy="48" r="7"/><circle class="n" cx="216" cy="48" r="7"/><circle class="m" cx="198" cy="88" r="7"/><text class="note" x="120.0" y="146" text-anchor="middle">each window = 1 sample</text></svg><figcaption><strong>Time window <code>tw_evaluation</code></strong>One sample per time-window graph, scored by reducing the losses of its edges. Malicious if it contains at least one attack event.</figcaption></figure>
<figure class="pm-eval-card"><svg viewBox="0 0 240 152" role="img"><rect class="win" x="8" y="22" width="68" height="86" rx="8"/><text class="lbl" x="42.0" y="16" text-anchor="middle">TW1</text><rect class="win" x="86" y="22" width="68" height="86" rx="8"/><text class="lbl" x="120.0" y="16" text-anchor="middle">TW2</text><rect class="win" x="164" y="22" width="68" height="86" rx="8"/><text class="lbl" x="198.0" y="16" text-anchor="middle">TW3</text><rect class="unit-m" x="82" y="18" width="154" height="94" rx="11"/><line class="e" x1="32.0" y1="48.0" x2="51.0" y2="48.0"/><line class="e" x1="27.3" y1="55.3" x2="38.3" y2="79.8"/><line class="e" x1="56.7" y1="55.3" x2="45.7" y2="79.8"/><circle class="n" cx="24" cy="48" r="7"/><circle class="n" cx="60" cy="48" r="7"/><circle class="n" cx="42" cy="88" r="7"/><line class="e" x1="110.0" y1="48.0" x2="129.0" y2="48.0"/><line class="e" x1="105.3" y1="55.3" x2="116.3" y2="79.8"/><line class="e" x1="134.7" y1="55.3" x2="123.7" y2="79.8"/><circle class="n" cx="102" cy="48" r="7"/><circle class="n" cx="138" cy="48" r="7"/><circle class="m" cx="120" cy="88" r="7"/><line class="e" x1="188.0" y1="48.0" x2="207.0" y2="48.0"/><line class="e" x1="183.3" y1="55.3" x2="194.3" y2="79.8"/><line class="e" x1="212.7" y1="55.3" x2="201.7" y2="79.8"/><circle class="n" cx="180" cy="48" r="7"/><circle class="n" cx="216" cy="48" r="7"/><circle class="m" cx="198" cy="88" r="7"/><text class="note" x="120.0" y="146" text-anchor="middle">TW2 + TW3 = 1 queue</text></svg><figcaption><strong>Queue <code>queue_evaluation</code></strong>Kairos: anomalous windows that share rare nodes are chained into a queue. All windows of a queue whose score exceeds <code>queue_threshold</code> are flagged.</figcaption></figure>
<div class="pm-eval-card pm-eval-legend"><span><i class="ben"></i>benign node</span><span><i class="mal"></i>attack node (ground truth)</span><span><i class="unit"></i>one sample, labeled benign</span><span><i class="unit unit-mal"></i>one sample, labeled malicious</span></div>
</div>


| `evaluation.used_method` | Sample | Score | Label: malicious if... |
|---|---|---|---|
| `node_evaluation` | A node, over the whole test set | Max (or mean, or 95th percentile) of the losses of its edges, as source (and as destination with `use_dst_node_loss`) | the node is in the ground truth |
| `node_tw_evaluation` | A (node, time window) pair | Same reduction, over the node's losses in that window | the node is an attack node of that window |
| `edge_evaluation` | An edge (event) | The loss of the edge | its source, destination or both are attack nodes, during an attack (`malicious_edge_selection`: `src_node`, `dst_node`, `both_nodes`); on network datasets, one of its events is labelled malicious |
| `tw_evaluation` | A time-window graph | Reduction of all the losses of the window | it contains at least one attack event |
| `queue_evaluation` | A time-window graph, through its queue | Score of the queue that contains it | it contains at least one attack event |
| `host_two_stage_evaluation` | An edge, predicted by the top-scored hosts | The loss of the edge | as `edge_evaluation` with `both_nodes`, or its event label (see [below](#host-centered-evaluation-lares)) |

!!! info "Which samples are evaluated?"
    Only samples seen during inference are evaluated: a node that never appears in the test graphs is neither scored nor counted as a false negative. Empty time windows are skipped.

`node_tw_evaluation` is closer to real-time detection: an alert must happen in the window of the attack, and a node that is benign most of the time and malicious in one window counts as one true positive and several true negatives.

### Queues (Kairos)

`queue_evaluation` with `used_method: kairos_idf_queue` reproduces the detection of Kairos, which requires edge-level losses:

1. In each test window, edges with a loss above the window's mean + 1.5 × std are anomalous. The window's loss is the mean loss of these edges.
2. Windows are chained into **queues** when they share a **rare** node, measured by its IDF over the graphs (network nodes are ignored). `include_test_set_in_IDF` also counts the test graphs.
3. The score of a queue is the product of (loss + 1) over its windows. All windows of queues scoring above `queue_threshold` are flagged.

The `provnet_lof_queue` method is not supported anymore.

### Host-centered evaluation (LARES)

`host_two_stage_evaluation` reproduces the two-stage evaluation of LARES, on edge-level losses:

1. Each host is scored by the maximum loss of its outgoing edges over the test period. The `alpha` highest host scores are split into two clusters with k-means, and the hosts of the high-score cluster are selected, along with the host ranked just below them.
2. The `beta` highest-scored outgoing edges of each selected host are predicted malicious.

With `root_attack_nodes`, only the malicious edges of these source hosts are positives, as in LARES (`dataset` uses the root sources of the dataset: `9660` on LANL, `201,402,501` on OPTC_NET). `test_end` ends the evaluated period earlier (LARES evaluates LANL on its first 14 days). Precision, recall and MCC come from the predictions of the two stages, AUC and AP from the edge losses; the rank of each root host is also reported. No threshold is used.

## Ground truth

Evaluation compares predictions against a **ground truth**. For provenance datasets, it is the UUIDs of the nodes involved in each attack, described below; network datasets label their events instead ([event-level labels](#event-level-labels-network-datasets)). The labeling of the DARPA TC attacks differs between prior works, and the choice of ground truth materially affects the reported metrics, so PIDSMaker ships several, selected with `evaluation.ground_truth_version` (default `orthrus`):

=== "CLI"

    ```shell
    python pidsmaker/main.py SYSTEM CADETS_E3 --evaluation.ground_truth_version threatrace
    ```

=== "YAML"

    ```yaml
    evaluation:
      ground_truth_version: threatrace
    ```

| Version | Source | Format | Coverage |
|---|---|---|---|
| `orthrus` *(default)* | Labeling used by [Orthrus](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) | One CSV per attack, rows of `uuid, {labels}, count` | All provenance datasets |
| `reapr` | Alternative labeling of the DARPA TC E3 datasets | Same as `orthrus` | E3: CADETS, THEIA, TRACE, FIVEDIRECTIONS |
| `threatrace` | Ground truth published by [ThreaTrace](https://arxiv.org/pdf/2111.04333) | One flat UUID-per-line `.txt` per dataset, treated as a single attack over the whole test period | E3: CADETS, THEIA, TRACE, FIVEDIRECTIONS |

The files live under `Ground_Truth/<version>/<dataset>/`. Changing the version only changes the labels read by evaluation: it doesn't change training or the graphs, so it re-evaluates an already-trained run.

- **Attacks.** Each dataset defines its attacks in `attack_to_time_window` (`DATASET_DEFAULT_CONFIG`, `pidsmaker/config/config.py`): one ground-truth file and one time range per attack. Attacks are what ADP and the per-attack metrics count.
- **Node labels.** A node is malicious if it appears in any attack file. UUIDs of the ground truth that are not in the graphs are skipped, and counted in the logs.
- **Window labels.** A time window is malicious if it contains a database event, inside an attack's time range, whose source or destination is an attack node.

Selecting a version that has no file for the dataset fails with a clear error. See [`Ground_Truth/threatrace/README.md`](https://github.com/ubc-provenance/PIDSMaker/blob/main/Ground_Truth/threatrace/README.md) for the per-dataset ThreaTrace node counts.

### Event-level labels (network datasets)

Datasets whose schema has a label column, such as [LANL and OPTC_NET](datasets.md#network-datasets), label their events in the database instead of shipping ground-truth files, and `ground_truth_version` has no effect on them. Edges carry the labels of their events (also after `merge_edges`), malicious nodes are the endpoints of malicious events, and, as these datasets define no `attack_to_time_window`, all malicious events form a single attack over the test period.

## Thresholds

`threshold_method` (in the section of the evaluation method) sets both how losses are reduced into a score and how the threshold is computed. Thresholds are computed on the **validation** losses unless stated otherwise.

| `threshold_method` | Score of a sample | Threshold | Used by |
|---|---|---|---|
| `max_val_loss` | Max loss | Max validation loss × `alpha` (`node_evaluation.max_val_loss.alpha`, node and edge evaluation only) | Velox, Orthrus, Kairos, R-Caid, network systems except LARES |
| `mean_val_loss` | Mean loss | Mean validation loss | |
| `percentile` | 95th percentile of the losses | Max validation loss × `alpha` | |
| `nodlink` | Max loss | 90th percentile of the validation losses | NodLink |
| `threatrace` | Max over time windows of log(1.5) − log(p_true/p_other), where p_true is the probability of the true node type and p_other the highest of the others; with `training.used_method: threatrace`, the min over the submodels | 0: flagged unless a model predicts the true node type with a probability ratio above 1.5 | ThreaTrace |
| `flash` | Max over time windows of 1 − c if the node type is predicted correctly and 1 + c otherwise, with c = (p₁ − p₂)/p₁ on the node type probabilities, min-max normalized per window. With `predict_node_type.flash_snapshots`, the min over the snapshots of the epochs so far | Flagged unless predicted correctly with c > 0.53 (score ≥ 0.47) | Flash |
| `magic` | Max over time windows of the mean distance to the `magic_knn_k` (10) nearest benign training embeddings, divided by the mean k-NN distance of the training nodes | Mean score over the test nodes | MAGIC |
| `ocrapt` | Max loss | Per node type: the top fraction of scores, set by `ocrapt_contamination` and `ocrapt_min_contamination` | OCR-APT |
| `fixed_zero` | Max loss | 0 (decision boundary of a binary classifier) | [Supervised](../features/supervised.md) objective |

`threatrace` and `flash` require the `predict_node_type` objective, and `magic` requires node-level evaluation with a node-level objective.

With node-level objectives, the scores of `threatrace`, `flash` and `magic` are computed at inference, and each node is flagged at evaluation, by the `NodeScorer` of the system (in `pidsmaker/detection/threatrace_detection.py`, `flash_detection.py` and `magic_knn.py`). A new threshold method with scores of its own subclasses `NodeScorer` (`pidsmaker/detection/node_scoring.py`) and registers it in `node_scorer_factory()` (`factory.py`).

### Clustering (Orthrus)

With `use_kmeans: True`, the threshold doesn't decide alone: the `kmeans_top_K` highest-scored nodes are split into two clusters with k-means on their scores, and only the nodes of the high-score cluster are predicted malicious. With `node_tw_evaluation`, clustering runs in each window that has at least one node above the threshold. Orthrus enables it with K = 30; `orthrus_non_snooped` disables it.

## Metrics

Every evaluation method reports the standard classification metrics, computed from the predictions (above the threshold) or directly from the scores:

| Metric | Meaning |
|---|---|
| `precision`, `recall`, `fscore`, `fpr`, `mcc`, `accuracy`, `balanced_acc` | Classification metrics of the thresholded predictions |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix |
| `ap`, `auc` | Average precision and ROC AUC of the scores: they don't depend on the threshold |
| `precision@K`, `recall@K`, `ap@K` | The same, among the K highest-scored samples (K = 10, 50, 100) |

Node and edge evaluation also report attack-centric metrics. They matter because attacks are tiny compared to benign activity: detecting one node of each attack with few false positives is often more useful than a high recall.

`adp_score`
:   **Attack Detection Precision.** Samples are ranked by score. At each rank, PIDSMaker computes the precision and the percentage of attacks with at least one detected node. ADP is the area under the curve of the attacks detected against precision: 1.0 means that every attack is detected before any false positive.

`discrimination`
:   For each attack, the gap between its highest score and the mean of the 10 highest benign scores (scores normalized by their max), averaged over attacks. Positive when attacks stand out from the most suspicious benign nodes.

`*_if_all_attacks_detected`
:   The precision, recall, TPs and FPs obtained with the lowest threshold that still detects every attack.

`percent_detected_attacks`, `tp_att_<i>`
:   The fraction of attacks detected by the threshold, and the true positives of each attack (node evaluation).

## Choosing the reported epoch

Every evaluated epoch is logged, and the metrics of the best one are reported, as selected by `evaluation.best_model_selection`:

| Value | Best epoch |
|---|---|
| `best_adp` *(default)* | Highest ADP, ties broken by discrimination |
| `best_discrimination` | Highest discrimination |
| `best_ap@10` | Highest AP among the 10 highest scores, ties broken by discrimination |
| `best_mcc` | Highest MCC, as in the code of LARES |
| `best_val_score` | Highest score on validation data computed by the objective, without labels: the AUC + AP of link prediction on the validation edges with `predict_edge_contrastive` (Euler's early stopping) |
| `last_epoch` | Last evaluated epoch, without labels |

With `best_adp`, `best_discrimination` and `best_ap@10`, time-window evaluations, which have no ADP, use the highest AP. Queue evaluation uses the highest precision, then recall.

With a label-free selection (`best_val_score` or `last_epoch`), `evaluation.evaluate_selected_epoch_only=True` evaluates only the selected epoch on the test set, instead of every epoch: faster on large test sets, but without per-epoch test metrics.

!!! warning "Most selections use test metrics"
    `best_adp`, `best_discrimination`, `best_ap@10` and `best_mcc` select the epoch on the metrics of the test set. Keep this in mind when comparing with systems that select their model on validation data only (`best_val_score`, `last_epoch`), and evaluate over [several runs](../features/instability.md).

## Outputs

Results are written to the `precision_recall_dir/` folder of the evaluation task, for each epoch: the scores and predictions (`scores_*.pkl`, `result_*.pth`, `stats_*.pth`) and figures, also sent to W&B:

- `scores_*.png` and `neat_scores_*.svg`: the score of every node, with attack nodes highlighted and the threshold.
- `adp_curve_*.png`: the curve whose area is the ADP.
- `discrim_curve_*.png` and `distrib_*.png`: the distribution of scores per node type.

Figures with one point per node or edge are slow to draw on large test sets: `evaluation.save_figures=False` skips them, and the metrics are computed either way. The network systems disable them by default.

## Settings of each system

| System | Evaluation | `threshold_method` | Clustering |
|---|---|---|---|
| Orthrus | `node_evaluation` | `max_val_loss` | k-means, K = 30 |
| Velox, `orthrus_non_snooped` | `node_evaluation` | `max_val_loss` | |
| Kairos | `node_evaluation` | `max_val_loss` | |
| R-Caid | `node_evaluation` | `max_val_loss` | |
| MAGIC | `node_evaluation` | `magic` | |
| Flash | `node_evaluation` | `flash` | |
| NodLink | `node_evaluation` | `nodlink` | |
| ProGrapher | `tw_evaluation` (`node_evaluation` from its key indicators) | `max_val_loss` | |
| ThreaTrace | `node_evaluation` | `threatrace` | |
| OCR-APT | `node_evaluation` | `ocrapt` | |
| Euler, Argus, Jbeil, E-GraphSAGE, Anomal-E | `edge_evaluation` (`both_nodes`) | `max_val_loss` | |
| LARES | `host_two_stage_evaluation` | | k-means on the top host scores (α = 3) |

Any system can be evaluated differently from the CLI, for example Kairos at the time-window level:

```shell
python pidsmaker/main.py kairos CADETS_E3 \
  --evaluation.used_method=tw_evaluation \
  --evaluation.tw_evaluation.threshold_method=max_val_loss
```

All arguments are listed in the [Tasks reference](../reference/tasks.md#stage-6-evaluation).
