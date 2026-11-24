"""Two-stage host-centered evaluation of edge scores, as in LARES (Bilot et al., ACSAC 2026).

1. Hosts are scored by the maximum score (loss) of their outgoing edges over the test period.
   The top `alpha` host scores are clustered in two with K-means, and the hosts of the cluster
   with the highest scores are selected, along with the host ranked just below them.
2. The `beta` highest-scored outgoing edges of each selected host are predicted malicious.

Edges are labelled by the label of their events (`y` of the loss files, i.e. datasets with
event labels such as LANL or OpTC flows). On datasets with node-level ground truth (e.g.
provenance datasets), an edge is malicious when both its nodes are malicious. With `root_attack_nodes`, only the malicious edges of
these sources are positives, as LARES does with the root sources of the attacks of each dataset
(e.g. host 9660 for LANL). `test_end` restricts the test period (LARES only evaluates LANL on its
first 14 days, i.e. until 2015-01-15 06:05:15 with the synthetic dates of the LANL loader).

Edge-level AUC and AP are computed from the raw edge scores, and precision, recall and MCC from
the predictions of the two stages.
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from pidsmaker.utils import labelling
from pidsmaker.utils.loss_files import read_losses
from pidsmaker.utils.utils import datetime_to_ns_time_US, listdir_sorted, log


def _load_edges(test_tw_path, cfg):
    event_labels = labelling.has_event_labels(cfg)
    cols = ["loss", "srcnode", "dstnode", "time"] + (["y"] if event_labels else [])
    df = pd.concat(
        [read_losses(os.path.join(test_tw_path, f), usecols=cols) for f in listdir_sorted(test_tw_path)],
        ignore_index=True,
    )
    if not event_labels:
        malicious = np.array([int(n) for n in labelling.get_ground_truth(cfg)[0]], dtype=np.int64)
        df["y"] = (np.isin(df.srcnode, malicious) & np.isin(df.dstnode, malicious)).astype(int)
    return df


def select_hosts(host_scores, alpha):
    """Hosts of the top K-means cluster of the `alpha` highest scores, plus the next host."""
    hosts = np.array(list(host_scores.keys()))
    scores = np.array(list(host_scores.values()), dtype=float)
    order = np.argsort(scores, kind="stable")
    top_scores, top_hosts = scores[order][-alpha:], hosts[order][-alpha:]
    if len(top_scores) < 2:
        return list(top_hosts)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(top_scores.reshape(-1, 1))
    high = np.argmax(km.cluster_centers_.flatten())
    idx = np.where(km.labels_ == high)[0]
    if idx[0] > 0:
        idx = np.concatenate([[idx[0] - 1], idx])
    return list(top_hosts[idx][::-1])


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, **kwargs):
    eval_cfg = cfg.evaluation.host_two_stage_evaluation
    alpha = int(eval_cfg.alpha or 3)
    beta = int(eval_cfg.beta or 350)
    roots = str(eval_cfg.root_attack_nodes or "").strip()
    if roots == "dataset":
        roots = list(getattr(cfg.dataset, "root_attack_nodes", None) or [])
        if not roots:
            log(f"No `root_attack_nodes` for dataset {cfg.dataset.name}: all malicious edges are positives")
    else:
        roots = [int(n) for n in roots.split(",") if n.strip()]
    test_end = (eval_cfg.test_end or "").strip()

    df = _load_edges(test_tw_path, cfg)
    if test_end:
        end_ns = datetime_to_ns_time_US(test_end, timezone=cfg.dataset.timezone)
        df = df[df.time < end_ns].reset_index(drop=True)

    src = df.srcnode.to_numpy().astype(np.int64)
    scores = df.loss.to_numpy()
    y = df.y.to_numpy().clip(max=1).astype(int)
    if roots:
        y = y * np.isin(src, roots)

    host_scores = df.groupby("srcnode").loss.max().to_dict()
    selected = select_hosts(host_scores, alpha)
    y_pred = np.zeros(len(df), dtype=int)
    for host in selected:
        out = np.where(src == host)[0]
        y_pred[out[np.argsort(scores[out], kind="stable")[-beta:]]] = 1

    tp = int(y_pred[y == 1].sum())
    fp = int(y_pred[y == 0].sum())
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    has_both = 0 < y.sum() < len(y)
    ranked = sorted(host_scores, key=host_scores.get, reverse=True)
    stats = {
        "precision": round(tp / max(tp + fp, 1), 5),
        "recall": round(tp / max(tp + fn, 1), 5),
        "fpr": round(fp / max(fp + tn, 1), 7),
        "mcc": round(float(matthews_corrcoef(y, y_pred)), 5) if has_both else 0.0,
        "auc": round(float(roc_auc_score(y, scores)), 5) if has_both else float("nan"),
        "ap": round(float(average_precision_score(y, scores)), 5) if has_both else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_edges": len(y),
        "num_malicious_edges": int(y.sum()),
        "selected_hosts": str([int(h) for h in selected]),
    }
    for root in roots:
        stats[f"rank_of_host_{root}"] = ranked.index(root) + 1 if root in host_scores else -1
    log(f"Selected hosts: {stats['selected_hosts']}")
    return stats
