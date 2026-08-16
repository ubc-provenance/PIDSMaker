import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from pidsmaker.utils.labelling import get_ground_truth
from pidsmaker.utils.utils import get_node_to_path_and_type, listdir_sorted, log


class OCRAPTEarlyStop:
    def __init__(self, num_node_types, patience=5, min_delta=0.01, max_delta=0.2):
        self.num_node_types = num_node_types
        self.patience = patience
        self.max_delta = max_delta
        self._first_epoch = [True] * num_node_types
        self._malicious_pct = [0.0] * num_node_types
        self._f1_baseline = [0.0] * num_node_types
        self._max_f1 = [0.0] * num_node_types
        self._max_auc = [0.0] * num_node_types
        self._max_tnr = [0.0] * num_node_types
        self._counter = [0] * num_node_types
        self._stopped = [False] * num_node_types

    def _update_one_type(self, nt, scores, labels, thr):
        if self._stopped[nt] or len(scores) == 0:
            return

        if self._first_epoch[nt]:
            self._malicious_pct[nt] = float(labels.mean())
            self._f1_baseline[nt] = (self._malicious_pct[nt] * 2) / (self._malicious_pct[nt] + 1)
            self._first_epoch[nt] = False

        pred = (scores > thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        auc = roc_auc_score(labels, scores) if labels.sum() > 0 else None
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None

        max_delta = self.max_delta
        if self._malicious_pct[nt] > 0.05:
            if f1 > self._max_f1[nt]:
                self._max_f1[nt] = f1
                self._counter[nt] = 0
            elif (f1 < self._f1_baseline[nt]) or (f1 < self._max_f1[nt] - max_delta):
                self._counter[nt] = 0
            elif f1 <= self._max_f1[nt]:
                self._counter[nt] += 1
                if self._counter[nt] >= self.patience:
                    self._stopped[nt] = True
        elif self._malicious_pct[nt] > 0:
            if auc is None:
                return
            if auc > self._max_auc[nt]:
                self._max_auc[nt] = auc
                self._counter[nt] = 0
            elif (auc <= 0.5) or (auc < self._max_auc[nt] - max_delta):
                self._counter[nt] = 0
            elif auc <= self._max_auc[nt]:
                self._counter[nt] += 1
                if self._counter[nt] >= self.patience:
                    self._stopped[nt] = True
        else:
            if tnr is None:
                return
            if tnr > self._max_tnr[nt]:
                self._max_tnr[nt] = tnr
                self._counter[nt] = 0
            elif (tnr <= 0.01) or (tnr < self._max_tnr[nt] - max_delta):
                self._counter[nt] = 0
            elif tnr <= self._max_tnr[nt]:
                self._counter[nt] += 1
                if self._counter[nt] >= self.patience:
                    self._stopped[nt] = True

    def _scores_by_type(self, edge_losses_dir, split, epoch, node_to_type, type_to_idx, gt_nids):
        d = os.path.join(edge_losses_dir, split, f"model_epoch_{epoch}")
        if not os.path.isdir(d):
            return None
        dfs = [pd.read_csv(os.path.join(d, f)) for f in listdir_sorted(d) if f.endswith(".csv")]
        if not dfs:
            return None
        node_score = pd.concat(dfs, ignore_index=True).groupby("node")["loss"].max()

        scores_by_type = [[] for _ in range(self.num_node_types)]
        labels_by_type = [[] for _ in range(self.num_node_types)]
        for nid, score in node_score.items():
            nt = type_to_idx.get(node_to_type.get(int(nid)))
            if nt is None:
                continue
            scores_by_type[nt].append(score)
            labels_by_type[nt].append(int(int(nid) in gt_nids))
        return scores_by_type, labels_by_type

    def check(self, cfg, epoch, min_contamination=0.001, max_contamination=0.05):
        gt_nids, _, _ = get_ground_truth(cfg)
        gt_nids = set(int(n) for n in gt_nids)
        node_to_type = {nid: info["type"] for nid, info in get_node_to_path_and_type(cfg).items()}
        type_names = sorted({t for t in node_to_type.values()})
        type_to_idx = {t: i for i, t in enumerate(type_names[: self.num_node_types])}

        # threshold from train scores (calibration), evaluated against val labels
        train = self._scores_by_type(cfg.training._edge_losses_dir, "train", epoch, node_to_type, type_to_idx, gt_nids)
        val = self._scores_by_type(cfg.training._edge_losses_dir, "val", epoch, node_to_type, type_to_idx, gt_nids)
        if train is None or val is None:
            return False
        train_scores_by_type, _ = train
        val_scores_by_type, val_labels_by_type = val

        for nt in range(self.num_node_types):
            train_scores = np.asarray(train_scores_by_type[nt], dtype=np.float64)
            val_scores = np.asarray(val_scores_by_type[nt], dtype=np.float64)
            val_labels = np.asarray(val_labels_by_type[nt], dtype=np.int64)
            if len(train_scores) == 0 or len(val_scores) == 0:
                continue
            # same per-type contamination as the final detection threshold
            # (data_statistics_contamination): val malicious fraction, clamped.
            contamination = min(max(float(val_labels.mean()), min_contamination), max_contamination)
            thr = float(np.percentile(train_scores, 100 * (1 - contamination)))
            self._update_one_type(nt, val_scores, val_labels, thr)

        all_stopped = all(self._stopped)
        if all_stopped:
            log(f"OCR-APT validation early stop triggered at epoch {epoch}")
        return all_stopped
