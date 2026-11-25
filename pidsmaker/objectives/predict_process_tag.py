"""Process tag prediction, the self-supervised objective of R-Caid (IEEE S&P 2024, Section 4.3).

Each process is annotated with the name of its executable (see `pidsmaker.utils.process_tags`),
and the GNN learns to predict it from its neighborhood and its roots: processes with the same
binary name get close embeddings. Only process nodes are classified (the paper found that
classifying all the nodes is detrimental); other nodes get a score of 0.

The classes are the tags of the processes of the training split, plus an "unknown" class for
the tags never seen in training, which is never a training target. Processes are scored either
as in the paper (`detection: kmeans`, see `ProcessTagPrediction`), which does not use the tags,
or by the loss of their tag (`detection: loss`), where a process running an executable never
seen in training gets a high loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.process_tags import process_tag
from pidsmaker.utils.utils import compute_class_weights, get_indexid2msg, get_split2nodes, log

PROCESS_TYPE = "subject"


def build_process_tag_classes(indexid2msg, train_nodes, process_type=PROCESS_TYPE):
    """(node id -> class array, number of known tags). Classes are numbered in order of
    first appearance in the training processes (sorted by node id); -1: not a process,
    `num_known`: process with an unknown tag."""
    ids = sorted(int(n) for n in indexid2msg)
    max_id = ids[-1] if ids else -1
    classes = np.full(max_id + 1, -1, dtype=np.int64)
    tag2class = {}
    train_nodes = {str(n) for n in train_nodes}
    for n in ids:
        node_type, label = indexid2msg[str(n)] if str(n) in indexid2msg else indexid2msg[n]
        if node_type == process_type and str(n) in train_nodes:
            tag2class.setdefault(process_tag(label), len(tag2class))
    num_known = len(tag2class)
    for n in ids:
        node_type, label = indexid2msg[str(n)] if str(n) in indexid2msg else indexid2msg[n]
        if node_type == process_type:
            classes[n] = tag2class.get(process_tag(label), num_known)
    return classes, num_known


def process_tag_classes_from_cfg(cfg):
    indexid2msg = get_indexid2msg(cfg)
    train_nodes = get_split2nodes(cfg)["train"]
    classes, num_known = build_process_tag_classes(indexid2msg, train_nodes)
    log(f"Process tags: {num_known} tags in training, {(classes >= 0).sum()} processes")
    return classes, num_known


def elbow_kmeans(x, k_values, seed=0, elbow_sample=5000):
    """K-Means of x with the K at the elbow of the inertia curve over `k_values` (the point the
    furthest from the line joining the first and last points, with normalized axes). The curve
    is computed on a random sample of `elbow_sample` points."""
    from sklearn.cluster import KMeans

    k_values = [k for k in k_values if k <= len(x)] or [1]
    sample = x
    if len(x) > elbow_sample:
        sample = x[np.random.default_rng(seed).choice(len(x), elbow_sample, replace=False)]
    inertia = np.array(
        [KMeans(n_clusters=k, n_init=3, random_state=seed).fit(sample).inertia_ for k in k_values]
    )
    best = len(k_values) - 1
    if len(k_values) >= 3:
        ks = np.array(k_values, dtype=float)
        ks = (ks - ks[0]) / (ks[-1] - ks[0])
        span = inertia[0] - inertia[-1]
        inertia = (inertia - inertia[-1]) / span if span > 0 else np.zeros_like(inertia)
        # Distance to the line from (0, 1) to (1, 0)
        best = int(np.argmax(np.abs(ks + inertia - 1)))
    return KMeans(n_clusters=k_values[best], n_init=3, random_state=seed).fit(x)


class ProcessTagPrediction(nn.Module):
    """Process tag prediction. Inference scores (`detection`):

    - `loss`: the cross-entropy of the tag of each process;
    - `kmeans`: as in the paper (Section 4.4), the distance of the embedding of each process to
      the median of its nearest cluster, clustered by K-Means (K chosen with the elbow method)
      on the embeddings of training processes. These are sampled during the last training epoch
      (up to `max_train_embeddings`, with dropout).
    """

    def __init__(self, decoder, node_classes, num_known, balanced_loss, detection="loss",
                 max_train_embeddings=20000, k_values=(2, 4, 6, 8, 10, 12, 16, 20, 25, 30)):
        super(ProcessTagPrediction, self).__init__()
        self.decoder = decoder
        self.num_classes = num_known + 1  # + unknown
        self.balanced_loss = balanced_loss
        self.register_buffer("node_classes", torch.as_tensor(node_classes, dtype=torch.long), persistent=False)
        if detection not in ("loss", "kmeans"):
            raise ValueError(f"Invalid detection {detection}")
        self.detection = detection
        self.max_train_embeddings = max_train_embeddings
        self.k_values = list(k_values)
        self._generator = torch.Generator().manual_seed(0)
        self.reset_state()

    def reset_state(self):
        """Called before each training epoch: the embeddings of the epoch are sampled again."""
        self._train_embs = []
        self._num_seen = 0
        self._medians = None

    def _sample_train_embeddings(self, h):
        """Uniform sample of the embeddings seen in the epoch: each one gets a random key, and
        those whose key is below `max_train_embeddings / number seen` are kept."""
        h = h.detach().float().cpu()
        keys = torch.rand(len(h), generator=self._generator)
        self._num_seen += len(h)
        ratio = min(1.0, self.max_train_embeddings / self._num_seen)
        self._train_embs.append((h, keys))
        self._train_embs = [(e[k < ratio], k[k < ratio]) for e, k in self._train_embs]
        self._medians = None

    def _cluster_medians(self, device):
        if self._medians is None:
            if not self._train_embs:
                raise RuntimeError("`detection: kmeans` needs the embeddings of a training epoch")
            x = torch.cat([e for e, _ in self._train_embs]).numpy()
            km = elbow_kmeans(x, self.k_values)
            medians = [np.median(x[km.labels_ == c], axis=0) for c in range(km.n_clusters)]
            log(f"Process tags: K-Means with K={km.n_clusters} on {len(x)} training processes")
            self._medians = torch.tensor(np.stack(medians), dtype=torch.float)
        return self._medians.to(device)

    def _classes(self, batch, num_nodes, device):
        n_id = batch.original_n_id.to(device)
        classes = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        valid = n_id < self.node_classes.numel()
        classes[valid] = self.node_classes[n_id[valid]]
        return classes

    def forward(self, h, inference, batch, **kwargs):
        classes = self._classes(batch, h.shape[0], h.device)
        is_process = classes >= 0

        if not inference:
            # Unknown tags (not in the training split) are never targets
            known = is_process & (classes < self.num_classes - 1)
            if not bool(known.any()):
                return {"loss": h.sum() * 0.0}
            if self.detection == "kmeans":
                self._sample_train_embeddings(h[known])
            logits = self.decoder(h[known])
            target = classes[known]
            weight = (
                compute_class_weights(F.one_hot(target, self.num_classes), num_classes=self.num_classes)
                if self.balanced_loss
                else None
            )
            return {"loss": F.cross_entropy(logits, target, weight=weight)}

        scores = torch.zeros(h.shape[0], dtype=torch.float, device=h.device)
        out = None
        if bool(is_process.any()):
            logits = self.decoder(h[is_process])
            out = F.log_softmax(logits, dim=1)
            if self.detection == "kmeans":
                dist = torch.cdist(h[is_process].float(), self._cluster_medians(h.device))
                scores[is_process] = dist.min(dim=1).values
            else:
                scores[is_process] = F.cross_entropy(logits, classes[is_process], reduction="none")
        return {"loss": scores, "out": out}
