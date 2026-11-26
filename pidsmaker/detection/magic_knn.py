"""MAGIC's detection module: k-nearest-neighbour outlier detection on node embeddings.

As in the original MAGIC (Jia et al., USENIX Security 2024, `model/eval.py`), the embeddings of
the benign training graphs form the reference set. Embeddings are standardized with the mean and
standard deviation of the reference set, and the anomaly score of a node is its mean Euclidean
distance to its k nearest benign neighbours divided by the mean k-NN distance of benign nodes.
A score of ~1 is as typical as a training node; outliers score well above 1.

The search is an exact brute-force search on the device of the reference set, by chunks. Many
training nodes share the exact same embedding (same type, same neighbourhood), so the reference
set is stored once per distinct embedding with its multiplicity: the k nearest distinct
embeddings always hold the k nearest nodes, and the k-NN distances are recovered exactly.
"""

import time

import torch

from pidsmaker.detection.node_scoring import NodeScorer
from pidsmaker.utils.utils import log

# MAGIC estimates the mean k-NN distance of the reference set on at most 50,000 of its nodes
MEAN_DISTANCE_SAMPLE = 50_000


class MagicKNNDetector:
    def __init__(self, k=10, max_reference_size=None, chunk_elems=2**28, seed=0):
        """
        Args:
            k: number of nearest neighbours (10 in the MAGIC paper).
            max_reference_size: if set, the reference set is a uniform random sample of at most
                this many training embeddings, to bound the cost of the search.
            chunk_elems: max number of query-reference distances computed at once.
            seed: seed of the sampling of the reference set.
        """
        self.k = k
        self.max_reference_size = max_reference_size
        self.chunk_elems = chunk_elems
        self.seed = seed
        self.reference = None  # distinct standardized embeddings (U, d)
        self.counts = None  # multiplicity of each distinct embedding (U,)
        self.num_reference_nodes = 0
        self.mean = None
        self.std = None
        self.mean_distance = None

    def fit(self, x_train: torch.Tensor):
        """Fits the detector on the embeddings of benign training nodes (N, d)."""
        x_train = x_train.float()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.max_reference_size and x_train.shape[0] > self.max_reference_size:
            idx = torch.randperm(x_train.shape[0], generator=gen)[: self.max_reference_size]
            x_train = x_train[idx.to(x_train.device)]

        self.mean = x_train.mean(dim=0)
        # Constant dimensions (e.g. dead units) would divide by zero
        self.std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_train = (x_train - self.mean) / self.std
        self.num_reference_nodes = x_train.shape[0]

        self.reference, inverse, self.counts = torch.unique(
            x_train, dim=0, return_inverse=True, return_counts=True
        )

        n = x_train.shape[0]
        sample = torch.randperm(n, generator=gen)[: min(MEAN_DISTANCE_SAMPLE, n)]
        # As in MAGIC, the sampled training nodes are part of the reference set: their nearest
        # neighbour is themselves, at distance 0
        dists = self._knn_mean_distance(self.reference[inverse[sample.to(inverse.device)]])
        mean_distance = dists.mean().item()
        self.mean_distance = mean_distance if mean_distance > 0 else 1e-9
        return self

    @torch.no_grad()
    def _knn_mean_distance(self, queries: torch.Tensor) -> torch.Tensor:
        """Mean distance of each (standardized) query to its k nearest reference nodes."""
        ref, counts = self.reference, self.counts
        k = min(self.k, self.num_reference_nodes)
        k_distinct = min(k, ref.shape[0])
        chunk = max(1, self.chunk_elems // max(1, ref.shape[0]))
        ref_sq = (ref * ref).sum(dim=1)
        out = []
        for i in range(0, queries.shape[0], chunk):
            q = queries[i : i + chunk]
            # ||q - r||^2 = ||q||^2 - 2 q.r + ||r||^2, clamped against negative rounding errors
            d2 = torch.addmm(ref_sq.unsqueeze(0), q, ref.T, alpha=-2)
            d2 += (q * q).sum(dim=1, keepdim=True)
            top = torch.topk(d2, k_distinct, dim=1, largest=False, sorted=True)
            del d2
            dist = top.values.clamp_min(0).sqrt()
            # Expands the distinct neighbours by their multiplicity, up to k nodes
            c = counts[top.indices]
            before = c.cumsum(dim=1) - c
            take = (k - before).clamp(min=0) - (k - before - c).clamp(min=0)
            out.append((dist * take).sum(dim=1) / k)
        if not out:
            return torch.zeros(0, device=ref.device)
        return torch.cat(out)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Anomaly score of each node embedding (N, d) -> (N,)."""
        if self.reference is None:
            raise RuntimeError("MagicKNNDetector must be fitted on benign embeddings first.")
        x = x.float().to(self.reference.device)
        # Identical test embeddings get the same score: each one is searched once
        x, inverse = torch.unique(x, dim=0, return_inverse=True)
        x = (x - self.mean) / self.std
        return (self._knn_mean_distance(x) / self.mean_distance)[inverse]


def magic_detection_embedding(model, h, cfg=None):
    """Node embedding scored by MAGIC's k-NN detector: the last encoder layer (`last`, the
    original code) or all the concatenated layers (`concat`, the paper)."""
    mode = (cfg.evaluation.node_evaluation.magic_embedding if cfg is not None else None) or "last"
    if mode == "last" and hasattr(model.encoder, "detection_embedding"):
        return model.encoder.detection_embedding(h)
    return h


@torch.no_grad()
def fit_magic_detector(cfg, model, train_data, device):
    """Fits MAGIC's k-NN detector on the embeddings of the benign training nodes."""
    from pidsmaker.detection.training_methods.training_loop import to_device_without_attacks

    model.eval()
    s = time.time()
    embs = []
    for dataset in train_data:
        for g in dataset:
            g = to_device_without_attacks(g, cfg, device)
            h, *_ = model.embed(g, inference=True)
            embs.append(magic_detection_embedding(model, h, cfg).float())
            del g
    x_train = torch.cat(embs)
    del embs
    node_eval = cfg.evaluation.node_evaluation
    detector = MagicKNNDetector(
        k=node_eval.magic_knn_k or 10,
        max_reference_size=node_eval.magic_max_reference_size or None,
        seed=cfg.training.seed or 0,
    ).fit(x_train)
    log(
        f"MAGIC k-NN detector fitted on {detector.num_reference_nodes:,} of {x_train.shape[0]:,} "
        f"benign training node embeddings ({detector.reference.shape[0]:,} distinct, "
        f"k={detector.k}, mean distance "
        f"{detector.mean_distance:.4f}) in {time.time() - s:.1f}s"
    )
    return detector


class MagicScorer(NodeScorer):
    """Writes the `magic_score` of the test nodes, thresholded by `threshold_method: magic`: the
    k-NN distance of their embeddings to those of the benign training nodes."""

    score_column = "magic_score"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.detector = None

    def prepare(self, model, train_data, split_names, device):
        if "test" in split_names:
            if train_data is None:
                raise ValueError("MAGIC detection requires the train split as k-NN reference set.")
            self.detector = fit_magic_detector(self.cfg, model, train_data, device)

    def columns(self, model, data, results, split, node, loss):
        if split != "test":
            return {"node": node, "loss": loss}
        if self.detector is None:
            raise ValueError("MAGIC detection requires a detector fitted on the train split.")
        x_test, *_ = model.embed(data, inference=True)
        score = self.detector.score(magic_detection_embedding(model, x_test, self.cfg))
        return {
            "node": node,
            "magic_score": score.cpu().numpy().astype(float),
            "loss": loss,
        }

    def node_prediction(self, values, thr):
        # Detected in the first time window of max score, if above the threshold
        max_score = 0
        pred_score = max(values["magic_score"])
        detected_tw = None
        for score, tw in zip(values["magic_score"], values["tw"]):
            if score > thr and score > max_score:
                max_score = score
                detected_tw = tw
        return pred_score, pred_score > thr, detected_tw
