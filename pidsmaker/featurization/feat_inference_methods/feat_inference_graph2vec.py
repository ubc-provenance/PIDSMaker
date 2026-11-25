"""Adds the graph2vec embedding and the rooted subgraphs (RSGs) of each snapshot to its graph.

graph2vec embeds whole graphs, not nodes: nodes only get their type as features, and each time
window graph written by `feat_inference` gets two graph-level attributes:

- `graph_emb` (1, d): the graph2vec embedding of the snapshot, inferred with the model trained
  in the featurization task (on the train split with `featurization.training_split=train`).
  All snapshots are inferred in the same way, including those of the training split, so that
  the embeddings of all splits follow the same distribution.
- `snapshot_rsg` (1, n, D + 2) int32: for each node of the snapshot (sorted by node ID), its ID
  followed by the vocabulary index of its RSG at each depth 0..D (-1 for RSGs unseen when
  training graph2vec). Used to rank the RSGs of anomalous snapshots (key indicators).

Both have a first dimension of size 1, so that they are kept whole when the edges of a graph are
selected (`CollatableTemporalData.index_select` slices the tensors whose first dimension is the
number of edges).
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.featurization_methods.featurization_graph2vec import (
    MODEL_FILE,
    infer_snapshot_embedding,
    snapshot_rsgs,
    snapshot_seed,
    snapshot_words,
)
from pidsmaker.utils.utils import log, log_start, log_tqdm

# Set in the parent process before forking the workers, to share the model without copies
_MODEL = None


def _embed_snapshot(args):
    """Worker: adds `graph_emb` and `snapshot_rsg` to the feat_inference output of a graph."""
    path, out_path, depth, epochs = args
    node_ids, labels = snapshot_rsgs(path, depth)
    vector = infer_snapshot_embedding(
        _MODEL, snapshot_words(labels), epochs=epochs, seed=snapshot_seed(path)
    )

    key_to_index = _MODEL.wv.key_to_index
    vocab_ids = np.vectorize(lambda w: key_to_index.get(w, -1), otypes=[np.int64])(labels)
    table = np.concatenate([node_ids[:, None], vocab_ids.T], axis=1).astype(np.int32)

    data = torch.load(out_path)
    data.graph_emb = torch.from_numpy(vector).float().view(1, -1)
    data.snapshot_rsg = torch.from_numpy(table).unsqueeze(0)
    torch.save(data, out_path)
    return float((vocab_ids < 0).mean())


def main(cfg, sorted_paths, out_dir):
    """Embeds the snapshots of `sorted_paths`, whose graphs were written to `out_dir`."""
    global _MODEL
    log_start(__file__)
    _MODEL = Doc2Vec.load(os.path.join(cfg.featurization._model_dir, MODEL_FILE))

    g2v_cfg = cfg.featurization.graph2vec
    tasks = []
    for path in sorted_paths:
        out_path = os.path.join(out_dir, f"{os.path.basename(path)}.TemporalData.simple")
        if os.path.exists(out_path):  # empty graphs have no output
            tasks.append((path, out_path, g2v_cfg.wl_iterations, g2v_cfg.infer_epochs))

    workers = max(1, g2v_cfg.num_workers)
    # Forked workers share the model loaded in this process
    fork = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=fork) as pool:
        oov = list(
            log_tqdm(
                pool.map(_embed_snapshot, tasks, chunksize=1),
                desc="Inferring graph2vec snapshot embeddings",
                total=len(tasks),
            )
        )
    if oov:
        log(f"Mean ratio of RSGs unseen during graph2vec training: {np.mean(oov):.3f}")
    _MODEL = None
