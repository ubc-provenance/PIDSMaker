"""
DeepWalk node embedding via Word2Vec Skip-gram on random walks.

Converts random walks from node IDs to node-label strings (via indexid2msg),
then trains gensim Word2Vec Skip-gram. This makes the model inductive at the
label level: any node whose (ntype, nlabel) was seen during training gets an
embedding, regardless of whether that specific node ID appeared in the walks.
"""

import os

import numpy as np
from gensim.models import Word2Vec

from pidsmaker.utils.utils import log


def _node_label_key(ntype, nlabel):
    """Canonical string key for a (ntype, nlabel) pair."""
    return f"{ntype}::{nlabel}"


def walks_to_label_sentences(walk_corpus, model_type="deepwalk"):
    """Convert a walk corpus to label-string sentences for Word2Vec.

    Each walk becomes a list of label-key strings. Nodes missing from
    indexid2msg are skipped (they have no label to embed).

    Args:
        walk_corpus: list of (walk_nodes, walk_edge_types, ds_indexid2msg, entity_pos)
        model_type: unused, kept for API consistency

    Returns:
        list of list[str] — one sentence per walk
    """
    sentences = []
    for walk_nodes, walk_edge_types, ds_indexid2msg, _epos in walk_corpus:
        sentence = []
        for node_id in walk_nodes:
            if node_id in ds_indexid2msg:
                ntype, nlabel = ds_indexid2msg[node_id]
                sentence.append(_node_label_key(ntype, nlabel))
        if len(sentence) >= 2:
            sentences.append(sentence)
    return sentences


def train_deepwalk(sentences, emb_dim, window=5, min_count=1, epochs=10,
                   workers=4, seed=42):
    """Train Word2Vec Skip-gram on label sentences.

    Args:
        sentences: list of list[str] — label-key sequences from walks
        emb_dim: embedding vector dimension
        window: Skip-gram context window size
        min_count: minimum label frequency to include
        epochs: number of training epochs
        workers: parallel training threads
        seed: random seed

    Returns:
        trained gensim Word2Vec model
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=emb_dim,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    log(f"DeepWalk trained: {len(model.wv)} labels, dim={emb_dim}, "
        f"{epochs} epochs, window={window}")
    return model


def save_deepwalk(model, out_dir, name="deepwalk.model"):
    """Save a trained Word2Vec model."""
    path = os.path.join(out_dir, name)
    model.save(path)
    log(f"DeepWalk model saved -> {path}")
    return path


def load_deepwalk(out_dir, name="deepwalk.model"):
    """Load a previously trained Word2Vec model."""
    path = os.path.join(out_dir, name)
    model = Word2Vec.load(path)
    log(f"DeepWalk model loaded <- {path} ({len(model.wv)} labels, dim={model.wv.vector_size})")
    return model


def deepwalk_embeddings(model, indexid2msg):
    """Build {node_id: np.ndarray} from a trained DeepWalk model.

    Looks up each node's (ntype, nlabel) in the Word2Vec vocabulary.
    Nodes whose label is not in the vocabulary get a zero vector.

    Returns:
        dict mapping int(node_id) -> np.ndarray of shape (emb_dim,)
    """
    emb_dim = model.wv.vector_size
    indexid2vec = {}
    n_hit, n_miss = 0, 0

    for node_id, (ntype, nlabel) in indexid2msg.items():
        key = _node_label_key(ntype, nlabel)
        if key in model.wv:
            emb = model.wv[key].astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            indexid2vec[int(node_id)] = emb
            n_hit += 1
        else:
            indexid2vec[int(node_id)] = np.zeros(emb_dim, dtype=np.float32)
            n_miss += 1

    log(f"DeepWalk embeddings: {n_hit} hits, {n_miss} misses out of {n_hit + n_miss} nodes")
    return indexid2vec
