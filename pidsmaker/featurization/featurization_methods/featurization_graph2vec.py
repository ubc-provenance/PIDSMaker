"""graph2vec embedding of whole time-window graphs (snapshots), as in ProGrapher.

Each snapshot is a "document" whose "words" are the rooted subgraphs (RSGs) of all its nodes,
extracted with the Weisfeiler-Lehman (WL) relabeling of ProGrapher (Algorithm 2 of the paper):
the RSG of a node at depth 0 is its type; at depth d, it combines its RSG at depth d-1 with the
set of RSGs at depth d-1 of its neighbors and the set of types of its edges. The graph is
considered undirected, and only node and edge types are used (no textual labels).

A doc2vec model (PV-DBOW with negative sampling) learns one vector per snapshot that predicts
its RSGs. The RSGs act as output vectors (`syn1neg`), later used to rank the RSGs of an anomalous
snapshot (key indicators, see `objectives/predict_snapshot_embedding.py`).

gensim truncates documents to 10,000 words, whereas snapshots have `(depth + 1)` words per node:
documents are therefore split into chunks of 10,000 words that share the same tag in training,
and the embedding of a snapshot is inferred over all its chunks (`infer_snapshot_embedding`).
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

import numpy as np
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec_inner import train_document_dbow
from gensim.models.keyedvectors import pseudorandom_weak_vector

from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import get_all_graphs_for_dates, log, log_start, log_tqdm

# Maximum number of words of a document processed by gensim (`MAX_DOCUMENT_LEN`)
GENSIM_MAX_DOCUMENT_LEN = 10000
MODEL_FILE = "graph2vec.model"


def snapshot_columns(path):
    """Node IDs (sorted), node types, edge endpoints (as rows of the node IDs) and edge types
    of the graph stored at `path`."""
    cols = load_graph_columns(path)
    if cols is not None:
        node_ids = np.asarray([int(n) for n in cols.node_ids], dtype=np.int64)
        node_types = [str(t) for t in cols.node_attrs["node_type"]]
        src = np.asarray(cols.src, dtype=np.int64)
        dst = np.asarray(cols.dst, dtype=np.int64)
        edge_types = [str(e) for e in cols.edge_attrs["label"]]
    else:  # legacy graphs stored as pickled networkx graphs
        graph = torch.load(path)
        nodes = list(graph.nodes)
        node2row = {n: i for i, n in enumerate(nodes)}
        node_ids = np.asarray([int(n) for n in nodes], dtype=np.int64)
        node_types = [str(graph.nodes[n]["node_type"]) for n in nodes]
        src, dst, edge_types = [], [], []
        for u, v, attr in graph.edges(data=True):
            src.append(node2row[u])
            dst.append(node2row[v])
            edge_types.append(str(attr.get("label")))
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)

    # Nodes sorted by ID, so that they can be found with a binary search
    order = np.argsort(node_ids, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    node_types = [node_types[i] for i in order]
    return node_ids[order], node_types, rank[src], rank[dst], edge_types


def _group_strings(rows, codes, strings, num_nodes):
    """For each node, the ","-joined `strings[codes]` of its unique (row, code) pairs, in the
    order of the codes (the strings are sorted, so this is the sorted set of strings)."""
    pairs = np.unique(np.stack([rows, codes], axis=1), axis=0) if len(rows) else np.zeros((0, 2), int)
    out = [""] * num_nodes
    if len(pairs) == 0:
        return out
    bounds = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(pairs)]])
    values = strings[pairs[:, 1]]
    for s, e in zip(starts, ends):
        out[pairs[s, 0]] = ",".join(values[s:e])
    return out


def _hash(label: str) -> str:
    return hashlib.blake2b(label.encode("utf-8"), digest_size=8).hexdigest()


def rooted_subgraphs(node_types, src, dst, edge_types, depth):
    """RSG labels of all nodes at depths 0..`depth` (Algorithm 2 of ProGrapher).

    Args:
        node_types: type of each node (n,)
        src, dst: rows of the endpoints of each edge (E,)
        edge_types: type of each edge (E,)
        depth: maximum depth D of the WL relabeling

    Returns:
        np.ndarray: (D + 1, n) array of string labels. Labels of depth 0 are the node types,
        labels of higher depths are hashes of the (sorted, deduplicated) labels they combine.
    """
    n = len(node_types)
    labels = np.asarray(node_types, dtype=object)
    all_labels = [labels]
    if depth == 0 or n == 0:
        return np.stack(all_labels)

    # The graph is undirected: each edge links both endpoints
    rows = np.concatenate([src, dst])
    nbrs = np.concatenate([dst, src])
    etype_strings, etype_codes = np.unique(np.asarray(edge_types, dtype=object), return_inverse=True)
    edge_part = _group_strings(rows, np.concatenate([etype_codes, etype_codes]), etype_strings, n)

    for _ in range(depth):
        label_strings, label_codes = np.unique(labels, return_inverse=True)
        nbr_part = _group_strings(rows, label_codes[nbrs], label_strings, n)
        labels = np.asarray(
            [_hash(f"{labels[i]}|{nbr_part[i]}|{edge_part[i]}") for i in range(n)], dtype=object
        )
        all_labels.append(labels)
    return np.stack(all_labels)


def snapshot_rsgs(path, depth):
    """(sorted node IDs, (D + 1, n) RSG labels) of the snapshot stored at `path`."""
    node_ids, node_types, src, dst, edge_types = snapshot_columns(path)
    return node_ids, rooted_subgraphs(node_types, src, dst, edge_types, depth)


def snapshot_words(labels):
    """The words of a snapshot document: the RSGs of each node at all depths (with duplicates,
    i.e. n x (D + 1) words, as ProGrapher)."""
    return labels.T.ravel().tolist()


def chunks(words, size=GENSIM_MAX_DOCUMENT_LEN):
    return [words[i : i + size] for i in range(0, len(words), size)]


def infer_snapshot_embedding(model, words, epochs, seed):
    """Embedding of a new snapshot, as `Doc2Vec.infer_vector` (PV-DBOW), but trained over all
    the chunks of the document instead of its first 10,000 words. RSGs absent from the
    vocabulary are ignored.

    Args:
        model: trained graph2vec (Doc2Vec) model
        words: words of the snapshot
        epochs: number of inference epochs
        seed: integer seed, making the inferred vector independent of other inferences
    """
    dim = model.dv.vector_size
    model.random = np.random.RandomState(seed)
    # gensim seeds this vector with Python's `hash` of the string by default, which is randomized
    # per process: the integer seed is used directly so the inferred vectors are reproducible.
    doctag_vectors = pseudorandom_weak_vector(
        dim, seed_string=str(seed), hashfxn=lambda _: seed
    ).reshape(1, dim)
    doctags_lockf = np.ones(1, dtype=np.float32)
    work = np.zeros(model.layer1_size, dtype=np.float32)

    alpha, min_alpha = model.alpha, model.min_alpha
    alpha_delta = (alpha - min_alpha) / max(epochs - 1, 1)
    doc_chunks = chunks(words)
    for _ in range(epochs):
        for chunk in doc_chunks:
            train_document_dbow(
                model, chunk, [0], alpha, work,
                learn_words=False, learn_hidden=False,
                doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf,
            )
        alpha -= alpha_delta
    return doctag_vectors[0].copy()


def snapshot_seed(path):
    """Deterministic seed of a snapshot, from its file name."""
    return int(hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()[:8], 16)


def _compact_rsgs(args):
    """Worker: the RSGs of a snapshot as (unique labels, codes) to limit pickling."""
    path, depth = args
    _, labels = snapshot_rsgs(path, depth)
    uniq, codes = np.unique(np.asarray(snapshot_words(labels), dtype=object), return_inverse=True)
    return uniq.tolist(), codes.astype(np.int32)


def get_graph2vec_corpus(cfg):
    """One tagged document per snapshot of the training splits, split into chunks of 10,000
    words. Strings are interned, so that each RSG is stored once in memory."""
    splits = get_splits_to_train_featurization(cfg)
    dates = list(chain.from_iterable(getattr(cfg.dataset, f"{split}_dates") for split in splits))
    paths = get_all_graphs_for_dates(cfg.transformation._graphs_dir, dates)
    depth = cfg.featurization.graph2vec.wl_iterations
    workers = max(1, cfg.featurization.graph2vec.num_workers)

    interned = {}
    corpus = []
    num_words = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_compact_rsgs, [(p, depth) for p in paths], chunksize=1)
        for path, (uniq, codes) in zip(log_tqdm(paths, desc="Extracting rooted subgraphs"), results):
            uniq = [interned.setdefault(w, w) for w in uniq]
            words = [uniq[c] for c in codes]
            num_words += len(words)
            tag = os.path.basename(path)
            corpus.extend(TaggedDocument(words=chunk, tags=[tag]) for chunk in chunks(words))
    log(f"graph2vec corpus: {len(paths)} snapshots, {num_words} RSGs, {len(interned)} unique RSGs")
    return corpus


def train_graph2vec(cfg, corpus):
    g2v_cfg = cfg.featurization.graph2vec
    model = Doc2Vec(
        vector_size=cfg.featurization.emb_dim,
        dm=0,  # PV-DBOW, as graph2vec
        dbow_words=0,  # RSG output vectors only, no word vectors
        negative=g2v_cfg.negative,
        hs=0,
        min_count=g2v_cfg.min_count,
        sample=0,
        alpha=g2v_cfg.alpha,
        workers=max(1, g2v_cfg.num_workers),
        seed=cfg.featurization.seed,
    )
    model.build_vocab(corpus)
    log(f"Training graph2vec ({cfg.featurization.epochs} epochs, {len(model.wv)} RSGs)...")
    model.train(corpus, total_examples=model.corpus_count, epochs=cfg.featurization.epochs)
    return model


def main(cfg):
    log_start(__file__)
    if cfg.featurization.multi_dataset_training:
        raise NotImplementedError("graph2vec does not support multi-dataset training.")

    corpus = get_graph2vec_corpus(cfg)
    model = train_graph2vec(cfg, corpus)

    model_dir = cfg.featurization._model_dir
    os.makedirs(model_dir, exist_ok=True)
    log(f"Saving graph2vec model to {model_dir}")
    model.save(os.path.join(model_dir, MODEL_FILE))
