import os
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.tokenization import fit_tokenizer, load_tokenizer
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_indexid2msg,
    log,
    log_start,
    log_tqdm,
)


# Each edge gives 3 texts to its source and destination nodes; a node keeps the texts of its
# first edges (in time order) of each graph, up to this number of texts
MAX_TEXTS_PER_NODE = 300


def _edge_columns(file_path):
    """(node IDs, src and dst node positions, edge labels, edge times) of a stored graph, in
    `graph.edges` order."""
    cols = load_graph_columns(file_path)
    if cols is not None:
        return (
            cols.node_ids,
            cols.src.astype(np.int64),
            cols.dst.astype(np.int64),
            cols.edge_attrs["label"] if len(cols.src) else [],
            cols.edge_attrs["time"] if len(cols.src) else [],
        )
    graph = torch.load(file_path)
    nodes = list(graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    src, dst, labels, times = [], [], [], []
    for u, v, key, attr in graph.edges(data=True, keys=True):
        src.append(pos[u])
        dst.append(pos[v])
        labels.append(attr["label"])
        times.append(attr["time"])
    return nodes, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), labels, times


def _time_order(times):
    """Edge indices sorted by `int(time)` (stable, as `sorted`)."""
    times = [int(t) for t in times]
    try:
        return np.argsort(np.array(times, dtype=np.int64), kind="stable")
    except OverflowError:
        return np.array(sorted(range(len(times)), key=times.__getitem__), dtype=np.int64)


def _graph_texts(file_path, indexid2msg, text_ids):
    """Texts of the nodes of one graph.

    Returns (node IDs, edges, triplets): the nodes in order of first appearance in
    the edges sorted by time; for each node, the (time-sorted) indices of the edges whose texts
    it keeps (its first MAX_TEXTS_PER_NODE / 3 edges, a self-loop counting twice); and for each
    edge the IDs in `text_ids` of its 3 texts (src, edge type, dst). `text_ids` maps each text
    (node type or None, label) to an ID, and is extended with the new texts.
    """
    nodes, src, dst, labels, times = _edge_columns(file_path)
    m = len(src)
    order = _time_order(times)
    src, dst = src[order], dst[order]
    labels = [labels[i] for i in order.tolist()]

    # Texts: (node type, label) of the endpoints, (None, edge type) of the edges
    used = np.unique(np.concatenate([src, dst])).tolist()
    node_text = np.zeros(len(nodes), dtype=np.int64)
    for i in used:
        node_type, msg = indexid2msg[nodes[i]]
        node_text[i] = text_ids.setdefault((node_type, msg), len(text_ids))
    op_text = np.array(
        [text_ids.setdefault((None, op), len(text_ids)) for op in labels], dtype=np.int64
    )
    triplets = np.stack([node_text[src], op_text, node_text[dst]], axis=1).reshape(m, 3)

    # Incidences of the nodes: (src of edge 0, dst of edge 0, src of edge 1, ...). A node keeps
    # the texts of its first incidences
    inc = np.empty(2 * m, dtype=np.int64)
    inc[0::2], inc[1::2] = src, dst
    by_node = np.argsort(inc, kind="stable")
    sorted_nodes = inc[by_node]
    starts = np.flatnonzero(np.r_[m > 0, sorted_nodes[1:] != sorted_nodes[:-1]])
    sizes = np.diff(np.append(starts, 2 * m))
    rank = np.arange(2 * m) - np.repeat(starts, sizes)
    max_edges = -(-MAX_TEXTS_PER_NODE // 3)  # texts are added 3 by 3 while there are fewer than MAX
    kept = rank < max_edges
    kept_edges = by_node[kept] // 2
    kept_sizes = np.minimum(sizes, max_edges)

    # Nodes in order of first incidence
    node_order = np.argsort(by_node[starts], kind="stable")
    group_nodes = sorted_nodes[starts].tolist()
    bounds = np.r_[0, np.cumsum(kept_sizes)].tolist()
    node_ids, node_edges = [], []
    for g in node_order.tolist():
        node_ids.append(nodes[group_nodes[g]])
        node_edges.append(kept_edges[bounds[g] : bounds[g + 1]])
    return node_ids, node_edges, triplets


def get_node2corpus(cfg, splits, fit=False):
    """
    Returns for each node the tokens of the (src label, edge type, dst label) triplets
    of its edges. With `fit=True`, the tokenizer is fitted on these texts (featurization),
    otherwise the tokenizer fitted during featurization is loaded (feat_inference).
    """
    indexid2msg = get_indexid2msg(cfg)

    dates = list(chain.from_iterable([getattr(cfg.dataset, f"{split}_dates") for split in splits]))
    sorted_paths = get_all_graphs_for_dates(cfg.transformation._graphs_dir, dates)

    # Texts, numbered in order of first appearance in the graphs
    text_ids = {}
    data_of_graphs = []
    for file_path in log_tqdm(sorted_paths, desc=f"Loading training data for {str(splits)}"):
        data_of_graphs.append(_graph_texts(file_path, indexid2msg, text_ids))
    texts = list(text_ids)
    del text_ids

    # Texts in order of first appearance in the nodes' texts (graph by graph, node by node)
    seen = np.zeros(len(texts), dtype=bool)
    text_order = []
    for _, node_edges, triplets in data_of_graphs:
        if not node_edges:
            continue
        seq = triplets[np.concatenate(node_edges)].ravel()
        uniq, first = np.unique(seq, return_index=True)
        uniq = uniq[np.argsort(first, kind="stable")]
        uniq = uniq[~seen[uniq]]
        seen[uniq] = True
        text_order.extend(uniq.tolist())

    if fit:
        tokenizer = fit_tokenizer(cfg, [texts[i] for i in text_order])
    else:
        tokenizer = load_tokenizer(cfg)

    # Tokens of each text, tokenized in order of first appearance
    tokens_of = [None] * len(texts)
    for i in log_tqdm(text_order, desc="Tokenizing corpus"):
        text_type, sentence = texts[i]
        tokens_of[i] = tokenizer.tokenize(sentence, text_type)

    node2corpus = defaultdict(list)
    triplet_tokens = {}
    for node_ids, node_edges, triplets in data_of_graphs:
        if not node_ids:
            continue
        # Tokens of the distinct (src, edge type, dst) text triplets of the graph
        uniq, inverse = np.unique(triplets, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        uniq_tokens = []
        for a, b, c in uniq.tolist():
            key = (a, b, c)
            tokens = triplet_tokens.get(key)
            if tokens is None:
                tokens = triplet_tokens[key] = tokens_of[a] + tokens_of[b] + tokens_of[c]
            uniq_tokens.append(tokens)
        for node_id, edges in zip(node_ids, node_edges):
            extend = node2corpus[node_id].extend
            for t in inverse[edges].tolist():
                extend(uniq_tokens[t])

    return node2corpus


class RepeatableIterator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for phrases in self.data:
            yield phrases


def main(cfg):
    log_start(__file__)

    training_files = get_splits_to_train_featurization(cfg)
    all_phrases = list(get_node2corpus(cfg=cfg, splits=training_files, fit=True).values())

    emb_dim = cfg.featurization.emb_dim
    epochs = cfg.featurization.epochs
    min_count = cfg.featurization.flash.min_count
    workers = cfg.featurization.flash.workers

    log("Training word2vec model...")
    model = Word2Vec(vector_size=emb_dim, min_count=min_count, workers=workers, epochs=epochs)
    model.build_vocab(RepeatableIterator(all_phrases), progress_per=10000)
    model.train(RepeatableIterator(all_phrases), total_examples=model.corpus_count, epochs=epochs)

    model_save_dir = cfg.featurization._model_dir
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(os.path.join(model_save_dir, "word2vec_model_final.model"))
