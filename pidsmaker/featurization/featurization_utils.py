from itertools import chain

import numpy as np
import pandas as pd
import torch
from gensim.models.doc2vec import TaggedDocument

from pidsmaker.config.pipeline import update_cfg_for_multi_dataset
from pidsmaker.tokenization import fit_tokenizer
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_indexid2msg,
    get_split2nodes,
    log,
    log_tqdm,
)


def get_splits_to_train_featurization(cfg):
    """
    Returns the splits on which train the embedding method.
    """
    training_split = cfg.featurization.training_split.strip()
    if training_split == "all":
        return ["train", "val", "test"]

    if training_split == "train":
        return ["train"]

    raise ValueError(f"Invalid training split {training_split}")


def get_corpus(cfg, doc2vec_format=False, gather_multi_dataset=False, label_fn=None):
    """
    Returns the tokenized labels of all nodes within a specific split.
    For training, the train split can be used to get only the corpus of train set.
    Labels are tokenized with `featurization.tokenization`, fitted on these labels.
    `label_fn(node_type, label)`, if given, replaces each label beforehand.
    """
    splits = get_splits_to_train_featurization(cfg)
    split2nodes = get_split2nodes(cfg, gather_multi_dataset=gather_multi_dataset)
    nodes_to_include = set().union(*(split2nodes[split] for split in splits))

    entries, tags = [], []
    node_labels = set()
    indexid2msg = get_indexid2msg(cfg, gather_multi_dataset=gather_multi_dataset)

    for node, msg in indexid2msg.items():
        node_type, node_label = msg
        if label_fn is not None:
            node_label = label_fn(node_type, node_label)

        if (node in nodes_to_include) and (node_label not in node_labels):
            node_labels.add(node_label)
            tags.append(node)
            entries.append((node_type, node_label))

    tokenizer = fit_tokenizer(cfg, entries)
    words = [tokenizer.tokenize(node_label, node_type) for node_type, node_label in entries]

    if doc2vec_format:
        words = [
            TaggedDocument(words=word_list, tags=[str(tag)]) for word_list, tag in zip(words, tags)
        ]

    return words


def get_corpus_from_pretrain_datasets(cfg):
    """
    Like get_corpus, but iterates over pretrain_datasets, merging tokenized node labels
    from all of them. Deduplicates across datasets by (node_type, node_label).
    """
    pretrain_datasets_str = cfg.featurization.pretrain_datasets
    datasets = [d.strip() for d in pretrain_datasets_str.split(",")]
    log(f"Loading corpus from pretrain_datasets: {datasets}")

    splits = get_splits_to_train_featurization(cfg)
    entries = []
    seen_labels = set()

    for dataset in datasets:
        dataset_cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)
        split2nodes = get_split2nodes(dataset_cfg)
        nodes_to_include = set().union(*(split2nodes[split] for split in splits))
        indexid2msg = get_indexid2msg(dataset_cfg)

        for node, msg in indexid2msg.items():
            node_type, node_label = msg
            label_key = (node_type, node_label)
            if (node in nodes_to_include) and (label_key not in seen_labels):
                seen_labels.add(label_key)
                entries.append(label_key)

    tokenizer = fit_tokenizer(cfg, entries)
    all_words = [tokenizer.tokenize(node_label, node_type) for node_type, node_label in entries]

    log(f"Pretrain corpus: {len(all_words)} unique (node_type, label) entries")
    return all_words


# Number of document entries built at once from a graph (bounds the memory used per graph)
_ENTRIES_CHUNK = 1 << 22


def _neighbor_columns(path):
    """What `get_corpus_using_neighbors_features` reads from a stored graph G.

    Returns (node_ids, node_entries, entries, bounds, succ): `node_ids` are the nodes in
    `G.nodes` order, `entries` the distinct (node_type, label) pairs of the nodes, and
    `entries[node_entries[i]]` the pair of node i. The neighbors of node i (`G.neighbors`, i.e.
    its distinct successors in adjacency order) are the node positions
    `succ[bounds[i]:bounds[i + 1]]`.
    """
    cols = load_graph_columns(path)
    if cols is None:
        return _neighbor_columns_of_graph(torch.load(path))
    node_ids = cols.node_ids
    n = len(node_ids)
    if n == 0:
        return node_ids, np.zeros(0, dtype=np.int64), [], np.zeros(1, dtype=np.int64), cols.dst
    type_codes, type_values = cols.node_attrs.factorize("node_type")
    label_codes, label_values = cols.node_attrs.factorize("label")
    num_labels = max(len(label_values), 1)
    pair_codes, node_entries = np.unique(type_codes * num_labels + label_codes, return_inverse=True)
    entries = [
        (type_values[c // num_labels], label_values[c % num_labels]) for c in pair_codes.tolist()
    ]

    # Edges are in `G.edges` order: grouped by source (in node order), then by destination (in
    # adjacency order), so the parallel edges to a neighbor are consecutive
    src, dst = cols.src, cols.dst
    del cols
    if len(src) > 1:
        parallel = src[1:] == src[:-1]
        parallel &= dst[1:] == dst[:-1]
        if parallel.any():
            first = np.empty(len(src), dtype=bool)
            first[0] = True
            np.logical_not(parallel, out=first[1:])
            src, dst = src[first], dst[first]
        del parallel
    bounds = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=bounds[1:])
    return node_ids, node_entries.reshape(-1), entries, bounds, dst


def _neighbor_columns_of_graph(G):
    """`_neighbor_columns` of a networkx graph (graphs stored as plain pickles)."""
    node_ids = list(G.nodes())
    pos = {node: i for i, node in enumerate(node_ids)}
    index = {}
    node_entries = np.fromiter(
        (
            index.setdefault((G.nodes[node]["node_type"], G.nodes[node]["label"]), len(index))
            for node in node_ids
        ),
        dtype=np.int64,
        count=len(node_ids),
    )
    sizes, succ = [], []
    for node in node_ids:
        neighbors = [pos[neighbor] for neighbor in G.neighbors(node)]
        sizes.append(len(neighbors))
        succ.extend(neighbors)
    bounds = np.zeros(len(node_ids) + 1, dtype=np.int64)
    np.cumsum(sizes, out=bounds[1:])
    return node_ids, node_entries, list(index), bounds, np.array(succ, dtype=np.int64)


def _graph_documents(columns, new_nodes, entry_ids):
    """Documents of the nodes at positions `new_nodes` of a graph (given by its
    `_neighbor_columns`): the (node_type, label) pair of the node, then those of its neighbors.

    The pairs are given as IDs in `entry_ids`, which maps each pair to an ID and is extended with
    the new pairs, numbered in order of first appearance in the documents. Yields chunks of
    about _ENTRIES_CHUNK entries, as (entries, bounds): document i of a chunk is
    `entries[bounds[i]:bounds[i + 1]]` (int32 IDs).
    """
    _, node_entries, entries, bounds, succ = columns
    local2id = np.full(len(entries), -1, dtype=np.int64)
    num_neighbors = bounds[new_nodes + 1] - bounds[new_nodes]
    doc_ends = np.cumsum(1 + num_neighbors)
    start = 0
    while start < len(new_nodes):
        # Documents start:stop, of about _ENTRIES_CHUNK entries (at least one document)
        offset = doc_ends[start - 1] if start else 0
        stop = max(start + 1, int(np.searchsorted(doc_ends, offset + _ENTRIES_CHUNK, "right")))
        nodes, counts = new_nodes[start:stop], num_neighbors[start:stop]
        # Positions in `succ` of the neighbors of these nodes, in order
        neighbor_starts = np.cumsum(counts) - counts
        neighbors = np.arange(counts.sum()) + np.repeat(bounds[nodes] - neighbor_starts, counts)
        doc_bounds = np.zeros(len(nodes) + 1, dtype=np.int64)
        doc_bounds[1:] = doc_ends[start:stop] - offset
        heads = np.zeros(doc_bounds[-1], dtype=bool)
        heads[doc_bounds[:-1]] = True
        seq = np.empty(len(heads), dtype=np.int64)
        seq[heads] = node_entries[nodes]
        seq[~heads] = node_entries[succ[neighbors]]
        del heads, neighbors
        for e in pd.unique(seq).tolist():  # in order of first appearance
            if local2id[e] < 0:
                local2id[e] = entry_ids.setdefault(entries[e], len(entry_ids))
        yield local2id[seq].astype(np.int32), doc_bounds
        start = stop


class NeighborsCorpus:
    """The documents of `get_corpus_using_neighbors_features`, stored compactly: each document
    is kept as the IDs (int32) of its (node_type, label) entries, each distinct entry being
    tokenized once, and its tokens are only concatenated when the corpus is iterated.

    Iterating gives the same documents as the list returned before (same tokens, tags and
    order), so Doc2Vec trained on it learns the same model; `len()` is the number of documents.
    """

    def __init__(self, chunks, tags, entry_tokens, doc2vec_format):
        self.chunks = chunks  # (entries, bounds) of consecutive documents
        self.tags = tags
        self.entry_tokens = entry_tokens
        self.doc2vec_format = doc2vec_format

    def __len__(self):
        return len(self.tags)

    def __iter__(self):
        tokens_of = self.entry_tokens.__getitem__
        tags = iter(self.tags)
        for entries, bounds in self.chunks:
            bounds = bounds.tolist()
            for i in range(len(bounds) - 1):
                doc_entries = entries[bounds[i] : bounds[i + 1]].tolist()
                words = list(chain.from_iterable(map(tokens_of, doc_entries)))
                if self.doc2vec_format:
                    yield TaggedDocument(words=words, tags=[next(tags)])
                else:
                    yield words


# Used in Rcaid
def get_corpus_using_neighbors_features(cfg, doc2vec_format=False):
    """
    Same but also adds the tokens of neighbors in the final tokens: the document of each node
    (in the graphs of the training dates, the first time it appears) is the (node_type, label)
    pairs of the node and of its neighbors (`G.neighbors`), tokenized.

    The graphs are read one at a time, and the documents are returned as a `NeighborsCorpus`
    (4 bytes per (node_type, label) entry of a document) instead of a list of token lists.
    """
    splits = get_splits_to_train_featurization(cfg)
    dates = list(chain.from_iterable([getattr(cfg.dataset, f"{split}_dates") for split in splits]))
    sorted_paths = get_all_graphs_for_dates(cfg.transformation._graphs_dir, dates)

    # Ordered deduplication of the entries keeps the fitting of trainable tokenizers deterministic
    chunks, tags, entry_ids = [], [], {}
    nodes = set()
    for path in log_tqdm(sorted_paths, desc="Get corpus with neighbors"):
        columns = _neighbor_columns(path)
        node_ids = columns[0]
        new_nodes = [i for i, node in enumerate(node_ids) if node not in nodes]
        nodes.update(node_ids)
        tags.extend(node_ids[i] for i in new_nodes)
        new_nodes = np.array(new_nodes, dtype=np.int64)
        chunks.extend(_graph_documents(columns, new_nodes, entry_ids))
        del columns, node_ids, new_nodes
    del nodes

    entries = list(entry_ids)
    del entry_ids
    tokenizer = fit_tokenizer(cfg, entries)
    entry_tokens = [tokenizer.tokenize(label, node_type) for node_type, label in entries]
    del entries

    return NeighborsCorpus(chunks, tags, entry_tokens, doc2vec_format)
