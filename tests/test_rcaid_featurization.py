"""R-Caid's doc2vec corpus (`get_corpus_using_neighbors_features`) gives the same documents as
its original implementation (kept below), and so the same Doc2Vec model, on small synthetic
graphs. Runs without a database:

    python -m pytest tests/test_rcaid_featurization.py
"""

import functools
import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from pidsmaker.featurization import featurization_utils
from pidsmaker.featurization.featurization_methods import featurization_doc2vec
from pidsmaker.preprocessing.transformation_methods import transformation_rcaid_pseudo_graph
from pidsmaker.tokenization import NLTKTokenizer
from pidsmaker.utils.graph_store import save_graph

pytestmark = pytest.mark.no_database


# --------------------------------------------------------------------------------------
# Original implementation
# --------------------------------------------------------------------------------------


def original_get_corpus_using_neighbors_features(sorted_paths, tokenizer, fitted, doc2vec_format):
    graph_list = [torch.load(path) for path in sorted_paths]

    documents = []
    nodes = set()
    for G in graph_list:
        for node in G.nodes():
            if node not in nodes:
                nodes.add(node)

                document = [(G.nodes[node]["node_type"], G.nodes[node]["label"])]
                for neighbor in G.neighbors(node):
                    document.append((G.nodes[neighbor]["node_type"], G.nodes[neighbor]["label"]))
                documents.append((node, document))

    unique_entries = dict.fromkeys(entry for _, document in documents for entry in document)
    fitted.append(list(unique_entries))

    words = []
    for node, document in documents:
        tokens = [
            token
            for node_type, label in document
            for token in tokenizer.tokenize(label, node_type)
        ]
        if doc2vec_format:
            words.append(TaggedDocument(words=tokens, tags=[node]))
        else:
            words.append(tokens)

    return words


# --------------------------------------------------------------------------------------
# Synthetic graphs
# --------------------------------------------------------------------------------------

TYPES = ["subject", "file", "netflow"]
LABELS = ["", "/bin/sh", "/usr/lib/x.so", "C:\\Windows\\a.dll", "/tmp/f g", "sh -c ls"]


def _label(rng, node_type):
    if node_type == "netflow":
        return f"10.0.{rng.randrange(3)}.{rng.randrange(4)}:{rng.choice([80, 443])}"
    # Few distinct labels: many nodes share a label, also with a different type
    return rng.choice(LABELS)


def _graphs(tmp_path, seed, storage, graph_class=nx.MultiDiGraph, num_graphs=5):
    """Graphs sharing nodes (sampled from a common pool) with a hub node that has an edge to
    most nodes, nodes without edges, self-loops, parallel edges and empty graphs."""
    rng = random.Random(seed)
    num_nodes = 80
    node_ids = [str(i) for i in range(num_nodes)] if seed % 2 == 0 else list(range(num_nodes))
    attrs = {}
    for n in node_ids:
        node_type = rng.choice(TYPES)
        attrs[n] = {"node_type": node_type, "label": _label(rng, node_type)}
    paths = []
    for g in range(num_graphs):
        G = graph_class()
        num_edges = 0 if g == 2 else rng.choice([1, 50, 300])
        nodes = rng.sample(node_ids, 60 if g == 0 else rng.randrange(1, 60)) if g != 3 else []
        for n in nodes:
            G.add_node(n, **attrs[n])
        hub = nodes[0] if nodes else None
        if hub is not None and g != 2:
            for v in rng.sample(nodes, int(0.8 * len(nodes))):  # the hub
                G.add_edge(hub, v, time=rng.randrange(1000))
        for _ in range(num_edges if nodes else 0):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if rng.random() < 0.1:
                v = u  # self-loop
            for _ in range(rng.choice([1, 1, 3])):  # parallel edges
                G.add_edge(u, v, time=rng.randrange(1000))
        path = tmp_path / f"g{g}"
        if storage == "legacy":
            torch.save(G, path)  # pickled networkx graph
        elif storage == "rcaid" and G.number_of_edges():
            # As stored by R-Caid's transformation
            rcaid_cfg = SimpleNamespace(use_pruning=g % 2)
            cfg = SimpleNamespace(transformation=SimpleNamespace(rcaid_pseudo_graph=rcaid_cfg))
            np.random.seed(g)
            transformation_rcaid_pseudo_graph.transform_and_save(G, cfg, str(path))
        else:
            save_graph(G, path)
        paths.append(str(path))
    return paths


def _cfg():
    return SimpleNamespace(
        featurization=SimpleNamespace(training_split="train"),
        dataset=SimpleNamespace(train_dates=["d"]),
        transformation=SimpleNamespace(_graphs_dir=""),
    )


def _get_corpus(monkeypatch, paths, tokenizer, fitted, doc2vec_format):
    m = featurization_utils
    monkeypatch.setattr(m, "get_all_graphs_for_dates", lambda base_dir, dates: paths)
    monkeypatch.setattr(m, "fit_tokenizer", lambda cfg, labels: fitted.append(labels) or tokenizer)
    return m.get_corpus_using_neighbors_features(_cfg(), doc2vec_format=doc2vec_format)


def _types(corpus):
    return [type(tag) for doc in corpus for tag in doc.tags]


@pytest.mark.parametrize("chunk", [None, 1, 7])
@pytest.mark.parametrize("doc2vec_format", [True, False])
@pytest.mark.parametrize("graph_class", [nx.MultiDiGraph, nx.DiGraph])
@pytest.mark.parametrize("storage", ["compact", "legacy", "rcaid"])
@pytest.mark.parametrize("seed", range(4))
def test_rcaid_corpus(tmp_path, monkeypatch, seed, storage, graph_class, doc2vec_format, chunk):
    if chunk is not None:  # entries ordered by chunks of documents
        monkeypatch.setattr(featurization_utils, "_ENTRIES_CHUNK", chunk)
    paths = _graphs(tmp_path, seed, storage, graph_class)
    tokenizer = NLTKTokenizer(type_aware=True)
    expected_fit = []
    expected = original_get_corpus_using_neighbors_features(
        paths, tokenizer, expected_fit, doc2vec_format
    )
    lengths = [len(doc.words if doc2vec_format else doc) for doc in expected]
    assert max(lengths) > 60  # the hub

    got_fit = []
    corpus = _get_corpus(monkeypatch, paths, tokenizer, got_fit, doc2vec_format)
    assert got_fit == expected_fit
    assert len(corpus) == len(expected)
    for _ in range(2):  # the corpus can be iterated several times
        got = list(corpus)
        assert got == expected
        assert [type(doc) for doc in got] == [type(doc) for doc in expected]
        if doc2vec_format:
            assert _types(got) == _types(expected)


def test_rcaid_doc2vec_model(tmp_path, monkeypatch):
    """Doc2Vec trained on the corpus is the same as trained on the original list (with one
    worker: several workers make the training non-deterministic)."""
    paths = _graphs(tmp_path, 0, "rcaid")
    tokenizer = NLTKTokenizer(type_aware=True)
    expected = original_get_corpus_using_neighbors_features(paths, tokenizer, [], True)
    corpus = _get_corpus(monkeypatch, paths, tokenizer, [], True)

    monkeypatch.setattr(featurization_doc2vec, "Doc2Vec", functools.partial(Doc2Vec, workers=1))
    models = [
        featurization_doc2vec.doc2vec(
            cfg=None,
            tagged_data=data,
            model_save_path=str(tmp_path / name),
            epochs=3,
            emb_dim=16,
            alpha=0.025,
        )
        for name, data in [("expected", expected), ("got", corpus)]
    ]
    expected_model, model = models
    assert model.corpus_count == expected_model.corpus_count
    assert model.corpus_total_words == expected_model.corpus_total_words
    assert model.wv.index_to_key == expected_model.wv.index_to_key
    assert model.dv.index_to_key == expected_model.dv.index_to_key
    assert np.array_equal(model.wv.vectors, expected_model.wv.vectors)
    assert np.array_equal(model.dv.vectors, expected_model.dv.vectors)
    assert np.array_equal(model.syn1neg, expected_model.syn1neg)
    for doc in expected:
        # infer_vector starts from a random vector seeded by the text, except for an empty text
        if " ".join(doc.words):
            assert np.array_equal(
                model.infer_vector(doc.words), expected_model.infer_vector(doc.words)
            )


def test_node_vectors_do_not_depend_on_inference_order():
    """Each node vector is the same whatever the nodes inferred before it, and a node
    without tokens gets a zero vector instead of an unseeded random one."""
    from pidsmaker.featurization.feat_inference_methods.feat_inference_doc2vec import (
        infer_node_vector,
    )

    docs = [TaggedDocument(words, [i]) for i, words in enumerate(
        [["usr", "bin", "sshd"], ["tmp", "x"], ["etc", "passwd"], ["usr", "lib"], [""]] * 5
    )]
    model = Doc2Vec(docs, vector_size=8, min_count=1, epochs=2, workers=1, seed=0)
    queries = [["usr", "bin", "sshd"], ["tmp", "x"], [], [""], ["etc", "passwd", "usr"]]

    forward = [infer_node_vector(model, q) for q in queries]
    backward = [infer_node_vector(model, q) for q in queries[::-1]][::-1]
    for a, b in zip(forward, backward):
        assert np.array_equal(a, b)
    assert not forward[2].any() and not forward[3].any()
