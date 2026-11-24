"""Flash's corpus building and node embedding give the same results as their original
implementation (kept below), on small synthetic graphs. Runs without a database:

    python -m pytest tests/test_flash_featurization.py
"""

import random
from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
from gensim.models import Word2Vec

from pidsmaker.featurization.feat_inference_methods import feat_inference_flash
from pidsmaker.featurization.featurization_methods import featurization_flash
from pidsmaker.tokenization import NLTKTokenizer
from pidsmaker.utils.graph_store import save_graph

pytestmark = pytest.mark.no_database


# --------------------------------------------------------------------------------------
# Original implementation
# --------------------------------------------------------------------------------------


def original_get_node2corpus(sorted_paths, indexid2msg, tokenizer, fitted):
    data_of_graphs = []
    for file_path in sorted_paths:
        graph = torch.load(file_path)
        sorted_edges = sorted(
            [
                (u, v, attr["label"], int(attr["time"]))
                for u, v, key, attr in graph.edges(data=True, keys=True)
            ],
            key=lambda x: x[3],
        )
        nodes, node_types, edges = defaultdict(list), {}, []
        for e in sorted_edges:
            src, dst, operation, t = e
            src_type, src_msg = indexid2msg[src]
            dst_type, dst_msg = indexid2msg[dst]
            properties = [(src_type, src_msg), (None, operation), (dst_type, dst_msg)]
            if len(nodes[src]) < 300:
                nodes[src].extend(properties)
            node_types[src] = src_type
            if len(nodes[dst]) < 300:
                nodes[dst].extend(properties)
            node_types[dst] = dst_type
            edges.append((src, dst))
        features, types, index_map = [], [], {}
        for node_id, props in nodes.items():
            features.append(props)
            types.append(node_types[node_id])
            index_map[node_id] = len(features) - 1
        data_of_graphs.append((features, types, list(index_map.keys())))

    texts = dict.fromkeys(text for graphs in data_of_graphs for msg in graphs[0] for text in msg)
    fitted.append(list(texts))

    token_cache = {}
    node2corpus = defaultdict(list)
    for graphs in data_of_graphs:
        for msg, node_type, node_id in zip(graphs[0], graphs[1], graphs[2]):
            for text in msg:
                if text not in token_cache:
                    text_type, sentence = text
                    token_cache[text] = tokenizer.tokenize(sentence, text_type)
                node2corpus[node_id].extend(token_cache[text])
    return node2corpus


def original_infer(document, w2vmodel, encoder):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    embedding_dim = w2vmodel.vector_size
    if not word_embeddings:
        return np.zeros(embedding_dim)
    output_embedding = torch.tensor(np.array(word_embeddings), dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)
    return np.mean(output_embedding.detach().cpu().numpy(), axis=0)


# --------------------------------------------------------------------------------------
# Synthetic graphs
# --------------------------------------------------------------------------------------


def _graphs(tmp_path, seed, num_graphs=4, legacy=False):
    rng = random.Random(seed)
    num_nodes = 60
    types = ["subject", "file", "netflow"]
    indexid2msg = {
        str(i): (
            types[i % 3],
            f"/usr/lib{i % 7}/x{i % 11}.so" if i % 3 != 2 else f"10.0.{i % 5}.{i}:80",
        )
        for i in range(num_nodes)
    }
    paths = []
    for g in range(num_graphs):
        G = nx.MultiDiGraph()
        nodes = [str(i) for i in rng.sample(range(num_nodes), 40)]
        for n in nodes:
            G.add_node(n, node_type=indexid2msg[n][0], label=indexid2msg[n][1])
        hub = nodes[0]  # with more than 100 edges: only its first 100 edges give texts
        for _ in range(rng.choice([0, 30, 200, 600]) if g else 400):
            u = hub if rng.random() < 0.4 else rng.choice(nodes)
            v = rng.choice(nodes) if rng.random() < 0.9 else u  # some self-loops
            if rng.random() < 0.5:
                u, v = v, u
            G.add_edge(
                u,
                v,
                time=10**18 + rng.randrange(50),  # many equal times: the sort must be stable
                label=rng.choice(["EVENT_READ", "EVENT_WRITE", "EVENT_EXECUTE"]),
                y=0,
            )
        path = tmp_path / f"g{g}"
        if legacy:
            torch.save(G, path)  # pickled networkx graph
        else:
            save_graph(G, path)
        paths.append(str(path))
    return paths, indexid2msg


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_flash_node2corpus(tmp_path, monkeypatch, seed, fit, legacy):
    paths, indexid2msg = _graphs(tmp_path, seed, legacy=legacy)
    tokenizer = NLTKTokenizer(type_aware=False)
    expected_fit = []
    expected = original_get_node2corpus(paths, indexid2msg, tokenizer, expected_fit)

    got_fit = []
    m = featurization_flash
    monkeypatch.setattr(m, "get_indexid2msg", lambda cfg: indexid2msg)
    monkeypatch.setattr(m, "get_all_graphs_for_dates", lambda base_dir, dates: paths)
    monkeypatch.setattr(m, "fit_tokenizer", lambda cfg, labels: got_fit.append(labels) or tokenizer)
    monkeypatch.setattr(m, "load_tokenizer", lambda cfg: tokenizer)
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(train_dates=["d"]), transformation=SimpleNamespace(_graphs_dir="")
    )
    got = m.get_node2corpus(cfg, ["train"], fit=fit)
    assert type(got) is type(expected)
    assert list(got.items()) == list(expected.items())
    if fit:
        assert got_fit == expected_fit


def test_flash_infer():
    rng = random.Random(0)
    words = [f"w{i}" for i in range(50)]
    sentences = [[rng.choice(words) for _ in range(20)] for _ in range(50)]
    model = Word2Vec(sentences, vector_size=30, min_count=1, workers=1, epochs=2, seed=0)
    encoder = feat_inference_flash.PositionalEncoder(30, max_len=1000)
    documents = [[], ["unknown"], ["w1"], ["w1", "unknown", "w2"] * 50]
    documents += [
        [rng.choice(words + ["oov"]) for _ in range(rng.randrange(1, 900))] for _ in range(30)
    ]
    for doc in documents:
        expected = original_infer(doc, model, encoder)
        got = feat_inference_flash.infer(doc, model, encoder)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_flash_infer_long_documents():
    """Documents of 100,000 words or more are not positionally encoded."""
    model = Word2Vec([["a", "b"]], vector_size=8, min_count=1, workers=1, epochs=1, seed=0)
    encoder = feat_inference_flash.PositionalEncoder(8, max_len=100000)
    for doc in (["a", "b", "c"] * 33333, ["a", "b", "c"] * 34000):
        expected = original_infer(doc, model, encoder)
        assert np.array_equal(feat_inference_flash.infer(doc, model, encoder), expected)
