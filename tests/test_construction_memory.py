"""Tests of the memory-saving structures of graph construction, which don't need a database."""

import os
from types import SimpleNamespace

import pytest
import torch

from pidsmaker.preprocessing.build_graph_methods.build_default_graphs import (
    compute_and_save_split2nodes,
)
from pidsmaker.utils.graph_store import save_edge_list

pytestmark = pytest.mark.no_database


def make_cfg(tmp_path):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_dates=["2018-04-02", "2018-04-03"], val_dates=["2018-04-04"],
            test_dates=["2018-04-05"],
        ),
        construction=SimpleNamespace(
            _graphs_dir=str(tmp_path / "graphs"), _dicts_dir=str(tmp_path / "dicts")
        ),
    )


def save_window(cfg, date, name, nodes):
    date_dir = os.path.join(cfg.construction._graphs_dir, f"graph_{date}")
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, name)
    src, dst = nodes[:-1], nodes[1:]
    edge_attrs = {"event_uuid": [f"e{i}" for i in range(len(src))], "time": list(range(len(src)))}
    node_attrs = {"node_type": ["subject"] * len(nodes), "label": ["x"] * len(nodes)}
    save_edge_list(path, nodes, node_attrs, src, dst, edge_attrs)
    return os.path.normpath(path)


def test_split2nodes_with_day_node_sets(tmp_path):
    """The graphs of a day sharing the set of the day's nodes give the same split2nodes as the
    nodes of each graph (recorded or reloaded from disk), and each shared set is merged once."""
    cfg = make_cfg(tmp_path)
    windows = {
        "2018-04-02": [["1", "2", "3"], ["3", "4"]],
        "2018-04-03": [["5", "6"], ["1", "7"]],
        "2018-04-04": [["8", "9"]],
        "2018-04-05": [["2", "10"], ["11", "12", "13"]],
    }
    per_graph, per_day = {}, {}
    for date, graphs in windows.items():
        day_nodes = set()
        for i, nodes in enumerate(graphs):
            path = save_window(cfg, date, f"w{i}", nodes)
            per_graph[path] = set(nodes)
            day_nodes.update(nodes)
            per_day[path] = day_nodes

    from_disk = compute_and_save_split2nodes(cfg)
    from_graphs = compute_and_save_split2nodes(cfg, file2nodes=per_graph)
    from_days = compute_and_save_split2nodes(cfg, file2nodes=per_day)
    expected = {
        "train": {"1", "2", "3", "4", "5", "6", "7"},
        "val": {"8", "9"},
        "test": {"2", "10", "11", "12", "13"},
    }
    assert from_disk == from_graphs == from_days == expected
    assert torch.load(os.path.join(cfg.construction._dicts_dir, "split2nodes.pkl")) == expected
