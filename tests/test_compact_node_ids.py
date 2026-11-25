"""Tests of `batching.compact_node_ids`, which don't need a database."""

import pytest
import torch
from torch_geometric.data import Data

from pidsmaker.utils import data_utils
from pidsmaker.utils.data_utils import (
    compact_node_ids,
    get_original_node_ids,
    set_original_node_ids,
    to_original_node_ids,
)

pytestmark = pytest.mark.no_database


@pytest.fixture(autouse=True)
def reset_node_ids():
    yield
    set_original_node_ids(None)


def graph(src, dst):
    return Data(src=torch.tensor(src), dst=torch.tensor(dst), t=torch.arange(len(src)))


def test_compaction_keeps_order_and_restores_ids():
    train = [[graph([10, 500, 7], [500, 3, 10])]]
    val = [[graph([42], [7])]]
    test = [[graph([1_000_000, 3], [42, 1_000_000])]]
    originals = [g.src.clone() for g in (train[0][0], val[0][0], test[0][0])], [
        g.dst.clone() for g in (train[0][0], val[0][0], test[0][0])
    ]

    node_ids = compact_node_ids([train, val, test])

    assert node_ids.tolist() == [3, 7, 10, 42, 500, 1_000_000]
    assert get_original_node_ids() is node_ids
    graphs = [train[0][0], val[0][0], test[0][0]]
    # Dense IDs, in the order of the original IDs
    assert torch.cat([torch.cat([g.src, g.dst]) for g in graphs]).unique().tolist() == list(range(6))
    for g, src, dst in zip(graphs, *originals):
        assert g.src.dtype == src.dtype
        assert torch.equal(torch.argsort(g.src, stable=True), torch.argsort(src, stable=True))
        assert torch.equal(to_original_node_ids(g.src), src)
        assert torch.equal(to_original_node_ids(torch.stack([g.src, g.dst])), torch.stack([src, dst]))


def test_no_compaction_keeps_ids():
    ids = torch.tensor([[5, 9], [9, 5]])
    assert data_utils.get_original_node_ids() is None
    assert to_original_node_ids(ids) is ids
