"""R-Caid model as described in the paper (Goyal et al., IEEE S&P 2024): encoder with root
attention, process tags and their objective. Runs without database."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pidsmaker.encoders.rcaid_encoder import RCaidGAT, split_pseudo_edges
from pidsmaker.objectives.predict_process_tag import (
    ProcessTagPrediction,
    build_process_tag_classes,
)
from pidsmaker.utils.process_tags import process_tag, truncate_process_name

pytestmark = pytest.mark.no_database


# --------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------


def test_split_pseudo_edges():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    types = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    local, pseudo = split_pseudo_edges(edge_index, types)
    assert local.tolist() == [[0, 2], [1, 0]] and pseudo.tolist() == [[1], [2]]
    # Without typed edges, all the edges are both
    for t in (None, torch.zeros(3, 2)):
        local, pseudo = split_pseudo_edges(edge_index, t)
        assert local is edge_index and pseudo is edge_index


def test_encoder_roots_send_their_input_features():
    torch.manual_seed(0)
    enc = RCaidGAT(in_dim=8, hid_dim=16, out_dim=4, dropout=0.0, num_layers=2).eval()
    x = torch.randn(5, 8)
    # Local edges 0->1->2->3, pseudo edge 4->3 (4 has no local edge)
    edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 3, 3]])
    types = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    out = enc(x, edge_index, edge_types=types)["h"]
    assert out.shape == (5, 4)
    # Node 3 depends on the input features of its pseudo-root
    x2 = x.clone()
    x2[4] += 1.0
    assert not torch.allclose(enc(x2, edge_index, edge_types=types)["h"][3], out[3])
    # Node 0 receives nothing: only its own features matter
    x3 = x.clone()
    x3[1:] += 1.0
    assert torch.allclose(enc(x3, edge_index, edge_types=types)["h"][0], out[0], atol=1e-6)


# --------------------------------------------------------------------------------------
# Process tags
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,tag,truncated",
    [
        ("/bin/dash /bin/sh -e /usr/lib/x", "dash", "/bin/"),
        ("/usr/sbin/sshd sshd: admin [priv]", "sshd", "/usr/sbin/"),
        ("None sleep", "sleep", ""),
        ("None", "", ""),
        ("", "", ""),
        ("firefox", "firefox", ""),
        ("C:\\Windows\\cmd.exe cmd /c", "cmd.exe", "C:/Windows/"),
    ],
)
def test_process_tags(label, tag, truncated):
    assert process_tag(label) == tag
    assert truncate_process_name(label) == truncated


def test_process_tag_objective():
    indexid2msg = {
        "0": ("subject", "None sleep"),
        "1": ("file", "/etc/passwd"),
        "2": ("subject", "/bin/sh /bin/sh -c x"),
        "3": ("subject", "None sleep"),
        "5": ("subject", "None evil"),  # only in test
    }
    classes, num_known = build_process_tag_classes(indexid2msg, train_nodes={"0", "1", "2", "3"})
    assert num_known == 2
    assert classes.tolist() == [0, -1, 1, 0, -1, 2]

    torch.manual_seed(0)
    obj = ProcessTagPrediction(torch.nn.Linear(4, 3), classes, num_known, balanced_loss=False)
    h = torch.randn(4, 4, requires_grad=True)
    batch = SimpleNamespace(original_n_id=torch.tensor([1, 0, 5, 2]))
    loss = obj(h=h, inference=False, batch=batch)["loss"]
    logits = obj.decoder(h)
    # Unknown tags are not targets
    expected = torch.nn.functional.cross_entropy(logits[[1, 3]], torch.tensor([0, 1]))
    assert torch.allclose(loss, expected)
    scores = obj(h=h, inference=True, batch=batch)["loss"]
    assert scores[0] == 0  # not a process
    assert torch.allclose(
        scores[[1, 2, 3]],
        torch.nn.functional.cross_entropy(logits[[1, 2, 3]], torch.tensor([0, 2, 1]), reduction="none"),
    )


def test_process_tag_kmeans_detection():
    """`kmeans`: distance to the median of the nearest cluster of the training embeddings."""
    classes = np.zeros(400, dtype=np.int64)
    obj = ProcessTagPrediction(torch.nn.Linear(2, 2), classes, 1, False, detection="kmeans",
                               max_train_embeddings=150, k_values=[1, 2, 3, 4])
    torch.manual_seed(0)
    # Two training clusters, around (0, 0) and (10, 10), seen in 4 graphs
    train = torch.cat([torch.randn(200, 2) * 0.1, torch.randn(200, 2) * 0.1 + 10])[torch.randperm(400)]
    for i in range(4):
        part = train[i * 100 : (i + 1) * 100]
        obj(h=part, inference=False, batch=SimpleNamespace(original_n_id=torch.arange(100)))
    assert 100 < sum(len(e) for e, _ in obj._train_embs) < 200  # 150 expected
    h = torch.tensor([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0], [0.0, 3.0]])
    scores = obj(h=h, inference=True, batch=SimpleNamespace(original_n_id=torch.arange(4)))["loss"]
    assert obj._cluster_medians("cpu").shape[0] == 2
    assert scores[0] < 0.5 and scores[1] < 0.5
    assert scores[2] > 6 and abs(scores[3] - 3) < 0.5
    obj.reset_state()
    with pytest.raises(RuntimeError):
        obj(h=h, inference=True, batch=SimpleNamespace(original_n_id=torch.arange(4)))
