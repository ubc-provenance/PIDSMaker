"""Tests of the TGN memory (pidsmaker/tgn.py), which don't need a database."""

import pytest
import torch

from pidsmaker.tgn import IdentityMessage, LastAggregator, TGNMemory


def make_memory(num_nodes, raw_msg_dim=6, memory_dim=8, time_dim=4, seed=0):
    torch.manual_seed(seed)
    memory = TGNMemory(
        num_nodes, raw_msg_dim, memory_dim, time_dim,
        message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(), device="cpu",
    )
    # A non-zero memory, so that the messages depend on the memory of their neighbors
    memory.memory.normal_()
    memory.last_update.random_(0, 50)
    return memory


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_chunked_flush_matches_single_update(chunk_size):
    """Flushing the message store in chunks of nodes (`model.eval()` on large graphs) gives the
    memory of a single update of all nodes: every chunk reads the memory before the update."""
    num_nodes, num_edges = 100, 400
    g = torch.Generator().manual_seed(1)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    t = torch.randint(50, 1000, (num_edges,), generator=g)
    raw_msg = torch.randn(num_edges, 6, generator=g)

    results = []
    for size in [num_nodes, chunk_size]:
        memory = make_memory(num_nodes)
        memory.flush_chunk_size = size
        memory.train()
        with torch.no_grad():
            # Messages are stored, not yet applied: the update of each batch only covers its nodes
            for batch in torch.arange(num_edges).split(50):
                memory.update_state(src[batch], dst[batch], t[batch], raw_msg[batch])
            memory.eval()
        results.append((memory.memory.clone(), memory.last_update.clone()))

    (mem_full, lu_full), (mem_chunked, lu_chunked) = results
    assert torch.equal(lu_full, lu_chunked)
    torch.testing.assert_close(mem_chunked, mem_full, rtol=0, atol=1e-6)
    assert not torch.allclose(mem_full, make_memory(num_nodes).memory)  # the flush changed it
