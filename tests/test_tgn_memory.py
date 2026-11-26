"""Tests of the TGN memory (pidsmaker/tgn.py), which don't need a database."""

import pytest
import torch

from pidsmaker.tgn import IdentityMessage, LastAggregator, TGNMemory

pytestmark = pytest.mark.no_database


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
    # Same computation, up to float32 rounding: the GRU's matrix products take other kernels
    # for 1 row than for 100 on some CPUs (up to 5e-6 apart on the CI runners)
    torch.testing.assert_close(mem_chunked, mem_full, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(mem_full, make_memory(num_nodes).memory)  # the flush changed it


def make_memories(num_nodes, seed=0):
    """A memory with the `LastMessageStore` and one with the `MessageStore`, in the same state."""
    memories = []
    for dense in [True, False]:
        torch.manual_seed(seed)
        memory = TGNMemory(
            num_nodes, 6, 8, 4,
            message_module=IdentityMessage(6, 8, 4),
            aggregator_module=LastAggregator(), device="cpu", dense_last_messages=dense,
        )
        memory.memory.normal_(generator=torch.Generator().manual_seed(seed))
        memory.last_update.random_(0, 50, generator=torch.Generator().manual_seed(seed))
        memories.append(memory)
    assert memories[0].dense_last_messages and not memories[1].dense_last_messages
    return memories


@pytest.mark.parametrize("train", [True, False])
def test_dense_last_messages_match_message_store(train):
    """Keeping only the last message of each node gives the memory, last updates and gradients
    of the store of all the messages followed by the `LastAggregator` (distinct timestamps)."""
    num_nodes, num_edges = 60, 600
    g = torch.Generator().manual_seed(2)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    # Distinct timestamps, not sorted within a batch
    t = 100 + torch.randperm(num_edges * 3, generator=g)[:num_edges]
    t = torch.cat([b.sort().values.flip(0) if i % 2 else b for i, b in enumerate(t.split(40))])
    raw_msg = torch.randn(num_edges, 6, generator=g)
    batches = torch.arange(num_edges).split(40)
    # Nodes of each batch and some neighbors
    n_ids = [
        torch.cat([src[b], dst[b], torch.randint(0, num_nodes, (10,), generator=g)]).unique()
        for b in batches
    ]

    dense, reference = make_memories(num_nodes)
    outputs = []
    for memory in [dense, reference]:
        memory.train(train)
        outs = []
        for batch, n_id in zip(batches, n_ids):
            s, d = src[batch], dst[batch]
            mem, last_update = memory(n_id)
            if train:
                mem.sum().backward()
            outs.append((mem.detach(), last_update))
            memory.update_state(s, d, t[batch], raw_msg[batch])
            memory.detach()
        grads = [p.grad.clone() for p in memory.parameters() if p.grad is not None]
        outputs.append((outs, memory.memory.clone(), memory.last_update.clone(), grads))

    (outs_a, mem_a, lu_a, grads_a), (outs_b, mem_b, lu_b, grads_b) = outputs
    for (m_a, l_a), (m_b, l_b) in zip(outs_a, outs_b):
        assert torch.equal(l_a, l_b)
        torch.testing.assert_close(m_a, m_b, rtol=0, atol=1e-6)
    assert torch.equal(lu_a, lu_b)
    torch.testing.assert_close(mem_a, mem_b, rtol=0, atol=1e-6)
    assert len(grads_a) == len(grads_b)
    for ga, gb in zip(grads_a, grads_b):
        torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-5)
