import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from torch_geometric.utils._scatter import scatter_argmax

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class MessageStore:
    """Vectorized per-node message store of TGN.

    For each node `i`, it holds the events `(src, dst, t, msg)` of the last
    `update()` call where `i` appeared in `src`, like the original PyG store
    `{i: (src, dst, t, msg)}`, but without a Python dict of `num_nodes` entries and
    without per-node Python loops: events are appended to growable buffers and each
    node points to its contiguous slice. Stale slices are compacted lazily.
    `get(n_id)` returns the concatenation of the slices of `n_id`, in the same
    order as the original implementation.
    """

    def __init__(self, num_nodes: int, msg_dim: int = None, device=None):
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.device = device
        self.reset()

    def reset(self, device=None):
        if device is not None:
            self.device = device
        d = self.device
        self.start = torch.zeros(self.num_nodes, dtype=torch.long, device=d)
        self.count = torch.zeros(self.num_nodes, dtype=torch.long, device=d)
        self.size = 0  # number of used rows in the buffers
        self.live = 0  # number of rows referenced by a node
        self._alloc(1024)

    def _alloc(self, capacity):
        d = self.device
        self.src = torch.empty(capacity, dtype=torch.long, device=d)
        self.dst = torch.empty(capacity, dtype=torch.long, device=d)
        self.t = torch.empty(capacity, dtype=torch.long, device=d)
        self.msg = (
            torch.empty((capacity, self.msg_dim), device=d) if self.msg_dim is not None else None
        )

    def _to(self, device):
        if self.start.device != device:
            self.device = device
            for k in ["start", "count", "src", "dst", "t", "msg"]:
                v = getattr(self, k)
                if v is not None:
                    setattr(self, k, v.to(device))

    def _compact_and_reserve(self, extra):
        """Keeps only live rows and ensures room for `extra` more rows."""
        if self.size + extra <= self.src.numel():
            return
        if self.live + extra <= self.src.numel() // 2:
            capacity = self.src.numel()  # compacting is enough
        else:
            capacity = max(2 * (self.live + extra), 1024)

        idx, starts = self._gather_index(torch.nonzero(self.count).view(-1))
        src, dst, t = self.src[idx], self.dst[idx], self.t[idx]
        msg = self.msg[idx] if self.msg is not None else None
        nodes = torch.nonzero(self.count).view(-1)

        self._alloc(capacity)
        n = idx.numel()
        self.src[:n], self.dst[:n], self.t[:n] = src, dst, t
        if msg is not None:
            self.msg[:n] = msg
        self.start[nodes] = starts
        self.size = self.live = n

    def _gather_index(self, n_id):
        """Returns the buffer rows of the slices of `n_id` (concatenated) and their new starts."""
        counts = self.count[n_id]
        offsets = torch.cumsum(counts, 0) - counts
        total = int(counts.sum())
        # Row k of node j's slice is at `start_j + k`, i.e. `position + (start_j - offset_j)`
        shift = torch.repeat_interleave(self.start[n_id] - offsets, counts, output_size=total)
        idx = torch.arange(total, device=counts.device) + shift
        return idx, offsets

    def update(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor = None):
        self._to(src.device)
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        num = src.numel()

        # Rows of the nodes being overwritten become stale
        self.live -= int(self.count[n_id].sum())
        self.count[n_id] = 0
        self._compact_and_reserve(num)

        lo, hi = self.size, self.size + num
        self.src[lo:hi] = src[perm]
        self.dst[lo:hi] = dst[perm]
        self.t[lo:hi] = t[perm]
        if self.msg is not None:
            self.msg[lo:hi] = raw_msg[perm]
        self.start[n_id] = lo + torch.cumsum(count, 0) - count
        self.count[n_id] = count
        self.size = hi
        self.live += num

    def get(self, n_id: Tensor):
        self._to(n_id.device)
        idx, _ = self._gather_index(n_id)
        msg = self.msg[idx] if self.msg is not None else None
        return self.src[idx], self.dst[idx], self.t[idx], msg


class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        device,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.device = device

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = MessageStore(num_nodes, raw_msg_dim, device)
        self.msg_d_store = MessageStore(num_nodes, raw_msg_dim, device)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp.
        """
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`.
        """
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        # Message store format: (src, dst, t, msg)
        self.msg_s_store.reset(self.memory.device)
        self.msg_d_store.reset(self.memory.device)

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, msg_store: MessageStore
    ):
        msg_store.update(src, dst, t, raw_msg)

    def _compute_msg(self, n_id: Tensor, msg_store: MessageStore, msg_module: Callable):
        src, dst, t, raw_msg = msg_store.get(n_id)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # NOTE: this line below was originaly here. However, it is called when calling model.eval()
            # during inference and generates a memory overflow on large datasets like CADETS_E5.
            # As we load directly the memory from a pkl file, I don't think we need this line anymore so we
            # removed it. If results become strange in the future, try to add this line back to ensure it
            # doen't affect performance.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


class TimeEncodingMemory(torch.nn.Module):
    """Custom lightweight class to only memorize past timestamps."""

    def __init__(self, num_nodes: int, time_dim: int, device):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.time_enc = TimeEncoder(time_dim)
        self.device = device

        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = MessageStore(num_nodes, None, device)
        self.msg_d_store = MessageStore(num_nodes, None, device)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.time_enc.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.last_update)
        self._reset_message_store()

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`.
        """
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        # Message store format: (src, dst, t)
        self.msg_s_store.reset(self.last_update.device)
        self.msg_d_store.reset(self.last_update.device)

    def _update_memory(self, n_id: Tensor):
        last_update = self.get_last_update(n_id)
        self.last_update[n_id] = last_update

    def _compute_last_update(self, n_id: Tensor, msg_store: MessageStore):
        src, dst, t, _ = msg_store.get(n_id)
        return t, src, dst

    def get_last_update(self, n_id: Tensor) -> Tensor:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        t_s, src_s, dst_s = self._compute_last_update(n_id, self.msg_s_store)
        t_d, src_d, dst_d = self._compute_last_update(n_id, self.msg_d_store)

        idx = torch.cat([src_s, src_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        dim_size = self.last_update.size(0)

        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        return last_update

    def _update_msg_store(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, msg_store: MessageStore
    ):
        msg_store.update(src, dst, t)

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # NOTE: this line below was originaly here. However, it is called when calling model.eval()
            # during inference and generates a memory overflow on large datasets like CADETS_E5.
            # As we load directly the memory from a pkl file, I don't think we need this line anymore so we
            # removed it. If results become strange in the future, try to add this line back to ensure it
            # doen't affect performance.
            self._update_memory(torch.arange(self.num_nodes, device=self.device))
            self._reset_message_store()
        super().train(mode)


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, directed=False, device=None):
        self.size = size
        self.directed = directed

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def to(self, device):
        self.neighbors = self.neighbors.to(device)
        self.e_id = self.e_id.to(device)
        self._assoc = self._assoc.to(device)
        return self

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        if self.directed:
            # New version: the graph is directed
            nodes = dst
            neighbors = src
            e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0), device=src.device)
        else:
            # Default version: the graph is undirected
            neighbors = torch.cat([src, dst], dim=0)
            nodes = torch.cat([dst, src], dim=0)
            e_id = torch.arange(
                self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
            ).repeat(2)

        self.cur_e_id += src.numel()

        # Sort nodes for consistent processing (preserves time order within nodes)
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        # Map nodes to dense indices
        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        # Compute the maximum number of edges for any node in the batch
        edge_counts = torch.bincount(self._assoc[nodes], minlength=n_id.numel())
        temp_size = edge_counts.max().item() if edge_counts.numel() > 0 else 1

        # Compute cumulative start indices
        cum_edge_counts = torch.cat([torch.tensor([0], device=nodes.device), edge_counts.cumsum(0)])

        # Compute local slot indices per node
        local_slots = torch.arange(nodes.size(0), device=nodes.device) - cum_edge_counts[self._assoc[nodes]]

        # Compute dense_id
        dense_id = local_slots + (self._assoc[nodes] * temp_size)

        # Initialize dense tensors with temporary size
        dense_e_id = e_id.new_full((n_id.numel() * temp_size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, temp_size)

        dense_neighbors = e_id.new_full((n_id.numel() * temp_size,), -1)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, temp_size)

        # Collect new and old interactions
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat([self.neighbors[n_id, : self.size], dense_neighbors], dim=-1)

        # Sort by e_id to keep the `size` most recent neighbors
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)
