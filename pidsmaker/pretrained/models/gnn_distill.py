"""
GNN Neighborhood Distillation for provenance graph entity pretraining.

A standalone model_type ("gnn_distill") that uses a T5 encoder (no decoder)
to learn entity embeddings by matching structurally-aware targets produced
by a GNN teacher. The GNN receives node features from an EMA (Exponential
Moving Average) copy of the student T5 encoder and aggregates 1-hop
neighborhoods. The center node is masked and reconstructed via SCE loss
(self-supervised signal for the GNN). At inference, only the T5 encoder
is used — mean-pooled hidden states serve as entity embeddings.

Components:
  ProvenanceGNNDistill    — T5 encoder-only wrapper + projection
  NeighborhoodGNNEncoder  — multi-layer TransformerConv with edge features
  NeighborhoodGNNDecoder  — lightweight decoder for masked reconstruction
  NeighborhoodGNNTeacher  — GNN teacher: edge embeddings + encoder + decoder
  DistillProjection       — 2-layer MLP projecting T5 output to GNN space
  build_edge_type_encoder — collects unique edge types across all samplers
  build_gnn_distill_batch — assembles batched PyG graph from 1-hop neighborhoods
"""

from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from transformers import T5Config, T5EncoderModel

from .bert import MODEL_SIZES
from .graphmae import sce_loss


# ── T5 encoder-only model with distillation ──────────────────────────────

def get_gnn_distill_encoder_config(
    vocab_size: int,
    model_size: str = "tiny",
    max_length: int = 512,
) -> T5Config:
    """Create T5Config for encoder-only use in GNN distillation."""
    params = MODEL_SIZES[model_size]
    return T5Config(
        vocab_size=vocab_size,
        d_model=params.H,
        d_ff=params.I,
        num_heads=params.A,
        num_layers=params.L,
        num_decoder_layers=0,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        is_encoder_decoder=False,
        use_cache=False,
        pad_token_id=0,
    )


class ProvenanceGNNDistill(nn.Module):
    """T5 encoder-only model for GNN neighborhood distillation pretraining.

    Uses only the T5 encoder to produce entity embeddings from tokens.
    A projection MLP maps the mean-pooled encoder output to the GNN
    embedding space. At inference, only the encoder is used (skip_cls=True).

    The GNN teacher is separate and managed by the training loop.
    """

    def __init__(self, config: T5Config, gnn_hidden_dim: int):
        super().__init__()
        self.config = config
        self.encoder = T5EncoderModel(config)
        self.projection = DistillProjection(config.d_model, gnn_hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Interface matching ProvenanceBERT's modified_fwd.

        For GNN distillation:
          - skip_cls=True: return encoder hidden states [B, L_enc, H]
            (used for embedding extraction at inference).
          - skip_cls=False: return mean-pooled encoder output projected
            to GNN space [B, gnn_hidden_dim]. The labels argument is
            unused (kept for interface compatibility); the actual loss
            is computed externally by comparing with GNN targets.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = enc_out.last_hidden_state  # [B, L, H]

        if skip_cls:
            return hidden_states

        # Mean-pool over non-padding tokens
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        # Project to GNN embedding space
        projected = self.projection(pooled)  # [B, gnn_hidden_dim]

        Output = namedtuple("Output", ["projected", "hidden_states"])
        return Output(projected=projected, hidden_states=hidden_states)

    @torch.no_grad()
    def mean_pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool encoder output (no projection). Used by EMA teacher."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = enc_out.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)


# ── GNN Encoder (with edge features) ─────────────────────────────────────

class NeighborhoodGNNEncoder(nn.Module):
    """Multi-layer GNN encoder using TransformerConv for edge feature support."""

    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(
            TransformerConv(in_dim, hidden_dim // num_heads, heads=num_heads,
                            edge_dim=edge_dim, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(
                TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                                edge_dim=edge_dim, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
        return x


# ── GNN Decoder (lightweight, for masked reconstruction) ─────────────────

class NeighborhoodGNNDecoder(nn.Module):
    """Lightweight TransformerConv decoder for feature reconstruction."""

    def __init__(self, hidden_dim, out_dim, edge_dim, num_layers=1, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out = out_dim if is_last else hidden_dim
            heads = 1 if is_last else num_heads
            self.layers.append(
                TransformerConv(hidden_dim, out // heads, heads=heads,
                                edge_dim=edge_dim, dropout=dropout,
                                concat=not is_last)
            )
            if not is_last:
                self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.elu(x)
        return x


# ── GNN Teacher ──────────────────────────────────────────────────────────

class NeighborhoodGNNTeacher(nn.Module):
    """GNN teacher that produces structurally-aware node embeddings.

    Node features come from an EMA copy of the student T5 encoder
    (pre-computed externally). The GNN only has learnable edge type
    embeddings. Trained via masked center node reconstruction.
    """

    def __init__(
        self,
        node_dim: int,
        num_edge_types: int,
        edge_emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_projection: bool = False,
    ):
        super().__init__()
        self.edge_projection = edge_projection

        # Learnable edge type embeddings
        self.edge_embedding = nn.Embedding(num_edge_types, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

        # Optional per-edge-type source node projection
        if edge_projection:
            self.edge_projections = nn.ModuleList([
                nn.Linear(node_dim, node_dim, bias=False)
                for _ in range(num_edge_types)
            ])

        # Encoder
        self.encoder = NeighborhoodGNNEncoder(
            node_dim, edge_emb_dim, hidden_dim,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
        )

        # Decoder (reconstructs back to node_dim = T5 d_model)
        self.decoder = NeighborhoodGNNDecoder(
            hidden_dim, node_dim, edge_emb_dim,
            num_layers=1, num_heads=num_heads, dropout=dropout,
        )

        # Encoder-to-decoder projection
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim)

        # Learnable [MASK] token replaces center nodes during training
        self.mask_token = nn.Parameter(torch.zeros(1, node_dim))
        nn.init.xavier_uniform_(self.mask_token)

    def forward(self, node_embeddings, edge_index, edge_type_indices,
                batch_vector, center_indices):
        """Training forward: mask center nodes, encode, reconstruct.

        Args:
            node_embeddings: [N, node_dim] pre-computed node features
                from the EMA teacher T5 encoder.
            edge_index: [2, E] int tensor — edge connectivity.
            edge_type_indices: [E] int tensor — edge type embedding indices.
            batch_vector: [N] int tensor — graph membership.
            center_indices: [B_valid] int tensor — flat index of each center node.

        Returns:
            gnn_loss: SCE reconstruction loss on center nodes.
            encoded: [N, hidden_dim] encoded node representations.
        """
        edge_attr = self.edge_embedding(edge_type_indices)

        # Mask center nodes only — replace with [MASK] token
        x = node_embeddings.clone()
        x[center_indices] = self.mask_token

        # Apply edge type-specific projections to source nodes
        if self.edge_projection:
            src_nodes = edge_index[0]  # [E]
            for etype_idx, proj in enumerate(self.edge_projections):
                mask = edge_type_indices == etype_idx
                if mask.any():
                    src_idx = src_nodes[mask]
                    x[src_idx] = proj(x[src_idx])

        # Encode with masked centers
        encoded = self.encoder(x, edge_index, edge_attr)

        # Decode for reconstruction loss
        dec_input = self.enc_to_dec(encoded)
        reconstructed = self.decoder(dec_input, edge_index, edge_attr)

        # SCE loss: reconstruct center nodes' original EMA embeddings
        gnn_loss = sce_loss(reconstructed[center_indices], node_embeddings[center_indices])

        return gnn_loss, encoded


# ── Distillation Projection ──────────────────────────────────────────────

class DistillProjection(nn.Module):
    """2-layer MLP projecting T5 encoder output to GNN embedding space."""

    def __init__(self, t5_hidden_dim: int, gnn_hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(t5_hidden_dim, t5_hidden_dim),
            nn.GELU(),
            nn.Linear(t5_hidden_dim, gnn_hidden_dim),
        )

    def forward(self, x):
        return self.proj(x)


# ── Edge type encoder ────────────────────────────────────────────────────

def build_edge_type_encoder(sampler_pairs) -> Dict[str, int]:
    """Collect unique edge type strings across all samplers.

    Each raw edge type (e.g. "EVENT_READ") produces two directed variants:
    "EVENT_READ_fwd" (original direction) and "EVENT_READ_rev" (reverse),
    so the GNN can distinguish incoming from outgoing edges.

    Returns:
        edge_type2idx: dict mapping directed edge_type_str → int index.
    """
    raw_edge_types = set()
    for sampler, _ in sampler_pairs:
        for edges in sampler.forward_adj.values():
            for edge in edges:
                # Real sampler: (dst, edge_type, time), Synthetic: (dst, edge_type)
                etype = edge[1]
                raw_edge_types.add(etype)
    directed = []
    for et in sorted(raw_edge_types):
        directed.append(f"{et}_fwd")
        directed.append(f"{et}_rev")
    return {et: i for i, et in enumerate(directed)}


# ── Batch construction ───────────────────────────────────────────────────

def build_gnn_distill_batch(
    entity_node_ids: List[str],
    sampler_indices: List[int],
    sampler_pairs: list,
    indexid2msg: dict,
    edge_type2idx: Dict[str, int],
    n_neighbors_min: int = 5,
    n_neighbors_max: int = 20,
    diverse: bool = True,
    device: str = "cpu",
    filter_noisy: bool = False,
) -> Optional[Tuple]:
    """Build a batched PyG graph of 1-hop neighborhoods for distillation.

    For each entity in the batch, extracts its 1-hop temporal
    neighborhood (with random neighbor count) and assembles them
    into a single batched graph.

    Args:
        entity_node_ids: [B] entity node ID strings from the T5 batch.
        sampler_indices: [B] index into sampler_pairs for each entity.
        sampler_pairs: list of (sampler, ds_indexid2msg) tuples.
        indexid2msg: combined {node_id: (node_type, label_str)} mapping.
        edge_type2idx: {edge_type_str: int} from build_edge_type_encoder.
        n_neighbors_min: min neighbors to sample per entity.
        n_neighbors_max: max neighbors to sample per entity.
        device: target device.

    Returns:
        Tuple of (node_labels, edge_index, edge_type_indices,
                  batch_vector, center_indices, valid_mask) or None.

        - node_labels: list of (ntype, nlabel) for each node (to be
          tokenized and encoded by the EMA teacher).
        - edge_index: [2, total_edges] graph connectivity.
        - edge_type_indices: [total_edges] edge type embedding indices.
        - batch_vector: [total_nodes] graph membership index.
        - center_indices: [n_valid] flat index of each center node.
        - valid_mask: [B] bool, True for entities with valid neighborhoods.
    """
    B = len(entity_node_ids)

    all_node_labels: List[Tuple] = []
    all_edge_src: List[int] = []
    all_edge_dst: List[int] = []
    all_edge_types: List[int] = []
    all_batch: List[int] = []
    center_indices: List[int] = []
    valid_mask = [False] * B
    node_offset = 0
    graph_idx = 0

    for i in range(B):
        sampler_idx = sampler_indices[i]
        if sampler_idx < 0:
            continue

        sampler, ds_indexid2msg = sampler_pairs[sampler_idx]
        entity_node = entity_node_ids[i]

        # Extract 1-hop neighborhood with random size
        result = sampler.sample_nhop_neighborhood(
            entity_node,
            n_neighbors_min=n_neighbors_min,
            n_neighbors_max=n_neighbors_max,
            diverse=diverse,
        )
        if result is None:
            continue

        center_node, unique_nodes, edges = result

        # Filter noisy edges (shared libs, /dev, /proc, etc.)
        if filter_noisy:
            from ..data.edge_filter import _is_noisy_file, _is_noisy_process
            filtered_edges = []
            for src, dst, etype in edges:
                # Check neighbor node (the one that isn't the center)
                neighbor = dst if src == center_node else src
                if neighbor in ds_indexid2msg:
                    ntype, nlabel = ds_indexid2msg[neighbor]
                    if ntype == "file" and _is_noisy_file(nlabel):
                        continue
                    if ntype == "subject" and _is_noisy_process(nlabel):
                        continue
                filtered_edges.append((src, dst, etype))
            edges = filtered_edges
            if not edges:
                continue
            # Rebuild unique_nodes from surviving edges
            visited = {center_node}
            for src, dst, _ in edges:
                visited.add(src)
                visited.add(dst)
            unique_nodes = [center_node] + [n for n in visited if n != center_node]

        # Build local node index
        local_id = {n: j + node_offset for j, n in enumerate(unique_nodes)}
        n_nodes = len(unique_nodes)

        # Node labels (to be tokenized + encoded by EMA teacher)
        for n in unique_nodes:
            lbl = ds_indexid2msg.get(n)
            if lbl is None:
                raise ValueError(f"Node {n} not found in indexid2msg")
            lbl = tuple(lbl) if isinstance(lbl, list) else lbl
            all_node_labels.append(lbl)
            all_batch.append(graph_idx)

        # Edge indices and types (directed: _fwd for original, _rev for reverse)
        for src, dst, etype in edges:
            if src in local_id and dst in local_id:
                all_edge_src.append(local_id[src])
                all_edge_dst.append(local_id[dst])
                all_edge_types.append(edge_type2idx.get(f"{etype}_fwd", 0))
                # Add reverse edge for message passing
                all_edge_src.append(local_id[dst])
                all_edge_dst.append(local_id[src])
                all_edge_types.append(edge_type2idx.get(f"{etype}_rev", 0))

        center_indices.append(local_id[center_node])
        valid_mask[i] = True
        node_offset += n_nodes
        graph_idx += 1

    if not center_indices:
        return None

    edge_index = torch.tensor([all_edge_src, all_edge_dst], dtype=torch.long, device=device)
    edge_type_indices = torch.tensor(all_edge_types, dtype=torch.long, device=device)
    batch_vector = torch.tensor(all_batch, dtype=torch.long, device=device)
    center_idx_tensor = torch.tensor(center_indices, dtype=torch.long, device=device)
    valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=device)

    return all_node_labels, edge_index, edge_type_indices, batch_vector, center_idx_tensor, valid_mask_tensor
