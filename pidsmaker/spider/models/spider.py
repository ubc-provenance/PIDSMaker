"""
SPIDER: class-supervised GNN teacher with T5 student distillation.

Architecture (training):

  Teacher (GNN + signature, trained with SupCon):
    L2_norm(gnn_encoded[center])   [hidden_dim]  ─┐
    L2_norm(signature_vector)      [num_labels]   ─┴─► concat ──► proj_head ──► SupCon
                                                           │
                                                      proj.detach()
                                                           │
  Student (sees only tokens):                              ▼
    Entity tokens ──► T5 Encoder ──► mean_pool ──► proj ──► L_distill (SCE)

  GNN receives EMA T5 embeddings for ALL nodes (center NOT masked).
  Center node identity is available to the GNN — no artificial handicap.

At inference, only the T5 encoder is used (mean-pooled, no heads).

Reuses NeighborhoodGNNEncoder, build_edge_type_encoder,
and build_gnn_distill_batch from model_gnn_distill.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphmae import sce_loss  # noqa: F401 — re-exported for training loop
from .gnn_distill import (
    NeighborhoodGNNEncoder,
    build_edge_type_encoder,  # noqa: F401
    build_gnn_distill_batch,  # noqa: F401
    get_gnn_distill_encoder_config as get_spider_encoder_config,  # same T5 config
)


# ── GNN Teacher ──────────────────────────────────────────────────────

class GNNClusterTeacher(nn.Module):
    """GNN teacher: encodes 1-hop neighborhood (center included),
    fuses with behavioral signature, projects for SupCon.

    No center masking — the GNN sees the center node's EMA T5
    embedding directly, giving it both identity and structural context.

    ``teacher_data`` controls which modalities are fused:
      - ``"signature,gnn_emb"`` (default): concatenate GNN center + signature
      - ``"gnn_emb"``: GNN center encoding only (no signature)
      - ``"signature"``: signature vector only (GNN encoder still built but unused)
    """

    def __init__(
        self,
        node_dim: int,
        num_edge_types: int,
        edge_emb_dim: int,
        hidden_dim: int,
        num_labels: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        teacher_data: str = "signature,gnn_emb",
    ):
        super().__init__()
        self.teacher_data = teacher_data
        self.use_gnn = "gnn_emb" in teacher_data
        self.use_sig = "signature" in teacher_data

        self.edge_embedding = nn.Embedding(num_edge_types, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

        # Single-layer GNN encoder (always built for checkpoint compat)
        self.encoder = NeighborhoodGNNEncoder(
            node_dim, edge_emb_dim, hidden_dim,
            num_layers=1, num_heads=num_heads, dropout=dropout,
        )

        # Projection head dimension depends on active modalities
        fused_dim = 0
        if self.use_gnn:
            fused_dim += hidden_dim
        if self.use_sig:
            fused_dim += num_labels
        self.proj_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, node_embeddings, edge_index, edge_type_indices,
                center_indices, signature_vectors):
        """Encode neighborhood, fuse with signature, project.

        Args:
            node_embeddings: [N, node_dim] from EMA teacher (center included).
            edge_index: [2, E] graph connectivity.
            edge_type_indices: [E] edge type indices.
            center_indices: [B_valid] flat index of center nodes.
            signature_vectors: [B_valid, num_labels] binary behavior vectors.

        Returns:
            proj: [B_valid, proj_dim] L2-normalized fused projection.
        """
        parts = []
        if self.use_gnn:
            edge_attr = self.edge_embedding(edge_type_indices)
            encoded = self.encoder(node_embeddings, edge_index, edge_attr)
            center_gnn = F.normalize(encoded[center_indices], p=2, dim=-1)
            parts.append(center_gnn)
        if self.use_sig:
            center_sig = F.normalize(signature_vectors.float(), p=2, dim=-1)
            parts.append(center_sig)

        fused = torch.cat(parts, dim=-1)
        return F.normalize(self.proj_head(fused), p=2, dim=-1)


# ── Student: reuse ProvenanceGNNDistill from model_gnn_distill ───────
from .gnn_distill import ProvenanceGNNDistill as ProvenanceGNNCluster  # noqa: F401, E402


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

# ── Student-only ablation heads ─────────────────────────────────

class StudentSignatureHead(nn.Module):
    """Student predicts binary behavior signature directly (no teacher)."""

    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, x):
        return self.head(x)  # [B, num_labels] raw logits


class StudentClassHead(nn.Module):
    """Student predicts entity class directly (no teacher)."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)  # [B, num_classes] raw logits


# ── Teacher classification head (BCE ablation) ─────────────────

class TeacherClassHead(nn.Module):
    """Classifies teacher projection into entity class (replaces SupCon)."""

    def __init__(self, proj_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        return self.head(x)  # [B, num_classes] raw logits


def supervised_contrastive_loss(
    projections: torch.Tensor,
    class_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon: pull same-class together, push different-class apart.

    Args:
        projections: [B, D] L2-normalized embeddings.
        class_labels: [B] integer class IDs.
        temperature: cosine similarity scaling.
    """
    B = projections.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=projections.device)

    cos_sim = torch.mm(projections, projections.t()) / temperature
    mask_self = torch.eye(B, dtype=torch.bool, device=projections.device)
    positive_mask = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)) & ~mask_self

    if not positive_mask.any():
        return torch.tensor(0.0, device=projections.device)

    cos_sim_masked = cos_sim.masked_fill(mask_self, float("-inf"))
    log_denom = torch.logsumexp(cos_sim_masked, dim=1)

    has_positives = positive_mask.any(dim=1)
    if not has_positives.any():
        return torch.tensor(0.0, device=projections.device)

    n_positives = positive_mask.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -(cos_sim_masked.masked_fill(~positive_mask, 0.0).sum(dim=1) / n_positives - log_denom)

    return loss_per_anchor[has_positives].mean()
