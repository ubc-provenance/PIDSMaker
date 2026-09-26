"""
Behavioral Embedding model for provenance graph entities.

Architecture:
                         ┌────────────────┐
  Entity text ─────────→ │   T5 Encoder    │──→  embedding (emb_dim)
                         └────────────────┘
                                │
                     ┌──────────┴──────────┐
                     ↓                     ↓
              ┌─────────────┐      ┌──────────────┐
              │  Multi-Label │      │  Contrastive  │
              │  Classifier  │      │     Head      │
              └─────────────┘      └──────────────┘

Training objective:
  Loss = λ₁ × BCE_loss + λ₂ × contrastive_loss

Task 1 (bce_mode="multilabel"): Multi-label BCE — predict the behavior
  signature binary vector.
Task 1 (bce_mode="class"): Cross-entropy — predict the signature class ID
  directly (one-of-N classification over unique signatures).
Task 2: Contrastive — pull entities with similar Jaccard signatures close,
         push dissimilar ones apart (NT-Xent / InfoNCE).

At inference, only the T5 encoder is used.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel

from .bert import MODEL_SIZES


# ── T5 Config ────────────────────────────────────────────────────────────

def get_behavior_encoder_config(
    vocab_size: int,
    model_size: str = "mini",
    max_length: int = 512,
) -> T5Config:
    """Create T5Config for encoder-only behavior embedding model."""
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


# ── Multi-Label Classifier Head ──────────────────────────────────────────

class MultiLabelHead(nn.Module):
    """Linear head: embedding → sigmoid over N behavior labels."""

    def __init__(self, emb_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, num_labels),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_labels] (pre-sigmoid)."""
        return self.classifier(embeddings)


# ── Classification Head (cross-entropy over signature classes) ────────

class ClassificationHead(nn.Module):
    """Linear head: embedding → logits over N signature classes."""

    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes] (pre-softmax)."""
        return self.classifier(embeddings)


# ── Contrastive Head (projection to unit sphere) ─────────────────────────

class ContrastiveHead(nn.Module):
    """Projects embeddings to a lower-dim space for contrastive learning."""

    def __init__(self, emb_dim: int, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized projections [B, proj_dim]."""
        return F.normalize(self.projector(embeddings), p=2, dim=-1)


# ── Main Model ───────────────────────────────────────────────────────────

class ProvenanceBehaviorModel(nn.Module):
    """T5 encoder + multi-label classifier + contrastive head.

    At inference, only the T5 encoder is needed — call with skip_cls=True
    to get mean-pooled embeddings.
    """

    def __init__(self, config: T5Config, num_labels: int, proj_dim: int = 128,
                 bce_mode: str = "multilabel", num_classes: int = 0):
        super().__init__()
        self.config = config
        self.encoder = T5EncoderModel(config)
        self.num_labels = num_labels
        self.emb_dim = config.d_model
        self.bce_mode = bce_mode

        # Training heads
        if bce_mode == "class":
            assert num_classes > 0, "num_classes required for bce_mode='class'"
            self.classification_head = ClassificationHead(config.d_model, num_classes)
            self.multilabel_head = None
        else:
            self.multilabel_head = MultiLabelHead(config.d_model, num_labels)
            self.classification_head = None
        self.contrastive_head = ContrastiveHead(config.d_model, proj_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool encoder output over non-padding tokens."""
        mask_exp = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Unified forward interface.

        skip_cls=True:  return raw encoder hidden states [B, L, H]
                        (for embedding extraction at inference).
        skip_cls=False: return (embedding, multilabel_logits, contrastive_proj).
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

        # Mean-pool
        embedding = self._mean_pool(hidden_states, attention_mask)  # [B, H]

        # Classification logits (multi-label or class)
        if self.bce_mode == "class":
            logits = self.classification_head(embedding)  # [B, num_classes]
        else:
            logits = self.multilabel_head(embedding)  # [B, num_labels]

        # Contrastive projection
        proj = self.contrastive_head(embedding)  # [B, proj_dim]

        Output = namedtuple("Output", ["embedding", "logits", "projection"])
        return Output(embedding=embedding, logits=logits, projection=proj)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: mean-pooled encoder embeddings (no training heads)."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(enc_out.last_hidden_state, attention_mask)


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def behavior_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for multi-label behavior prediction.

    Args:
        logits: [B, N] raw logits from multi-label head.
        targets: [B, N] binary target vectors.

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def behavior_ce_loss(
    logits: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for direct signature class prediction.

    Args:
        logits: [B, num_classes] raw logits from classification head.
        class_labels: [B] integer class IDs.

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, class_labels)


def behavior_contrastive_loss(
    projections: torch.Tensor,
    class_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon) using class identity.

    Positives = entities sharing the same behavioral class (identical
    signature). Negatives = entities from different classes.

    Args:
        projections: [B, D] L2-normalized embeddings from contrastive head.
        class_labels: [B] integer class IDs (same ID = same behavioral class).
        temperature: temperature for cosine similarity scaling.

    Returns:
        Scalar contrastive loss.
    """
    B = projections.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=projections.device)

    # Cosine similarity matrix (already L2-normalized)
    cos_sim = torch.mm(projections, projections.t()) / temperature  # [B, B]

    # Mask out self-similarities
    mask_self = torch.eye(B, dtype=torch.bool, device=projections.device)

    # Positive mask: same class, excluding self
    positive_mask = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)) & ~mask_self

    # If no positives exist in this batch, skip
    if not positive_mask.any():
        return torch.tensor(0.0, device=projections.device)

    # Log-sum-exp over all non-self pairs (denominator)
    cos_sim_masked = cos_sim.masked_fill(mask_self, float("-inf"))
    log_denom = torch.logsumexp(cos_sim_masked, dim=1)  # [B]

    # For each anchor with positives: mean of log(exp(sim_pos) / denom)
    # = mean of (sim_pos - log_denom) over positive pairs
    has_positives = positive_mask.any(dim=1)
    if not has_positives.any():
        return torch.tensor(0.0, device=projections.device)

    # Per-anchor: mean log-prob over its positives
    # Mask out non-positives with -inf before computing mean
    pos_sim = cos_sim_masked.masked_fill(~positive_mask, float("-inf"))
    # For each anchor, compute mean over its positive pairs
    n_positives = positive_mask.float().sum(dim=1).clamp(min=1)  # [B]
    # sum of (sim_pos_j - log_denom) for each positive j
    loss_per_anchor = -(pos_sim.masked_fill(~positive_mask, 0.0).sum(dim=1) / n_positives - log_denom)

    return loss_per_anchor[has_positives].mean()


def behavior_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    projections: torch.Tensor,
    class_labels: torch.Tensor,
    bce_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    temperature: float = 0.07,
    bce_mode: str = "multilabel",
    entity_class_labels: torch.Tensor = None,
    contrastive_target: str = "signature",
) -> tuple:
    """Combined BCE/CE + contrastive loss.

    Args:
        bce_mode: "multilabel" for BCE over binary vectors,
                  "class" for cross-entropy over entity class IDs.
        entity_class_labels: [B] integer entity class IDs (coarse
                             functional classes, ~70).
        contrastive_target: "signature" uses fine-grained signature class IDs
                            for contrastive positives; "entity_class" uses
                            coarse entity class IDs instead, clustering
                            entities by functional class.

    Returns:
        (total_loss, bce_loss_val, contrastive_loss_val)
    """
    if bce_mode == "class":
        bce = behavior_ce_loss(logits, entity_class_labels)
    else:
        bce = behavior_bce_loss(logits, targets)

    if contrastive_target == "entity_class":
        con_labels = entity_class_labels
    else:
        con_labels = class_labels

    contrastive = behavior_contrastive_loss(
        projections, con_labels,
        temperature=temperature,
    )
    total = bce_weight * bce + contrastive_weight * contrastive
    return total, bce, contrastive
