"""
RoPEBERT implementation for provenance graph pretraining.

Combines the best of both BERT and ModernBERT:
- RoPE (Rotary Position Embeddings) for relative position encoding
- Full global attention at every layer (no sliding window, no flash attention)
- RMSNorm pre-normalization for stable training
- GeGLU activation in MLP layers
- No bias terms in linear layers (except decoder)
- Additive dataset embedding for multi-dataset pretraining
"""

import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .bert import MODEL_SIZES, FineTuneCLS, FineTuneLP


from .rope_utils import RMSNorm, RotaryEmbedding, GeGLU


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RoPEBERTConfig:
    """Configuration for RoPEBERT model.

    Compared to ModernBERTConfig:
    - No sliding window / flash attention params
    - Single rope_theta for all layers
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


# ============================================================================
# Attention with RoPE (full global, no sliding window)
# ============================================================================

class RoPEBERTAttention(nn.Module):
    """Full global multi-head attention with RoPE.

    Unlike ModernBERTAttention:
    - Always uses full global attention (no sliding window)
    - No flash attention (standard softmax)
    - Single RoPE theta for all layers
    """

    def __init__(self, config: RoPEBERTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_probs_dropout_prob

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states)  # [B, L, 3*H]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        q, k = self.rotary_emb(q, k, seq_len)

        # Full global attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), -1e9)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, v)  # [B, heads, L, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)
        return attn_output


# ============================================================================
# Transformer Layer with Pre-Normalization
# ============================================================================

class RoPEBERTLayer(nn.Module):
    """Pre-norm transformer layer: x = x + Attn(Norm(x)), x = x + MLP(Norm(x))."""

    def __init__(self, config: RoPEBERTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.attention = RoPEBERTAttention(config)
        self.mlp = GeGLU(config.hidden_size, config.intermediate_size, config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.attn_norm(hidden_states)
        attn_output = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_output

        normed = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + mlp_output
        return hidden_states


# ============================================================================
# Embeddings
# ============================================================================

class RoPEBERTEmbeddings(nn.Module):
    """Token embeddings without positional embeddings.

    Position encoding is handled by RoPE in the attention layers.
    """

    def __init__(self, config: RoPEBERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ============================================================================
# Encoder
# ============================================================================

class RoPEBERTEncoder(nn.Module):
    """Stack of RoPEBERT transformer layers (all identical)."""

    def __init__(self, config: RoPEBERTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEBERTLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# ============================================================================
# ProvenanceRoPEBERT (top-level model)
# ============================================================================

class ProvenanceRoPEBERT(nn.Module):
    """RoPEBERT for masked language modeling on provenance graph walks.

    Architecture:
    1. Token embeddings (position via RoPE in attention)
    2. Stack of pre-norm transformer layers with full global attention
    3. Final RMSNorm
    4. Prediction head for masked token prediction
    """

    def __init__(self, config: RoPEBERTConfig):
        super().__init__()
        self.config = config

        self.embeddings = RoPEBERTEmbeddings(config)
        self.encoder = RoPEBERTEncoder(config)
        self.final_norm = RMSNorm(config.hidden_size)

        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.GELU(),
            RMSNorm(config.hidden_size),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights first, then tie
        self.apply(self._init_weights)
        self.decoder.weight = self.embeddings.word_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = (2.0 / (module.weight.shape[0] + module.weight.shape[1])) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)

        hidden_states = self.prediction_head(hidden_states)
        logits = self.decoder(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return loss, logits

        return logits

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Interface matching ProvenanceBERT and ProvenanceModernBERT."""
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        encoder_hidden = self.final_norm(hidden_states)

        if skip_cls:
            return encoder_hidden

        pred_hidden = self.prediction_head(encoder_hidden)
        logits = self.decoder(pred_hidden)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if return_loss:
            return loss

        Output = namedtuple("Output", ["loss", "logits"])
        return Output(loss=loss, logits=logits)


# ============================================================================
# Model size presets
# ============================================================================



def get_ropebert_config(
    vocab_size: int,
    model_size: str = "tiny",
    max_position_embeddings: int = 512,
    rope_theta: float = 10000.0,
) -> RoPEBERTConfig:
    """Create RoPEBERTConfig for a given model size."""
    params = MODEL_SIZES[model_size]
    return RoPEBERTConfig(
        vocab_size=vocab_size,
        hidden_size=params.H,
        num_hidden_layers=params.L,
        num_attention_heads=params.A,
        intermediate_size=params.I,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
    )


def ProvenanceRoPEBERTFineTuneCLS(config, pretrained_state_dict=None, device="cpu", freeze_backbone=True, margin=2.0):
    return FineTuneCLS(ProvenanceRoPEBERT, config, pretrained_state_dict, device, freeze_backbone, margin)

def ProvenanceRoPEBERTFineTuneLP(config, pretrained_state_dict=None, device="cpu", freeze_backbone=False):
    return FineTuneLP(ProvenanceRoPEBERT, config, pretrained_state_dict, device, freeze_backbone)
