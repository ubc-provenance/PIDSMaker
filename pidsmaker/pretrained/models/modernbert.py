"""
ModernBERT implementation for provenance graph pretraining.

Incorporates modern transformer improvements:
- RoPE (Rotary Position Embeddings) instead of absolute positional embeddings
- RMSNorm instead of LayerNorm
- GeGLU activation in MLP layers
- Pre-normalization architecture
- Alternating global/local sliding window attention
- No bias terms in linear layers (except decoder)
- Flash Attention 2 support for efficiency

Based on: "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast,
Memory Efficient, and Long Context Finetuning and Inference" (ACL 2025)
GitHub: https://github.com/AnswerDotAI/ModernBERT
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .bert import MODEL_SIZES, FineTuneCLS, FineTuneLP



@dataclass
class ModernBERTConfig:
    """Configuration for ModernBERT model.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension (H)
        num_hidden_layers: Number of transformer layers (L)
        num_attention_heads: Number of attention heads (A)
        intermediate_size: MLP intermediate dimension (I)
        max_position_embeddings: Maximum sequence length
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout
        global_attn_every_n_layers: Use global attention every N layers (others use local)
        local_attention_window: Sliding window size for local attention
        global_rope_theta: RoPE base frequency for global attention layers
        local_rope_theta: RoPE base frequency for local attention layers
        rope_scaling: Optional RoPE scaling factor
        use_flash_attention: Whether to use Flash Attention (if available)
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    global_attn_every_n_layers: int = 3
    local_attention_window: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    use_flash_attention: bool = True

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


from .rope_utils import RMSNorm, RotaryEmbedding, GeGLU


# ============================================================================
# Multi-Head Attention with RoPE
# ============================================================================

class ModernBERTAttention(nn.Module):
    """Multi-head attention with RoPE and optional sliding window.

    Args:
        config: Model configuration
        layer_id: Layer index (determines if global or local attention)
    """

    def __init__(self, config: ModernBERTConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_probs_dropout_prob

        # Determine if this is a global or local attention layer
        self.is_global = (layer_id % config.global_attn_every_n_layers == 0)
        self.attention_window = None if self.is_global else config.local_attention_window

        # QKV projection (no bias)
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

        # Output projection (no bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

        # RoPE embeddings (different base for global vs local)
        rope_base = config.global_rope_theta if self.is_global else config.local_rope_theta
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=rope_base
        )

        # Try to import flash attention
        self.use_flash = config.use_flash_attention
        if self.use_flash:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                self.use_flash = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional sliding window attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] or [batch, 1, seq_len, seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv(hidden_states)  # [batch, seq_len, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        q, k = self.rotary_emb(q, k, seq_len)

        # Compute attention
        if self.use_flash and self.training:
            # Use Flash Attention if available (more efficient)
            # Note: Flash Attention expects [batch, seq_len, num_heads, head_dim]
            q = q.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply sliding window for local attention
            window_size = (-1, -1) if self.is_global else (self.attention_window // 2, self.attention_window // 2)

            attn_output = self.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                window_size=window_size,
                causal=False,  # BERT is bidirectional
            )
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        else:
            # Standard scaled dot-product attention
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        return attn_output

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with optional sliding window.

        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_heads, seq_len, head_dim]
            v: [batch, num_heads, seq_len, head_dim]
            attention_mask: Optional mask

        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Scaled dot-product: Q @ K^T / sqrt(d_k)
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, seq_len]

        # Apply sliding window mask for local attention
        if not self.is_global and self.attention_window is not None:
            # Create sliding window mask
            window_mask = self._create_sliding_window_mask(seq_len, self.attention_window, q.device)
            attn_scores = attn_scores.masked_fill(~window_mask, -1e9)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Use large negative value instead of -inf to prevent NaN
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), -1e9)

        # Softmax and apply to values (use float32 for stability)
        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)
        # Replace any NaN with 0 (can happen if entire row is masked)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        return attn_output

    @staticmethod
    def _create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create a sliding window attention mask.

        Args:
            seq_len: Sequence length
            window_size: Window size (tokens can attend to window_size // 2 on each side)
            device: Device to create mask on

        Returns:
            Boolean mask [1, 1, seq_len, seq_len] where True means attend
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)

        # Compute distance between all pairs of positions
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

        # Allow attention within window_size // 2 on each side
        half_window = window_size // 2
        mask = (distance.abs() <= half_window)  # [seq_len, seq_len]

        # Add batch and head dimensions
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


# ============================================================================
# Transformer Layer with Pre-Normalization
# ============================================================================

class ModernBERTLayer(nn.Module):
    """Transformer layer with pre-normalization architecture.

    Pre-norm (used here):
        x = x + Attention(Norm(x))
        x = x + MLP(Norm(x))

    This is more stable than post-norm and is standard in modern LLMs.
    """

    def __init__(self, config: ModernBERTConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # Pre-normalization
        self.attn_norm = RMSNorm(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size)

        # Attention and MLP
        self.attention = ModernBERTAttention(config, layer_id=layer_id)
        self.mlp = GeGLU(config.hidden_size, config.intermediate_size, config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-normalization.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Pre-norm attention with residual
        normed = self.attn_norm(hidden_states)
        attn_output = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_output

        # Pre-norm MLP with residual
        normed = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + mlp_output

        return hidden_states


# ============================================================================
# ModernBERT Embeddings (no positional embeddings - using RoPE instead)
# ============================================================================

class ModernBERTEmbeddings(nn.Module):
    """Token embeddings without positional embeddings.

    ModernBERT uses RoPE in the attention layers instead of adding
    positional embeddings at the input.
    """

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Layer norm after embeddings (as mentioned in ModernBERT paper)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens.

        Args:
            input_ids: [batch, seq_len] token indices

        Returns:
            Embeddings [batch, seq_len, hidden_size]
        """
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ============================================================================
# ModernBERT Encoder
# ============================================================================

class ModernBERTEncoder(nn.Module):
    """Stack of ModernBERT transformer layers."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            ModernBERTLayer(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask

        Returns:
            Final hidden states [batch, seq_len, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


# ============================================================================
# ModernBERT Model for Masked Language Modeling
# ============================================================================

class ProvenanceModernBERT(nn.Module):
    """ModernBERT for masked language modeling on provenance graph walks.

    Architecture:
    1. Token embeddings (no positional embeddings)
    2. Stack of transformer layers with:
       - Pre-normalization (RMSNorm)
       - RoPE in attention
       - Alternating global/local attention
       - GeGLU activation in MLP
    3. Final layer norm
    4. Prediction head for masked token prediction
    """

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = ModernBERTEmbeddings(config)

        # Transformer encoder
        self.encoder = ModernBERTEncoder(config)

        # Final layer norm (for pre-norm architecture)
        self.final_norm = RMSNorm(config.hidden_size)

        # Prediction head (with bias in final layer - only place with bias)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.GELU(),
            RMSNorm(config.hidden_size),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights FIRST
        self.apply(self._init_weights)

        # Tie weights between embeddings and decoder AFTER initialization
        self.decoder.weight = self.embeddings.word_embeddings.weight

    def _init_weights(self, module):
        """Initialize weights following ModernBERT guidelines."""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for better gradient flow
            # Scaled by 1/sqrt(2) for residual connections
            std = (2.0 / (module.weight.shape[0] + module.weight.shape[1])) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Smaller std for embeddings to prevent initial instability
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
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs (with [MASK] applied)
            attention_mask: [batch, seq_len] mask (1 for real tokens, 0 for padding)
            labels: [batch, seq_len] target token IDs (-100 for non-predicted positions)

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits [batch, seq_len, vocab_size]
        """
        # Embed tokens
        hidden_states = self.embeddings(input_ids)

        # Pass through encoder
        hidden_states = self.encoder(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Prediction head
        hidden_states = self.prediction_head(hidden_states)
        logits = self.decoder(hidden_states)

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
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
        """Convenience wrapper matching the original ProvenanceBERT interface.

        Args:
            input_ids: [B, L] token IDs (with [MASK] applied)
            attention_mask: [B, L] bool, True for real tokens
            labels: [B, L] target token IDs, -100 for non-predicted positions
            return_loss: if True, return scalar loss; else return tuple
            skip_cls: if True, return hidden states instead of prediction logits

        Returns:
            If skip_cls: hidden states [B, L, H]
            If return_loss: scalar loss
            Otherwise: tuple of (loss, logits)
        """
        # Get embeddings and encode
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        encoder_hidden = self.final_norm(hidden_states)

        if skip_cls:
            return encoder_hidden

        # Get predictions
        pred_hidden = self.prediction_head(encoder_hidden)
        logits = self.decoder(pred_hidden)

        # Compute loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if return_loss:
            return loss

        # Return named tuple for compatibility
        from collections import namedtuple
        Output = namedtuple('Output', ['loss', 'logits'])
        return Output(loss=loss, logits=logits)


# ============================================================================
# Model size presets (matching original BERT sizes)
# ============================================================================




def get_modernbert_config(
    vocab_size: int,
    model_size: str = "tiny",
    max_position_embeddings: int = 512,
    global_attn_every_n_layers: int = 3,
    local_attention_window: int = 128,
) -> ModernBERTConfig:
    """Create ModernBERTConfig for a given model size.

    Args:
        vocab_size: Size of the tokenizer vocabulary
        model_size: One of "tiny", "mini", "med", "baseline"
        max_position_embeddings: Maximum sequence length
        global_attn_every_n_layers: Use global attention every N layers
        local_attention_window: Sliding window size for local attention

    Returns:
        ModernBERTConfig instance
    """
    params = MODEL_SIZES[model_size]

    return ModernBERTConfig(
        vocab_size=vocab_size,
        hidden_size=params.H,
        num_hidden_layers=params.L,
        num_attention_heads=params.A,
        intermediate_size=params.I,
        max_position_embeddings=max_position_embeddings,
        global_attn_every_n_layers=global_attn_every_n_layers,
        local_attention_window=local_attention_window,
    )


def ProvenanceModernBERTFineTuneCLS(config, pretrained_state_dict=None, device="cpu", freeze_backbone=True, margin=2.0):
    return FineTuneCLS(ProvenanceModernBERT, config, pretrained_state_dict, device, freeze_backbone, margin)

def ProvenanceModernBERTFineTuneLP(config, pretrained_state_dict=None, device="cpu", freeze_backbone=False):
    return FineTuneLP(ProvenanceModernBERT, config, pretrained_state_dict, device, freeze_backbone)
