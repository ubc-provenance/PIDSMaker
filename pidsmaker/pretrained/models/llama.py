"""
Decoder-only LLaMA model for provenance graph pretraining.

Uses HuggingFace's LlamaForCausalLM with random initialization (no pretrained
English weights). Pretrained with next-token prediction on linearized provenance
walks, then used directly for perplexity-based anomaly detection — no
fine-tuning needed.

Architecture advantages over MLM encoders for real-time detection:
- Causal attention aligns with streaming provenance events
- Every token provides training signal (vs ~15% for MLM)
- Perplexity gives direct, unsupervised anomaly scoring
"""

from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LlamaForCausalLM

from .bert import MODEL_SIZES


def get_llama_config(
    vocab_size: int,
    model_size: str = "tiny",
    max_position_embeddings: int = 512,
    rope_theta: float = 10000.0,
) -> LlamaConfig:
    """Create LlamaConfig for a given model size.

    Uses the same size presets as BERT/RoPEBERT (tiny/mini/med/baseline).
    """
    params = MODEL_SIZES[model_size]
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=params.H,
        num_hidden_layers=params.L,
        num_attention_heads=params.A,
        intermediate_size=params.I,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        # Match other models: no extra features
        num_key_value_heads=params.A,  # no GQA, full MHA
        use_cache=False,  # not needed during training
    )


class ProvenanceLLaMA(nn.Module):
    """Decoder-only LLaMA for causal language modeling on provenance walks.

    Wraps HuggingFace LlamaForCausalLM following the same pattern as
    ProvenanceBERT wraps BertForMaskedLM.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.llama = LlamaForCausalLM(config)

    @property
    def device(self):
        return next(self.parameters()).device

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Interface matching ProvenanceBERT's modified_fwd.

        For causal LM, the model internally shifts logits and labels so that
        token[t] predicts token[t+1]. Pass labels=input_ids for standard
        next-token prediction.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] bool, True for real tokens
            labels: [B, L] target token IDs (typically = input_ids for causal LM).
                    Positions with -100 are ignored in loss.
            return_loss: if True, return scalar loss; else return namedtuple(loss, logits)
            skip_cls: if True, return hidden states [B, L, H] instead of LM output
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        if skip_cls:
            out = self.llama.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return out.last_hidden_state

        out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if return_loss:
            return out.loss

        Output = namedtuple("Output", ["loss", "logits"])
        return Output(loss=out.loss, logits=out.logits)
