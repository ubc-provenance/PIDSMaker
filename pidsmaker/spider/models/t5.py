"""
T5 encoder-decoder model for provenance graph entity pretraining.

Uses HuggingFace's T5ForConditionalGeneration with random initialization
(no pretrained English weights). The encoder processes entity tokens (type +
label) and the decoder autoregressively predicts temporal random walks
(forward and backward context from that entity).

At inference, only the encoder is used — mean-pooled encoder hidden states
serve as entity embeddings for downstream GNN pipelines.

Architecture:
  Encoder (bidirectional, relative position bias):
      entity tokens → entity embedding
  Decoder (autoregressive, relative position bias + cross-attention):
      [FORWARD/BACKWARD] + walk tokens → next-token prediction

Both encoder and decoder use T5's learned relative positional encoding,
which handles variable-length inputs naturally without absolute embeddings.
"""

from collections import namedtuple

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from .bert import MODEL_SIZES


def get_t5_config(
    vocab_size: int,
    model_size: str = "tiny",
    max_length: int = 512,
) -> T5Config:
    """Create T5Config for a given model size.

    Uses the same size presets as BERT/RoPEBERT (tiny/mini/med/baseline).
    """
    params = MODEL_SIZES[model_size]
    return T5Config(
        vocab_size=vocab_size,
        d_model=params.H,
        d_ff=params.I,
        num_heads=params.A,
        num_layers=params.L,
        num_decoder_layers=params.L,
        # T5 uses learned relative position bias — no absolute embeddings
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        is_encoder_decoder=True,
        use_cache=False,  # not needed during training
        pad_token_id=0,  # matches our [PAD] token at ID 0
        decoder_start_token_id=0,
    )


class ProvenanceT5(nn.Module):
    """T5 encoder-decoder for entity pretraining on provenance walks.

    Wraps HuggingFace T5ForConditionalGeneration following the same pattern
    as ProvenanceBERT wraps BertForMaskedLM.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.t5 = T5ForConditionalGeneration(config)

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

        For T5 encoder-decoder:
          - input_ids: [B, L_enc] encoder input (entity tokens)
          - attention_mask: [B, L_enc] bool, True for real encoder tokens
          - labels: [B, L_dec] decoder target tokens. Positions with -100 are
                    ignored in loss. T5 internally shifts labels right to create
                    decoder_input_ids.
          - skip_cls: if True, return encoder hidden states [B, L_enc, H]
                      (used for embedding extraction at inference)
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        if skip_cls:
            out = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return out.last_hidden_state

        out = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if return_loss:
            return out.loss

        Output = namedtuple("Output", ["loss", "logits"])
        return Output(loss=out.loss, logits=out.logits)
