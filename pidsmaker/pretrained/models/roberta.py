"""
RoBERTa models for provenance graph pretraining and fine-tuning.

RoBERTa differs from BERT in:
  - Dynamic masking (already used in the SPIDER pipeline)
  - No next-sentence prediction (already omitted)
  - RobertaLMHead: dense → GELU → LayerNorm → projection (vs BERT's simpler head)
  - Position embeddings offset by padding_idx + 1
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaLMHead,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput

from .bert import MODEL_SIZES, FineTuneCLS, FineTuneLP


def get_roberta_config(vocab_size: int, model_size: str = "tiny", max_position_embeddings: int = 512) -> RobertaConfig:
    """Create RobertaConfig for a given model size."""
    params = MODEL_SIZES[model_size]
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=params.H,
        num_hidden_layers=params.L,
        num_attention_heads=params.A,
        intermediate_size=params.I,
        max_position_embeddings=max_position_embeddings + 2,  # RoBERTa offset
    )


class ProvenanceRoBERTa(RobertaPreTrainedModel):
    """RoBERTa for masked language modeling on provenance graph walks.

    Uses RobertaModel (encoder) + RobertaLMHead (prediction head with
    dense → GELU → LayerNorm → projection).
    """

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Convenience wrapper matching CyberGFM's modified_fwd interface."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            skip_cls=skip_cls,
        )

        if skip_cls:
            return out  # hidden states [B,L,H]
        if return_loss:
            return out.loss
        return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_cls: bool = False,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if skip_cls:
            return sequence_output

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def ProvenanceRoBERTaFineTuneCLS(config, pretrained_state_dict=None, device="cpu", freeze_backbone=True, margin=2.0):
    return FineTuneCLS(ProvenanceRoBERTa, config, pretrained_state_dict, device, freeze_backbone, margin)

def ProvenanceRoBERTaFineTuneLP(config, pretrained_state_dict=None, device="cpu", freeze_backbone=False):
    return FineTuneLP(ProvenanceRoBERTa, config, pretrained_state_dict, device, freeze_backbone)
