"""
BERT models for provenance graph pretraining and fine-tuning.

Follows CyberGFM's architecture: HuggingFace BertForMaskedLM for pretraining,
with CLS and LP fine-tuning heads for anomaly detection.
"""

from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput


# Model size presets (matching CyberGFM)
MODEL_SIZES = {
    "tiny": SimpleNamespace(H=128, L=2, A=2, I=512),
    "mini": SimpleNamespace(H=256, L=4, A=4, I=1024),
    "med": SimpleNamespace(H=512, L=8, A=8, I=2048),
    "baseline": SimpleNamespace(H=768, L=12, A=12, I=3072),
}


def get_bert_config(vocab_size: int, model_size: str = "tiny", max_position_embeddings: int = 512) -> BertConfig:
    """Create BertConfig for a given model size."""
    params = MODEL_SIZES[model_size]
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=params.H,
        num_hidden_layers=params.L,
        num_attention_heads=params.A,
        intermediate_size=params.I,
        max_position_embeddings=max_position_embeddings,
    )


class ProvenanceBERT(BertPreTrainedModel):
    """BERT for masked language modeling on provenance graph walks.

    Directly follows CyberGFM's GraphBertForMaskedLM pattern:
    BertModel (encoder) + BertOnlyMLMHead (prediction head).
    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def modified_fwd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_loss: bool = True,
        skip_cls: bool = False,
    ):
        """Convenience wrapper matching CyberGFM's modified_fwd interface.

        Args:
            input_ids: [B, L] token IDs (with [MASK] applied)
            attention_mask: [B, L] bool, True for real tokens
            labels: [B, L] target token IDs, -100 for non-predicted positions
            return_loss: if True, return scalar loss; else return full output
            skip_cls: if True, return hidden states instead of prediction logits
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        position_ids = torch.arange(
            input_ids.size(1), device=self.device
        ).unsqueeze(0).expand(input_ids.size(0), -1)

        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
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

        outputs = self.bert(
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

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
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


def compute_cls_loss(pred: torch.Tensor, labels: torch.Tensor, margin: float, device) -> torch.Tensor:
    """Compute CLS fine-tuning loss (shared by BERT and ModernBERT heads).

    If margin > 0: margin ranking loss — anomalous logit must exceed the normal
    logit by at least `margin`. Produces an unbounded score range at inference.
    If margin <= 0: standard BCE with logits loss.

    Args:
        pred: [B, 1] anomaly logits from the classifier head
        labels: [B, 1] float, 1 for anomalous, 0 for normal
        margin: ranking margin threshold (use BCE when <= 0)
        device: torch device
    """
    if margin > 0:
        pred = pred.squeeze(-1)                       # [B]
        labels_1d = labels.squeeze(-1).to(device)    # [B]
        pred_pos = pred[labels_1d == 0]              # normal edges:    want low score
        pred_neg = pred[labels_1d == 1]              # anomalous edges: want high score
        min_len = min(len(pred_pos), len(pred_neg))
        return F.relu(margin - (pred_neg[:min_len] - pred_pos[:min_len])).mean()
    else:
        return nn.BCEWithLogitsLoss()(pred, labels.float().to(device))


class FineTuneCLS(nn.Module):
    """Classification-based fine-tuning for anomaly detection.

    Backbone-agnostic: wraps any model that exposes `modified_fwd(...)`.
    Appends [CLS] token to walk+edge+dst sequence,
    uses the [CLS] representation for binary classification.

    Subclass or use directly by passing backbone_cls:
        model = FineTuneCLS(ProvenanceBERT, config, ...)
    """

    def __init__(self, backbone_cls, config, pretrained_state_dict=None, device="cpu", freeze_backbone=True, margin=2.0):
        super().__init__()
        self.fm = backbone_cls(config)
        if pretrained_state_dict is not None:
            self.fm.load_state_dict(pretrained_state_dict)
        self.fm = self.fm.to(device)

        if freeze_backbone:
            for param in self.fm.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1, device=device),
        )
        self.device = device
        self.margin = margin

    def predict(self, input_ids, attention_mask, target_mask):
        """Get anomaly scores for the [CLS] positions."""
        hidden = self.fm.modified_fwd(
            input_ids, attention_mask,
            labels=torch.full_like(input_ids, -100),
            return_loss=False, skip_cls=True,
        )
        cls_hidden = hidden[target_mask]  # [B, H]
        return self.classifier(cls_hidden)

    def forward(self, input_ids, attention_mask, target_mask, labels):
        pred = self.predict(input_ids, attention_mask, target_mask)
        return compute_cls_loss(pred, labels, self.margin, self.device)


class FineTuneLP(nn.Module):
    """Link prediction fine-tuning for anomaly detection.

    Backbone-agnostic: wraps any model that exposes `modified_fwd(...)`.
    Uses masked destination node prediction. Anomaly score is
    1 - P(true destination | context walk + edge type).

    Subclass or use directly by passing backbone_cls:
        model = FineTuneLP(ProvenanceBERT, config, ...)
    """

    def __init__(self, backbone_cls, config, pretrained_state_dict=None, device="cpu", freeze_backbone=False):
        super().__init__()
        self.fm = backbone_cls(config)
        if pretrained_state_dict is not None:
            self.fm.load_state_dict(pretrained_state_dict)
        self.fm = self.fm.to(device)

        if freeze_backbone:
            for param in self.fm.parameters():
                param.requires_grad = False

        self.config = config
        self.device = device

    def forward(self, input_ids, attention_mask, target_mask, target_token_ids):
        return self.fm.modified_fwd(
            input_ids, attention_mask, labels=target_token_ids, return_loss=True
        )

    @torch.no_grad()
    def anomaly_score(self, input_ids, attention_mask, target_mask, target_token_ids):
        """Compute per-edge anomaly scores via mean cross-entropy on masked tokens."""
        out = self.fm.modified_fwd(
            input_ids, attention_mask,
            labels=torch.full_like(input_ids, -100),
            return_loss=False,
        )
        logits = out.logits  # [B, L, V]
        B, L, V = logits.shape

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_pos_loss = loss_fct(logits.view(-1, V), target_token_ids.view(-1)).view(B, L)

        valid = (target_token_ids >= 0) & target_mask
        num_valid = valid.float().sum(dim=1).clamp(min=1)
        scores = (per_pos_loss * valid.float()).sum(dim=1) / num_valid

        return scores


# Backwards-compatible aliases
def ProvenanceBERTFineTuneCLS(config, pretrained_state_dict=None, device="cpu", freeze_backbone=True, margin=2.0):
    return FineTuneCLS(ProvenanceBERT, config, pretrained_state_dict, device, freeze_backbone, margin)

def ProvenanceBERTFineTuneLP(config, pretrained_state_dict=None, device="cpu", freeze_backbone=False):
    return FineTuneLP(ProvenanceBERT, config, pretrained_state_dict, device, freeze_backbone)
