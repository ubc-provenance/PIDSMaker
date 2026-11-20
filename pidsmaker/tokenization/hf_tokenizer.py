"""Tokenizers of HuggingFace language models (GPT-2, Llama, OPT, ...)."""

from typing import List, Optional

from .base import Tokenizer
from .substitution import Substitutor, get_substitutions

AUTO_MODEL_NAME = "auto"


def get_hf_tokenizer_name(cfg) -> str:
    """HuggingFace id of the tokenizer set in `featurization.tokenization.huggingface`.

    `model_name: auto` selects the tokenizer of the pretrained language model
    (`featurization.pretrained.model_type` and `model_size`).
    """
    model_name = cfg.featurization.tokenization.huggingface.model_name
    if model_name != AUTO_MODEL_NAME:
        return model_name

    from pidsmaker.pretrained.models.hf_pretrained import HF_MODEL_TYPES, get_hf_model_name

    pretrained_cfg = cfg.featurization.pretrained
    if cfg.featurization.used_method != "pretrained" or pretrained_cfg.model_type not in HF_MODEL_TYPES:
        raise ValueError(
            "`featurization.tokenization.huggingface.model_name: auto` is only valid with a "
            f"HuggingFace pretrained model ({', '.join(HF_MODEL_TYPES)}). "
            "Set a HuggingFace model id instead (e.g. openai-community/gpt2)."
        )
    return get_hf_model_name(pretrained_cfg.model_type, pretrained_cfg.model_size)


def load_hf_tokenizer(name_or_dir: str):
    """Loads a HuggingFace tokenizer, using EOS as padding token when there is none."""
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    return hf_tokenizer


def get_hf_substitutor(cfg) -> Substitutor:
    return Substitutor(get_substitutions(cfg, "huggingface"))


class HuggingFaceTokenizer(Tokenizer):
    """Subword tokens of a HuggingFace tokenizer, truncated to `max_seq_len`.
    Substituted words are replaced by their placeholder text before tokenizing."""

    name = "huggingface"

    def __init__(self, cfg):
        self.model_name = get_hf_tokenizer_name(cfg)
        self.max_seq_len = cfg.featurization.tokenization.huggingface.max_seq_len
        self.hf_tokenizer = load_hf_tokenizer(self.model_name)
        self.substitutor = get_hf_substitutor(cfg)

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        label = self.substitutor.substitute_text(label, node_type)
        return self.hf_tokenizer.tokenize(label)[: self.max_seq_len]
