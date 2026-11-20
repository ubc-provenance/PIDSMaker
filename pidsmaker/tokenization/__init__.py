"""Tokenization of node labels, configured in `featurization.tokenization`.

`used_method` selects one of `TOKENIZERS`; each one is parametrized by the
sub-dict of the same name, e.g.:

    featurization:
      used_method: word2vec
      tokenization:
        used_method: nltk
        nltk:
          type_aware: True
          substitutions: none   # or e.g. hash, guid, number (see substitution.py)
"""

from pidsmaker.utils.utils import log

from .base import NODE_TYPES, Tokenizer
from .bpe_tokenizer import BPETokenizer, build_provenance_bpe
from .hierarchical_tokenizer import HierarchicalTokenizer
from .hf_tokenizer import (
    HuggingFaceTokenizer,
    get_hf_substitutor,
    get_hf_tokenizer_name,
    load_hf_tokenizer,
)
from .label_tokenizer import WholeLabelTokenizer, label_key
from .nltk_tokenizer import NLTKTokenizer
from .substitution import SUBSTITUTION_METHODS, Substitutor, get_substitutions


def _substitutor(cfg, method):
    return Substitutor(get_substitutions(cfg, method))


def _build_nltk(cfg):
    return NLTKTokenizer(
        type_aware=cfg.featurization.tokenization.nltk.type_aware,
        substitutor=_substitutor(cfg, "nltk"),
    )


TOKENIZERS = {
    "nltk": _build_nltk,
    "hierarchical": lambda cfg: HierarchicalTokenizer(_substitutor(cfg, "hierarchical")),
    "whole_label": lambda cfg: WholeLabelTokenizer(_substitutor(cfg, "whole_label")),
    "bpe": BPETokenizer,
    "huggingface": HuggingFaceTokenizer,
}


def get_tokenization_method(cfg) -> str:
    return cfg.featurization.tokenization.used_method.strip()


def build_tokenizer(cfg) -> Tokenizer:
    """Builds the tokenizer set in `featurization.tokenization.used_method`.

    Trainable tokenizers are returned empty: use `fit_tokenizer` (featurization) or
    `load_tokenizer` (feat_inference) instead.
    """
    method = get_tokenization_method(cfg)
    if method == "none":
        raise ValueError(
            f"Featurization method `{cfg.featurization.used_method}` tokenizes node labels, "
            "but `featurization.tokenization.used_method` is `none`."
        )
    if method not in TOKENIZERS:
        raise ValueError(f"Invalid tokenization method `{method}`.")
    return TOKENIZERS[method](cfg)


def fit_tokenizer(cfg, labels) -> Tokenizer:
    """Builds the tokenizer and, if trainable, fits it on the `(node_type, label)` pairs
    and saves it in `featurization._model_dir` for feat_inference."""
    tokenizer = build_tokenizer(cfg)
    if tokenizer.trainable:
        log(f"Fitting {tokenizer.name} tokenizer...")
        tokenizer.fit(labels)
        tokenizer.save(cfg.featurization._model_dir)
    return tokenizer


def load_tokenizer(cfg) -> Tokenizer:
    """Builds the tokenizer used during featurization, loading its vocabulary if trainable."""
    tokenizer = build_tokenizer(cfg)
    if tokenizer.trainable:
        tokenizer.load(cfg.featurization._model_dir)
    return tokenizer


__all__ = [
    "NODE_TYPES",
    "SUBSTITUTION_METHODS",
    "Substitutor",
    "TOKENIZERS",
    "Tokenizer",
    "BPETokenizer",
    "HierarchicalTokenizer",
    "HuggingFaceTokenizer",
    "NLTKTokenizer",
    "WholeLabelTokenizer",
    "build_provenance_bpe",
    "build_tokenizer",
    "fit_tokenizer",
    "get_hf_substitutor",
    "get_hf_tokenizer_name",
    "get_substitutions",
    "get_tokenization_method",
    "label_key",
    "load_hf_tokenizer",
    "load_tokenizer",
]
