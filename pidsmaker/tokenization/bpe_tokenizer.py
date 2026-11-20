"""Provenance-specific BPE tokenization, as in SPIDER.

Labels are first split with domain rules (path components, flags, hashes, port
categories, IP categories, ...) and the remaining words are encoded with byte-pair
encoding learned on the training labels. The implementation lives in
`pidsmaker.pretrained.data.tokenizer_bpe`; this module builds it from the config and
exposes it through the common `Tokenizer` interface.
"""

from typing import List, Optional

from .base import Tokenizer
from .substitution import get_substitutions


def build_provenance_bpe(cfg, **overrides):
    """Builds a `ProvenanceTokenizerBPE` from `featurization.tokenization.bpe`.

    Used by the pretrained encoders, which need the full BPE tokenizer (special
    tokens, walks, masking) rather than the string interface of `BPETokenizer`.
    `overrides` are passed to the constructor (e.g. `strip_entity_type`).
    """
    from pidsmaker.pretrained.data.tokenizer_bpe import ProvenanceTokenizerBPE

    bpe_cfg = cfg.featurization.tokenization.bpe
    kwargs = dict(
        max_seq_len=bpe_cfg.max_seq_len,
        bpe_vocab_size=bpe_cfg.vocab_size,
        mode=bpe_cfg.mode,
        substitutions=get_substitutions(cfg, "bpe"),
        canonicalize_neighbors=bpe_cfg.canonicalize_neighbors,
    )
    kwargs.update(overrides)
    return ProvenanceTokenizerBPE(cfg, **kwargs)


class BPETokenizer(Tokenizer):
    """String tokens of `ProvenanceTokenizerBPE`, truncated to `max_seq_len`.

    Node labels start with their entity type token ([PROC], [FILE], [SOCK]).
    Texts with `node_type=None` are mapped to their edge type token when they are
    an edge type, and are split on whitespace + BPE otherwise.
    """

    name = "bpe"
    trainable = True

    def __init__(self, cfg):
        self.bpe = build_provenance_bpe(cfg)

    def fit(self, labels):
        corpus = [(node_type, label) for node_type, label in labels if node_type is not None]
        self.bpe.build_vocab({}, walk_label_corpus=corpus)
        return self

    def _encode_words(self, words: List[str]) -> List[int]:
        ids = []
        for word in words:
            if word in self.bpe.token2id:
                ids.append(self.bpe.token2id[word])
            else:
                ids.extend(self.bpe._token_to_id(sub) for sub in self.bpe._apply_bpe(word))
        return ids

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        if not self.bpe.token2id:
            raise RuntimeError("The BPE tokenizer must be fitted or loaded before tokenizing.")
        if node_type is None:
            ids = self.bpe.tokenize_edge(label) or self._encode_words(
                self.bpe._pretokenize_node_simple(None, label)
            )
        else:
            ids = self.bpe.tokenize_node(node_type, label)
        return [self.bpe.id2token[i] for i in ids[: self.bpe.max_seq_len]]

    def save(self, model_dir: str):
        self.bpe.save(self.get_path(model_dir))

    def load(self, model_dir: str):
        max_seq_len = self.bpe.max_seq_len
        self.bpe.load(self.get_path(model_dir))
        self.bpe.max_seq_len = max_seq_len  # the config value prevails, as for pretrained models
        return self
