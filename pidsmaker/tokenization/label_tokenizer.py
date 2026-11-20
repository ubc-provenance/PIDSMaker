"""The whole label of a node is a single token, as in DeepWalk and Node2Vec."""

from typing import List, Optional

from .base import Tokenizer
from .substitution import Substitutor


def label_key(node_type: Optional[str], label: str) -> str:
    """The token of a node: its type and label, so equal labels of different types differ."""
    return f"{node_type}::{label}"


class WholeLabelTokenizer(Tokenizer):
    """Substituted words are replaced by their placeholder, so e.g. `/tmp/tmpAb3Xk9q2` and
    `/tmp/tmpQ8mZr1Lp` become the same token with the `random` substitution."""

    name = "whole_label"

    def __init__(self, substitutor: Optional[Substitutor] = None):
        self.substitutor = substitutor or Substitutor()

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        return [label_key(node_type, self.substitutor.substitute_text(label, node_type))]
