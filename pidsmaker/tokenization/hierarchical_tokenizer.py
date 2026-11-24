"""Hierarchical tokenization of paths and IPs, as in Kairos.

A label gives one token per level of its hierarchy:
`/usr/bin/ls` -> ["", "/usr", "/usr/bin", "/usr/bin/ls"] and
`10.0.0.1` -> ["10", "10.0", "10.0.0", "10.0.0.1"].
"""

from typing import List, Optional

from pidsmaker.config.schemas import label_format

from .base import Tokenizer
from .substitution import Substitutor


def _hierarchical_prefixes(label: str, separator: str) -> List[str]:
    prefixes = []
    for part in label.strip().split(separator):
        prefixes.append(prefixes[-1] + separator + part if prefixes else part)
    return prefixes


def path2higlist(path: str) -> List[str]:
    return _hierarchical_prefixes(path, "/")


def ip2higlist(ip: str) -> List[str]:
    return _hierarchical_prefixes(ip, ".")


class HierarchicalTokenizer(Tokenizer):
    """Path labels (e.g. processes and files) are split on '/', other labels on '.'.
    Substituted words are replaced by their placeholder before splitting."""

    name = "hierarchical"

    def __init__(self, substitutor: Optional[Substitutor] = None):
        self.substitutor = substitutor or Substitutor()

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        label = self.substitutor.substitute_text(label, node_type)
        if label_format(node_type) == "path":
            return path2higlist(label)
        return ip2higlist(label)
