"""Word-level tokenization with NLTK.

Used by Orthrus (word2vec), R-Caid (doc2vec), NodLink (fasttext) and FLASH (word2vec).
"""

import re
from typing import List, Optional

from nltk.tokenize import word_tokenize

from .base import Tokenizer
from .substitution import Substitutor

_BACKSLASHES = re.compile(r"\\+")


def tokenize_path(label: str) -> List[str]:
    """Process and file labels: backslashes become '/' and '/' is kept as a token."""
    return word_tokenize(_BACKSLASHES.sub("/", label).replace("/", " / "))


def tokenize_netflow(label: str) -> List[str]:
    """Netflow labels: IP octets, ':' and '.' are separate tokens."""
    return word_tokenize(label.replace(":", " : ").replace(".", " . "))


def tokenize_any(label: str) -> List[str]:
    """Labels of any type: splits on '/', ':' and '.'."""
    new_label = _BACKSLASHES.sub("/", label)
    return word_tokenize(new_label.replace("/", " / ").replace(":", " : ").replace(".", " . "))


class NLTKTokenizer(Tokenizer):
    """NLTK `word_tokenize` after splitting labels on path and IP separators.

    Args:
        type_aware: if True, paths (`subject`, `file`) are split on '/' and IPs
            (`netflow`) on ':' and '.'. An empty label gives the token "".
            If False, all labels are split on '/', ':' and '.' (as in FLASH).
            Texts with `node_type=None` (e.g. edge types) are always tokenized
            in the type-agnostic way.
        substitutor: replaces the matching words of labels by placeholder tokens,
            which are kept as single tokens.
    """

    name = "nltk"

    def __init__(self, type_aware: bool = True, substitutor: Optional[Substitutor] = None):
        self.type_aware = type_aware
        self.substitutor = substitutor or Substitutor()

    def _tokenize_text(self, label: str, node_type: Optional[str]) -> List[str]:
        if not self.type_aware or node_type is None:
            return tokenize_any(label)
        if label == "":
            return [""]
        if node_type in ("subject", "file"):
            return tokenize_path(label)
        if node_type == "netflow":
            return tokenize_netflow(label)
        raise ValueError(f"Invalid node type {node_type}")

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        pieces = self.substitutor.substitute(label, node_type)
        if not any(is_placeholder for _, is_placeholder in pieces):
            return self._tokenize_text(label, node_type)

        # Text between placeholders is tokenized as usual, placeholders are single tokens
        tokens, text = [], ""
        for piece, is_placeholder in pieces:
            if is_placeholder:
                if text:
                    tokens.extend(self._tokenize_text(text, node_type))
                    text = ""
                tokens.append(piece)
            else:
                text += piece
        if text:
            tokens.extend(self._tokenize_text(text, node_type))
        return tokens
