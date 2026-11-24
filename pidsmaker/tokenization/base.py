"""Common interface of all tokenizers.

A tokenizer turns the textual label of a node into a list of string tokens.
Featurization methods (word2vec, doc2vec, fasttext, flash, ...) only rely on this
interface, so any of them can be combined with any tokenizer through
`featurization.tokenization.used_method`.
"""

import os
from typing import Iterable, List, Optional, Tuple

from pidsmaker.config.schemas import all_node_types

# Node types of all dataset schemas (see `pidsmaker/config/schemas.py`)
NODE_TYPES = tuple(all_node_types())

# Name of the file where trainable tokenizers are stored in `featurization._model_dir`.
TOKENIZER_FILE = "tokenizer.pt"


class Tokenizer:
    """Base class of tokenizers.

    `tokenize(label, node_type)` receives the label of a node and its type (e.g. `subject`,
    `file`, `netflow` or `host`). `node_type=None` marks a text that is not the
    label of a node, e.g. the edge type that FLASH adds to its node documents.

    Trainable tokenizers (e.g. BPE) learn their vocabulary in `fit()` during the
    featurization task; it is saved next to the featurization model and loaded back
    during feat_inference, so both tasks tokenize labels the same way.
    """

    name: str = None
    trainable: bool = False

    def fit(self, labels: Iterable[Tuple[Optional[str], str]]):
        """Learns the vocabulary from `(node_type, label)` pairs. No-op if not trainable."""
        return self

    def tokenize(self, label: str, node_type: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    def save(self, model_dir: str):
        """Saves the learned vocabulary in `model_dir`. No-op if not trainable."""

    def load(self, model_dir: str):
        """Loads the vocabulary saved by `save()`. No-op if not trainable."""
        return self

    @staticmethod
    def get_path(model_dir: str):
        return os.path.join(model_dir, TOKENIZER_FILE)

    def __call__(self, label: str, node_type: Optional[str] = None) -> List[str]:
        return self.tokenize(label, node_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
