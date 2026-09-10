"""Online featurization: embedding the nodes of a live time window.

The offline `feat_inference` task embeds every node of a dataset in one pass,
because it knows them all up front. A live stream does not: nodes appear as
processes start and files are touched, and a detector must embed them the moment
they show up.

What makes this possible is that the featurization models PIDSMaker trains are
*inductive over text*: word2vec, fasttext and doc2vec embed a node from its
label, so a path never seen during training still gets a vector (out-of-vocabulary
words contribute zero, subwords still help in fasttext). The featurizers that are
not inductive - the ones that need random walks over the whole graph - cannot be
served online, and say so explicitly rather than silently returning wrong vectors.
"""

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from pidsmaker.utils.utils import log, tokenize_arbitrary_label, tokenize_label

# Featurization methods that embed a node from its own label, and can therefore
# embed nodes that did not exist at training time.
LABEL_METHODS = ("word2vec", "fasttext", "doc2vec", "hierarchical_hashing")
# Methods that use no learned model at all.
TYPE_ONLY_METHODS = ("only_type", "only_ones")
# Methods that embed a node from its neighborhood, recomputed per window.
NEIGHBORHOOD_METHODS = ("flash",)
# Methods that need a global pass over the dataset (random walks, dataset-wide
# min/max normalization) and have no sound streaming equivalent.
UNSUPPORTED_METHODS = ("temporal_rw", "alacarte", "ocrapt_features")


def build_online_featurizer(cfg):
    """Builds the featurizer matching `featurization.used_method`.

    Args:
        cfg: Full pipeline config; the featurization model is loaded from the
            same `featurization._model_dir` the offline run wrote it to.

    Returns:
        OnlineFeaturizer: The featurizer to call on every time window.
    """
    method = cfg.featurization.used_method.strip()

    if method in TYPE_ONLY_METHODS:
        return TypeOnlyFeaturizer(cfg, method)
    if method in LABEL_METHODS:
        return LabelFeaturizer(cfg, method)
    if method in NEIGHBORHOOD_METHODS:
        return FlashFeaturizer(cfg)
    if method in UNSUPPORTED_METHODS:
        raise NotImplementedError(
            f"Featurization method {method!r} cannot be served online: it derives node features "
            f"from a global pass over the dataset (random walks / dataset-wide statistics), which "
            f"a live stream cannot provide. Train the system with one of "
            f"{LABEL_METHODS + TYPE_ONLY_METHODS + NEIGHBORHOOD_METHODS} to run it in real time."
        )
    raise ValueError(f"Unknown featurization method {method!r}")


class OnlineFeaturizer:
    """Embeds the nodes of one time window.

    Args:
        cfg: Full pipeline config.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.emb_dim = cfg.featurization.emb_dim

    def embed_window(self, graph, indexid2msg) -> Optional[Dict[str, np.ndarray]]:
        """Returns `{index_id: vector}` for the nodes of this window.

        Returning None means "this system uses no node embedding" (`only_type`,
        `only_ones`), which the edge-message builder handles the same way the
        offline `feat_inference` task does.
        """
        raise NotImplementedError


class TypeOnlyFeaturizer(OnlineFeaturizer):
    """No embedding at all: nodes are described by their type only."""

    def __init__(self, cfg, method):
        super().__init__(cfg)
        self.method = method
        self.emb_dim = 0

    def embed_window(self, graph, indexid2msg):
        return None


class LabelFeaturizer(OnlineFeaturizer):
    """Embeds each node from its textual label, with the model trained offline.

    Vectors are cached per label: on a real host the same paths recur constantly,
    so the cache absorbs most of the cost of a window.

    Args:
        cfg: Full pipeline config.
        method: One of `LABEL_METHODS`.
        cache_size: Most labels to keep embedded (0 = unbounded).
    """

    def __init__(self, cfg, method, cache_size: int = 500_000):
        super().__init__(cfg)
        self.method = method
        self.cache_size = cache_size
        self._cache = {}
        self._model = None
        self._load_model()

    def _load_model(self):
        model_dir = self.cfg.featurization._model_dir

        if self.method == "word2vec":
            import os

            from gensim.models import Word2Vec

            from pidsmaker.featurization.feat_inference_methods.feat_inference_word2vec import (
                cal_word_weight,
            )

            self._model = Word2Vec.load(os.path.join(model_dir, "word2vec.model"))
            self._cal_word_weight = cal_word_weight
            self._decline_rate = self.cfg.featurization.word2vec.decline_rate

        elif self.method == "fasttext":
            import os

            from gensim.models import FastText

            self._model = FastText.load(os.path.join(model_dir, "fasttext.pkl"))

        elif self.method == "doc2vec":
            import os

            from gensim.models.doc2vec import Doc2Vec

            self._model = Doc2Vec.load(os.path.join(model_dir, "doc2vec_model.model"))

        elif self.method == "hierarchical_hashing":
            from sklearn.feature_extraction import FeatureHasher

            # Stateless: the hashing trick needs no trained model, which is why
            # this featurizer works online with no caveat at all.
            self._model = FeatureHasher(n_features=self.emb_dim, input_type="string")

        log(f"Loaded online featurizer: {self.method} (emb_dim={self.emb_dim})")

    def embed_window(self, graph, indexid2msg):
        vectors = {}
        for index_id in graph.nodes():
            node_type, label = indexid2msg[index_id]
            key = (node_type, label)
            vector = self._cache.get(key)
            if vector is None:
                vector = self._embed_label(node_type, label)
                if self.cache_size and len(self._cache) >= self.cache_size:
                    self._cache.clear()
                self._cache[key] = vector
            vectors[index_id] = vector
        return vectors

    def _embed_label(self, node_type: str, label: str) -> np.ndarray:
        """Embeds one label, mirroring the offline `feat_inference` method exactly."""
        if self.method == "hierarchical_hashing":
            from pidsmaker.featurization.feat_inference_methods.feat_inference_HFH import (
                ip2higlist,
                list2str,
                path2higlist,
            )

            higlist = path2higlist(label) if node_type in ("subject", "file") else ip2higlist(label)
            dense_vector = self._model.fit_transform([list2str(higlist)]).toarray()
            return _normalize(dense_vector).squeeze().astype(np.float32)

        tokens = tokenize_label(label, node_type)

        if self.method == "word2vec":
            zeros = np.zeros((self.emb_dim,))
            weights = self._cal_word_weight(len(tokens), self._decline_rate)
            word_vectors = [
                self._model.wv[word] if word in self._model.wv else zeros for word in tokens
            ]
            weighted = [w * vec for w, vec in zip(weights, word_vectors)]
            return _normalize(np.mean(weighted, axis=0)).astype(np.float32)

        if self.method == "fasttext":
            # fasttext is subword-based: even a path never seen during training has
            # a vector, which is exactly what a live stream needs.
            word_vectors = [self._model.wv[word] for word in tokens]
            return _normalize(np.mean(word_vectors, axis=0)).astype(np.float32)

        if self.method == "doc2vec":
            return _normalize(self._model.infer_vector(tokens)).astype(np.float32)

        raise ValueError(f"Unhandled label featurization method {self.method!r}")


class FlashFeaturizer(OnlineFeaturizer):
    """Flash's featurizer: a node is embedded from the events it takes part in.

    Offline, each node's "document" accumulates over every graph of the dataset.
    Online, it accumulates over every window seen so far, capped so that a
    long-running detector does not grow without bound.

    Args:
        cfg: Full pipeline config.
        max_corpus_tokens: Most tokens kept per node.
        max_properties_per_window: Same per-window cap as the offline corpus builder.
    """

    def __init__(self, cfg, max_corpus_tokens: int = 3000, max_properties_per_window: int = 300):
        super().__init__(cfg)
        import os

        from gensim.models import Word2Vec

        from pidsmaker.featurization.feat_inference_methods.feat_inference_flash import (
            PositionalEncoder,
            infer,
        )

        self._model = Word2Vec.load(
            os.path.join(cfg.featurization._model_dir, "word2vec_model_final.model")
        )
        self._infer = infer
        self._encoder = PositionalEncoder(cfg.featurization.emb_dim)
        self.max_corpus_tokens = max_corpus_tokens
        self.max_properties_per_window = max_properties_per_window
        self._node2corpus = defaultdict(list)
        self._token_cache = {}
        log(f"Loaded online featurizer: flash (emb_dim={self.emb_dim})")

    def embed_window(self, graph, indexid2msg):
        sorted_edges = sorted(
            [
                (u, v, attr["label"], int(attr["time"]))
                for u, v, _, attr in graph.edges(data=True, keys=True)
            ],
            key=lambda x: x[3],
        )

        properties_this_window = defaultdict(int)
        for src, dst, operation, _ in sorted_edges:
            src_type, src_msg = indexid2msg[src]
            dst_type, dst_msg = indexid2msg[dst]
            properties = [src_msg, operation, dst_msg]

            for node in (src, dst):
                if properties_this_window[node] >= self.max_properties_per_window:
                    continue
                properties_this_window[node] += len(properties)
                corpus = self._node2corpus[node]
                for sentence in properties:
                    if sentence not in self._token_cache:
                        self._token_cache[sentence] = tokenize_arbitrary_label(sentence)
                    corpus.extend(self._token_cache[sentence])
                if len(corpus) > self.max_corpus_tokens:
                    del corpus[: len(corpus) - self.max_corpus_tokens]

        return {
            node: self._infer(self._node2corpus[node], self._model, self._encoder).astype(
                np.float32
            )
            for node in graph.nodes()
        }


def _normalize(vector):
    """L2-normalizes a vector, as every offline `feat_inference` method does."""
    return vector / (np.linalg.norm(vector) + 1e-12)
