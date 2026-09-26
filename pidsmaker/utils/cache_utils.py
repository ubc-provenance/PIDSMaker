"""Generic artifact caching system for pipeline tasks.

Provides deterministic hash-based caching for any intermediate artifacts
(tokenizers, walk corpora, embeddings, etc.) with automatic parameter tracking.
"""

import hashlib
import json
import os
import pickle
from typing import Any, Callable, Dict, Optional

from pidsmaker.utils.utils import log


class ArtifactCache:
    """Hash-based cache for pipeline artifacts with automatic versioning.

    Example:
        # Cache a walk corpus
        cache = ArtifactCache(name="walk_corpus", output_dir=model_dir)
        corpus_params = {'walk_length': 10, 'num_walks': 5, 'datasets': ['train']}

        corpus = cache.load(params=corpus_params)
        if corpus is None:
            corpus = build_corpus(...)
            cache.save(data=corpus, params=corpus_params)
    """

    def __init__(self, name: str, output_dir: str, hash_length: int = 12):
        """Initialize artifact cache.

        Args:
            name: Artifact name (e.g., 'tokenizer', 'walk_corpus')
            output_dir: Directory to store cached artifacts
            hash_length: Length of hash string (default 12)
        """
        self.name = name
        self.output_dir = output_dir
        self.hash_length = hash_length
        os.makedirs(output_dir, exist_ok=True)

    def compute_hash(self, params: Dict[str, Any]) -> str:
        """Compute deterministic hash from parameters.

        Args:
            params: Dictionary of parameters that define this artifact

        Returns:
            Hexadecimal hash string of specified length
        """
        # Ensure deterministic ordering
        params_str = json.dumps(params, sort_keys=True)
        full_hash = hashlib.md5(params_str.encode()).hexdigest()
        return full_hash[:self.hash_length]

    def get_paths(self, cache_hash: str) -> tuple[str, str]:
        """Get artifact and config file paths for a given hash.

        Args:
            cache_hash: Hash string identifying the artifact

        Returns:
            (artifact_path, config_path)
        """
        artifact_path = os.path.join(self.output_dir, f"{self.name}_{cache_hash}.pkl")
        config_path = os.path.join(self.output_dir, f"{self.name}_config_{cache_hash}.json")
        return artifact_path, config_path

    def load(
        self,
        params: Dict[str, Any],
        loader_fn: Optional[Callable[[str], Any]] = None
    ) -> Optional[Any]:
        """Try to load cached artifact if it exists.

        Args:
            params: Parameters that define this artifact
            loader_fn: Optional custom loader function. If None, uses pickle.
                      Function signature: (file_path: str) -> artifact

        Returns:
            Loaded artifact if cache exists, None otherwise
        """
        cache_hash = self.compute_hash(params)
        artifact_path, config_path = self.get_paths(cache_hash)

        if not os.path.exists(artifact_path) or not os.path.exists(config_path):
            return None

        try:
            log(f"Found cached {self.name}: {artifact_path}")

            # Load artifact using custom loader or pickle
            if loader_fn:
                artifact = loader_fn(artifact_path)
            else:
                with open(artifact_path, 'rb') as f:
                    artifact = pickle.load(f)

            log(f"Loaded cached {self.name} (hash: {cache_hash})")
            return artifact

        except Exception as e:
            log(f"Failed to load cached {self.name}: {e}")
            return None

    def save(
        self,
        data: Any,
        params: Dict[str, Any],
        saver_fn: Optional[Callable[[Any, str], None]] = None
    ) -> str:
        """Save artifact with its configuration.

        Args:
            data: Artifact data to save
            params: Parameters that define this artifact
            saver_fn: Optional custom saver function. If None, uses pickle.
                     Function signature: (artifact, file_path: str) -> None

        Returns:
            Cache hash identifying the saved artifact
        """
        cache_hash = self.compute_hash(params)
        artifact_path, config_path = self.get_paths(cache_hash)

        # Save artifact using custom saver or pickle
        if saver_fn:
            saver_fn(data, artifact_path)
        else:
            with open(artifact_path, 'wb') as f:
                pickle.dump(data, f)

        log(f"Saved {self.name} cache: {artifact_path}")

        # Save configuration for debugging/reproducibility
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=2)

        log(f"Saved {self.name} config: {config_path}")

        return cache_hash

    def exists(self, params: Dict[str, Any]) -> bool:
        """Check if cached artifact exists for given parameters.

        Args:
            params: Parameters that define this artifact

        Returns:
            True if cache exists, False otherwise
        """
        cache_hash = self.compute_hash(params)
        artifact_path, config_path = self.get_paths(cache_hash)
        return os.path.exists(artifact_path) and os.path.exists(config_path)

    def clear(self, params: Optional[Dict[str, Any]] = None):
        """Clear cached artifacts.

        Args:
            params: If provided, only clears artifact with these params.
                   If None, clears all artifacts with this name.
        """
        if params:
            cache_hash = self.compute_hash(params)
            artifact_path, config_path = self.get_paths(cache_hash)
            for path in [artifact_path, config_path]:
                if os.path.exists(path):
                    os.remove(path)
                    log(f"Cleared cache: {path}")
        else:
            # Clear all artifacts with this name
            for file in os.listdir(self.output_dir):
                if file.startswith(f"{self.name}_"):
                    path = os.path.join(self.output_dir, file)
                    os.remove(path)
                    log(f"Cleared cache: {path}")


def extract_cache_params(cfg_section, param_keys: list[str]) -> Dict[str, Any]:
    """Extract caching parameters from config section.

    Useful helper to extract only cache-relevant params from a config object.

    Args:
        cfg_section: Config section (e.g., cfg.featurization.feat_training.spider.corpus)
        param_keys: List of parameter names to extract

    Returns:
        Dictionary of extracted parameters
    """
    params = {}
    for key in param_keys:
        if hasattr(cfg_section, key):
            value = getattr(cfg_section, key)
            # Convert to JSON-serializable types
            if hasattr(value, 'tolist'):  # numpy array
                value = value.tolist()
            params[key] = value
    return params


def config_to_params(cfg_section, exclude_private: bool = True) -> Dict[str, Any]:
    """Automatically extract all parameters from a config section.

    Dynamically extracts all attributes from a config object, filtering out
    private attributes (starting with _) and methods by default.

    Args:
        cfg_section: Config section (e.g., cfg.featurization.feat_training.spider.corpus)
        exclude_private: If True, exclude attributes starting with '_'

    Returns:
        Dictionary of all extracted parameters

    Example:
        # Instead of manually building params dict:
        corpus_params = {
            'walk_length': cfg.corpus.walk_length,
            'num_walks': cfg.corpus.num_walks,
            ...
        }

        # Simply do:
        corpus_params = config_to_params(cfg.featurization.feat_training.spider.corpus)
    """
    params = {}

    # Try to use config's built-in conversion if available (OmegaConf, etc.)
    if hasattr(cfg_section, 'to_dict'):
        return cfg_section.to_dict()

    # Fallback: extract attributes manually
    for key in dir(cfg_section):
        # Skip private attributes and methods
        if exclude_private and key.startswith('_'):
            continue

        # Skip methods and special attributes
        try:
            value = getattr(cfg_section, key)
            if callable(value):
                continue

            # Convert to JSON-serializable types
            if hasattr(value, 'tolist'):  # numpy array
                value = value.tolist()
            elif hasattr(value, 'to_dict'):  # nested config
                value = value.to_dict()

            params[key] = value
        except (AttributeError, TypeError):
            continue

    return params
