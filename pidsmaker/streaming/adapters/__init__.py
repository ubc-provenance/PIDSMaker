"""Registry of provenance adapters.

`--stream_adapter` selects one by name. Registering a new source is a one-line
change here plus the adapter itself.
"""

from pidsmaker.streaming.adapters.base import ProvenanceAdapter
from pidsmaker.streaming.adapters.spade import SpadeKafkaAdapter

ADAPTERS = {
    SpadeKafkaAdapter.name: SpadeKafkaAdapter,
}


def get_adapter_class(name: str):
    """Returns the adapter class registered under `name`."""
    name = name.strip()
    if name not in ADAPTERS:
        raise ValueError(f"Unknown stream adapter {name!r}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def build_adapter(name: str, **options) -> ProvenanceAdapter:
    """Instantiates the adapter registered under `name` with the given options."""
    return get_adapter_class(name)(**options)


__all__ = [
    "ADAPTERS",
    "ProvenanceAdapter",
    "SpadeKafkaAdapter",
    "build_adapter",
    "get_adapter_class",
]
