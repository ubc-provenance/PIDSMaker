"""Persisting what a stream has taught us about nodes.

A provenance producer does not repeat itself: SPADE's `Deduplicate` screen
publishes each vertex once, and every later edge refers to it by hash alone. That
is efficient, but it means a consumer that joins the topic afterwards - a detector
started once the model has been trained, or one restarted after a crash - receives
edges whose endpoints it has never seen, and has to drop them.

The fix is to carry the node knowledge across runs. The ingestion run writes what
it learned to a state file; the detector loads it and starts out already knowing
every node the capture contained.

Only the mapping is stored, never labels: node labels depend on
`construction.node_label_features`, so the detector rebuilds them from the raw
attributes with its own configuration.
"""

import os
from typing import Dict, Optional, Tuple

import torch

from pidsmaker.streaming.records import StreamNode
from pidsmaker.utils.utils import log

STATE_FILE = "stream_state.pkl"


def default_state_path(artifact_dir: str, database: str) -> str:
    """Returns where a dataset's stream state lives by default."""
    return os.path.join(artifact_dir, "streaming", database, STATE_FILE)


def save_stream_state(path: str, adapter, key_to_node: Dict[str, Tuple[str, dict]]) -> None:
    """Writes the node knowledge accumulated by a stream run.

    Args:
        path: Destination file.
        adapter: The adapter, whose `hash_to_key` resolves a producer's vertex
            identifier to the node it was folded into.
        key_to_node: `{node key: (node type, raw attributes)}`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "adapter": adapter.name,
        "hash_to_key": dict(getattr(adapter, "hash_to_key", {})),
        "key_to_node": key_to_node,
    }
    torch.save(state, path)
    log(
        f"Stream state written to {path} "
        f"({len(state['hash_to_key'])} vertex ids, {len(key_to_node)} nodes)"
    )


def load_stream_state(path: str) -> Optional[dict]:
    """Reads a state file, or returns None when it does not exist."""
    if not path or not os.path.isfile(path):
        return None
    return torch.load(path)


def apply_stream_state(state: dict, adapter, builder) -> int:
    """Restores node knowledge into a running adapter and graph builder.

    Args:
        state: Output of `load_stream_state()`.
        adapter: The adapter to seed with the known vertex identifiers.
        builder: The graph builder, which assigns an index and rebuilds a label
            for every restored node.

    Returns:
        int: How many nodes were restored.
    """
    if state is None:
        return 0
    if state.get("adapter") != adapter.name:
        raise ValueError(
            f"Stream state was written by the {state.get('adapter')!r} adapter but is being "
            f"loaded into {adapter.name!r}. Node identifiers are adapter-specific."
        )

    for key, (node_type, attrs) in state["key_to_node"].items():
        builder.add_node(StreamNode(key=key, node_type=node_type, attrs=attrs))

    adapter.hash_to_key.update(state["hash_to_key"])

    log(
        f"Restored {len(state['key_to_node'])} nodes and {len(state['hash_to_key'])} vertex ids "
        "from the stream state: edges whose vertices were published before this run started "
        "can still be resolved."
    )
    return len(state["key_to_node"])
