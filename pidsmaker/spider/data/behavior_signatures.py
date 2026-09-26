"""
Behavior signature extraction for provenance graph entities.

A behavior signature is the set of unique (direction, event_type,
neighbor_entity_type, neighbor_class) tuples observed in an entity's
1-hop neighborhood.  Signatures are sets (no counts) so they capture
*what kinds of interactions* an entity has, not how frequently.

Example — nginx:
  {OUT:EVENT_CONNECT:netflow:port_https,
   OUT:EVENT_WRITE:file:log_web,
   IN:EVENT_EXECUTE:subject:webserver,
   IN:EVENT_RECVFROM:netflow:port_https}

The label vocabulary is built from all observed signatures across the
training corpus.  Each entity's target is a sparse binary vector over
this vocabulary, used by model_behavior.py for multi-label BCE training.
"""

import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

from .entity_classes import classify_entity


# ═══════════════════════════════════════════════════════════════════════════
# SIGNATURE LABEL
# ═══════════════════════════════════════════════════════════════════════════

# A single behavior label: (direction, event_type, neighbor_node_type, neighbor_class)
# e.g. ("OUT", "EVENT_WRITE", "file", "log_web")
BehaviorLabel = Tuple[str, str, str, str]

# A behavior signature is a frozenset of such labels
BehaviorSignature = FrozenSet[BehaviorLabel]


def _direction_from_edge(center_node: str, src: str, dst: str) -> str:
    """Determine if an edge is outgoing or incoming relative to center."""
    if src == center_node:
        return "OUT"
    return "IN"


def _neighbor_from_edge(center_node: str, src: str, dst: str) -> str:
    """Return the neighbor node (the one that isn't the center)."""
    return dst if src == center_node else src


# ═══════════════════════════════════════════════════════════════════════════
# SIGNATURE EXTRACTION FROM GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def extract_signature_from_neighborhood(
    center_node: str,
    edges: List[Tuple[str, str, str]],
    indexid2msg: dict,
) -> BehaviorSignature:
    """Extract a behavior signature from a node's 1-hop neighborhood edges.

    Args:
        center_node: the entity node ID.
        edges: list of (src, dst, edge_type) tuples from the neighborhood.
        indexid2msg: {node_id: (node_type, label_str)} mapping.

    Returns:
        A frozenset of BehaviorLabel tuples.
    """
    labels: Set[BehaviorLabel] = set()

    for src, dst, edge_type in edges:
        direction = _direction_from_edge(center_node, src, dst)
        neighbor = _neighbor_from_edge(center_node, src, dst)

        if neighbor not in indexid2msg:
            continue

        neighbor_type, neighbor_label = indexid2msg[neighbor]
        neighbor_class = classify_entity(neighbor_type, neighbor_label)

        labels.add((direction, edge_type, neighbor_type, neighbor_class))

    return frozenset(labels)


def extract_full_signature(
    center_node: str,
    sampler,
    indexid2msg: dict,
    filter_noisy: bool = False,
) -> BehaviorSignature:
    """Extract the FULL behavior signature from ALL edges of a node.

    Unlike extract_signature_from_neighborhood (which uses a sampled
    subset), this iterates over all forward and backward adjacency
    entries to build the complete signature.  Used for building the
    training dataset (computed once, reused every epoch).

    Args:
        center_node: the entity node ID.
        sampler: ProvenanceWalkSampler with forward_adj/backward_adj.
        indexid2msg: {node_id: (node_type, label_str)} mapping.
        filter_noisy: if True, skip noisy edges.

    Returns:
        A frozenset of BehaviorLabel tuples.
    """
    if filter_noisy:
        from .edge_filter import _is_noisy_file, _is_noisy_process

    labels: Set[BehaviorLabel] = set()

    # Forward edges: center → neighbor
    if center_node in sampler.forward_adj:
        for edge in sampler.forward_adj[center_node]:
            neighbor = edge[0]
            edge_type = edge[1]

            if neighbor not in indexid2msg:
                continue

            neighbor_type, neighbor_label = indexid2msg[neighbor]

            if filter_noisy:
                if neighbor_type == "file" and _is_noisy_file(neighbor_label):
                    continue
                if neighbor_type == "subject" and _is_noisy_process(neighbor_label):
                    continue

            neighbor_class = classify_entity(neighbor_type, neighbor_label)
            labels.add(("OUT", edge_type, neighbor_type, neighbor_class))

    # Backward edges: neighbor → center
    if center_node in sampler.backward_adj:
        for edge in sampler.backward_adj[center_node]:
            neighbor = edge[0]
            edge_type = edge[1]

            if neighbor not in indexid2msg:
                continue

            neighbor_type, neighbor_label = indexid2msg[neighbor]

            if filter_noisy:
                if neighbor_type == "file" and _is_noisy_file(neighbor_label):
                    continue
                if neighbor_type == "subject" and _is_noisy_process(neighbor_label):
                    continue

            neighbor_class = classify_entity(neighbor_type, neighbor_label)
            labels.add(("IN", edge_type, neighbor_type, neighbor_class))

    return frozenset(labels)


# ═══════════════════════════════════════════════════════════════════════════
# LABEL VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════

class BehaviorLabelVocab:
    """Maps behavior labels ↔ integer indices for the multi-label classifier.

    The vocabulary is built once from all training signatures and frozen.
    Each entity's target is a binary vector of size len(vocab).
    """

    def __init__(self):
        self.label2idx: Dict[BehaviorLabel, int] = {}
        self.idx2label: Dict[int, BehaviorLabel] = {}

    def build_from_signatures(self, signatures: Dict[str, BehaviorSignature]):
        """Build vocabulary from a dict of {entity_key: signature}."""
        all_labels: Set[BehaviorLabel] = set()
        for sig in signatures.values():
            all_labels.update(sig)

        # Sort for deterministic ordering
        sorted_labels = sorted(all_labels)
        self.label2idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    @property
    def size(self) -> int:
        return len(self.label2idx)

    def signature_to_vector(self, signature: BehaviorSignature) -> torch.Tensor:
        """Convert a signature to a binary vector."""
        vec = torch.zeros(self.size, dtype=torch.float32)
        for label in signature:
            if label in self.label2idx:
                vec[self.label2idx[label]] = 1.0
        return vec

    def signature_to_indices(self, signature: BehaviorSignature) -> List[int]:
        """Convert a signature to a list of active label indices."""
        return [self.label2idx[label] for label in signature if label in self.label2idx]

    def save(self, path: str):
        """Save vocabulary to a text file."""
        with open(path, "w") as f:
            f.write(f"# BehaviorLabelVocab: {self.size} labels\n")
            for idx in range(self.size):
                direction, event, ntype, nclass = self.idx2label[idx]
                f.write(f"{idx}\t{direction}:{event}:{ntype}:{nclass}\n")

    def load(self, path: str):
        """Load vocabulary from a text file."""
        self.label2idx = {}
        self.idx2label = {}
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                idx = int(parts[0])
                label_parts = parts[1].split(":")
                if len(label_parts) != 4:
                    continue
                label = tuple(label_parts)
                self.label2idx[label] = idx
                self.idx2label[idx] = label


# ═══════════════════════════════════════════════════════════════════════════
# BATCH-LEVEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def jaccard_similarity(sig_a: BehaviorSignature, sig_b: BehaviorSignature) -> float:
    """Compute Jaccard similarity between two behavior signatures."""
    if not sig_a and not sig_b:
        return 1.0
    intersection = len(sig_a & sig_b)
    union = len(sig_a | sig_b)
    if union == 0:
        return 1.0
    return intersection / union


def batch_jaccard_matrix_from_vectors(target_vecs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Jaccard similarity from binary target vectors (vectorized).

    Args:
        target_vecs: [B, N] binary (0/1) tensor.

    Returns:
        [B, B] float tensor of Jaccard similarities.
    """
    vecs = target_vecs.float()
    # intersection[i,j] = sum of min(a_i, b_j) = dot product for binary vecs
    intersection = vecs @ vecs.t()
    # |A| + |B| for each pair
    sizes = vecs.sum(dim=1)  # [B]
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - intersection
    jaccard = intersection / union.clamp(min=1e-8)
    # Fix 0/0 case (both empty → Jaccard = 1)
    both_empty = (sizes.unsqueeze(1) == 0) & (sizes.unsqueeze(0) == 0)
    jaccard[both_empty] = 1.0
    return jaccard


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_behavior_dataset(
    sampler_pairs: list,
    combined_indexid2msg: dict,
    filter_noisy: bool = False,
    min_signature_size: int = 2,
) -> Tuple[
    Dict[Tuple[str, str], BehaviorSignature],
    Dict[Tuple[str, str], List[Tuple[str, int, List[int]]]],
    BehaviorLabelVocab,
]:
    """Build the complete behavior signature dataset from all graph samplers.

    Extracts the full 1-hop behavior signature for every entity across all
    datasets.  Entities with the same (node_type, label_str) are grouped
    together (they share the same signature, since classification is
    text-based).

    Args:
        sampler_pairs: list of (sampler, ds_indexid2msg) tuples.
        combined_indexid2msg: merged {prefixed_node_id: (type, label)} mapping.
        filter_noisy: whether to skip noisy edges.
        min_signature_size: skip entities with fewer behavior labels.

    Returns:
        label_to_signature: {(node_type, label_str): frozenset_of_labels}
        label_to_entities: {(node_type, label_str): [(node_id, sampler_idx, token_ids)]}
            Note: token_ids is an empty list here; will be filled by the caller
            after tokenizer is built.
        vocab: BehaviorLabelVocab built from all signatures.
    """
    # Step 1: For each unique entity label, collect its full signature
    # across ALL graphs (union of signatures from different graph instances)
    label_to_signature: Dict[Tuple[str, str], Set[BehaviorLabel]] = defaultdict(set)
    label_to_entities: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)

    if filter_noisy:
        from .edge_filter import _is_noisy_file, _is_noisy_process

    for sidx, (sampler, ds_indexid2msg) in enumerate(sampler_pairs):
        for node in sampler.nodes:
            if node not in ds_indexid2msg:
                continue

            ntype, nlabel = ds_indexid2msg[node]

            # Skip noisy center entities
            if filter_noisy:
                if ntype == "file" and _is_noisy_file(nlabel):
                    continue
                if ntype == "subject" and _is_noisy_process(nlabel):
                    continue

            label_key = (ntype, nlabel)

            # Extract full signature (union of all edges)
            sig = extract_full_signature(
                node, sampler, ds_indexid2msg, filter_noisy=filter_noisy
            )
            label_to_signature[label_key].update(sig)
            label_to_entities[label_key].append((node, sidx))

    # Step 2: Filter by minimum signature size and freeze
    final_signatures = {}
    final_entities = {}
    for label_key, sig_set in label_to_signature.items():
        if len(sig_set) >= min_signature_size:
            final_signatures[label_key] = frozenset(sig_set)
            final_entities[label_key] = label_to_entities[label_key]

    # Step 3: Build label vocabulary
    vocab = BehaviorLabelVocab()
    vocab.build_from_signatures(final_signatures)

    return final_signatures, final_entities, vocab


def dump_entity_classifications(
    sampler_pairs: list,
    filter_noisy: bool = False,
    output_path: str = "entity_classifications.txt",
):
    """Dump entity class attributions to a file for analysis.

    Writes a report with the same format as the standalone classify_entities.py:
    summary table, per-class entity listings, and unclassified entities.

    Args:
        sampler_pairs: list of (sampler, ds_indexid2msg) tuples.
        filter_noisy: whether to skip noisy entities.
        output_path: where to write the report.
    """
    from collections import Counter

    if filter_noisy:
        from .edge_filter import _is_noisy_file, _is_noisy_process

    # Collect unique entities and classify them
    classified = defaultdict(list)    # class -> [(entity_type, label)]
    unclassified = defaultdict(list)  # entity_type -> [label]
    entity_type_map = {"subject": "PROC", "file": "FILE", "netflow": "SOCK"}

    seen = set()
    for _sidx, (sampler, ds_indexid2msg) in enumerate(sampler_pairs):
        for node in sampler.nodes:
            if node not in ds_indexid2msg:
                continue

            ntype, nlabel = ds_indexid2msg[node]

            if filter_noisy:
                if ntype == "file" and _is_noisy_file(nlabel):
                    continue
                if ntype == "subject" and _is_noisy_process(nlabel):
                    continue

            label_key = (ntype, nlabel)
            if label_key in seen:
                continue
            seen.add(label_key)

            entity_class = classify_entity(ntype, nlabel)
            display_type = entity_type_map.get(ntype, ntype.upper())

            if entity_class == "unknown":
                unclassified[display_type].append(nlabel)
            else:
                classified[entity_class].append((display_type, nlabel))

    # Write report
    total_classified = sum(len(v) for v in classified.values())
    total_unclassified = sum(len(v) for v in unclassified.values())
    total = total_classified + total_unclassified

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ENTITY CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total unique entities: {total}\n")
        if total > 0:
            f.write(f"  Classified:   {total_classified} ({100*total_classified/total:.1f}%)\n")
            f.write(f"  Unclassified: {total_unclassified} ({100*total_unclassified/total:.1f}%)\n\n")
        else:
            f.write("  No entities found.\n\n")

        # Per-type breakdown
        type_counts = Counter()
        for items in classified.values():
            for etype, _ in items:
                type_counts[etype] += 1
        type_uncl = {etype: len(items) for etype, items in unclassified.items()}
        for etype in ["PROC", "FILE", "SOCK"]:
            c = type_counts.get(etype, 0)
            u = type_uncl.get(etype, 0)
            t = c + u
            if t > 0:
                f.write(f"  {etype}: {c}/{t} classified ({100*c/t:.1f}%)\n")
        f.write("\n")

        # Summary table sorted by count
        f.write("-" * 60 + "\n")
        f.write(f"{'CLASS':<35} {'COUNT':>8}\n")
        f.write("-" * 60 + "\n")
        for cls in sorted(classified.keys(), key=lambda c: (-len(classified[c]), c)):
            f.write(f"{cls:<35} {len(classified[cls]):>8}\n")
        f.write("-" * 60 + "\n\n")

        # Detailed per-class listings
        f.write("=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATIONS\n")
        f.write("=" * 80 + "\n\n")

        for cls in sorted(classified.keys()):
            items = classified[cls]
            f.write(f"\n--- {cls} ({len(items)} entities) ---\n")
            for etype, name in sorted(items, key=lambda x: (x[0], x[1])):
                f.write(f"  [{etype}] {name}\n")

        # Unclassified
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("UNCLASSIFIED ENTITIES\n")
        f.write("=" * 80 + "\n\n")

        for etype in ["PROC", "FILE", "SOCK"]:
            items = unclassified.get(etype, [])
            f.write(f"\n--- Unclassified {etype} ({len(items)} entities) ---\n")
            for name in sorted(items):
                f.write(f"  [{etype}] {name}\n")


def dump_behavior_signatures(
    label_to_signature: Dict[Tuple[str, str], BehaviorSignature],
    output_path: str = "behavior_signatures.txt",
):
    """Dump every entity's behavior signature in human-readable format.

    For each entity, lists its class and all signature labels grouped
    by direction (OUT / IN), showing the event type, neighbor entity
    type, and neighbor class for each interaction.

    Args:
        label_to_signature: {(node_type, label_str): frozenset_of_labels}
        output_path: where to write the report.
    """
    from collections import Counter

    entity_type_map = {"subject": "PROC", "file": "FILE", "netflow": "SOCK"}

    # Compute stats
    sig_sizes = [len(sig) for sig in label_to_signature.values()]
    all_labels_flat: Set[BehaviorLabel] = set()
    for sig in label_to_signature.values():
        all_labels_flat.update(sig)

    # Count how often each label appears across entities
    label_freq = Counter()
    for sig in label_to_signature.values():
        for lbl in sig:
            label_freq[lbl] += 1

    with open(output_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("BEHAVIOR SIGNATURE REPORT\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Entities with signatures: {len(label_to_signature):,}\n")
        f.write(f"Unique behavior labels:   {len(all_labels_flat):,}\n")
        if sig_sizes:
            f.write(f"Signature size:           min={min(sig_sizes)}, "
                    f"median={sorted(sig_sizes)[len(sig_sizes)//2]}, "
                    f"max={max(sig_sizes)}, "
                    f"mean={sum(sig_sizes)/len(sig_sizes):.1f}\n")
        f.write("\n")

        # Global label frequency table
        f.write("-" * 90 + "\n")
        f.write(f"{'BEHAVIOR LABEL':<65} {'ENTITIES':>8}\n")
        f.write("-" * 90 + "\n")
        for lbl, count in label_freq.most_common():
            direction, event, ntype, nclass = lbl
            f.write(f"  {direction}:{event}:{ntype}:{nclass:<45} {count:>8}\n")
        f.write("-" * 90 + "\n\n")

        # Per-entity signatures
        f.write("=" * 90 + "\n")
        f.write("PER-ENTITY SIGNATURES\n")
        f.write("=" * 90 + "\n")

        # Sort by entity type then label
        sorted_entities = sorted(
            label_to_signature.items(),
            key=lambda x: (x[0][0], x[0][1]),
        )

        for (ntype, nlabel), sig in sorted_entities:
            display_type = entity_type_map.get(ntype, ntype.upper())
            entity_class = classify_entity(ntype, nlabel)

            f.write(f"\n--- [{display_type}] {nlabel} ---\n")
            f.write(f"    class: {entity_class}\n")
            f.write(f"    signature size: {len(sig)}\n")

            # Group by direction
            out_labels = sorted(
                (e, nt, nc) for d, e, nt, nc in sig if d == "OUT"
            )
            in_labels = sorted(
                (e, nt, nc) for d, e, nt, nc in sig if d == "IN"
            )

            if out_labels:
                f.write(f"    OUT ({len(out_labels)}):\n")
                for event, neighbor_type, neighbor_class in out_labels:
                    nt_display = entity_type_map.get(neighbor_type, neighbor_type)
                    f.write(f"      → {event:<25} [{nt_display}] {neighbor_class}\n")

            if in_labels:
                f.write(f"    IN ({len(in_labels)}):\n")
                for event, neighbor_type, neighbor_class in in_labels:
                    nt_display = entity_type_map.get(neighbor_type, neighbor_type)
                    f.write(f"      ← {event:<25} [{nt_display}] {neighbor_class}\n")
