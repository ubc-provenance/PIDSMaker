#!/usr/bin/env python3
"""
Build 1-hop neighborhood walks from auditd provenance graph (entities.tsv + edges.tsv).

Reads the output of audit_to_provenance.py and produces walks in the same format
as the synthetic training data:
    [TYPE] label | EVENT | [TYPE] label | EVENT | [TYPE] label ...

Each unique entity gets one entry showing all its direct neighbors grouped by
direction (outgoing / incoming) and event type.

Usage:
    python3 build_neighborhoods.py --input provenance_data/ --output neighborhoods.txt
    python3 build_neighborhoods.py --input provenance_data/ --output neighborhoods.txt --walks walks.txt
"""

import argparse
import os
import sys
from collections import defaultdict


def load_graph(input_dir):
    """Load entities.tsv and edges.tsv into memory."""
    entities = {}  # entity_id -> (entity_type, entity_text)
    edges = []     # (src_id, event_type, dst_id, timestamp)

    entities_path = os.path.join(input_dir, 'entities.tsv')
    edges_path = os.path.join(input_dir, 'edges.tsv')

    # Load entities
    with open(entities_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                eid, etype, etext = parts[0], parts[1], parts[2]
                entities[eid] = (etype, etext)

    # Load edges
    with open(edges_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                src, evt, dst, ts = parts[0], parts[1], parts[2], parts[3]
                edges.append((src, evt, dst, float(ts)))

    return entities, edges


def build_adjacency(entities, edges):
    """Build adjacency lists: for each entity, track outgoing and incoming edges."""
    # outgoing[entity_id] = [(event_type, neighbor_id, timestamp), ...]
    outgoing = defaultdict(list)
    # incoming[entity_id] = [(event_type, neighbor_id, timestamp), ...]
    incoming = defaultdict(list)

    for src, evt, dst, ts in edges:
        if src in entities and dst in entities:
            outgoing[src].append((evt, dst, ts))
            incoming[dst].append((evt, src, ts))

    return outgoing, incoming


def format_entity(entities, eid):
    """Format entity as [TYPE] label."""
    etype, etext = entities[eid]
    return f"[{etype}] {etext}"


def deduplicate_entity_text(entities):
    """
    Group entity IDs that share the same (type, text) — these are the same
    logical entity created multiple times (e.g. same file opened many times).
    Returns: text_key -> [entity_ids], and canonical_id map.
    """
    groups = defaultdict(list)
    for eid, (etype, etext) in entities.items():
        key = (etype, etext)
        groups[key].append(eid)
    return groups


def write_neighborhoods(entities, outgoing, incoming, output_path):
    """Write readable 1-hop neighborhood file grouped by entity label."""

    # Group by unique (type, text) to merge duplicate entity IDs
    text_groups = deduplicate_entity_text(entities)

    # Separate by entity type
    by_type = defaultdict(list)
    for (etype, etext), eids in text_groups.items():
        by_type[etype].append((etext, eids))

    # Sort within each type
    for etype in by_type:
        by_type[etype].sort(key=lambda x: x[0])

    total_unique = len(text_groups)
    total_with_edges = 0

    with open(output_path, 'w') as f:
        f.write(f"# 1-Hop Neighborhoods from auditd provenance graph\n")
        f.write(f"# Unique entities (by label): {total_unique}\n")
        f.write(f"# Entity IDs (with duplicates): {len(entities)}\n")
        f.write(f"#\n")

        for etype in ['PROC', 'FILE', 'SOCK']:
            if etype not in by_type:
                continue

            entries = by_type[etype]
            f.write(f"\n{'='*80}\n")
            f.write(f"  [{etype}] — {len(entries)} unique entities\n")
            f.write(f"{'='*80}\n\n")

            for etext, eids in entries:
                # Merge outgoing/incoming across all IDs for this entity
                merged_out = defaultdict(int)  # (event, neighbor_type, neighbor_text) -> count
                merged_in = defaultdict(int)

                for eid in eids:
                    for evt, nid, ts in outgoing.get(eid, []):
                        if nid in entities:
                            ntype, ntext = entities[nid]
                            merged_out[(evt, ntype, ntext)] += 1
                    for evt, nid, ts in incoming.get(eid, []):
                        if nid in entities:
                            ntype, ntext = entities[nid]
                            merged_in[(evt, ntype, ntext)] += 1

                if not merged_out and not merged_in:
                    continue

                total_with_edges += 1
                total_out = sum(merged_out.values())
                total_in = sum(merged_in.values())

                f.write(f"--- [{etype}] {etext} ---\n")

                if merged_out:
                    f.write(f"  OUTGOING ({total_out}):\n")
                    for (evt, ntype, ntext), count in sorted(merged_out.items(), key=lambda x: -x[1]):
                        f.write(f"    --{evt}--> [{ntype}] {ntext}")
                        if count > 1:
                            f.write(f"  x{count}")
                        f.write(f"\n")

                if merged_in:
                    f.write(f"  INCOMING ({total_in}):\n")
                    for (evt, ntype, ntext), count in sorted(merged_in.items(), key=lambda x: -x[1]):
                        f.write(f"    <--{evt}-- [{ntype}] {ntext}")
                        if count > 1:
                            f.write(f"  x{count}")
                        f.write(f"\n")

                f.write(f"\n")

    print(f"Wrote {total_with_edges} entity neighborhoods to {output_path}")
    print(f"  ({total_unique} unique labels, {len(entities)} entity IDs)")


def write_walks(entities, outgoing, incoming, walks_path):
    """
    Generate training walks in the provenance walk format:
      [TYPE] entity | EVENT | [TYPE] neighbor

    For each entity, emit one walk line per (event, neighbor) edge.
    This produces triplets suitable for the T5 pretraining task.
    """
    walk_count = 0

    with open(walks_path, 'w') as f:
        for eid, (etype, etext) in sorted(entities.items()):
            # Outgoing edges: entity -> event -> neighbor
            for evt, nid, ts in outgoing.get(eid, []):
                if nid in entities:
                    ntype, ntext = entities[nid]
                    f.write(f"[{etype}] {etext} | {evt} | [{ntype}] {ntext}\n")
                    walk_count += 1

            # Incoming edges: neighbor -> event -> entity
            for evt, nid, ts in incoming.get(eid, []):
                if nid in entities:
                    ntype, ntext = entities[nid]
                    f.write(f"[{ntype}] {ntext} | {evt} | [{etype}] {etext}\n")
                    walk_count += 1

    print(f"Wrote {walk_count} walk triplets to {walks_path}")


def print_stats(entities, outgoing, incoming):
    """Print summary statistics."""
    text_groups = deduplicate_entity_text(entities)

    by_type = defaultdict(int)
    for (etype, _), _ in text_groups.items():
        by_type[etype] += 1

    # Degree stats per unique entity
    degrees = []
    for (etype, etext), eids in text_groups.items():
        out_neighbors = set()
        in_neighbors = set()
        for eid in eids:
            for evt, nid, ts in outgoing.get(eid, []):
                if nid in entities:
                    ntype, ntext = entities[nid]
                    out_neighbors.add((evt, ntype, ntext))
            for evt, nid, ts in incoming.get(eid, []):
                if nid in entities:
                    ntype, ntext = entities[nid]
                    in_neighbors.add((evt, ntype, ntext))
        degrees.append((len(out_neighbors), len(in_neighbors)))

    # Event type distribution
    event_counts = defaultdict(int)
    for eid in outgoing:
        for evt, nid, ts in outgoing[eid]:
            event_counts[evt] += 1

    out_degrees = [d[0] for d in degrees]
    in_degrees = [d[1] for d in degrees]
    total_degrees = [d[0]+d[1] for d in degrees]

    print(f"\n{'='*60}")
    print(f"  PROVENANCE GRAPH STATISTICS")
    print(f"{'='*60}")
    print(f"  Entity IDs:         {len(entities):>8,}")
    print(f"  Unique labels:      {len(text_groups):>8,}")
    print(f"  Total edges:        {sum(len(v) for v in outgoing.values()):>8,}")
    print(f"")
    print(f"  By type:")
    for t in ['PROC', 'FILE', 'SOCK']:
        if t in by_type:
            print(f"    [{t}]: {by_type[t]:>8,}")
    print(f"")
    if degrees:
        print(f"  Degree stats (unique neighbors per label):")
        print(f"    Out:   min={min(out_degrees)}, max={max(out_degrees)}, avg={sum(out_degrees)/len(out_degrees):.1f}")
        print(f"    In:    min={min(in_degrees)}, max={max(in_degrees)}, avg={sum(in_degrees)/len(in_degrees):.1f}")
        print(f"    Total: min={min(total_degrees)}, max={max(total_degrees)}, avg={sum(total_degrees)/len(total_degrees):.1f}")
    print(f"")
    print(f"  Edge event distribution:")
    for evt, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"    {evt:25s} {count:>10,}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Build 1-hop neighborhoods from auditd provenance graph')
    parser.add_argument('--input', '-i', required=True,
                        help='Directory containing entities.tsv and edges.tsv')
    parser.add_argument('--output', '-o', required=True,
                        help='Output file for readable neighborhoods')
    parser.add_argument('--walks', '-w', default=None,
                        help='Optional: output file for walk triplets (training format)')
    args = parser.parse_args()

    # Verify input files exist
    for fname in ['entities.tsv', 'edges.tsv']:
        fpath = os.path.join(args.input, fname)
        if not os.path.exists(fpath):
            print(f"Error: {fpath} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading graph from {args.input}/...")
    entities, edges = load_graph(args.input)
    print(f"  Loaded {len(entities)} entities, {len(edges)} edges")

    outgoing, incoming = build_adjacency(entities, edges)
    print_stats(entities, outgoing, incoming)

    write_neighborhoods(entities, outgoing, incoming, args.output)

    if args.walks:
        write_walks(entities, outgoing, incoming, args.walks)

    print("\nDone.")


if __name__ == '__main__':
    main()