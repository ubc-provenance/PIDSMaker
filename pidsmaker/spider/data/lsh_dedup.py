"""Structural-group near-duplicate removal for tokenized walk corpora.

Groups walks by their structural token sequence (bracket-enclosed tokens and
EVENT_* tokens from the full encoder+decoder walk), then within each group
removes walks whose content tokens (everything else) differ by at most
``max_token_diff`` tokens from an already-kept walk.

A "shared ≥ 1" guard prevents collapsing walks whose content tokens are
completely disjoint (e.g. different single-process walks like ``sed`` vs ``sh``).
"""

import os
from collections import defaultdict
from itertools import combinations

from pidsmaker.utils.utils import log


# ── Internal helpers ─────────────────────────────────────────────────────────

def _greedy_diverse(members, full_walks, contents, max_diff):
    """Hash-based greedy diversity selection within a structural group.

    For each kept content set K we store lookup sets so that redundancy
    checks are O(|content|²) per walk instead of O(n_kept):

      exact_kept:  K                          → catches diff=0
      leave1_kept: K − {tok}                  → catches diff=1 (K has extra)
      leave2_kept: K − {t1, t2}               → catches diff=2 (K has 2 extra)

    A new walk S is redundant iff |S △ K| ≤ max_diff for some kept K:
      d=0:  S ∈ exact
      d=1:  S ∈ leave1  OR  S−{t} ∈ exact           (requires |S| ≥ 2)
      d=2:  S ∈ leave2  OR  S−{t} ∈ leave1 (swap!)   (requires |S| ≥ 2)
            OR  S−{t1,t2} ∈ exact                     (requires |S| ≥ 3)
    """
    if len(members) == 1:
        return [members[0]]

    # Longest decoder first → keep more informative walks
    order = sorted(members, key=lambda i: -len(full_walks[i]))

    kept = []
    exact_kept  = set()
    leave1_kept = set()
    leave2_kept = set() if max_diff >= 2 else None

    for i in order:
        ct = contents[i]

        # diff == 0: exact content match
        if ct in exact_kept:
            continue

        n_ct = len(ct)

        if max_diff >= 1 and n_ct >= 2:
            # diff == 1, K has 1 extra: S = K−{tok} → S ∈ leave1
            if ct in leave1_kept:
                continue
            # diff == 1, S has 1 extra: S−{tok} = K → S−{tok} ∈ exact
            if any((ct - {t}) in exact_kept for t in ct):
                continue

        if max_diff >= 2 and n_ct >= 2:
            # diff == 2, K has 2 extra: S = K−{t1,t2} → S ∈ leave2
            if ct in leave2_kept:
                continue
            # diff == 2, swap: S−{t} = K−{t'} → S−{t} ∈ leave1
            if any((ct - {t}) in leave1_kept for t in ct):
                continue
            # diff == 2, S has 2 extra: S−{t1,t2} = K → need |S| ≥ 3
            if n_ct >= 3:
                ct_list = list(ct)
                if any(
                    (ct - {ct_list[a], ct_list[b]}) in exact_kept
                    for a in range(len(ct_list))
                    for b in range(a + 1, len(ct_list))
                ):
                    continue

        # Keep — register signatures
        kept.append(i)
        exact_kept.add(ct)
        if max_diff >= 1:
            for t in ct:
                leave1_kept.add(ct - {t})
        if max_diff >= 2:
            for pair in combinations(ct, 2):
                leave2_kept.add(ct - set(pair))

    return kept


# ── Public API ───────────────────────────────────────────────────────────────

def walk_dedup(pretokenized_t5, t5_dump_data, out_dir,
               structural_token_ids, max_token_diff=2, id2token=None):
    """Near-duplicate removal using structural grouping + content-token diff.

    Parameters
    ----------
    pretokenized_t5 : list of (entity_ids, decoder_ids)
        Token-ID sequences for each walk.
    t5_dump_data : list of tuples
        Metadata for each walk (used only for report generation and filtering).
    out_dir : str
        Directory to write the dedup report.
    structural_token_ids : frozenset[int]
        Token IDs considered structural (bracket-enclosed [X] and EVENT_* tokens).
        Content tokens are everything else.
    max_token_diff : int
        Maximum symmetric difference in content tokens for two walks to be
        considered near-duplicates (default 2).

    Returns
    -------
    (filtered_pretokenized_t5, filtered_t5_dump_data)
    """
    n = len(pretokenized_t5)
    log(f"Walk dedup: processing {n:,} walks (max_token_diff={max_token_diff})...")

    is_structural = structural_token_ids.__contains__

    # ── Build full walks, structural keys, and content fingerprints ────────
    full_walks  = []  # full token-ID sequence (entity + decoder)
    struct_keys = []  # tuple of structural token IDs in order
    contents    = []  # frozenset of content token IDs

    for entry in pretokenized_t5:
        entity_ids, decoder_ids = entry[0], entry[1]
        full = list(entity_ids) + list(decoder_ids)
        full_walks.append(full)
        struct_keys.append(tuple(t for t in full if is_structural(t)))
        contents.append(frozenset(t for t in full if not is_structural(t)))

    # ── Group by structural key ───────────────────────────────────────────
    struct_groups = defaultdict(list)
    for i, sk in enumerate(struct_keys):
        struct_groups[sk].append(i)

    log(f"Walk dedup: {len(struct_groups):,} structural groups")

    # ── Greedy diversity within each group ────────────────────────────────
    to_keep = set()
    for members in struct_groups.values():
        kept = _greedy_diverse(members, full_walks, contents, max_token_diff)
        to_keep.update(kept)

    n_keep   = len(to_keep)
    n_remove = n - n_keep
    log(f"Walk dedup: {n:,} → {n_keep:,} walks "
        f"({n_remove:,} removed, {n_remove / n * 100:.1f}%)")

    # ── Write report ──────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "walk_dedup_report.txt")
    with open(report_path, "w", buffering=1 << 20) as f:
        f.write("# Walk Dedup Report (structural-group + content-diff)\n")
        f.write(f"# max_token_diff={max_token_diff}\n")
        f.write(f"# structural_token_ids: {len(structural_token_ids)} tokens\n")
        f.write(f"# Walks before: {n:,}  kept: {n_keep:,}  "
                f"removed: {n_remove:,} ({n_remove / n * 100:.1f}%)\n\n")

        # Group walks by entity for readable output
        entity_order = []
        entity_indices = defaultdict(list)
        for i, entry in enumerate(pretokenized_t5):
            key = tuple(entry[0])
            if not entity_indices[key]:
                entity_order.append(key)
            entity_indices[key].append(i)

        _tok = (lambda tid: id2token.get(tid, f"<{tid}>")) if id2token else str

        for ent_key in entity_order:
            indices = entity_indices[ent_key]
            n_k = sum(1 for i in indices if i in to_keep)
            n_r = len(indices) - n_k
            ent_str = " ".join(_tok(t) for t in ent_key)
            f.write(f"=== entity[{ent_str}] ===  "
                    f"{len(indices)} walks → keep {n_k}, remove {n_r}\n")
            for i in indices:
                tag = "KEEP  " if i in to_keep else "REMOVE"
                walk_str = " ".join(_tok(t) for t in full_walks[i])
                f.write(f"  {tag}  {walk_str}\n")
            f.write("\n")

    log(f"Walk dedup report written to {report_path}")

    # ── Filter ────────────────────────────────────────────────────────────
    to_remove = set(range(n)) - to_keep
    filtered_pretokenized = [x for i, x in enumerate(pretokenized_t5) if i not in to_remove]
    filtered_dump = [x for i, x in enumerate(t5_dump_data) if i not in to_remove]

    return filtered_pretokenized, filtered_dump
