"""Compute mean entity label length (in tokens) and total token count
across all entities in the SPIDER pretrain corpus.

Definition: per entity, tokens = tokenize_node(node_type, label).
The full encoder input would be [CLS] + tokens + [SEP] (= len + 2),
both numbers are reported.
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_ROOT  = "/home/artifacts/preprocessing"
TOKENIZER_PATH  = "/home/pids/weights/foundation-small/tokenizer.pt"

OS_OF = {
    "PROVENANCE_BENIGN": "Linux",
    "TRACE_E3":          "Linux",
    "TRACE_E5":          "Linux",
    "CADETS_E5":         "FreeBSD",
    "CLEARSCOPE_E3":     "Android",
    "optc_h201":         "Windows",
}


def find_build_dir(ds_name):
    base = os.path.join(ARTIFACTS_ROOT, ds_name, "build_graphs")
    cands = []
    for h in os.listdir(base):
        path = os.path.join(base, h)
        done = os.path.join(path, "done.txt")
        idx  = os.path.join(path, "indexid2msg", "indexid2msg.pkl")
        if os.path.exists(done) and os.path.exists(idx):
            cands.append((os.path.getmtime(done), path))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


# ── Load tokenizer ─────────────────────────────────────────────────────────
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE
# Build minimal cfg shell needed by tokenizer (only get_rel2id is consulted).
# Use the spider cfg so edge tokens line up with what the live model uses.
import argparse
from pidsmaker.config.pipeline import (get_default_cfg, get_yml_file,
    merge_cfg_and_check_syntax, overwrite_cfg_with_args, set_shortcut_variables, set_immutable_cfg)

_args = argparse.Namespace(model="spider", dataset="CADETS_E5",
    artifact_dir_in_container="", test_mode=False, wandb=False, cpu=True,
    tuning_mode="none", experiment="none", tuning_file_path="", tuned=False,
    exp="", tags="", sweep_id="", force_restart="", restart_from_scratch=False,
    save_graph_preprocessing=False, from_weights=False)
_cfg = get_default_cfg(_args)
merge_cfg_and_check_syntax(_cfg, get_yml_file("spider"))
overwrite_cfg_with_args(_cfg, _args)
set_shortcut_variables(_cfg)
set_immutable_cfg(_cfg)

tok = ProvenanceTokenizerBPE(_cfg)
tok.load(TOKENIZER_PATH)
print(f"  vocab size: {len(tok.token2id):,}")
print(f"  mode: {tok.mode}")
print(f"  max_seq_len: {tok.max_seq_len}")

max_seq_len = getattr(tok, 'max_seq_len', 60)

# ── Per-dataset entity tokenization ────────────────────────────────────────
overall_count = 0
overall_token_sum = 0
overall_truncated_sum = 0
overall_per_type_lens = defaultdict(list)

per_os = defaultdict(lambda: {"n": 0, "tokens": 0, "trunc_tokens": 0,
                              "lens": [], "type_lens": defaultdict(list)})
per_ds = {}

for ds_name, os_name in OS_OF.items():
    print(f"\n[{ds_name} / {os_name}]")
    bd = find_build_dir(ds_name)
    idx_path = os.path.join(bd, "indexid2msg", "indexid2msg.pkl")
    indexid2msg = torch.load(idx_path, map_location="cpu", weights_only=False)
    print(f"  {len(indexid2msg):,} entities — tokenizing...")

    n = 0
    token_sum = 0           # raw label-token count (no CLS/SEP)
    trunc_sum = 0           # capped at max_seq_len-2 (room for CLS, SEP)
    lens = []
    type_lens = defaultdict(list)

    for _nid, val in indexid2msg.items():
        ntype, label = val[0], val[1]
        ids = tok.tokenize_node(ntype, label)
        ln  = len(ids)
        # encoder cap: [CLS] + content + [SEP] ≤ max_seq_len
        capped = min(ln, max_seq_len - 2)
        n += 1
        token_sum += ln
        trunc_sum += capped
        lens.append(ln)
        type_lens[ntype].append(ln)

    mean_len = np.mean(lens)
    p50 = np.median(lens)
    p95 = np.percentile(lens, 95)
    p99 = np.percentile(lens, 99)
    max_l = max(lens)

    print(f"  mean_tokens/entity = {mean_len:6.2f}   "
          f"median={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={max_l}")
    for ntype in sorted(type_lens):
        ml = np.mean(type_lens[ntype])
        print(f"    {ntype:<10} (n={len(type_lens[ntype]):>10,}): mean={ml:6.2f}  "
              f"max={max(type_lens[ntype])}")
    print(f"  total label tokens     : {token_sum:>14,}")
    print(f"  total tokens (CLS+SEP) : {token_sum + 2*n:>14,}  "
          f"(+{2*n:,} for [CLS]/[SEP])")
    print(f"  total tokens (truncated@{max_seq_len-2}): {trunc_sum:>14,}")

    per_ds[ds_name] = {"n": n, "tokens": token_sum, "trunc": trunc_sum,
                       "mean_len": mean_len, "max_len": max_l}
    p = per_os[os_name]
    p["n"] += n
    p["tokens"] += token_sum
    p["trunc_tokens"] += trunc_sum
    p["lens"].extend(lens)
    for k, v in type_lens.items():
        p["type_lens"][k].extend(v)
        overall_per_type_lens[k].extend(v)

    overall_count += n
    overall_token_sum += token_sum
    overall_truncated_sum += trunc_sum

    del indexid2msg, lens, type_lens

# ── Per-OS rollup ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PER-OS ROLLUP")
print(f"{'='*70}")
print(f"{'OS':<10} {'Entities':>12} {'MeanTok':>8} {'MedTok':>7} {'P95':>5} "
      f"{'TotalTokens':>16} {'TotTrunc':>14}")
for os_name in ["Linux", "FreeBSD", "Windows", "Android"]:
    d = per_os[os_name]
    if d["n"] == 0: continue
    arr = d["lens"]
    print(f"{os_name:<10} {d['n']:>12,} {np.mean(arr):>7.2f}  "
          f"{int(np.median(arr)):>6}  {int(np.percentile(arr,95)):>4}  "
          f"{d['tokens']:>16,} {d['trunc_tokens']:>14,}")

# ── Global totals ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  GLOBAL TOTALS")
print(f"{'='*70}")
print(f"  Total entities                : {overall_count:>14,}")
mean_all = overall_token_sum / overall_count
print(f"  Mean tokens / entity (label)  : {mean_all:>14.3f}")
print(f"  Median tokens / entity        : "
      f"{int(np.median([np.median(per_os[o]['lens']) for o in per_os if per_os[o]['n']>0])):>14}")
print(f"  Total label tokens            : {overall_token_sum:>14,}")
print(f"  Total tokens incl. [CLS][SEP] : {overall_token_sum + 2*overall_count:>14,}")
print(f"  Total tokens, truncated@{max_seq_len-2}    : {overall_truncated_sum:>14,}")

print(f"\n  Per-node-type means (across all datasets):")
for ntype in sorted(overall_per_type_lens):
    arr = overall_per_type_lens[ntype]
    print(f"    {ntype:<10} (n={len(arr):>12,}): "
          f"mean={np.mean(arr):6.2f}  median={int(np.median(arr))}  "
          f"p95={int(np.percentile(arr,95))}  p99={int(np.percentile(arr,99))}  "
          f"max={max(arr)}")
