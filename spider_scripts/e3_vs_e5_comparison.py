"""Compare E3 vs E5 dataset variants for each DARPA TC team.

For each pair (e.g. CADETS_E3 vs CADETS_E5), reports:
  - Subject (process) label overlap     — proxy for "same applications"
  - File path overlap & path-prefix overlap — proxy for "same host / same OS install"
  - Netflow IP overlap                  — proxy for "same network endpoints"
  - Top-N most frequent process names side-by-side

Reuses the build_dir discovery logic from pretrain_corpus_stats.py.
"""

import os
from collections import Counter, defaultdict

import torch

ARTIFACTS_ROOT = "/home/artifacts/preprocessing"
PAIRS = [
    ("CADETS_E3",     "CADETS_E5"),
    ("CLEARSCOPE_E3", "CLEARSCOPE_E5"),
    ("THEIA_E3",      "THEIA_E5"),
    ("TRACE_E3",      "TRACE_E5"),
]


def find_build_dir(ds_name):
    base = os.path.join(ARTIFACTS_ROOT, ds_name, "build_graphs")
    if not os.path.isdir(base):
        return None
    cands = []
    for h in os.listdir(base):
        path = os.path.join(base, h)
        done = os.path.join(path, "done.txt")
        idx  = os.path.join(path, "indexid2msg", "indexid2msg.pkl")
        if os.path.exists(done) and os.path.exists(idx):
            cands.append((os.path.getmtime(done), path))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def load_indexid2msg(ds_name):
    bd = find_build_dir(ds_name)
    if bd is None:
        return None
    path = os.path.join(bd, "indexid2msg", "indexid2msg.pkl")
    return torch.load(path, map_location="cpu", weights_only=False)


def split_by_type(indexid2msg):
    """Return dict[node_type] -> Counter(label -> count)."""
    out = defaultdict(Counter)
    for _nid, val in indexid2msg.items():
        ntype, label = val[0], val[1]
        out[ntype][label] += 1
    return out


def path_prefix(label, depth=2):
    """Return the first `depth` path components of a unix path label."""
    if not label.startswith("/"):
        return None
    parts = label.strip("/").split("/")
    return "/" + "/".join(parts[:depth])


def netflow_ip(label):
    """Heuristic: keep only the IP part of a netflow label.
    Labels look like 'IP:PORT' or 'IP→IP:PORT'. Strip ports and arrows."""
    s = label.replace("→", " ").replace("->", " ")
    parts = s.split()
    ips = []
    for p in parts:
        if ":" in p:
            p = p.split(":")[0]
        if p and any(c.isdigit() for c in p):
            ips.append(p)
    return ips


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def compare_pair(ds_a, ds_b):
    print(f"\n{'='*78}")
    print(f"  {ds_a}  vs  {ds_b}")
    print(f"{'='*78}")

    print(f"  Loading {ds_a}...", end="", flush=True)
    a = load_indexid2msg(ds_a)
    print(f" {len(a):,} nodes" if a else " MISSING")
    print(f"  Loading {ds_b}...", end="", flush=True)
    b = load_indexid2msg(ds_b)
    print(f" {len(b):,} nodes" if b else " MISSING")
    if a is None or b is None:
        return

    by_a = split_by_type(a)
    by_b = split_by_type(b)

    # ── 1. Subject (application) overlap ────────────────────────────────────
    sa = set(by_a["subject"].keys())
    sb = set(by_b["subject"].keys())
    inter = sa & sb
    print(f"\n  [1] APPLICATIONS (subject labels)")
    print(f"      |E3| = {len(sa):>8,}   |E5| = {len(sb):>8,}   shared = {len(inter):>6,}")
    print(f"      Jaccard         = {jaccard(sa, sb):.4f}")
    print(f"      coverage(E3∩/|E3|) = {len(inter)/max(1,len(sa)):.4f}   "
          f"(E3∩/|E5|) = {len(inter)/max(1,len(sb)):.4f}")

    # Top-20 processes side by side
    top_a = by_a["subject"].most_common(15)
    top_b = by_b["subject"].most_common(15)
    print(f"\n      Top-15 processes (by node count):")
    print(f"      {'rk':>3}  {ds_a:<35} {ds_b:<35}")
    for i in range(15):
        la, ca = top_a[i] if i < len(top_a) else ("", 0)
        lb, cb = top_b[i] if i < len(top_b) else ("", 0)
        in_other_a = "✓" if la in sb else " "
        in_other_b = "✓" if lb in sa else " "
        print(f"      {i+1:>3}  {la[:28]:<28} {ca:>6,} {in_other_a}  "
              f"{lb[:28]:<28} {cb:>6,} {in_other_b}")
    print(f"      (✓ = label also exists in the other engagement)")

    # ── 2. File path overlap ────────────────────────────────────────────────
    fa = set(by_a["file"].keys())
    fb = set(by_b["file"].keys())
    finter = fa & fb
    print(f"\n  [2] FILE PATHS")
    print(f"      |E3| = {len(fa):>8,}   |E5| = {len(fb):>8,}   shared = {len(finter):>6,}")
    print(f"      Jaccard         = {jaccard(fa, fb):.4f}")
    print(f"      coverage(E3∩/|E3|) = {len(finter)/max(1,len(fa)):.4f}   "
          f"(E3∩/|E5|) = {len(finter)/max(1,len(fb)):.4f}")

    # Path-prefix overlap (first 2 components — proxy for OS install layout)
    pref_a = Counter()
    pref_b = Counter()
    for lbl in fa:
        p = path_prefix(lbl, depth=2)
        if p:
            pref_a[p] += 1
    for lbl in fb:
        p = path_prefix(lbl, depth=2)
        if p:
            pref_b[p] += 1
    pref_inter = set(pref_a) & set(pref_b)
    pref_union = set(pref_a) | set(pref_b)
    print(f"\n      Top-2-level path-prefix vocabulary:")
    print(f"        |E3-prefixes|={len(pref_a):>5}  |E5-prefixes|={len(pref_b):>5}  "
          f"shared={len(pref_inter):>4}  Jaccard={len(pref_inter)/max(1,len(pref_union)):.4f}")
    print(f"      Top-15 path prefixes (E3 / E5):")
    top_pa = pref_a.most_common(15)
    top_pb = pref_b.most_common(15)
    print(f"        {'rk':>3}  {ds_a:<35} {ds_b:<35}")
    for i in range(15):
        la, ca = top_pa[i] if i < len(top_pa) else ("", 0)
        lb, cb = top_pb[i] if i < len(top_pb) else ("", 0)
        flag_a = "✓" if la in pref_b else " "
        flag_b = "✓" if lb in pref_a else " "
        print(f"        {i+1:>3}  {la[:28]:<28} {ca:>7,} {flag_a}  "
              f"{lb[:28]:<28} {cb:>7,} {flag_b}")

    # ── 3. Netflow IP overlap ───────────────────────────────────────────────
    ips_a = set()
    ips_b = set()
    for lbl in by_a["netflow"]:
        ips_a |= set(netflow_ip(lbl))
    for lbl in by_b["netflow"]:
        ips_b |= set(netflow_ip(lbl))
    n_inter = ips_a & ips_b
    print(f"\n  [3] NETFLOW IP endpoints")
    print(f"      |E3-IPs| = {len(ips_a):>8,}   |E5-IPs| = {len(ips_b):>8,}   "
          f"shared = {len(n_inter):>6,}   Jaccard={jaccard(ips_a, ips_b):.4f}")
    if n_inter:
        sample = sorted(n_inter)[:10]
        print(f"      Sample shared IPs: {sample}")

    # ── 4. Verdict heuristic ────────────────────────────────────────────────
    print(f"\n  [4] VERDICT (heuristic)")
    j_app  = jaccard(sa, sb)
    j_pref = len(pref_inter)/max(1,len(pref_union))
    if j_app > 0.30 and j_pref > 0.50:
        verdict = "LIKELY SAME HOST/SETUP — high process & path-prefix overlap"
    elif j_app > 0.10 or j_pref > 0.30:
        verdict = "PARTIAL OVERLAP — same OS family or shared baseline apps, but different deployments"
    else:
        verdict = "DIFFERENT HOST/SETUP — disjoint applications and OS layout"
    print(f"      Subject Jaccard = {j_app:.4f},  Path-prefix Jaccard = {j_pref:.4f}")
    print(f"      → {verdict}")


if __name__ == "__main__":
    print("E3 vs E5 dataset comparison — DARPA TC engagements")
    print("Comparing process applications, file-system layout, and network endpoints.\n")
    for a, b in PAIRS:
        compare_pair(a, b)
    print(f"\nDone.")
