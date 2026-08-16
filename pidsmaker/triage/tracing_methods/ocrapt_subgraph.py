"""OCR-APT anomalous-subgraph triage (arXiv:2510.15188, Sec 5.2, Algorithm 1)."""

import math
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from pidsmaker.utils.labelling import get_ground_truth
from pidsmaker.utils.utils import get_node_to_path_and_type, get_split_to_files, listdir_sorted, log

# Seed-eligible node types
SHORT_TYPES = {"process", "flow", "net", "netflowobject", "file", "module",
               "netflow", "subject"}
SEVERITY_ORDER = ["Negligible", "Minor", "Moderate", "Significant", "Critical"]


def _classify(score):
    if score <= 1:
        return "Negligible"
    if score <= 10:
        return "Minor"
    if score <= 100:
        return "Moderate"
    if score <= 1000:
        return "Significant"
    return "Critical"


def _node_type_order(nt):
    low = str(nt).lower()
    if low in ("flow", "net", "netflowobject", "netflow"):
        return 1
    if low.split("_")[-1] == "process" or low == "subject":
        return 2
    if low.split("_")[0] == "file":
        return 3
    return 4


def _load_graphs(cfg):
    base_dir = cfg.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)
    graphs = {}
    for split, paths in split_to_files.items():
        graphs[split] = [torch.load(p, weights_only=False) for p in paths]
        log(f"  [{split}] loaded {len(graphs[split])} graph files")
    return graphs


def _best_epoch(cfg):
    in_dir = cfg.evaluation._precision_recall_dir
    stats_files = [f for f in os.listdir(in_dir) if f.startswith("stats_") and f.endswith(".pth")]
    if not stats_files:
        raise FileNotFoundError(f"No stats_*.pth found under {in_dir}. Run evaluation first.")
    best_mcc, best_epoch = float("-inf"), None
    for f in stats_files:
        model_epoch_dir = f[len("stats_"):-len(".pth")]
        stats = torch.load(os.path.join(in_dir, f), weights_only=False)
        mcc = stats.get("mcc", float("-inf"))
        if isinstance(mcc, float) and math.isnan(mcc):
            mcc = float("-inf")  # NaN mcc still counts as a candidate, just the worst one
        if best_epoch is None or mcc > best_mcc:
            best_mcc, best_epoch = mcc, model_epoch_dir
    return best_epoch


def _train_reference_range_by_type(cfg, best_epoch, node_to_type):
    train_dir = os.path.join(cfg.training._edge_losses_dir, "train", best_epoch)
    if not os.path.isdir(train_dir):
        return {}
    dfs = [pd.read_csv(os.path.join(train_dir, f)) for f in listdir_sorted(train_dir)
           if f.endswith(".csv")]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    by_type = defaultdict(list)
    for nid, loss in zip(df["node"], df["loss"]):
        t = node_to_type.get(int(nid))
        if t is not None:
            by_type[t].append(loss)
    return {t: (min(v), max(v)) for t, v in by_type.items()}


def main(cfg):
    o = cfg.triage.ocrapt
    max_edges = o.max_edges

    best_epoch = _best_epoch(cfg)
    scores_file = os.path.join(cfg.evaluation._precision_recall_dir, f"scores_{best_epoch}.pkl")
    d = torch.load(scores_file, weights_only=False)
    nodes = [int(n) for n in d["nodes"]]
    pred_scores = np.asarray(d["pred_scores"], dtype=np.float64)
    y_preds = np.asarray(d["y_preds"])

    testing = set(nodes)
    anomalies = {n for n, y in zip(nodes, y_preds) if y}
    log(f"  anomalies (node_evaluation, threshold_method=ocrapt): {len(anomalies)} / testing {len(testing)}")

    pids2type = {nid: info["type"] for nid, info in get_node_to_path_and_type(cfg).items()}
    range_by_type = _train_reference_range_by_type(cfg, best_epoch, pids2type)

    node_types = np.array([pids2type.get(n) for n in nodes])
    prob = np.zeros(len(nodes), dtype=np.float64)
    for t, (lo, hi) in range_by_type.items():
        mask = node_types == t
        if not mask.any() or hi <= lo:
            continue
        prob[mask] = np.clip((pred_scores[mask] - lo) / (hi - lo), 0.0, 1.0)
    prob_by_pid = dict(zip(nodes, prob.tolist()))
    score_by_pid = dict(zip(nodes, pred_scores.tolist()))

    gt_nids, _, _ = get_ground_truth(cfg)
    malicious = set(int(n) for n in gt_nids) & testing
    log(f"  malicious (GT in test): {len(malicious)}")

    # load graphs
    graphs_by_split = _load_graphs(cfg)
    subgraph_obj = nx.MultiDiGraph()
    rows, cols, all_nodes = [], [], set()
    for split, graphs in graphs_by_split.items():
        is_test = split == "test"
        for g in graphs:
            for u, v, attr in g.edges(data=True):
                u, v = int(u), int(v)
                all_nodes.add(u); all_nodes.add(v)
                rows.append(u); cols.append(v)
                if is_test and u in testing and v in testing and (u in anomalies or v in anomalies):
                    subgraph_obj.add_edge(u, v, action=attr.get("label", "NA"))
    subgraph_obj.add_nodes_from(anomalies)
    del graphs_by_split
    log(f"  subgraph_obj: {subgraph_obj.number_of_nodes()} nodes, "
        f"{subgraph_obj.number_of_edges()} edges")

    # full undirected adjacency for 2-hop
    node_list = sorted(all_nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    r = np.fromiter((idx[x] for x in rows), dtype=np.int64, count=len(rows))
    c = np.fromiter((idx[x] for x in cols), dtype=np.int64, count=len(cols))
    A = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, N))
    A = (A + A.T).tocsr()
    del rows, cols, r, c

    # subgraph construction
    correlated = set()

    def filt(neighbours, visited):
        s = neighbours & anomalies
        if o.correlate_anomalous_once:
            s = s - correlated
        return s - visited

    def sample_by_edge_type(sg):
        df = nx.to_pandas_edgelist(sg, edge_key="ekey")
        actions = df["action"].unique().tolist()
        w = {}
        for et in actions:
            n_et = len(df[df["action"] == et])
            w[et] = int(n_et / len(df) * max_edges) if n_et > int(max_edges / len(actions)) else int(n_et)
        sample = pd.concat([df.loc[df["action"] == et].sample(w[et], random_state=0) for et in actions])
        edges = [tuple(e) for e in sample[["source", "target", "ekey"]].values]
        sampled = sg.edge_subgraph(edges).copy()
        out = [sampled.subgraph(n).copy() for n in nx.weakly_connected_components(sampled) if len(n) > 1]
        sg.remove_nodes_from(set(sampled.nodes()))
        out.extend([sg.subgraph(n).copy() for n in nx.weakly_connected_components(sg)
                    if len(n) > 1 and 0 < sg.subgraph(n).number_of_edges() <= max_edges])
        return out

    def partition(big):
        comms = nx.community.louvain_communities(big, resolution=1, seed=0)
        tmp = [big.subgraph(n).copy() for n in comms if len(n) >= o.min_nodes]
        tmp = [s for s in tmp if s.number_of_edges() > 2]
        if not tmp:
            tmp = [big]
        res = []
        for s in tmp:
            res.extend(sample_by_edge_type(s) if s.number_of_edges() > max_edges else [s])
        return res

    def expand(seed):
        conn = {n for n, _ in subgraph_obj.in_edges(seed) if n != seed}
        conn.update({n for _, n in subgraph_obj.out_edges(seed) if n != seed})
        if not conn:
            yield None; return
        nodes_ = {seed}
        nodes_ |= filt(conn, {seed})
        visited = {seed} | nodes_
        if o.num_hops >= 1:
            for cn in sorted(conn):
                nb = {n for n, _ in subgraph_obj.in_edges(cn)}
                a1 = filt(nb, visited)
                nb = {n for _, n in subgraph_obj.out_edges(cn)}
                a1 |= filt(nb, visited)
                if a1:
                    nodes_ |= a1; nodes_.add(cn); visited |= nodes_
        if len(nodes_) >= o.min_nodes:
            sg = subgraph_obj.subgraph(nodes_).copy()
            lst = partition(sg) if (sg.number_of_edges() > max_edges and max_edges != 0) else [sg]
            for s in lst:
                correlated.update(s.nodes())
                yield s
        else:
            yield None

    def remove_identical(lst):
        rm = set()
        for i1, s1 in enumerate(lst):
            for i2, s2 in enumerate(lst):
                if i1 != i2 and i2 not in rm and s1.number_of_nodes() == s2.number_of_nodes() \
                        and s1.number_of_edges() == s2.number_of_edges():
                    if nx.to_pandas_edgelist(s1, edge_key="ekey").equals(
                            nx.to_pandas_edgelist(s2, edge_key="ekey")):
                        rm.add(i1); break
        return [s for i, s in enumerate(lst) if i not in rm]

    # top-K per type by score, then sort by percentile severity
    init_disc = {next(iter(comp)) for comp in nx.weakly_connected_components(subgraph_obj) if len(comp) == 1}
    init_corr = [a for a in anomalies if a not in init_disc]
    by_type = defaultdict(list)
    for a in init_corr:
        by_type[pids2type.get(a, "?")].append(a)
    seeds = []
    for t, members in by_type.items():
        if str(t).split("_")[-1].lower() not in SHORT_TYPES:
            continue
        members.sort(key=lambda a: score_by_pid.get(a, 0.0), reverse=True)
        seeds.extend(members[: o.top_k])
    seeds.sort(key=lambda a: (-prob_by_pid.get(a, 0.0), _node_type_order(pids2type.get(a, "?"))))
    n_mal_seed = sum(1 for s in seeds if s in malicious)
    log(f"  seeds: {len(seeds)} ({n_mal_seed} malicious)")

    subgraphs = []
    unvisited = set(init_corr)
    for s in seeds:
        subgraphs.extend([sg for sg in expand(s) if sg is not None])
        unvisited -= correlated
        if not unvisited:
            break
    if o.remove_duplicated_subgraph:
        subgraphs = remove_identical(subgraphs)
    log(f"  constructed {len(subgraphs)} subgraphs, correlated {len(correlated & anomalies)} anomalies")

    # severity filter
    keep_levels = set(SEVERITY_ORDER[SEVERITY_ORDER.index(o.abnormality_level):])
    def sev(sg):
        return float(sum(prob_by_pid.get(n, 0.0) for n in (set(sg.nodes()) & anomalies)))
    kept = [sg for sg in subgraphs if _classify(sev(sg)) in keep_levels]
    mp = set()
    for sg in kept:
        mp |= set(sg.nodes())
    mp &= testing
    log(f"  kept {len(kept)}/{len(subgraphs)} severe subgraphs, predicted-malicious={len(mp)}")

    # 2-hop-relaxed node metric
    from pidsmaker.detection.evaluation_methods.evaluation_utils import two_hop_relaxed_metrics
    GP = np.zeros(N, dtype=bool); GP[[idx[n] for n in malicious if n in idx]] = True
    MP = np.zeros(N, dtype=bool); MP[[idx[n] for n in mp if n in idx]] = True
    stats = two_hop_relaxed_metrics(MP, GP, A)
    stats["subgraphs_kept"] = len(kept)
    return stats
