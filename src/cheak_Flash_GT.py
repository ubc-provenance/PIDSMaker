from provnet_utils import *
from config import *

import numpy as np
from labelling import get_ground_truth
from detection.evaluation_methods.evaluation_utils import compute_tw_labels
import networkx as nx
import datetime
import json

def get_uuid2nids(cur):
    queries = {
        "file": "SELECT index_id, node_uuid FROM file_node_table;",
        "netflow": "SELECT index_id, node_uuid FROM netflow_node_table;",
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;"
    }
    uuid2nids = {}
    nid2uuid = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            uuid2nids[row[1]] = row[0]
            nid2uuid[row[0]] = row[1]

    return uuid2nids, nid2uuid

def main(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    nid2msg = get_indexid2msg(cur)

    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [int(nid) for nid in GP_nids]

    with open("./src/cadets.json", "r") as json_file:
        GT_mal = set(json.load(json_file))

    node_unseen = set()
    node_seen = set()

    for uuid in GT_mal:
        if uuid not in uuid2nids:
            node_unseen.add(uuid)
        else:
            node_seen.add(int(uuid2nids[uuid]))

    log(f"Number of unseen nodes: {len(node_unseen)}")
    log(f"Number of seen nodes: {len(node_seen)}")

    none_file_number = 0
    node_not_None = set()
    for nid in node_seen:
        if nid2msg[int(nid)] == ['file', 'None']:
            none_file_number += 1
        else:
            log(nid, " : ", nid2msg[int(nid)])
            node_not_None.add(nid)

    log(f"There are {none_file_number} None files")
    log("==" * 30)

    log(f"There are {len(set(GPs)-set(node_seen))} nodes in our ground truth but not in their file")
    for nid in set(GPs)-set(node_seen):
        log(f"{nid} : {nid2msg[nid]}")
    log("==" * 30)

    log(f"There are {len(set(node_not_None)-set(GPs))} non-None nodes in their ground truth but not in ours")
    for nid in set(node_not_None)-set(GPs):
        log(f"{nid} : {nid2msg[nid]}")
    log("==" * 30)

    log(f"There are {len(set(node_seen) & set(GPs))} nodes in both ground truth")
    for nid in set(node_seen) & set(GPs):
        log(f"{nid} : {nid2msg[nid]}")
    log("==" * 30)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)