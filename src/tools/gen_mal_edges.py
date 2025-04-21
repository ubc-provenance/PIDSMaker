import sys
import os 
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import get_runtime_required_args, get_yml_cfg
from provnet_utils import datetime_to_ns_time_US, init_database_connection
import labelling

def gen_mal_edges(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = labelling.get_uuid2nids(cur)

    for attack_tuple in cfg.dataset.attack_to_time_window:

        mal_node_file = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        mal_edge_file = mal_node_file.split("/")[0] + "/edge_" + mal_node_file.split("/")[1][5:]
        result_edges = []

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, mal_node_file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))

        rows = labelling.get_events(cur, start_time, end_time)
        for row in rows:
            src_idx_id = row[1]
            ope = row[2]
            dst_idx_id = row[4]
            event_uuid = row[5]
            timestamp_rec = row[6]

            if src_idx_id in ground_truth_nids and dst_idx_id in ground_truth_nids:
                mal_edge = [
                    nid2uuid[int(src_idx_id)],
                    ope,
                    nid2uuid[int(dst_idx_id)],
                    timestamp_rec,
                    event_uuid
                ]
                result_edges.append(mal_edge)
        
        with open(os.path.join(cfg._ground_truth_dir, mal_edge_file), "w") as f:
            writer = csv.writer(f)
            for row in result_edges:
                writer.writerow(row)

        print(f"{len(result_edges)} malicious edges written into {mal_edge_file}")

def main(cfg):
    gen_mal_edges(cfg)

if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    cfg = get_yml_cfg(args)
    main(cfg)


