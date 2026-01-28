import csv
import os

from pidsmaker.config import (
    get_runtime_required_args,
    get_yml_cfg,
)
from pidsmaker.utils.utils import (
    init_database_connection,
)


def get_uuid2msg(cfg):
    cur, connnect = init_database_connection(cfg)
    queries = {
        "file": "SELECT node_uuid, index_id, path FROM file_node_table;",
        "netflow": "SELECT node_uuid, index_id, src_addr, dst_addr, src_port, dst_port FROM netflow_node_table;",
        "subject": "SELECT node_uuid, index_id, path, cmd FROM subject_node_table;",
    }

    uuid2msg = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            if node_type == "file":
                node_uuid, index_id, path = row
                uuid2msg[node_uuid] = {
                    "node_type": node_type,
                    "index_id": index_id,
                    "path": path,
                }
            elif node_type == "netflow":
                node_uuid, index_id, src_addr, dst_addr, src_port, dst_port = row
                uuid2msg[node_uuid] = {
                    "node_type": node_type,
                    "index_id": index_id,
                    "src_addr": src_addr,
                    "dst_addr": dst_addr,
                    "src_port": src_port,
                    "dst_port": dst_port,
                }
            elif node_type == "subject":
                node_uuid, index_id, path, cmd = row
                uuid2msg[node_uuid] = {
                    "node_type": node_type,
                    "index_id": index_id,
                    "path": path,
                    "cmd": cmd,
                }
    return uuid2msg


def main(cfg):
    uuid2msg = get_uuid2msg(cfg)

    for file in cfg.dataset.ground_truth_relative_path:
        print(f"Processing file {os.path.join(cfg._ground_truth_dir, file)}")
        attack_uuids = set()
        with open(os.path.join(cfg._ground_truth_dir, file), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # _, node_uuid, _, label = row[0], row[1].strip(), row[2], row[3].strip()
                _, node_uuid, _, _, label = row[0], row[1].strip(), row[2], row[3], row[4].strip()

                print(f"Processing node {node_uuid} with label {label}")
                if label == "attack":
                    attack_uuids.add(node_uuid)

        print(f"Found {len(attack_uuids)} attack nodes in {file}")

        with open(os.path.join(cfg._ground_truth_dir, file), "w") as f:
            writer = csv.writer(f)
            for node_uuid in attack_uuids:
                print(node_uuid)
                if uuid2msg[node_uuid]["node_type"] == "file":
                    path = uuid2msg[node_uuid]["path"]
                    msg = {"file": path}
                elif uuid2msg[node_uuid]["node_type"] == "netflow":
                    src_addr = uuid2msg[node_uuid]["src_addr"]
                    dst_addr = uuid2msg[node_uuid]["dst_addr"]
                    src_port = uuid2msg[node_uuid]["src_port"]
                    dst_port = uuid2msg[node_uuid]["dst_port"]
                    msg = {"netflow": f"{src_addr}:{src_port} -> {dst_addr}:{dst_port}"}
                elif uuid2msg[node_uuid]["node_type"] == "subject":
                    path = uuid2msg[node_uuid]["path"]
                    cmd = uuid2msg[node_uuid]["cmd"]
                    msg = {"subject": f"{path} {cmd}"}

                index_id = uuid2msg[node_uuid]["index_id"]
                print([node_uuid, str(msg), index_id])
                writer.writerow([node_uuid, str(msg), index_id])

    print(f"Finish!")


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    cfg = get_yml_cfg(args)

    main(cfg)
