"""OpTC dataset preprocessing - create the Parquet dataset from JSON logs.

Parses OpTC provenance JSON files and writes the nodes (subjects, files, netflows) and events
(edges) of a host for graph construction to a Parquet dataset (`<datasets_dir>/<database>`, see
`pidsmaker/utils/database.py`), with the tables of the Postgres databases of the OpTC datasets.
All the files of `--raw_dir` (and its subfolders) are read:

    python dataset_preprocessing/optc/create_database_optc.py orthrus DATASET --raw_dir /data/
"""

import argparse
import json
import os

from tqdm import tqdm

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.utils.database import parquet_dataset_dir
from pidsmaker.utils.dataset_utils import OPTC_hostname_map, get_rel2id, get_schema
from pidsmaker.utils.dataset_writer import INTEGER, DatasetBuilder, schema_tables
from pidsmaker.utils.utils import (
    OPTC_datetime_to_timestamp_US,
    get_all_filelist,
    log,
    stringtomd5,
)


def save_nodes(cfg, builder, dataset_dir):
    hostname = OPTC_hostname_map[cfg.dataset.name]
    all_paths = get_all_filelist(dataset_dir)

    rel2id = get_rel2id(cfg)

    subject_uuid2attr = {}
    netflow_uuid2attr = {}
    file_uuid2attr = {}

    for i, file in enumerate(all_paths):
        with open(file, "r") as f:
            for line in tqdm(f, desc=f"Extracting nodes from {i}-th/{len(all_paths)} file."):
                line = line.replace("\\\\", "/")
                temp_dic = json.loads(line.strip())

                if temp_dic["hostname"].split(".")[0] != hostname:
                    continue

                if temp_dic["action"] not in rel2id:
                    continue

                if temp_dic["object"] not in ["FILE", "FLOW", "PROCESS"]:
                    continue

                src_uuid = temp_dic["actorID"]
                dst_uuid = temp_dic["objectID"]

                # Process src_node
                if "image_path" in temp_dic["properties"]:
                    subject_uuid2attr[src_uuid] = (temp_dic["properties"]["image_path"], None)
                else:
                    subject_uuid2attr[src_uuid] = (None, None)

                # Process dst_node
                if temp_dic["object"] == "FILE":
                    if "file_path" in temp_dic["properties"]:
                        file_uuid2attr[dst_uuid] = temp_dic["properties"]["file_path"]
                    else:
                        file_uuid2attr[dst_uuid] = None
                elif temp_dic["object"] == "FLOW":
                    if "src_ip" in temp_dic["properties"]:
                        src_ip = temp_dic["properties"]["src_ip"]
                    else:
                        src_ip = None
                    if "dest_ip" in temp_dic["properties"]:
                        dst_ip = temp_dic["properties"]["dest_ip"]
                    else:
                        dst_ip = None
                    if "src_port" in temp_dic["properties"]:
                        src_port = temp_dic["properties"]["src_port"]
                    else:
                        src_port = None
                    if "dest_port" in temp_dic["properties"]:
                        dst_port = temp_dic["properties"]["dest_port"]
                    else:
                        dst_port = None

                    netflow_uuid2attr[dst_uuid] = (src_ip, src_port, dst_ip, dst_port)
                elif temp_dic["object"] == "PROCESS":
                    if "parent_image_path" in temp_dic["properties"]:
                        path = temp_dic["properties"]["parent_image_path"]
                    else:
                        path = None
                    if "command_line" in temp_dic["properties"]:
                        cmd = temp_dic["properties"]["command_line"]
                    else:
                        cmd = None
                    subject_uuid2attr[dst_uuid] = (path, cmd)

    index_id = 0
    uuid2index_id = {}

    # Save subject_nodes
    datalist = []
    for sub_uuid, sub_attr in tqdm(
        subject_uuid2attr.items(), desc=f"Processing datalist for subject nodes"
    ):
        temp_data = [sub_uuid, stringtomd5(str(sub_attr)), sub_attr[0], sub_attr[1], index_id]
        datalist.append(temp_data)
        uuid2index_id[sub_uuid] = index_id
        index_id += 1

    log(f"Start saving subject nodes.")
    builder.table("subject_node_table").add_rows(datalist)
    log(f"Finished saving subject nodes.")
    del subject_uuid2attr
    del datalist

    # Save file_nodes
    datalist = []
    for file_uuid, file_attr in tqdm(file_uuid2attr.items(), desc=f"Processing file nodes"):
        temp_data = [file_uuid, stringtomd5(str(file_attr)), file_attr, index_id]
        datalist.append(temp_data)
        uuid2index_id[file_uuid] = index_id
        index_id += 1

    log(f"Start saving file nodes.")
    builder.table("file_node_table").add_rows(datalist)
    log(f"Finished saving file nodes.")
    del file_uuid2attr
    del datalist

    # Save netflow_nodes
    datalist = []
    for net_uuid, net_attr in tqdm(netflow_uuid2attr.items(), desc=f"Processing net nodes"):
        temp_data = [
            net_uuid,
            stringtomd5(str(net_attr)),
            net_attr[0],
            net_attr[1],
            net_attr[2],
            net_attr[3],
            index_id,
        ]
        datalist.append(temp_data)
        uuid2index_id[net_uuid] = index_id
        index_id += 1

    log(f"Start saving net nodes.")
    builder.table("netflow_node_table").add_rows(datalist)
    log(f"Finished saving net nodes.")
    del netflow_uuid2attr
    del datalist

    return uuid2index_id


def save_events(cfg, builder, uuid2index_id, dataset_dir):
    hostname = OPTC_hostname_map[cfg.dataset.name]
    all_paths = get_all_filelist(dataset_dir)

    rel2id = get_rel2id(cfg)

    events = builder.table("event_table")

    for i, file in enumerate(all_paths):
        with open(file, "r") as f:
            datalist = []
            for line in tqdm(f, desc=f"Extracting events from {i}-th/{len(all_paths)} file."):
                line = line.replace("\\\\", "/")
                temp_dic = json.loads(line.strip())

                reverse_flag = False

                if temp_dic["hostname"].split(".")[0] != hostname:
                    continue

                operation = temp_dic["action"]
                if operation not in rel2id:
                    continue

                if operation in ["READ"]:
                    reverse_flag = True

                src_uuid = temp_dic["actorID"]
                dst_uuid = temp_dic["objectID"]
                if (src_uuid not in uuid2index_id) or (dst_uuid not in uuid2index_id):
                    continue
                else:
                    src_index_id = uuid2index_id[src_uuid]
                    dst_index_id = uuid2index_id[dst_uuid]

                event_uuid = temp_dic["id"]

                if (temp_dic["object"] == "FLOW") and ("direction" in temp_dic["properties"]):
                    if temp_dic["properties"]["direction"] == "inbound":
                        reverse_flag = True

                timestr = temp_dic["timestamp"]
                timestamp = OPTC_datetime_to_timestamp_US(timestr, timezone=cfg.dataset.timezone)

                if reverse_flag:
                    temp_data = [
                        dst_uuid,
                        str(dst_index_id),
                        operation,
                        src_uuid,
                        str(src_index_id),
                        event_uuid,
                        timestamp,
                    ]
                else:
                    temp_data = [
                        src_uuid,
                        str(src_index_id),
                        operation,
                        dst_uuid,
                        str(dst_index_id),
                        event_uuid,
                        timestamp,
                    ]

                datalist.append(temp_data)

            log(f"Start saving events for {i}-th/{len(all_paths)} file.")
            events.add_rows(datalist)
            log(f"Finish saving events for {i}-th/{len(all_paths)} file.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s MODEL DATASET [--raw_dir RAW_DIR] [--overwrite] [framework options, "
        "e.g. --datasets_dir]",
    )
    parser.add_argument("--raw_dir", default="/data/", help="Folder of the JSON files")
    parser.add_argument("--overwrite", action="store_true", help="Replaces an existing dataset")
    args, unknown = parser.parse_known_args()
    cfg = get_yml_cfg(get_runtime_required_args(args=unknown))

    dataset_dir = parquet_dataset_dir(cfg)
    # Tables of the Postgres databases of the OpTC datasets, whose `_id` is a SERIAL
    with DatasetBuilder(
        dataset_dir,
        schema_tables(get_schema(cfg), serial_type=INTEGER),
        source="dataset_preprocessing/optc/create_database_optc.py",
        overwrite=args.overwrite,
        log=log,
    ) as builder:
        uuid2index_id = save_nodes(cfg, builder, args.raw_dir)
        log("Finished saving nodes.")

        save_events(cfg, builder, uuid2index_id, args.raw_dir)
        log("Finished saving events.")
    log(f"Dataset {dataset_dir} created.")


if __name__ == "__main__":
    main()
