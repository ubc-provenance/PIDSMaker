"""DARPA TC E5 dataset preprocessing - create the Parquet dataset from JSON logs.

Parses DARPA TC Engagement 5 provenance JSON files and writes the nodes and events for graph
construction to a Parquet dataset (`<datasets_dir>/<database>`, see
`pidsmaker/utils/database.py`), with the tables of the published DARPA TC datasets. The files
of the dataset are listed in `filelist.py`, and read from `--raw_dir`:

    python dataset_preprocessing/darpa_tc/create_database_e5.py orthrus DATASET --raw_dir /data/
"""

import argparse
import os
import re

from tqdm import tqdm

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.utils.database import parquet_dataset_dir
from pidsmaker.utils.dataset_utils import edge_reversed, exclude_edge_type, get_schema
from pidsmaker.utils.dataset_writer import INTEGER, DatasetBuilder, schema_tables
from pidsmaker.utils.utils import log, stringtomd5

try:
    from . import filelist
except ImportError:  # run as a script
    import filelist


def store_netflow(file_path, table, index_id, filelist):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}',
                            line,
                        )[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        hashstr = stringtomd5(nodeid)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    net_uuid2hash = {}
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(",") + [index_id])
            net_uuid2hash[i] = netobj2hash[i][0]
            index_id += 1

    table.add_rows(datalist)

    return index_id, net_uuid2hash, datalist


def store_subject(file_path, table, index_id, filelist):
    # Parse data from logs
    scusess_count = 0
    fail_count = 0
    subject_obj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "schema.avro.cdm20.Subject" in line:
                    subject_uuid = re.findall(
                        'avro.cdm20.Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"path":"(.*?)"',
                        line,
                    )
                    try:
                        subject_obj2hash[subject_uuid[0][0]] = [
                            subject_uuid[0][-1],
                            subject_uuid[0][-3],
                        ]  # {uuid:[path, cmd]}
                        scusess_count += 1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]] = "null"
                        except:
                            pass
                        fail_count += 1
    # Store into database
    datalist = []
    subject_uuid2hash = {}
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            datalist.append(
                [i] + [stringtomd5(i)] + subject_obj2hash[i] + [index_id]
            )  # ([uuid, hashstr, path, cmdLine, index_id]) and hashstr=stringtomd5(uuid)
            subject_uuid2hash[i] = stringtomd5(i)
            index_id += 1

    table.add_rows(datalist)

    return index_id, subject_uuid2hash, datalist


def store_file(file_path, table, index_id, filelist):
    file_obj2hash = {}
    fail_count = 0
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "avro.cdm20.FileObject" in line:
                    Object_uuid = re.findall(
                        'avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"', line
                    )
                    try:
                        file_obj2hash[Object_uuid[0][0]] = Object_uuid[0][-1]
                    except:
                        fail_count += 1

    datalist = []
    file_uuid2hash = {}
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(i), file_obj2hash[i]] + [index_id])
            file_uuid2hash[i] = stringtomd5(i)
            index_id += 1

    table.add_rows(datalist)

    return index_id, file_uuid2hash, datalist


def create_node_list(netflow_rows, subject_rows, file_rows):
    """{hash_id: index_id} of the rows of the node tables (hash_id is their second column and
    index_id their last one), in the order of the tables."""
    nodeid2msg = {}
    for rows in (netflow_rows, subject_rows, file_rows):
        for i in rows:
            nodeid2msg[i[1]] = i[-1]

    return nodeid2msg  # {hash_id:index_id}


def store_event(
    file_path,
    table,
    reverse,
    nodeid2msg,
    subject_uuid2hash,
    file_uuid2hash,
    net_uuid2hash,
    filelist,
):
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type not in exclude_edge_type:
                        subject_uuid = re.findall(
                            '"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line
                        )
                        predicateObject_uuid = re.findall(
                            '"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line
                        )

                        if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                            if subject_uuid[0] in subject_uuid2hash and (
                                predicateObject_uuid[0] in subject_uuid2hash
                                or predicateObject_uuid[0] in file_uuid2hash
                                or predicateObject_uuid[0] in net_uuid2hash
                            ):
                                event_uuid = re.findall(
                                    '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event":{"uuid":"(.*?)",',
                                    line,
                                )[0]
                                time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                                time_rec = int(time_rec)
                                subjectId = subject_uuid2hash[subject_uuid[0]]
                                if predicateObject_uuid[0] in file_uuid2hash:
                                    objectId = file_uuid2hash[predicateObject_uuid[0]]
                                elif predicateObject_uuid[0] in net_uuid2hash:
                                    objectId = net_uuid2hash[predicateObject_uuid[0]]
                                else:
                                    objectId = subject_uuid2hash[predicateObject_uuid[0]]
                                if relation_type in reverse:
                                    table.add(
                                        [
                                            objectId,
                                            str(nodeid2msg[objectId]),
                                            relation_type,
                                            subjectId,
                                            str(nodeid2msg[subjectId]),
                                            event_uuid,
                                            time_rec,
                                        ]
                                    )
                                else:
                                    table.add(
                                        [
                                            subjectId,
                                            str(nodeid2msg[subjectId]),
                                            relation_type,
                                            objectId,
                                            str(nodeid2msg[objectId]),
                                            event_uuid,
                                            time_rec,
                                        ]
                                    )


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

    files = filelist.get_filelist(cfg.dataset.name)
    if files is None:
        raise ValueError(f"No list of files for {cfg.dataset.name} in filelist.py")
    raw_dir = os.path.join(args.raw_dir, "")

    dataset_dir = parquet_dataset_dir(cfg)
    # Tables of the DARPA TC datasets, whose `_id` is an integer
    with DatasetBuilder(
        dataset_dir,
        schema_tables(get_schema(cfg), serial_type=INTEGER),
        source="dataset_preprocessing/darpa_tc/create_database_e5.py",
        overwrite=args.overwrite,
        log=log,
    ) as builder:
        index_id = 0

        log("Processing netflow data")
        index_id, net_uuid2hash, netflow_rows = store_netflow(
            file_path=raw_dir,
            table=builder.table("netflow_node_table"),
            index_id=index_id,
            filelist=files,
        )

        log("Processing subject data")
        index_id, subject_uuid2hash, subject_rows = store_subject(
            file_path=raw_dir,
            table=builder.table("subject_node_table"),
            index_id=index_id,
            filelist=files,
        )

        log("Processing file data")
        index_id, file_uuid2hash, file_rows = store_file(
            file_path=raw_dir,
            table=builder.table("file_node_table"),
            index_id=index_id,
            filelist=files,
        )

        log("Extracting the node list")
        nodeid2msg = create_node_list(netflow_rows, subject_rows, file_rows)
        del netflow_rows, subject_rows, file_rows

        log("Processing the events")
        store_event(
            file_path=raw_dir,
            table=builder.table("event_table"),
            reverse=edge_reversed,
            nodeid2msg=nodeid2msg,
            subject_uuid2hash=subject_uuid2hash,
            file_uuid2hash=file_uuid2hash,
            net_uuid2hash=net_uuid2hash,
            filelist=files,
        )
    log(f"Dataset {dataset_dir} created.")


if __name__ == "__main__":
    main()
