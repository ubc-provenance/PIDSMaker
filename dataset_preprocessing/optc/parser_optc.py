import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser_config import (
    DATA_DIR,
    DATABASE_DEFAULT_CONFIG,
    DATASET_DEFAULT_CONFIG,
    OPTC_hostname_map as hostname_map,
    OPTC_reversed_type as reversed_type_optc,
    OPTC_node_type_used as node_type_used,
    OPTC_rel2id as rel2id,
)
from optc_utils import *
import argparse
import os
from tqdm import tqdm
from psycopg2 import extras as ex
import json

def save_nodes(args):
    hostname = hostname_map[args.host]
    dataset_dir = os.path.join(DATA_DIR, args.host + '/')
    all_paths = get_all_filelist(dataset_dir)

    subject_uuid2attr = {}
    netflow_uuid2attr = {}
    file_uuid2attr = {}

    for i, file in enumerate(all_paths):
        with open(file, 'r') as f:
            for line in tqdm(f, desc=f"Extracting nodes from {i}-th/{len(all_paths)} file."):
                line = line.replace('\\\\', '/')
                temp_dic = json.loads(line.strip())

                if temp_dic['hostname'].split('.')[0] != hostname:
                    continue

                if temp_dic['action'] not in rel2id:
                    continue

                if temp_dic['object'] not in node_type_used:
                    continue

                src_uuid = temp_dic['actorID']
                dst_uuid = temp_dic['objectID']

                # Process src_node
                if 'image_path' in temp_dic['properties']:
                    subject_uuid2attr[src_uuid] = (temp_dic['properties']['image_path'], None)
                else:
                    subject_uuid2attr[src_uuid] = (None, None)

                # Process dst_node
                if temp_dic['object'] == 'FILE':
                    if 'file_path' in temp_dic['properties']:
                        file_uuid2attr[dst_uuid] = temp_dic['properties']['file_path']
                    else:
                        file_uuid2attr[dst_uuid] = None
                elif temp_dic['object'] == 'FLOW':
                    if 'src_ip' in temp_dic['properties']:
                        src_ip = temp_dic['properties']['src_ip']
                    else:
                        src_ip = None
                    if 'dest_ip' in temp_dic['properties']:
                        dst_ip = temp_dic['properties']['dest_ip']
                    else:
                        dst_ip = None
                    if 'src_port' in temp_dic['properties']:
                        src_port = temp_dic['properties']['src_port']
                    else:
                        src_port = None
                    if 'dest_port' in temp_dic['properties']:
                        dst_port = temp_dic['properties']['dest_port']
                    else:
                        dst_port = None

                    netflow_uuid2attr[dst_uuid] = (src_ip, src_port, dst_ip, dst_port)
                elif temp_dic['object'] == 'PROCESS':
                    if 'parent_image_path' in temp_dic['properties']:
                        path = temp_dic['properties']['parent_image_path']
                    else:
                        path = None
                    if 'command_line' in temp_dic['properties']:
                        cmd = temp_dic['properties']['command_line']
                    else:
                        cmd = None
                    subject_uuid2attr[dst_uuid] = (path, cmd)

    index_id = 0
    cur, connect = init_database_connection(args.host)
    uuid2index_id = {}

    # Save subject_nodes
    datalist = []
    for sub_uuid, sub_attr in tqdm(subject_uuid2attr.items(), desc=f"Processing datalist for subject nodes"):
        temp_data = [sub_uuid, stringtomd5(str(sub_attr)), sub_attr[0], sub_attr[1], index_id]
        datalist.append(temp_data)
        uuid2index_id[sub_uuid] = index_id
        index_id += 1

    print(f"Start saving subject nodes.")
    sql = '''insert into subject_node_table
                             values %s
                '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"Finished saving subject nodes.")
    del subject_uuid2attr
    del datalist

    # Save file_nodes
    datalist = []
    for file_uuid, file_attr in tqdm(file_uuid2attr.items(), desc=f"Processing file nodes"):
        temp_data = [file_uuid, stringtomd5(str(file_attr)), file_attr, index_id]
        datalist.append(temp_data)
        uuid2index_id[file_uuid] = index_id
        index_id += 1

    print(f"Start saving file nodes.")
    sql = '''insert into file_node_table
                             values %s
                '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"Finished saving file nodes.")
    del file_uuid2attr
    del datalist

    # Save netflow_nodes
    datalist = []
    for net_uuid, net_attr in tqdm(netflow_uuid2attr.items(), desc=f"Processing net nodes"):
        temp_data = [net_uuid, stringtomd5(str(net_attr)), net_attr[0], net_attr[1], net_attr[2], net_attr[3], index_id]
        datalist.append(temp_data)
        uuid2index_id[net_uuid] = index_id
        index_id += 1

    print(f"Start saving net nodes.")
    sql = '''insert into netflow_node_table
                             values %s
                '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"Finished saving net nodes.")
    del netflow_uuid2attr
    del datalist

    return uuid2index_id

def save_events(args, uuid2index_id):
    hostname = hostname_map[args.host]
    dataset_dir = os.path.join(DATA_DIR, args.host + '/')
    all_paths = get_all_filelist(dataset_dir)

    cur, connect = init_database_connection(args.host)

    for i, file in enumerate(all_paths):
        with open(file, 'r') as f:
            datalist = []
            for line in tqdm(f, desc=f"Extracting events from {i}-th/{len(all_paths)} file."):
                line = line.replace('\\\\', '/')
                temp_dic = json.loads(line.strip())

                reverse_flag = False

                if temp_dic['hostname'].split('.')[0] != hostname:
                    continue

                operation = temp_dic['action']
                if operation not in rel2id:
                    continue

                if operation in reversed_type_optc:
                    reverse_flag = True

                src_uuid = temp_dic['actorID']
                dst_uuid = temp_dic['objectID']
                if (src_uuid not in uuid2index_id) or (dst_uuid not in uuid2index_id):
                    continue
                else:
                    src_index_id = uuid2index_id[src_uuid]
                    dst_index_id = uuid2index_id[dst_uuid]

                event_uuid = temp_dic['id']

                if (temp_dic['object'] == 'FLOW') and ('direction' in temp_dic['properties']):
                    if temp_dic['properties']['direction'] == 'inbound':
                        reverse_flag = True

                timestr = temp_dic['timestamp']
                timestamp = datetime_to_timestamp_US(timestr)

                if reverse_flag:
                    temp_data = [dst_uuid, dst_index_id, operation, src_uuid, src_index_id, event_uuid, timestamp]
                else:
                    temp_data = [src_uuid, src_index_id, operation, dst_uuid, dst_index_id, event_uuid, timestamp]

                datalist.append(temp_data)

            print(f"Start saving events for {i}-th/{len(all_paths)} file.")
            sql = '''insert into event_table
                                        values %s
                        '''
            ex.execute_values(cur, sql, datalist, page_size=50000)
            connect.commit()
            print(f"Finish saving events for {i}-th/{len(all_paths)} file.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host', help='The name of host to be processed.')
    args = parser.parse_args()

    uuid2index_id = save_nodes(args)
    print(f"Finished saving nodes.")

    save_events(args, uuid2index_id)
    print(f"Finished saving events.")

