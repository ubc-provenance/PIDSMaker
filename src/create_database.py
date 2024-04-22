import os
import re
import torch
from tqdm import tqdm
import hashlib


from config import *
from provnet_utils import *

filelist = [
    'ta1-theia-1-e5-official-1.bin.1.json',
    'ta1-theia-1-e5-official-1.bin.1.json.1',
    'ta1-theia-1-e5-official-1.bin.1.json.2',
    'ta1-theia-1-e5-official-1.bin.2.json',
    'ta1-theia-1-e5-official-1.bin.json',
    'ta1-theia-1-e5-official-1.bin.json.1',
    'ta1-theia-1-e5-official-1.bin.json.2',
    'ta1-theia-1-e5-official-1.json.1',
    'ta1-theia-1-e5-official-1.json.2',
    'ta1-theia-1-e5-official-1.json',
    'ta1-theia-1-e5-official-2.bin.10.json',
    'ta1-theia-1-e5-official-2.bin.10.json.1',
    'ta1-theia-1-e5-official-2.bin.10.json.2',
    'ta1-theia-1-e5-official-2.bin.11.json',
    'ta1-theia-1-e5-official-2.bin.11.json.1',
    'ta1-theia-1-e5-official-2.bin.11.json.2',
    'ta1-theia-1-e5-official-2.bin.12.json',
    'ta1-theia-1-e5-official-2.bin.12.json.1',
    'ta1-theia-1-e5-official-2.bin.12.json.2',
    'ta1-theia-1-e5-official-2.bin.13.json',
    'ta1-theia-1-e5-official-2.bin.13.json.1',
    'ta1-theia-1-e5-official-2.bin.13.json.2',
    'ta1-theia-1-e5-official-2.bin.14.json',
    'ta1-theia-1-e5-official-2.bin.14.json.1',
    'ta1-theia-1-e5-official-2.bin.14.json.2',
    'ta1-theia-1-e5-official-2.bin.15.json',
    'ta1-theia-1-e5-official-2.bin.15.json.1',
    'ta1-theia-1-e5-official-2.bin.15.json.2',
    'ta1-theia-1-e5-official-2.bin.16.json',
    'ta1-theia-1-e5-official-2.bin.16.json.1',
    'ta1-theia-1-e5-official-2.bin.16.json.2',
    'ta1-theia-1-e5-official-2.bin.17.json',
    'ta1-theia-1-e5-official-2.bin.17.json.1',
    'ta1-theia-1-e5-official-2.bin.17.json.2',
    'ta1-theia-1-e5-official-2.bin.18.json',
    'ta1-theia-1-e5-official-2.bin.18.json.1',
    'ta1-theia-1-e5-official-2.bin.18.json.2',
    'ta1-theia-1-e5-official-2.bin.19.json',
    'ta1-theia-1-e5-official-2.bin.19.json.1',
    'ta1-theia-1-e5-official-2.bin.19.json.2',
    'ta1-theia-1-e5-official-2.bin.1.json',
    'ta1-theia-1-e5-official-2.bin.1.json.1',
    'ta1-theia-1-e5-official-2.bin.1.json.2',
    'ta1-theia-1-e5-official-2.bin.20.json',
    'ta1-theia-1-e5-official-2.bin.20.json.1',
    'ta1-theia-1-e5-official-2.bin.20.json.2',
    'ta1-theia-1-e5-official-2.bin.21.json',
    'ta1-theia-1-e5-official-2.bin.21.json.1',
    'ta1-theia-1-e5-official-2.bin.21.json.2',
    'ta1-theia-1-e5-official-2.bin.22.json',
    'ta1-theia-1-e5-official-2.bin.22.json.1',
    'ta1-theia-1-e5-official-2.bin.22.json.2',
    'ta1-theia-1-e5-official-2.bin.23.json',
    'ta1-theia-1-e5-official-2.bin.23.json.1',
    'ta1-theia-1-e5-official-2.bin.23.json.2',
    'ta1-theia-1-e5-official-2.bin.24.json',
    'ta1-theia-1-e5-official-2.bin.24.json.1',
    'ta1-theia-1-e5-official-2.bin.24.json.2',
    'ta1-theia-1-e5-official-2.bin.25.json',
    'ta1-theia-1-e5-official-2.bin.25.json.1',
    'ta1-theia-1-e5-official-2.bin.25.json.2',
    'ta1-theia-1-e5-official-2.bin.26.json',
    'ta1-theia-1-e5-official-2.bin.26.json.1',
    'ta1-theia-1-e5-official-2.bin.26.json.2',
    'ta1-theia-1-e5-official-2.bin.27.json',
    'ta1-theia-1-e5-official-2.bin.27.json.1',
    'ta1-theia-1-e5-official-2.bin.27.json.2',
    'ta1-theia-1-e5-official-2.bin.28.json',
    'ta1-theia-1-e5-official-2.bin.28.json.1',
    'ta1-theia-1-e5-official-2.bin.28.json.2',
    'ta1-theia-1-e5-official-2.bin.29.json',
    'ta1-theia-1-e5-official-2.bin.29.json.1',
    'ta1-theia-1-e5-official-2.bin.29.json.2',
    'ta1-theia-1-e5-official-2.bin.2.json',
    'ta1-theia-1-e5-official-2.bin.2.json.1',
    'ta1-theia-1-e5-official-2.bin.2.json.2',
    'ta1-theia-1-e5-official-2.bin.30.json',
    'ta1-theia-1-e5-official-2.bin.30.json.1',
    'ta1-theia-1-e5-official-2.bin.30.json.2',
    'ta1-theia-1-e5-official-2.bin.31.json',
    'ta1-theia-1-e5-official-2.bin.31.json.1',
    'ta1-theia-1-e5-official-2.bin.31.json.2',
    'ta1-theia-1-e5-official-2.bin.32.json',
    'ta1-theia-1-e5-official-2.bin.32.json.1',
    'ta1-theia-1-e5-official-2.bin.32.json.2',
    'ta1-theia-1-e5-official-2.bin.33.json',
    'ta1-theia-1-e5-official-2.bin.33.json.1',
    'ta1-theia-1-e5-official-2.bin.33.json.2',
    'ta1-theia-1-e5-official-2.bin.34.json',
    'ta1-theia-1-e5-official-2.bin.34.json.1',
    'ta1-theia-1-e5-official-2.bin.34.json.2',
    'ta1-theia-1-e5-official-2.bin.35.json',
    'ta1-theia-1-e5-official-2.bin.35.json.1',
    'ta1-theia-1-e5-official-2.bin.35.json.2',
    'ta1-theia-1-e5-official-2.bin.36.json',
    'ta1-theia-1-e5-official-2.bin.36.json.1',
    'ta1-theia-1-e5-official-2.bin.36.json.2',
    'ta1-theia-1-e5-official-2.bin.37.json',
    'ta1-theia-1-e5-official-2.bin.37.json.1',
    'ta1-theia-1-e5-official-2.bin.37.json.2',
    'ta1-theia-1-e5-official-2.bin.38.json',
    'ta1-theia-1-e5-official-2.bin.38.json.1',
    'ta1-theia-1-e5-official-2.bin.38.json.2',
    'ta1-theia-1-e5-official-2.bin.39.json',
    'ta1-theia-1-e5-official-2.bin.3.json',
    'ta1-theia-1-e5-official-2.bin.3.json.1',
    'ta1-theia-1-e5-official-2.bin.3.json.2',
    'ta1-theia-1-e5-official-2.bin.4.json',
    'ta1-theia-1-e5-official-2.bin.4.json.1',
    'ta1-theia-1-e5-official-2.bin.4.json.2',
    'ta1-theia-1-e5-official-2.bin.5.json',
    'ta1-theia-1-e5-official-2.bin.5.json.1',
    'ta1-theia-1-e5-official-2.bin.5.json.2',
    'ta1-theia-1-e5-official-2.bin.6.json',
    'ta1-theia-1-e5-official-2.bin.6.json.1',
    'ta1-theia-1-e5-official-2.bin.6.json.2',
    'ta1-theia-1-e5-official-2.bin.7.json',
    'ta1-theia-1-e5-official-2.bin.7.json.1',
    'ta1-theia-1-e5-official-2.bin.7.json.2',
    'ta1-theia-1-e5-official-2.bin.8.json',
    'ta1-theia-1-e5-official-2.bin.8.json.1',
    'ta1-theia-1-e5-official-2.bin.8.json.2',
    'ta1-theia-1-e5-official-2.bin.9.json',
    'ta1-theia-1-e5-official-2.bin.9.json.1',
    'ta1-theia-1-e5-official-2.bin.9.json.2',
    'ta1-theia-1-e5-official-2.bin.json',
    'ta1-theia-1-e5-official-2.bin.json.1',
    'ta1-theia-1-e5-official-2.bin.json.2'
]


def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def store_netflow(file_path, cur, connect):
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
                            line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        hashstr = stringtomd5(nodeproperty)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(","))

    sql = '''insert into netflow_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

def store_subject(file_path, cur, connect):
    # Parse data from logs
    scusess_count = 0
    fail_count = 0
    subject_obj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "schema.avro.cdm20.Subject" in line:
                    subject_uuid = re.findall('avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"', line)
                    try:
                        subject_obj2hash[subject_uuid[0][0]] = subject_uuid[0][-1]
                        scusess_count += 1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]] = "null"
                        except:
                            pass
                        fail_count += 1
    # Store into database
    datalist = []
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(subject_obj2hash[i]), subject_obj2hash[i]])

    sql = '''insert into subject_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

def store_file(file_path, cur, connect):
    file_obj2hash = {}
    fail_count = 0
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "avro.cdm20.FileObject" in line:
                    Object_uuid = re.findall('avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"', line)
                    try:
                        file_obj2hash[Object_uuid[0][0]] = Object_uuid[0][-1]
                    except:
                        fail_count += 1

    datalist = []
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(file_obj2hash[i]), file_obj2hash[i]])

    sql = '''insert into file_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

def create_node_list(cur, connect):
    node_list = {}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["file", i[-1]]
    file_uuid2hash = {}
    for i in records:
        file_uuid2hash[i[0]] = i[1]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["subject", i[-1]]
    subject_uuid2hash = {}
    for i in records:
        subject_uuid2hash[i[0]] = i[1]

    # netflow
    sql = """
    select * from netflow_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["netflow", i[-2] + ":" + i[-1]]
    net_uuid2hash = {}
    for i in records:
        net_uuid2hash[i[0]] = i[1]

    node_list_database = []
    node_index = 0
    for i in node_list:
        node_list_database.append([i] + node_list[i] + [node_index])
        node_index += 1

    sql = '''insert into node2id
                         values %s
            '''
    ex.execute_values(cur, sql, node_list_database, page_size=10000)
    connect.commit()

    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash

def write_event_in_DB(cur, connect, datalist):
    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur,sql, datalist,page_size=10000)
    connect.commit()

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash):
    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type not in exclude_edge_type:
                        subject_uuid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line)
                        predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line)

                        if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                            if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in subject_uuid2hash or
                                                                         predicateObject_uuid[0] in file_uuid2hash or
                                                                         predicateObject_uuid[0] in net_uuid2hash):
                                event_uuid = re.findall('{"datum":{"com.bbn.tc.schema.avro.cdm20.Event":{"uuid":"(.*?)",', line)[0]
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
                                    datalist.append(
                                        [objectId, nodeid2msg[objectId], relation_type, subjectId,
                                         nodeid2msg[subjectId], event_uuid, time_rec])
                                else:
                                    datalist.append(
                                        [subjectId, nodeid2msg[subjectId], relation_type, objectId,
                                         nodeid2msg[objectId], event_uuid, time_rec])

    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=50000)
    connect.commit()


if __name__ == "__main__":
    cur, connect = init_database_connection()

    # There will be 36747 netflow nodes stored in the table
    print("Processing netflow data")
    store_netflow(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 1264440 subject nodes stored in the table
    print("Processing subject data")
    store_subject(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 984937 file nodes stored in the table
    print("Processing file data")
    store_file(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 967388 entities stored in the table
    print("Extracting the node list")
    nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash = create_node_list(cur=cur, connect=connect)

    # There will be 140994662 events stored in the table
    print("Processing the events")
    store_event(
        file_path=raw_dir,
        cur=cur,
        connect=connect,
        reverse=edge_reversed,
        nodeid2msg=nodeid2msg,
        subject_uuid2hash=subject_uuid2hash,
        file_uuid2hash=file_uuid2hash,
        net_uuid2hash=net_uuid2hash
    )