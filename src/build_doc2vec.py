from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

from provnet_utils import *

def get_indexid2msg(cur):
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        remote_address = i[4] + ':' + i[5]
        index_id = i[-1] # int
        indexid2msg[index_id] = ['netflow', remote_address]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        path = i[2]
        cmd = i[3]
        index_id = i[-1]
        indexid2msg[index_id] = ['subject', cmd]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        path = i[2]
        index_id = i[-1]
        indexid2msg[index_id] = ['file', path]

    return indexid2msg #{index_id: [node_type, msg]}

def main(cfg):
    #TODO: add logger

    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)



if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
