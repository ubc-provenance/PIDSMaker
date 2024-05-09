from config import *
from provnet_utils import *

def get_label2indexid(indexid2msg):
    label2indexid = {}
    for indexid, msg_list in indexid2msg.items():
        node_type = msg_list[0]
        msg = msg_list[1]
        if node_type == "file":
            spl = msg.strip().split('/')
            label ='/' + spl[1] + '/*'
            if label not in label2indexid:
                label2indexid[label] = []
            label2indexid[label].append(indexid)
    return label2indexid


def main(cfg):
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur, use_port=False, use_cmd=False)

    label2indexid = get_label2indexid(indexid2msg)

    for label, indexids in label2indexid.items():
        print(f"{label}\t{len(indexids)}")

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)