from pidsmaker.config import *
from pidsmaker.utils import *


def get_node_attrs(cur):
    uuid2attrs = {}

    sql1 = "select * from subject_node_table;"
    cur.execute(sql1)
    rows1 = cur.fetchall()
    for i in rows1:
        uuid2attrs[i[0]] = {"index_id": i[-1], "node_type": "subject", "msg": i[2] + " " + i[3]}

    sql2 = "select * from file_node_table;"
    cur.execute(sql2)
    rows2 = cur.fetchall()
    for i in rows2:
        uuid2attrs[i[0]] = {"index_id": i[-1], "node_type": "file", "msg": i[2]}

    sql3 = "select * from netflow_node_table;"
    cur.execute(sql3)
    row3 = cur.fetchall()
    for i in row3:
        uuid2attrs[i[0]] = {"index_id": i[-1], "node_type": "netflow", "msg": i[4] + ":" + i[5]}

    return uuid2attrs


def modify_ground_truth_file(file_path, save_path, uuid2attrs):
    with open(save_path, "w") as save:
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, start=1):
                if line_num >= 2:
                    node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
                    print(line)
                    new_msg = {uuid2attrs[node_uuid]["node_type"]: uuid2attrs[node_uuid]["msg"]}
                    new_index_id = uuid2attrs[node_uuid]["index_id"]
                    new_line = node_uuid + ", " + str(new_msg) + ", " + str(new_index_id)
                    print(new_line)
                    save.write(new_line + "\n")


def main(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2attrs = get_node_attrs(cur)

    old_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path)
    new_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path_new)

    modify_ground_truth_file(old_path, new_path, uuid2attrs)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
