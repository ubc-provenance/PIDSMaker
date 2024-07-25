from config import *
from provnet_utils import *
from tqdm import tqdm


def get_tasks(evaluation_results):
    tw_to_poi = {}

    for tw, nid_to_result in evaluation_results.items():

        for nid, result in nid_to_result.items():
            score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]

            if y_hat == 1 and y_true == 0:
                if tw not in tw_to_poi:
                    tw_to_poi[tw] = []
                tw_to_poi[tw].append(str(nid))

    return tw_to_poi

def check_if_in_trainset(fps,tw_to_graphdir,cur):
    indexid2msg = get_node_infos(cur)
    fp_to_samename = {}
    for fp in tqdm(list(fps),desc='get same-name nodes of FPs'):
        if fp not in fp_to_samename:
            fp_to_samename[fp] = set()
        msg = indexid2msg[fp]
        for nid, m in indexid2msg.items():
            if msg == m:
                fp_to_samename[fp].add(nid)

    fp_to_tw = {}
    sn_to_tw = {}

    for tw, graphdir in tqdm(tw_to_graphdir.items(),desc="check if fp or same-name node is in train sets"):
        graph = torch.load(graphdir)

        for fp in fps:
            if fp not in fp_to_tw:
                fp_to_tw[fp] = set()
            if fp not in sn_to_tw:
                sn_to_tw[fp] = set()

            sns = list(fp_to_samename[fp] - {fp})

            if fp in graph.nodes():
                fp_to_tw[fp].add(tw)

            for sn in sns:
                if sn in graph.nodes():
                    sn_to_tw[fp].add(sn)

    for fp in fps:
        print("==" * 20)
        print(f"FP node {fp}:")
        print(f"Feature of {fp} is: {indexid2msg[fp]}")
        if len(fp_to_tw[fp]) == 0:
            print("It does not appear in any training graph")
        else:
            print(f"It appear in training time windows:")
            print(list(fp_to_tw[fp]))

        if len(sn_to_tw[fp]) == 0:
            print("Its same-name nodes do not appear in any training graph")
        else:
            print(f"Its same-name nodes appear in time windows:")
            print(list(sn_to_tw[fp]))

        print("==" * 20)



def get_node_infos(cur):
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of netflow nodes: {len(records)}")

    for i in records:
        remote_ip = str(i[4])
        remote_port = str(i[5])
        index_id = str(i[-1]) # int
        msg = 'netflow' + ' ' + remote_ip + ':' + remote_port
        indexid2msg[index_id] = msg

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of process nodes: {len(records)}")

    for i in records:
        path = str(i[2])
        cmd = str(i[3])
        index_id = str(i[-1])
        msg = 'subject' + ' ' + path + ' ' +cmd
        indexid2msg[index_id] = msg

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of file nodes: {len(records)}")

    for i in records:
        path = str(i[2])
        index_id = str(i[-1])
        msg = 'file' + ' ' + path
        indexid2msg[index_id] = msg

    return indexid2msg

def main(cfg):
    cur, connect = init_database_connection(cfg)

    in_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")

    best_ap, best_stats = 0.0, {}
    best_model_epoch = listdir_sorted(test_losses_dir)[-1]
    for model_epoch_dir in listdir_sorted(test_losses_dir):

        stats_file = os.path.join(in_dir, f"stats_{model_epoch_dir}.pth")
        stats = torch.load(stats_file)
        if stats["ap"] > best_ap:
            best_ap = stats["ap"]
            best_model_epoch = model_epoch_dir
    results_file = os.path.join(in_dir, f"result_{best_model_epoch}.pth")
    results = torch.load(results_file)

    sorted_tw_paths = sorted(os.listdir(os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, 'train')))
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    tw_to_graphdir = {}
    for tw, tw_file in enumerate(sorted_tw_paths):
        timestr = tw_file[:-20]
        day = timestr[8:10].lstrip('0')
        graph_dir = os.path.join(base_dir, f"graph_{day}/{timestr}")

        tw_to_graphdir[tw] = graph_dir

    tw_to_fps = get_tasks(results)
    fps = set()
    for tw, fps_list in tw_to_fps.items():
        for fp in fps_list:
            fps.add(fp)

    check_if_in_trainset(fps,tw_to_graphdir,cur)
    print("Finish checking!")

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)