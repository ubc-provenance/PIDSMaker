import os

from config import *
from provnet_utils import *
import torch

from .depimpact_utils import DEPIMPACT, visualize_dependency_graph, log_with_pid

import multiprocessing
import labelling

import time
import csv

def get_tasks(evaluation_results):
    tw_to_poi = {}
    tw_to_node_score = {}
    for tw, nid_to_result in evaluation_results.items():
        tw_to_node_score[tw] = {}
        for nid, result in nid_to_result.items():
            score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
            tw_to_node_score[tw][str(nid)] = score
            if y_hat == 1:
                if tw not in tw_to_poi:
                    tw_to_poi[tw] = []
                tw_to_poi[tw].append(int(nid))

    return tw_to_poi, tw_to_node_score

def split_list(lst, n):
    avg = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

def worker_func(task_list, worker_num):
    pid = os.getpid()
    log(f"Start worker {str(worker_num)} with pid {pid}")

    result = []

    for task in task_list:
        tw, graph_dir, poi, node_to_score, used_method, score_method  = task[0], task[1], task[2], task[3], task[4], task[5]

        # load graph
        graph = torch.load(graph_dir)

        start_time = time.time()
        # init depimpact
        log_with_pid(f"start tracing poi {str(poi)} in time window {str(tw)}")
        dep = DEPIMPACT(graph, str(poi), node_to_score, used_method, score_method)

        # get subgraph nodes
        subgraph_nodes = dep.run()
        log_with_pid(f"finish tracing poi {str(poi)} in time window {str(tw)}")

        end_time = time.time()
        time_taken = end_time - start_time

        result.append((tw, graph_dir, subgraph_nodes, poi, time_taken))

    log(f"Finish worker {str(worker_num)} with pid {pid}")

    return result

def run(tasks,
        workers):
    workload_list = split_list(tasks, workers)

    arg_list = []
    for i in range(len(workload_list)):
        args = (
            workload_list[i],
            i
        )
        arg_list.append(args)

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.starmap(worker_func, arg_list)

    log(f"Finish tracing work")
    all_results = []
    for result in results:
        all_results.extend(result)

    return all_results

def main(evaluation_results,
         tw_to_timestr,
         cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    ground_truth_nids, _, _ = labelling.get_ground_truth(cfg) # int

    used_method = cfg.triage.tracing.depimpact.used_method
    score_method = cfg.triage.tracing.depimpact.score_method

    tw_to_poi, tw_to_node_score = get_tasks(evaluation_results)
    tasks = []
    for tw, pois in tw_to_poi.items():
        timestr = tw_to_timestr[tw]
        day = timestr[8:10].lstrip('0')
        graph_dir = os.path.join(base_dir, f"graph_{day}/{timestr}")

        for poi in pois:
            tasks.append((tw, graph_dir, poi, tw_to_node_score[tw], used_method, score_method))

    workers = cfg.triage.tracing.depimpact.workers

    all_results = run(tasks, workers)

    log("==" * 20)
    log("Results of each poi tracing")
    tw_to_info = {}
    detailed_info = []
    for result in all_results:
        tw, graph_dir, subgraph_nodes, poi, time_taken = result[0], result[1], result[2], result[3], result[4]

        #check tracing result of each poi
        log('--' * 20)
        if int(poi) in ground_truth_nids:
            log(f"POI {str(poi)} in time window {str(tw)} is a TP POI")
            tp_or_fp = 'TP'
        else:
            log(f"POI {str(poi)} in time window {str(tw)} is a FP POI")
            tp_or_fp = 'FP'
        log(f"Tracing poi {str(poi)} in time window {str(tw)} leads to:")
        tps = 0
        fps = 0
        for n in list(subgraph_nodes):
            if int(n) in ground_truth_nids:
                tps += 1
            else:
                fps += 1
        time_taken_str = f"{time_taken:.2f}"
        detailed_info.append((int(poi), tw, tp_or_fp, tps, fps, time_taken_str))
        log(f"TPS: {tps}, FPS: {fps}")
        log(f"{time_taken:.2f} seconds is taken")
        log('--' * 20)

        if int(tw) not in tw_to_info:
            tw_to_info[int(tw)] = {}
            tw_to_info[int(tw)]['graph_dir'] = graph_dir
            tw_to_info[int(tw)]['subgraph_nodes'] = set()
        tw_to_info[int(tw)]['subgraph_nodes'] |= set(subgraph_nodes)
    log("==" * 20)

    out_dir = cfg.triage.tracing._tracing_graph_dir
    os.makedirs(out_dir, exist_ok=True)

    detailed_info_filename = used_method + '_' + score_method + "_detailed.csv"
    detailed_info_dir = os.path.join(out_dir, detailed_info_filename)
    save_detailed_infos(detailed_info, detailed_info_dir)

    out_file = os.path.join(out_dir, "results.pth")
    torch.save(tw_to_info, out_file)

    all_traced_nodes = set()

    for tw, info in tw_to_info.items():
        origin_graph = torch.load(info["graph_dir"])
        subgraph = origin_graph.subgraph(info["subgraph_nodes"]).copy()

        all_traced_nodes |= set(subgraph.nodes())

        if cfg.triage.tracing.depimpact.visualize:
            log(f"Visualize graph for tw {str(tw)}")
            visualize_dependency_graph(dependency_graph=subgraph,
                                       ground_truth_nids=ground_truth_nids,
                                       poi=set(tw_to_poi[tw]),
                                       tw=str(tw),
                                       out_dir=out_dir,
                                       cfg=cfg)

    return tw_to_info, all_traced_nodes

def save_detailed_infos(infos, out_dir):
    poi, tw, tp_or_fp, tps, fps, time_taken_str = [], [], [], [], [], []
    sorted_infos = sorted(infos, key=lambda x: x[0])
    for info in sorted_infos:
        poi.append(info[0])
        tw.append(info[1])
        tp_or_fp.append(info[2])
        tps.append(info[3])
        fps.append(info[4])
        time_taken_str.append(info[5])
    with open(out_dir, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(poi)
        writer.writerow(tw)
        writer.writerow(tp_or_fp)
        writer.writerow(tps)
        writer.writerow(fps)
        writer.writerow(time_taken_str)

