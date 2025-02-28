from config import get_rel2id
from .evaluation_utils import *
from labelling import get_ground_truth_edges, get_attack_to_mal_edges

def get_edge_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_edges = get_ground_truth_edges(cfg)
    threshold_method = cfg.detection.evaluation.edge_evaluation.threshold_method
    if threshold_method == 'magic':
        thr = get_threshold(test_tw_path, threshold_method)
    else:
        thr = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {thr:.3f}")

    scores, y_true, y_hat, src_dst_t_type = [], [], [], []
    edge_type_map = get_rel2id(cfg)
    filelist = listdir_sorted(test_tw_path)
    for file in log_tqdm(sorted(filelist), desc="Compute edge labels"):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        
        for line in df:
            srcnode = str(line['srcnode'])
            dstnode = str(line['dstnode'])
            loss = line['loss']
            t = line['time']
            edge_type = edge_type_map[line['edge_type']]
            edge = (srcnode, dstnode, t, edge_type)
            
            scores.append(loss)
            y_true.append(int(edge in ground_truth_edges))
            y_hat.append(int(loss > thr))
            src_dst_t_type.append(edge)
            
    return scores, y_true, y_hat, src_dst_t_type


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    scores, y_truth, y_preds, src_dst_t_type = get_edge_predictions(val_tw_path, test_tw_path, cfg, **kwargs)
    attack_to_mal_edges = get_attack_to_mal_edges(cfg)
    
    log(f"Found {sum(y_truth)} / {sum(len(edges) for edges in attack_to_mal_edges.values())} malicious edges")
    
    edge2attack = transform_attack2nodes_to_node2attacks(attack_to_mal_edges)
    
    out_dir = cfg.detection.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    adp_img_file = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
    
    log(f"Saving figures to {out_dir}...")
    adp_score = plot_detected_attacks_vs_precision(scores, src_dst_t_type, edge2attack, y_truth, adp_img_file)
    discrim_scores = compute_discrimination_score(scores, src_dst_t_type, edge2attack, y_truth)
    plot_discrimination_metric(scores, y_truth, discrim_img_file)
    discrim_tp = compute_discrimination_tp(scores, src_dst_t_type, edge2attack, y_truth)
    plot_simple_scores(scores, y_truth, simple_scores_img_file)
    plot_scores_with_paths_edge_level(scores, y_truth, src_dst_t_type, tw_to_malicious_nodes, edge2attack, scores_img_file, cfg)
    stats = classifier_evaluation(y_truth, y_preds, scores)
    
    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(scores, src_dst_t_type, attack_to_mal_edges)
    stats["fps_if_all_attacks_detected"] = fps
    stats["tps_if_all_attacks_detected"] = tps
    stats["precision_if_all_attacks_detected"] = precision
    stats["recall_if_all_attacks_detected"] = recall
    
    stats["adp_score"] = round(adp_score, 3)
    
    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)
        
    stats = {**stats, **discrim_tp}
    
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
    torch.save({
        "pred_scores": scores,
        "y_preds": y_preds,
        "y_truth": y_truth,
        "edges": src_dst_t_type,
        "edge2attack": edge2attack,
    }, scores_file)
    
    stats["scores_file"] = scores_file
    
    return stats
    