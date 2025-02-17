from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch
import cudf
import tracemalloc
from cuml.neighbors import NearestNeighbors


@torch.no_grad()
def test_edge_level(
        data,
        full_data,
        model,
        split,
        model_epoch_file,
        cfg,
        device,
):
    model.eval()

    time_with_loss = {}  # key: time，  value： the losses
    edge_list = None
    start_time = data.t[0]
    all_losses = []

    # NOTE: warning, this may reindex the data is TGN is not used
    batch_loader = batch_loader_factory(cfg, data, test_mode=True)
    
    validation = split == "val"

    for batch in batch_loader:
        results = model(batch, full_data, inference=True, validation=validation)
        each_edge_loss = results["loss"]
        all_losses.extend(each_edge_loss.cpu().numpy().tolist())

        # If the data has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index
        
        # edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        
        srcnodes = edge_index[0, :].cpu().numpy()
        dstnodes = edge_index[1, :].cpu().numpy()
        t_vars = batch.t.cpu().numpy()
        losses = each_edge_loss.cpu().numpy()

        # if 1 in batch.y:
        #     log(f"Mean score of fake malicious edges: {losses[batch.y.cpu() == 1].mean():.4f}")
        #     log(f"Mean score of benign malicious edges: {losses[batch.y.cpu() == 0].mean():.4f}")
        
        edge_df = pd.DataFrame({
            'loss': losses.astype(float),
            'srcnode': srcnodes.astype(int),
            'dstnode': dstnodes.astype(int),
            'time': t_vars.astype(int)
        })
        if edge_list is None:
            edge_list = edge_df
        else:
            edge_list = pd.concat([edge_list, edge_df])


    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list["time"].max())

    logs_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    edge_list.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')
    return all_losses

    # log(f'Time: {time_interval}, Loss: {losses:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')

@torch.no_grad()
def test_node_level(
    data,
    full_data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    node_list = []
    start_time = data.t[0]
    end_time = data.t[-1]
    losses = []
    start = time.perf_counter()

    loader = batch_loader_factory(cfg, data)
    
    validation = split == "val"

    for batch in loader:
        batch = batch.to(device)
        
        results = model(batch, full_data, inference=True, validation=validation)
        loss = results["loss"]
        losses.extend(loss.cpu().numpy().tolist())

        # ThreaTrace code
        if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
            out = results["out"]
            pred = out.max(1)[1]
            pro = F.softmax(out, dim=1)
            pro1 = pro.max(1)
            for i in range(len(out)):
                pro[i][pro1[1][i]] = -1
            pro2 = pro.max(1)
            
            node_type_num = batch.node_type.argmax(1)
            for i in range(len(out)):
                if pro2[0][i] != 0:
                    score = pro1[0][i] / pro2[0][i]
                else:
                    score = pro1[0][i] / 1e-5
                score = torch.log(score + 1e-12) # we do that or the score is much too high
                score = max(score.item(), 0)
            
                node = batch.original_n_id[i].item()
                correct_pred = int((node_type_num[i] == pred[i]).item())

                temp_dic = {
                    'node': node,
                    'loss': float(loss[i].item()),
                    'threatrace_score': score,
                    'correct_pred': correct_pred,
                }
                node_list.append(temp_dic)
                
        # Flash code
        elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
            out = results["out"]
            pred = out.max(1)[1]
            sorted, indices = out.sort(dim=1, descending=True)
            eps = 1e-6
            conf = (sorted[:, 0] - sorted[:, 1]) / (sorted[:, 0] + eps)
            conf = (conf - conf.min()) / conf.max() if conf.max() > 0 else conf

            node_type_num = batch.node_type.argmax(1)
            for i in range(len(out)):
                score = max(conf[i].item(), 0)

                node = batch.original_n_id[i].item()
                correct_pred = int((node_type_num[i] == pred[i]).item())

                temp_dic = {
                    'node': node,
                    'loss': float(loss[i].item()),
                    'flash_score': score,
                    'correct_pred': correct_pred,
                }
                node_list.append(temp_dic)
            
        # Magic codes
        elif cfg.detection.evaluation.node_evaluation.threshold_method == "magic":
            os.makedirs(cfg.detection.gnn_training._magic_dir, exist_ok=True)
            if split == 'val':
                x_train = model.embed(batch, full_data, inference=True).cpu().numpy()
                num_nodes = x_train.shape[0]
                sample_size = 5000 if num_nodes > 5000 else num_nodes
                sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
                x_train_sampled = x_train[sample_indices]
                x_train_mean = x_train_sampled.mean(axis=0)
                x_train_std = x_train_sampled.std(axis=0)
                x_train_sampled = (x_train_sampled - x_train_mean) / x_train_std

                torch.cuda.empty_cache()
                x_train_sampled = cudf.DataFrame.from_records(x_train_sampled)

                n_neighbors = 10
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                nbrs.fit(x_train_sampled)
                idx = list(range(x_train_sampled.shape[0]))
                random.shuffle(idx)
                try:
                    sample = x_train_sampled.iloc[idx[:min(50000, x_train_sampled.shape[0])]].to_pandas()
                    distances_train, _ = nbrs.kneighbors(
                        sample,
                        n_neighbors=min(len(sample), n_neighbors))
                except KeyError as e:
                    log(f"KeyError encountered: {e}")
                    log(f"Available columns in x_train: {x_train_sampled.columns}")
                    raise
                mean_distance_train = distances_train.mean().mean()
                if mean_distance_train == 0:
                    mean_distance_train = 1e-9
                torch.cuda.empty_cache()

                train_distance_file = os.path.join(cfg.detection.gnn_training._magic_dir, "train_distance.txt")
                with open(train_distance_file, "a") as f:
                    f.write(f"{mean_distance_train}\n")

                for i, node in enumerate(batch.original_n_id):
                    temp_dic = {
                        'node': node.item(),
                        'loss': float(loss[i].item()),
                    }
                    node_list.append(temp_dic)

            elif split == 'test':
                train_distance_file = os.path.join(cfg.detection.gnn_training._magic_dir, "train_distance.txt")
                mean_distance_train = calculate_average_from_file(train_distance_file)

                x_test = model.embed(batch, full_data, inference=True).cpu().numpy()
                num_nodes = x_test.shape[0]
                sample_size = 5000 if num_nodes > 5000 else num_nodes
                sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
                x_test_sampled = x_test[sample_indices]
                x_test_mean = x_test_sampled.mean(axis=0)
                x_test_std = x_test_sampled.std(axis=0)
                x_test_sampled = (x_test_sampled - x_test_mean) / x_test_std

                torch.cuda.empty_cache()
                x_test_sampled = cudf.DataFrame.from_records(x_test_sampled)

                n_neighbors = 10
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                nbrs.fit(x_test_sampled)

                distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
                distances = distances.mean(axis=1)
                # distances = distances.to_numpy()
                score = distances / mean_distance_train
                score = score.tolist()

                for i, node in enumerate(batch.original_n_id):
                    temp_dic = {
                        'node': node.item(),
                        'magic_score': float(score[i]),
                        'loss': float(loss[i].item()),
                    }
                    node_list.append(temp_dic)
        
        else:
            for i, node in enumerate(batch.original_n_id):
                temp_dic = {
                    'node': node.item(),
                    'loss': float(loss[i].item()),
                }
                node_list.append(temp_dic)
                

    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(end_time)

    end = time.perf_counter()
    logs_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(node_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')
    return losses

    # log(f'Time: {time_interval}, Loss: {losses:.4f}, Nodes_count: {node_count}, Cost Time: {(end - start):.2f}s')


def main(cfg, model, val_data, test_data, full_data, epoch, split, logging=True):
    set_seed(cfg)
    
    if split == "all":
        splits = [(val_data, "val"), (test_data, "test")]
    elif split == "val":
        splits = [(val_data, "val")]
    elif split == "test":
        splits = [(test_data, "test")]
    else:
        raise ValueError(f"Invalid split {split}")
    
    device = cfg.detection.gnn_training.inference_device.strip()
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid inference device {device}")

    device = torch.device(device)
    use_cuda = device == torch.device("cuda")
    model.to_device(device)

    model_epoch_file = f"model_epoch_{epoch}"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    val_score = 0.0
    peak_inference_cpu_mem = 0
    peak_inference_gpu_mem = 0
    tpb = []
    split2loss = {}

    for graphs, split_name in splits:
        desc = "Validation" if split_name == "val" else "Testing"

        tracemalloc.start()
        
        all_losses = []
        for g in log_tqdm(graphs, desc=desc, logging=logging):
            g.to(device=device)
            
            s = time.time()
            test_fn = test_node_level if cfg._is_node_level else test_edge_level
            losses = test_fn(
                data=g,
                full_data=full_data,
                model=model,
                split=split_name,
                model_epoch_file=model_epoch_file,
                cfg=cfg,
                device=device,
            )
            all_losses.extend(losses)
            tpb.append(time.time() - s)
            
            g.to("cpu")  # Move graph back to CPU to free GPU memory for next batch
            if use_cuda:
                torch.cuda.empty_cache()
            
        _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
        peak_inference_cpu_mem = max(peak_inference_cpu_mem, peak_inference_cpu_memory / (1024 ** 3))
        tracemalloc.stop()
        
        if use_cuda:
            peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            peak_inference_gpu_mem = max(peak_inference_gpu_mem, peak_inference_gpu_memory)
            torch.cuda.reset_peak_memory_stats(device=device)
        
        mean_loss = np.mean(all_losses)
        split2loss[split_name] = mean_loss
        
        if split_name == "val":
            val_score = model.get_val_ap()
            if logging:
                log(f'[@epoch{epoch:02d}] Validation finished - Val Loss: {mean_loss:.4f} - Val Score: {val_score:.4f}', return_line=True)
        else:
            if logging:
                log(f'[@epoch{epoch:02d}] Test finished - Test Loss: {mean_loss:.4f}', return_line=True)

    del model
    
    stats = {
        "val_score": val_score,
        "val_loss": split2loss.get("val", None),
        "test_loss": split2loss.get("test", None),
        "peak_inference_cpu_memory": peak_inference_cpu_mem,
        "peak_inference_gpu_memory": peak_inference_gpu_mem,
        "time_per_batch_inference": np.mean(tpb),
    }
    return stats


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
