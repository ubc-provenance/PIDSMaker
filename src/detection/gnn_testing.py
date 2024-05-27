from tqdm import tqdm

from encoders import TGNEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    log("Warning: the device is CPU instead of CUDA")

@torch.no_grad()
def test(
    data,
    model,
    graph_reindexer,
    nodeid2msg,
    split,
    model_epoch_file,
    logger,
    cfg,
):
    model.eval()
    
    time_with_loss = {}  # key: time，  value： the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()
    
    batch_loader = batch_loader_factory(cfg, data, graph_reindexer)
    
    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        each_edge_loss = model(batch, data, inference=True)
        tot_loss += each_edge_loss.sum().item()
        
        num_events = each_edge_loss.shape[0]
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        for i in range(num_events):
            srcnode = int(batch.edge_index[0, i])
            dstnode = int(batch.edge_index[1, i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(batch.t[i])
            edge_type_idx = edge_types[i].item()
            edge_type = rel2id[edge_type_idx]
            loss = each_edge_loss[i]

            temp_dic = {
                'loss': float(loss),
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'edge_type': edge_type,
                'time': t_var,
            }
            edge_list.append(temp_dic)

        event_count += num_events
    tot_loss /= event_count

        # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list[-1]["time"])

    end = time.perf_counter()
    model_epoch_file = model_epoch_file.split(".")[0]
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    log_file = open(os.path.join(logs_dir, time_interval + ".txt"), 'w')

    log(
        f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
    edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
    for e in edge_list:
        log_file.write(str(e))
        log_file.write("\n")
    log_file.close()
    edge_list.clear()


def main(cfg):
    logger = get_logger(
        name="gnn_testing",
        filename=os.path.join(cfg.detection.gnn_testing._logs_dir, "gnn_testing.log"))

    # load the map between nodeID and node labels
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    
    val_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="val")
    test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")
    
    graph_reindexer = GraphReindexer(
        num_nodes=cfg.dataset.max_node_num,
        device=device,
    )

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = listdir_sorted(gnn_models_dir)

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        model = torch.load(os.path.join(gnn_models_dir, trained_model), map_location=device)
        
        # TODO: we may want to move the validation set into the training for early stopping
        for graphs, split in [
            (val_data, "val"),
            (test_data, "test"),
        ]:
            log(f"    Testing {split} set...")
            for g in tqdm(graphs, desc=f"{split} set with {trained_model}"):
                g.to(device)
                test(
                    data=g.clone(),
                    model=model,
                    graph_reindexer=graph_reindexer,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    logger=logger,
                    cfg=cfg,
                )
                
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
