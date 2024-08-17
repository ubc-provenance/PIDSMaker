from tqdm import tqdm

from encoders import TGNEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch


@torch.no_grad()
def test(
        data,
        full_data,
        model,
        nodeid2msg,
        split,
        model_epoch_file,
        cfg,
        device,
):
    model.eval()

    time_with_loss = {}  # key: time，  value： the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()

    # NOTE: warning, this may reindex the graph is TGN is not used
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        each_edge_loss = model(batch, full_data, inference=True)
        tot_loss += each_edge_loss.sum().item()

        # If the graph has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index
        
        num_events = each_edge_loss.shape[0]
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        for i in range(num_events):
            srcnode = int(edge_index[0, i])
            dstnode = int(edge_index[1, i])

            srcmsg = nodeid2msg[srcnode]
            dstmsg = nodeid2msg[dstnode]
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
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(edge_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

    log(
        f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')


def main(cfg):
    # load the map between nodeID and node labels
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    nodeid2msg = {k: str(v) for k, v in nodeid2msg.items()}  # pre-compute because it's too slow in main loop

    _, val_data, test_data, full_data = load_all_datasets(cfg)

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = listdir_sorted(gnn_models_dir)

    device = get_device(cfg)

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        torch.cuda.empty_cache()
        model = build_model(data_sample=test_data[0], device=device, cfg=cfg)
        model = load_model(model, os.path.join(gnn_models_dir, trained_model), cfg, map_location=device)

        # TODO: we may want to move the validation set into the training for early stopping
        for graphs, split in [
            (val_data, "val"),
            (test_data, "test"),
        ]:
            log(f"    Testing {split} set...")
            for g in tqdm(graphs, desc=f"{split} set with {trained_model}"):
                g.to(device=device)
                test(
                    data=g,
                    full_data=full_data,
                    model=model,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    cfg=cfg,
                    device=device,
                )
                g.to("cpu")

        del model


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
