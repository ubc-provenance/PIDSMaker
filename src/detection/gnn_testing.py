##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################
from tqdm import tqdm

from provnet_utils import *
from config import *
from model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def test_tw(inference_data,
            model,
            nodeid2msg,
            split,
            model_epoch_file,
            logger,
            cfg,
            ):
    model.eval()

    if isinstance(model.encoder, TGNEncoder):
        model.encoder.reset_state()
    
    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    start_time = inference_data.t[0]
    event_count = 0

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    word_embedding_dim = cfg.featurization.embed_nodes.emb_dim
    batch_size = cfg.detection.gnn_training.encoder.tgn.tgn_batch_size
    
    for batch in inference_data.seq_batches(batch_size=batch_size):

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, dst]).unique()
        edge_index = torch.stack([src, dst])
        h_src = msg[:, :word_embedding_dim] # TODO: replace by x_src, x_dst
        h_dst = msg[:, -word_embedding_dim:]

        each_edge_loss = model(edge_index, t, h_src, h_dst, msg, inference=True)

        node_embedding_dim = word_embedding_dim
        for i in range(len(each_edge_loss)):
            srcnode = int(src[i])
            dstnode = int(dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            loss = each_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(loss)
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        event_count += len(batch.src)

        # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

    end = time.perf_counter()
    time_with_loss[time_interval] = {'loss': loss,

                                     'nodes_count': len(unique_nodes),
                                     'total_edges': total_edges,
                                     'costed_time': (end - start)}

    model_epoch_file = model_epoch_file.split(".")[0]
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    log = open(os.path.join(logs_dir, time_interval + ".txt"), 'w')

    for e in edge_list:
        loss += e['loss']

    loss = loss / event_count
    logger.info(
        f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
    edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
    for e in edge_list:
        log.write(str(e))
        log.write("\n")
    log.close()
    edge_list.clear()


# @torch.no_grad()
# def test(inference_data,
#          memory,
#          gnn,
#          src_recon,
#          dst_recon,
#          neighbor_loader,
#          nodeid2msg,
#          path
#          ):
#     if os.path.exists(path):
#         pass
#     else:
#         os.mkdir(path)

#     memory.eval()
#     gnn.eval()
#     src_recon.eval()
#     dst_recon.eval()

#     memory.reset_state()  # Start with a fresh memory.
#     neighbor_loader.reset_state()  # Start with an empty graph.

#     time_with_loss = {}  # key: time，  value： the losses
#     total_loss = 0
#     edge_list = []

#     unique_nodes = torch.tensor([]).to(device=device)
#     total_edges = 0

#     start_time = inference_data.t[0]
#     event_count = 0

#     # Record the running time to evaluate the performance
#     start = time.perf_counter()

#     for batch in inference_data.seq_batches(batch_size=BATCH):

#         src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
#         unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
#         total_edges += BATCH

#         n_id = torch.cat([src, pos_dst]).unique()
#         n_id, edge_index, e_id = neighbor_loader(n_id)
#         assoc[n_id] = torch.arange(n_id.size(0), device=device)

#         z, last_update = memory(n_id)
#         z = gnn(z, edge_index)

#         y_pred_src = src_recon(z[assoc[src]])
#         y_pred_dst = dst_recon(z[assoc[pos_dst]])

#         y_true_src = []
#         y_true_dst = []
#         for m in msg:
#             y_true_src.append(m[:word_embedding_dim])
#             y_true_dst.append(m[-word_embedding_dim:])
#         y_true_src = torch.stack(y_true_src).to(device)
#         y_true_dst = torch.stack(y_true_dst).to(device)

#         loss_src = criterion(y_pred_src, y_true_src)
#         loss_dst = criterion(y_pred_dst, y_true_dst)

#         loss = loss_src + loss_dst

#         total_loss += float(loss) * batch.num_events

#         # update the edges in the batch to the memory and neighbor_loader
#         memory.update_state(src, pos_dst, t, msg)
#         neighbor_loader.insert(src, pos_dst)

#         # compute the loss for each edge
#         each_edge_loss = cal_pos_edges_loss(y_pred_src, y_true_src, y_pred_dst, y_true_dst)

#         for i in range(len(y_pred_src)):
#             srcnode = int(src[i])
#             dstnode = int(pos_dst[i])

#             srcmsg = str(nodeid2msg[srcnode])
#             dstmsg = str(nodeid2msg[dstnode])
#             t_var = int(t[i])
#             edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
#             edge_type = rel2id[edgeindex]
#             loss = each_edge_loss[i]

#             temp_dic = {}
#             temp_dic['loss'] = float(loss)
#             temp_dic['srcnode'] = srcnode
#             temp_dic['dstnode'] = dstnode
#             temp_dic['srcmsg'] = srcmsg
#             temp_dic['dstmsg'] = dstmsg
#             temp_dic['edge_type'] = edge_type
#             temp_dic['time'] = t_var

#             edge_list.append(temp_dic)

#         event_count += len(batch.src)
#         if t[-1] > start_time + time_window_size: # TODO: use cfg here
#             # Here is a checkpoint, which records all edge losses in the current time window
#             time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

#             end = time.perf_counter()
#             time_with_loss[time_interval] = {'loss': loss,

#                                              'nodes_count': len(unique_nodes),
#                                              'total_edges': total_edges,
#                                              'costed_time': (end - start)}

#             log = open(path + "/" + time_interval + ".txt", 'w')

#             for e in edge_list:
#                 loss += e['loss']

#             loss = loss / event_count
#             logger.info(
#                 f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
#             edge_list = sorted(edge_list, key=lambda x: x['loss'],
#                                reverse=True)  # Rank the results based on edge losses
#             for e in edge_list:
#                 log.write(str(e))
#                 log.write("\n")
#             event_count = 0
#             total_loss = 0
#             start_time = t[-1]
#             log.close()
#             edge_list.clear()

#     return time_with_loss, memory, neighbor_loader

def main(cfg):
    logger = get_logger(
        name="gnn_testing",
        filename=os.path.join(cfg.detection.gnn_testing._logs_dir, "gnn_testing.log"))

    # load the map between nodeID and node labels
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)

    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    val_graphs_dir = os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "val")
    test_graphs_dir = os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "test")

    # For each model trained at a given epoch, we test
    all_trained_models = listdir_sorted(gnn_models_dir)

    for trained_model in all_trained_models:
        logger.info(f"Testing with model {trained_model}...")
        model = torch.load(os.path.join(gnn_models_dir, trained_model), map_location=device)
        
        # TODO: we may want to move the validation set into the training for early stopping
        for graphs_dir, split in [
            (val_graphs_dir, "val"),
            (test_graphs_dir, "test"),
        ]:
            logger.info(f"    Testing {split} set... (files into {graphs_dir})")
            filelist = os.listdir(graphs_dir)
            for file in tqdm(sorted(filelist)):
                filepath = os.path.join(graphs_dir, file)
                g = torch.load(filepath)
                g.to(device)
                test_tw(inference_data=g,
                        model=model,
                        nodeid2msg=nodeid2msg,
                        split=split,
                        model_epoch_file=trained_model,
                        logger=logger,
                        cfg=cfg,)
                del g

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
