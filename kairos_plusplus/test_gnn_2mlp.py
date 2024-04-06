##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################
import argparse
import logging

from provnet_utils import *
from config import *
from model import *

# Setting for logging
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'reconstruction.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def cal_pos_edges_loss(y_pred_src, y_true_src, y_pred_dst, y_true_dst):
    loss = []
    for i in range(len(y_pred_src)):
        src_loss = criterion(y_pred_src[i], y_true_src[i])
        dst_loss = criterion(y_pred_dst[i], y_true_dst[i])

        loss.append(src_loss + dst_loss)
    return torch.tensor(loss)

@torch.no_grad()
def test_tw(inference_data,
            memory,
            gnn,
            src_recon,
            dst_recon,
            neighbor_loader,
            nodeid2msg,
            path
            ):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    memory.eval()
    gnn.eval()
    src_recon.eval()
    dst_recon.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    start_time = inference_data.t[0]
    event_count = 0

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, edge_index)

        y_pred_src = src_recon(z[assoc[src]])
        y_pred_dst = dst_recon(z[assoc[pos_dst]])

        y_true_src = []
        y_true_dst = []
        for m in msg:
            y_true_src.append(m[:word_embedding_dim])
            y_true_dst.append(m[-word_embedding_dim:])
        y_true_src = torch.stack(y_true_src).to(device)
        y_true_dst = torch.stack(y_true_dst).to(device)

        loss_src = criterion(y_pred_src, y_true_src)
        loss_dst = criterion(y_pred_dst, y_true_dst)

        loss = loss_src + loss_dst

        total_loss += float(loss) * batch.num_events

        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        each_edge_loss = cal_pos_edges_loss(y_pred_src, y_true_src, y_pred_dst, y_true_dst)

        for i in range(len(y_pred_src)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

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

    log = open(path + "/" + time_interval + ".txt", 'w')

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

    return time_with_loss, memory, neighbor_loader


@torch.no_grad()
def test(inference_data,
         memory,
         gnn,
         src_recon,
         dst_recon,
         neighbor_loader,
         nodeid2msg,
         path
         ):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    memory.eval()
    gnn.eval()
    src_recon.eval()
    dst_recon.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0

    start_time = inference_data.t[0]
    event_count = 0

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, edge_index)

        y_pred_src = src_recon(z[assoc[src]])
        y_pred_dst = dst_recon(z[assoc[pos_dst]])

        y_true_src = []
        y_true_dst = []
        for m in msg:
            y_true_src.append(m[:word_embedding_dim])
            y_true_dst.append(m[-word_embedding_dim:])
        y_true_src = torch.stack(y_true_src).to(device)
        y_true_dst = torch.stack(y_true_dst).to(device)

        loss_src = criterion(y_pred_src, y_true_src)
        loss_dst = criterion(y_pred_dst, y_true_dst)

        loss = loss_src + loss_dst

        total_loss += float(loss) * batch.num_events

        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        each_edge_loss = cal_pos_edges_loss(y_pred_src, y_true_src, y_pred_dst, y_true_dst)

        for i in range(len(y_pred_src)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

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
        if t[-1] > start_time + time_window_size:
            # Here is a checkpoint, which records all edge losses in the current time window
            time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval] = {'loss': loss,

                                             'nodes_count': len(unique_nodes),
                                             'total_edges': total_edges,
                                             'costed_time': (end - start)}

            log = open(path + "/" + time_interval + ".txt", 'w')

            for e in edge_list:
                loss += e['loss']

            loss = loss / event_count
            logger.info(
                f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x: x['loss'],
                               reverse=True)  # Rank the results based on edge losses
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    return time_with_loss, memory, neighbor_loader

if __name__ == "__main__":
    logger.info("Start logging.")


    # load the map between nodeID and node labels
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # load trained model
    memory, gnn, neighbor_loader, src_recon, dst_recon = torch.load(f"{gnn_models_dir}/gnn_2mlp_models_epoch50.pt",map_location=device)

    # test the datasets
    # May 11
    graph_5_11 = torch.load(f'{vec_graphs_dir}/val/e5-graph-11-00_00_00.TemporalData.simple').to(device='cpu')
    graph_5_11.to(device)
    _, memory, neighbor_loader = test(inference_data=graph_5_11,
                                      memory=memory,
                                      gnn=gnn,
                                      src_recon=src_recon,
                                      dst_recon=dst_recon,
                                      neighbor_loader=neighbor_loader,
                                      nodeid2msg=nodeid2msg,
                                      path=artifact_dir + "graph_5_11")
    del graph_5_11
    torch.cuda.empty_cache()

    # May 14
    filelist = os.listdir(f"{vec_graphs_dir}/test/graph_5_14/")
    for file in sorted(filelist):
        filepath = f"{vec_graphs_dir}/test/graph_5_14/" + file
        g = torch.load(filepath)
        g.to(device)
        _, memory, neighbor_loader = test_tw(inference_data=g,
                                             memory=memory,
                                             gnn=gnn,
                                             src_recon=src_recon,
                                             dst_recon=dst_recon,
                                             neighbor_loader=neighbor_loader,
                                             nodeid2msg=nodeid2msg,
                                             path=artifact_dir + "graph_5_14")
        del g

    # May 15
    filelist = os.listdir(f"{vec_graphs_dir}/test/graph_5_15/")
    for file in sorted(filelist):
        filepath = f"{vec_graphs_dir}/test/graph_5_15/" + file
        g = torch.load(filepath)
        g.to(device)
        _, memory, neighbor_loader = test_tw(inference_data=g,
                                             memory=memory,
                                             gnn=gnn,
                                             src_recon=src_recon,
                                             dst_recon=dst_recon,
                                             neighbor_loader=neighbor_loader,
                                             nodeid2msg=nodeid2msg,
                                             path=artifact_dir + "graph_5_15")
        del g