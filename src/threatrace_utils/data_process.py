from provnet_utils import *
from config import *

import torch
from torch_geometric.data import Data, InMemoryDataset

class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass
    def _process(self):
        pass

def load_train_graph(path):
    nx_graph = torch.load(path)

    edge_feature_map = {}
    for k, v in rel2id.items():
        if isinstance(k, str):
            edge_feature_map[k] = v - 1

    node_label_map = {}
    for k, v in ntype2id.items():
        if isinstance(k, str):
            node_label_map[k] = v - 1

    node_to_id = {}
    node_to_type = {}
    node_list = []
    nid = 0
    for node, data in nx_graph.nodes(data=True):
        node_to_id[node] = nid
        node_to_type[node] = data['node_type']
        node_list.append(node)
        nid += 1

    edge_s = []
    edge_e = []
    provenance = []
    for u, v, k, data in nx_graph.edges(keys=True, data=True):
        src_nid = node_to_id[u]
        src_type = node_label_map[node_to_type[u]]
        dst_nid = node_to_id[v]
        dst_type = node_label_map[node_to_type[v]]
        edge_type = edge_feature_map[data['label']]
        timestamp = str(data['time'])
        temp = [src_nid, src_type, dst_nid, dst_type, edge_type, timestamp]

        edge_s.append(src_nid)
        edge_e.append(dst_nid)
        provenance.append(temp)

    edge_feature_num = len(edge_feature_map.items())
    node_label_num = len(node_label_map.items())

    x_list = []
    y_list = []

    for i in range(len(node_list)):
        x_list.append([0]*edge_feature_num*2)
        y_list.append(0)

    for temp in provenance:
        srcId = temp[0]
        srcType = temp[1]
        dstId = temp[2]
        dstType = temp[3]
        edge = temp[4]

        x_list[srcId][edge] += 1
        y_list[srcId] = srcType
        x_list[dstId][edge + edge_feature_num] +=1
        y_list[dstId] = dstType

    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data1 = Data(x=x, y=y, edge_index=edge_index)

    return data1, edge_feature_num * 2, node_label_num, 0, 0, node_list
