import torch


def compute_hetero_features(g, node_map, edge_map):
    x_dict, edge_index_dict = {}, {}
    node_types = g.node_type_argmax
    edge_types = g.edge_type_argmax
    
    # 1: Map node types to indices
    unique_node_types = torch.unique(node_types)
    node_type_to_indices = {}
    
    for n_type in unique_node_types:
        mask = node_types == n_type
        indices = torch.where(mask)[0]
        
        n_type_str = node_map[n_type.item()]
        node_type_to_indices[n_type_str] = indices
        x_dict[n_type_str] = g.x[mask]
    
    # 2: Handle edge types
    src_types = g.node_type_src_argmax
    dst_types = g.node_type_dst_argmax
    
    triplets = torch.stack([src_types, edge_types, dst_types], dim=1)
    unique_triplets = torch.unique(triplets, dim=0)
    
    for triplet in unique_triplets:
        src_t, e_t, dst_t = triplet.tolist()
        src_t_str, dst_t_str, e_t_str = node_map[src_t], node_map[dst_t], edge_map[e_t]
        
        # Mask for edges matching this triplet
        mask = (src_types == src_t) & (edge_types == e_t) & (dst_types == dst_t)
        edge_indices = torch.where(mask)[0]  # Indices of matching edges
        hetero_edge_index = g.edge_index[:, edge_indices]
        
        # Remap node indices to local indices within their type groups
        src_local_indices = torch.searchsorted(node_type_to_indices[src_t_str], hetero_edge_index[0])
        dst_local_indices = torch.searchsorted(node_type_to_indices[dst_t_str], hetero_edge_index[1])
        remapped_edge_index = torch.stack([src_local_indices, dst_local_indices], dim=0)
        
        edge_index_dict[(src_t_str, e_t_str, dst_t_str)] = remapped_edge_index
    
    return x_dict, edge_index_dict
    
def hetero_to_homo_features(x_dict, node_types, node_map, device, num_nodes, out_dim):
    """Maps nodes from x_dict to original node order as x"""
    x_homo = torch.zeros(num_nodes, out_dim, device=device)

    for node_type, embeddings in x_dict.items():
        type_idx = node_map[node_type]
        mask = node_types == type_idx
        indices = torch.where(mask)[0]
        x_homo[indices] = embeddings
        
    return x_homo
    
def get_metadata(possible_events, node_map):
    """Creates the metadata tuple used by PyG's HeteroData"""
    node_types = [entity for entity in node_map if not isinstance(entity, int)]
    
    edge_triplets = []
    for (src, dst), events in possible_events.items():
        for event in events:
            edge_triplets.append((src, event, dst))
    
    return (node_types, edge_triplets)
