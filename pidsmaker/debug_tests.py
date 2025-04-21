import torch


def debug_test_batching(original_data_list, batched_data):
    """Ensures that graph batching works correctly by comparing the features and
    edge indices of the batched data to the original data."""
    node_offset, edge_offset = 0, 0
    for i, original_data in enumerate(original_data_list):
        orig_n_id = original_data.n_id_tgn
        orig_x = original_data.x_tgn
        num_nodes = orig_n_id.size(0)

        # Check if features match
        batched_x_slice = batched_data.x_tgn[node_offset : node_offset + num_nodes]
        assert torch.allclose(batched_x_slice, orig_x), f"Feature mismatch in graph {i}"

        # Check if edge indices are correctly offset
        for attr in ["edge_index_tgn", "edge_type_tgn"]:
            orig = getattr(original_data, attr)
            batched = getattr(batched_data, attr)
            if "edge_index" in attr:
                batched_slice = batched[:, edge_offset : edge_offset + orig.size(1)]
                orig = orig + node_offset
            else:
                batched_slice = batched[edge_offset : edge_offset + orig.size(0)]

            assert torch.all(batched_slice == orig), f"Edge index mismatch in graph {i}"

        node_offset += num_nodes
        edge_offset += original_data.edge_index_tgn.size(1)
