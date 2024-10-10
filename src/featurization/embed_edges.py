from config import *
from provnet_utils import *

from .embed_edges_methods import (
    embed_edges_word2vec,
    embed_edges_doc2vec,
    embed_edges_HFH,
    embed_edges_feature_word2vec,
    embed_edges_TRW,
    provd_embed_paths,
)

def embed_edges(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg):

    for path in tqdm(sorted_paths, desc="Computing edge embeddings"):
        graph = torch.load(path)
        sorted_edges = sorted(graph.edges(data=True, keys=True), key=lambda t: t[3]["time"])

        src, dst, msg, t = [], [], [], []
        for u, v, k, attr in sorted_edges:
            src.append(int(u))
            dst.append(int(v))
            t.append(int(attr["time"]))

            # Only types
            if indexid2vec is None:
                msg.append(torch.cat([
                    ntype2oh[graph.nodes[u]['node_type']],
                    etype2oh[attr["label"]],
                    ntype2oh[graph.nodes[v]['node_type']],
                ]))
                
            # Types + node embeddings
            else:
                msg.append(torch.cat([
                    ntype2oh[graph.nodes[u]['node_type']],
                    torch.from_numpy(indexid2vec[u]),
                    etype2oh[attr["label"]],
                    ntype2oh[graph.nodes[v]['node_type']],
                    torch.from_numpy(indexid2vec[v])
                ]))

        data = TemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float),
        )

        os.makedirs(out_dir, exist_ok=True)
        file = path.split("/")[-1]
        torch.save(data, os.path.join(out_dir, f"{file}.TemporalData.simple"))

def get_indexid2vec(cfg):
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "provd":
        provd_embed_paths.main(cfg)
    if method == "word2vec":
        return embed_edges_word2vec.main(cfg)
    if method == "doc2vec":
        return embed_edges_doc2vec.main(cfg)
    if method == "hierarchical_hashing":
        return embed_edges_HFH.main(cfg)
    if method == "feature_word2vec":
        return embed_edges_feature_word2vec.main(cfg)
    if method == "only_type":
        return None
    if method == "temporal_rw":
        return embed_edges_TRW.main(cfg)
    if method == "flash" or method == 'magic':
        raise EnvironmentError("TODO (see with Baoxiang)")
    
    raise ValueError(f"Invalid node embedding method {method}")

def main(cfg):
    rel2id = get_rel2id(cfg)
    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)
    
    base_dir = cfg.preprocessing.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)
    
    # Here we get a mapping {node_id => embedding vector}
    indexid2vec = get_indexid2vec(cfg)
    
    # Create edges for Train, Val, Test sets
    for split, sorted_paths in split_to_files.items():
        embed_edges(
            indexid2vec=indexid2vec,
            etype2oh=etype2onehot,
            ntype2oh=ntype2onehot,
            sorted_paths=sorted_paths,
            out_dir=os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, f"{split}/"),
            cfg=cfg
        )


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)