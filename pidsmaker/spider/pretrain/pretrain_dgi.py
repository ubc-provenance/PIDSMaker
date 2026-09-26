"""DGI pretraining branch for SPIDER."""

import os
import random
import time

import torch

from pidsmaker.utils.utils import log
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer


def pretrain_dgi(cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device, batch_size,
                 model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                 normalize_netflow_ips, canonicalize_neighbors, spider_path):
    from ..models.dgi import DGI, save_dgi
    from ..models.graphmae import T5NodeEncoder, neighborhoods_to_pyg_batch
    from ..models.gnn_distill import get_gnn_distill_encoder_config

    emb_dim = cfg.featurization.feat_training.emb_dim
    dgi_cfg = spider_cfg.dgi
    num_layers = dgi_cfg.num_layers
    num_heads = dgi_cfg.num_heads
    n_min = dgi_cfg.neighborhood_min
    n_max = dgi_cfg.neighborhood_max
    epochs = dgi_cfg.epochs
    lr = dgi_cfg.lr

    # ── Build or load tokenizer ────────────────────────────────────
    if spider_path:
        tok_path = os.path.join(spider_path, "tokenizer.pt")
        log(f"Loading tokenizer from {tok_path}")
        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(tok_path)
        tokenizer.max_seq_len = max_seq_len
    else:
        walk_label_corpus = []
        for sampler, ds_indexid2msg in sampler_pairs:
            for node in sampler.nodes:
                if node in ds_indexid2msg:
                    walk_label_corpus.append(ds_indexid2msg[node])

        tokenizer = ProvenanceTokenizer(cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size, mode=tokenizer_mode,
                                        normalize_netflow_ips=normalize_netflow_ips,
                                        canonicalize_neighbors=canonicalize_neighbors)
        tokenizer.build_vocab(combined_indexid2msg, walk_label_corpus=walk_label_corpus)
    log(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))

    # ── Build T5 node encoder ──────────────────────────────────────
    t5_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
    t5_hidden_dim = t5_config.d_model
    t5_encoder = T5NodeEncoder(t5_config).to(device)
    log(f"T5 node encoder: d_model={t5_hidden_dim}, layers={t5_config.num_layers}")

    # Use T5 hidden dim as GNN input dim
    in_dim = t5_hidden_dim

    # Build model
    model = DGI(
        in_dim=in_dim,
        hidden_dim=emb_dim,
        num_encoder_layers=num_layers,
        num_heads=num_heads,
    ).to(device)
    log(f"DGI model: {num_layers} encoder layers, {num_heads} heads")
    log(f"Parameters: GNN={sum(p.numel() for p in model.parameters()):,}, "
        f"T5={sum(p.numel() for p in t5_encoder.parameters()):,}")

    opt = torch.optim.Adam(
        list(model.parameters()) + list(t5_encoder.parameters()),
        lr=lr,
    )

    # Collect all nodes across all samplers
    all_nodes_per_sampler = []
    for sampler, ds_indexid2msg in sampler_pairs:
        nodes_with_backward = [n for n in sampler.nodes if n in sampler.backward_adj]
        all_nodes_per_sampler.append((sampler, ds_indexid2msg, nodes_with_backward))
    total_nodes = sum(len(ns) for _, _, ns in all_nodes_per_sampler)
    log(f"DGI: {total_nodes:,} nodes with backward neighbors across {len(sampler_pairs)} graph(s)")

    # Training loop
    log_path = os.path.join(out_dir, "dgi_pretrain_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,loss,time\n")

    for epoch in range(1, epochs + 1):
        epoch_st = time.time()
        model.train()
        t5_encoder.eval()  # T5 encoder in eval mode (used as frozen featurizer per batch)
        epoch_loss = 0.0
        n_batches = 0
        label_cache = {}  # Reset label cache each epoch

        # Shuffle nodes each epoch
        for sampler, ds_indexid2msg, nodes in all_nodes_per_sampler:
            random.shuffle(nodes)

            # Sample neighborhoods and batch them
            batch_neighborhoods = []
            for node in nodes:
                nh = sampler.sample_temporal_neighborhood(node, n_min=n_min, n_max=n_max)
                if nh is not None:
                    batch_neighborhoods.append(nh)

                if len(batch_neighborhoods) >= batch_size:
                    x, edge_index, batch_idx, node_ids = neighborhoods_to_pyg_batch(
                        batch_neighborhoods, ds_indexid2msg, t5_encoder, tokenizer, label_cache, device,
                    )
                    if x is not None and edge_index.size(1) > 0:
                        loss = model(x, edge_index)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                    batch_neighborhoods = []

            # Remaining neighborhoods
            if batch_neighborhoods:
                x, edge_index, batch_idx, node_ids = neighborhoods_to_pyg_batch(
                    batch_neighborhoods, ds_indexid2msg, t5_encoder, tokenizer, label_cache, device,
                )
                if x is not None and edge_index.size(1) > 0:
                    loss = model(x, edge_index)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_st

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{elapsed:.1f}\n")

        log(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")
        if epoch % 10 == 0 or epoch == epochs:
            save_dgi(model, t5_encoder, out_dir)

    log(f"DGI pretraining complete. Final loss: {avg_loss:.4f}")
