"""GraphMAE pretraining branch for SPIDER."""

import os
import random
import time

import torch

from pidsmaker.utils.utils import log
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer


def pretrain_graphmae(cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device, batch_size,
                      model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                      normalize_netflow_ips, canonicalize_neighbors, spider_path):
    from ..models.graphmae import (
        GraphMAE, T5NodeEncoder, neighborhoods_to_pyg_batch, save_graphmae,
    )
    from ..models.gnn_distill import get_gnn_distill_encoder_config

    emb_dim = cfg.featurization.emb_dim
    gm_cfg = spider_cfg.graphmae
    gm_num_layers = gm_cfg.num_layers
    gm_num_heads = gm_cfg.num_heads
    gm_decoder_layers = gm_cfg.decoder_num_layers
    gm_mask_rate = gm_cfg.mask_rate
    gm_replace_rate = gm_cfg.replace_rate
    gm_n_min = gm_cfg.neighborhood_min
    gm_n_max = gm_cfg.neighborhood_max
    gm_epochs = gm_cfg.epochs
    gm_lr = gm_cfg.lr

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
    model = GraphMAE(
        in_dim=in_dim,
        hidden_dim=emb_dim,
        num_encoder_layers=gm_num_layers,
        num_decoder_layers=gm_decoder_layers,
        num_heads=gm_num_heads,
        mask_rate=gm_mask_rate,
        replace_rate=gm_replace_rate,
    ).to(device)
    log(f"GraphMAE model: {gm_num_layers} encoder layers, {gm_decoder_layers} decoder layers, "
        f"{gm_num_heads} heads, mask={gm_mask_rate}, replace={gm_replace_rate}")
    log(f"Parameters: GNN={sum(p.numel() for p in model.parameters()):,}, "
        f"T5={sum(p.numel() for p in t5_encoder.parameters()):,}")

    opt = torch.optim.Adam(
        list(model.parameters()) + list(t5_encoder.parameters()),
        lr=gm_lr,
    )

    # Collect all nodes across all samplers
    all_nodes_per_sampler = []
    for sampler, ds_indexid2msg in sampler_pairs:
        nodes_with_backward = [n for n in sampler.nodes if n in sampler.backward_adj]
        all_nodes_per_sampler.append((sampler, ds_indexid2msg, nodes_with_backward))
    total_nodes = sum(len(ns) for _, _, ns in all_nodes_per_sampler)
    log(f"GraphMAE: {total_nodes:,} nodes with backward neighbors across {len(sampler_pairs)} graph(s)")

    # Training loop
    log_path = os.path.join(out_dir, "graphmae_pretrain_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,loss,time\n")

    for epoch in range(1, gm_epochs + 1):
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
                nh = sampler.sample_temporal_neighborhood(node, n_min=gm_n_min, n_max=gm_n_max)
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

        log(f"Epoch {epoch}/{gm_epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")
        if epoch % 10 == 0 or epoch == gm_epochs:
            save_graphmae(model, t5_encoder, out_dir)

    log(f"GraphMAE pretraining complete. Final loss: {avg_loss:.4f}")
