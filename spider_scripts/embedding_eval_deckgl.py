#!/usr/bin/env python3
"""
Embedding Evaluation: Deck.gl GPU-Accelerated Visualization
============================================================
Preprocesses embeddings + UMAP coordinates and exports binary data
for a Deck.gl-based HTML viewer that handles 500K+ points at 60fps.

Uses the same model/tokenizer/sampling pipeline as embedding_eval_clusters.py,
then exports optimized binary + JSON for the Deck.gl viewer.
"""

import sys, os, json, struct, time
sys.path.insert(0, '/home/pids')

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from umap import UMAP

ENTITY_TYPES = ['subject', 'file', 'netflow']  # Options: 'subject', 'file', 'netflow'
MODEL_TYPE = 'behavior_cluster'  # Options: 't5', 'gnn_distill', 'behavior_cluster'
USE_CACHE = False      # Set True to reuse cached embeddings
CACHE_DIR = '/home/pids/notebooks/cluster_plots/cache'
OUT_DIR = '/home/pids/notebooks/cluster_plots/deckgl'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = '/home/artifacts/featurization/CADETS_E3+CADETS_E5+CLEARSCOPE_E5+PROVENANCE_BENIGN+THEIA_E3+TRACE_E3+TRACE_E5+optc_h201/feat_training/1fb6d800bab33856988afdc96766f0f4e15f2b3bdaf47bc026e4dc5a5435785e/stored_models'
EVAL_JSON = '/home/pids/eval_set_behavior.json'

DATASET_GRAPH_DIRS = {
    'CADETS_E3':       '/home/artifacts/preprocessing/CADETS_E3/build_graphs/caa79c683d91cf8cc35e12c28f0138e935069b84dcc28797bcb23d11c5217695',
    'CADETS_E5':       '/home/artifacts/preprocessing/CADETS_E5/build_graphs/7439d8ab235e9b5b0c5e9670ccbeaccb67f82eb6bf0abeef0799bea9042d7fb1',
    'CLEARSCOPE_E5':   '/home/artifacts/preprocessing/CLEARSCOPE_E5/build_graphs/537d373ee10384e57f5d42feaf5e6d98d1bed21753751e2ca2561335dd2c0ea5',
    'PROVENANCE_BENIGN': '/home/artifacts/preprocessing/PROVENANCE_BENIGN/build_graphs/4bde7dc110365008686f1e0ac91dd64ca94f420ffcc628aeb313b0ae38ae340d',
    'TRACE_E3':        '/home/artifacts/preprocessing/TRACE_E3/build_graphs/354e92a24a4aa94ceb53be366f0ccf60a23d4caa26438288d464da189c050dd3',
    'THEIA_E3':        '/home/artifacts/preprocessing/THEIA_E3/build_graphs/354e92a24a4aa94ceb53be366f0ccf60a23d4caa26438288d464da189c050dd3',
    'optc_h201':       '/home/artifacts/preprocessing/optc_h201/build_graphs/c22bd80871b9e27fccb3c878fafb5cbdf3ceed976b7e600833903c512f58bb05',
}

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load entity metadata from ALL pretrain datasets
# ──────────────────────────────────────────────────────────────────────────────
unique_labels = []
label_datasets = []
label_set = set()

for ds_name, ds_dir in DATASET_GRAPH_DIRS.items():
    pkl_path = f'{ds_dir}/indexid2msg/indexid2msg.pkl'
    indexid2msg = torch.load(pkl_path, map_location='cpu', weights_only=False)
    n_before = len(unique_labels)
    for nid, (ntype, nlabel) in indexid2msg.items():
        key = (ntype, nlabel)
        if key not in label_set:
            label_set.add(key)
            unique_labels.append(key)
            label_datasets.append(ds_name)
    print(f'  {ds_name}: {len(indexid2msg):>10,} entities  '
          f'-> {len(unique_labels) - n_before:,} new unique labels')
    del indexid2msg

label_types = [t for t, _ in unique_labels]
print(f'\nTotal unique (type, label) pairs: {len(unique_labels):,}')

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load T5 model + tokenizer (only if not using cache)
# ──────────────────────────────────────────────────────────────────────────────
cache_file = os.path.join(CACHE_DIR, 'embeddings_umap.npz')

if USE_CACHE and os.path.exists(cache_file):
    print(f'\nLoading cached embeddings + UMAP from {cache_file}')
    cached = np.load(cache_file, allow_pickle=True)
    bg_umap = cached['bg_umap']
    eval_umap = cached['eval_umap']
    fit_bg_idx = cached['fit_bg_idx']
    extra_bg_idx = cached['extra_bg_idx']
    eval_embeddings = cached['eval_embeddings']
    print(f'  Loaded: {len(bg_umap):,} background + {len(eval_umap)} eval UMAP coords')

    # Filter cached background points by ENTITY_TYPES
    all_bg_idx_cached = np.concatenate([fit_bg_idx, extra_bg_idx])
    type_mask = np.array([label_types[i] in ENTITY_TYPES for i in all_bg_idx_cached])
    bg_umap = bg_umap[type_mask]
    # Rebuild fit/extra idx consistently
    n_fit_orig = len(fit_bg_idx)
    fit_mask = type_mask[:n_fit_orig]
    extra_mask = type_mask[n_fit_orig:]
    fit_bg_idx = fit_bg_idx[fit_mask]
    extra_bg_idx = extra_bg_idx[extra_mask]
    print(f'  After ENTITY_TYPES filter: {len(bg_umap):,} background points')
else:
    # Full encode pipeline
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE

    tokenizer = object.__new__(ProvenanceTokenizerBPE)
    tokenizer.expand_netflow_ips = False
    tokenizer.mask_rate = 0.15
    tokenizer.load(f'{MODEL_DIR}/tokenizer.pt')

    model_state = torch.load(f'{MODEL_DIR}/pretrain_mini_best.pt', map_location='cpu',
                             weights_only=False)

    if MODEL_TYPE == 'behavior_cluster':
        from pidsmaker.spider.models.behavior import ProvenanceBehaviorModel, get_behavior_encoder_config
        from pidsmaker.spider.data.behavior_signatures import BehaviorLabelVocab
        config = get_behavior_encoder_config(tokenizer.vocab_size, 'mini')
        bvocab = BehaviorLabelVocab()
        bvocab_path = f'{MODEL_DIR}/behavior_vocab.txt'
        if os.path.exists(bvocab_path):
            bvocab.load(bvocab_path)
        model = ProvenanceBehaviorModel(config, num_labels=max(bvocab.size, 1), proj_dim=256)
    elif MODEL_TYPE == 'gnn_distill':
        from pidsmaker.spider.models.gnn_distill import ProvenanceGNNDistill, get_gnn_distill_encoder_config
        config = get_gnn_distill_encoder_config(tokenizer.vocab_size, 'mini')
        model = ProvenanceGNNDistill(config, 128)
    elif MODEL_TYPE == 'spider':
        from pidsmaker.spider.models.spider import ProvenanceGNNCluster, get_spider_encoder_config
        config = get_spider_encoder_config(tokenizer.vocab_size, 'mini')
        model = ProvenanceGNNCluster(config, 256)
    else:  # t5
        from pidsmaker.spider.models.t5 import ProvenanceT5, get_t5_config
        config = get_t5_config(tokenizer.vocab_size, 'mini')
        model = ProvenanceT5(config)

    model.load_state_dict(model_state, strict=False)
    model = model.to(DEVICE).eval()

    BATCH_SIZE = 512

    def encode_entities(node_types, labels):
        all_token_ids = []
        for ntype, label in zip(node_types, labels):
            ids = tokenizer.tokenize_node(ntype, label)
            if not ids:
                ids = [tokenizer.pad_id]
            all_token_ids.append(ids)

        n_total = len(all_token_ids)
        all_embeddings = []
        with torch.no_grad():
            for batch_start in range(0, n_total, BATCH_SIZE):
                if batch_start % (BATCH_SIZE * 100) == 0 and n_total > 10000:
                    print(f'  encoding {batch_start:,}/{n_total:,} ...')
                batch = all_token_ids[batch_start:batch_start + BATCH_SIZE]
                max_len = min(max(len(t) for t in batch), tokenizer.max_seq_len)
                input_ids = torch.full((len(batch), max_len), tokenizer.pad_id,
                                       dtype=torch.long)
                attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
                for i, tids in enumerate(batch):
                    seq_len = min(len(tids), max_len)
                    input_ids[i, :seq_len] = torch.tensor(tids[:seq_len])
                    attention_mask[i, :seq_len] = True
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                hidden_states = model.modified_fwd(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=torch.full_like(input_ids, -100), skip_cls=True)
                mask_expanded = attention_mask.unsqueeze(-1).float()
                mean_pooled = ((hidden_states * mask_expanded).sum(dim=1) /
                               mask_expanded.sum(dim=1).clamp(min=1)).cpu().numpy()
                all_embeddings.append(mean_pooled)

        emb = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return emb / norms

    # Load eval set
    with open(EVAL_JSON) as f:
        eval_data_raw = json.load(f)
    EVAL_TYPE_MAP = {'PROC': 'subject', 'FILE': 'file', 'SOCK': 'netflow', 'NETFLOW': 'netflow'}
    eval_data_raw = [e for e in eval_data_raw
                     if EVAL_TYPE_MAP.get(e['entity_type']) in ENTITY_TYPES]
    eval_node_types = [EVAL_TYPE_MAP[e['entity_type']] for e in eval_data_raw]
    eval_labels = [e['entity_text'] for e in eval_data_raw]

    # Sample background — dedup by token sequence first (like training),
    # then cap per class for balance. This removes the socket flood
    # (679K port_reg → ~59 unique tokens) and ensures every class is
    # well-represented in the UMAP fit.
    MAX_PER_CLASS = 200
    N_FIT_RATIO = 0.35  # fraction of deduped set used for UMAP fit
    rng = np.random.RandomState(42)

    label_datasets_arr = np.array(label_datasets)
    label_types_arr = np.array(label_types)
    ds_names_unique = list(DATASET_GRAPH_DIRS.keys())

    # Cache the balanced sample to avoid re-tokenizing every run
    sample_cache = os.path.join(CACHE_DIR, 'balanced_sample.npz')
    if os.path.exists(sample_cache):
        print(f'  Loading cached balanced sample from {sample_cache}')
        cached_sample = np.load(sample_cache)
        fit_bg_idx = cached_sample['fit_bg_idx']
        extra_bg_idx = cached_sample['extra_bg_idx']
        # Filter by active ENTITY_TYPES
        all_bg_idx = np.concatenate([fit_bg_idx, extra_bg_idx])
        type_mask = np.array([label_types[i] in ENTITY_TYPES for i in all_bg_idx])
        n_fit_orig = len(fit_bg_idx)
        fit_bg_idx = fit_bg_idx[type_mask[:n_fit_orig]]
        extra_bg_idx = extra_bg_idx[type_mask[n_fit_orig:]]
        all_bg_idx = np.concatenate([fit_bg_idx, extra_bg_idx])
        print(f'  Fit: {len(fit_bg_idx):,}, Extra: {len(extra_bg_idx):,}, '
              f'Total: {len(all_bg_idx):,}')
    else:
        # Build index of valid entities (matching active entity types)
        valid_mask = np.zeros(len(label_types), dtype=bool)
        for etype in ENTITY_TYPES:
            valid_mask |= (label_types_arr == etype)
        valid_idx = np.where(valid_mask)[0]
        print(f'  Valid entities before dedup: {len(valid_idx):,}')

        # Dedup: keep one index per unique token sequence
        token_sig_to_idx = {}
        for pos in valid_idx:
            ntype, nlabel = unique_labels[pos]
            tids = tokenizer.tokenize_node(ntype, nlabel)
            if not tids:
                continue
            sig = tuple(tids)
            if sig not in token_sig_to_idx:
                token_sig_to_idx[sig] = pos
        deduped_idx = np.array(list(token_sig_to_idx.values()))
        print(f'  After token dedup: {len(deduped_idx):,}')

        # Classify and cap per class
        from pidsmaker.spider.data.entity_classes import classify_entity as _classify
        deduped_classes = np.array([_classify(label_types[i], unique_labels[i][1])
                                    for i in deduped_idx])

        capped_idx = []
        for cls in np.unique(deduped_classes):
            cls_positions = np.where(deduped_classes == cls)[0]
            rng.shuffle(cls_positions)
            capped_idx.extend(cls_positions[:MAX_PER_CLASS].tolist())
        rng.shuffle(capped_idx)
        capped_idx = np.array(capped_idx)
        bg_global_idx = deduped_idx[capped_idx]  # indices into unique_labels
        print(f'  After {MAX_PER_CLASS}/class cap: {len(bg_global_idx):,} '
              f'({len(np.unique(deduped_classes[capped_idx]))} classes)')

        # Split into fit / extra
        n_fit_sample = min(int(len(bg_global_idx) * N_FIT_RATIO), len(bg_global_idx))
        fit_bg_idx = bg_global_idx[:n_fit_sample]
        extra_bg_idx = bg_global_idx[n_fit_sample:]
        all_bg_idx = np.concatenate([fit_bg_idx, extra_bg_idx])
        print(f'  Fit: {len(fit_bg_idx):,}, Extra: {len(extra_bg_idx):,}, '
              f'Total: {len(all_bg_idx):,}')

        # Save for next run
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(sample_cache, fit_bg_idx=fit_bg_idx, extra_bg_idx=extra_bg_idx)
        print(f'  Saved balanced sample to {sample_cache}')

    # Encode
    print(f'\nEncoding {len(eval_data_raw)} eval entities...')
    eval_embeddings = encode_entities(eval_node_types, eval_labels)

    sampled_types = [label_types[i] for i in all_bg_idx]
    sampled_labels = [unique_labels[i][1] for i in all_bg_idx]
    print(f'Encoding {len(all_bg_idx):,} background entities...')
    bg_embeddings = encode_entities(sampled_types, sampled_labels)

    n_fit = len(fit_bg_idx)
    fit_bg_emb = bg_embeddings[:n_fit]
    extra_bg_emb = bg_embeddings[n_fit:]

    fit_emb = np.vstack([fit_bg_emb, eval_embeddings])
    n_fit_bg = len(fit_bg_emb)

    print('Fitting UMAP...')
    t0 = time.time()
    reducer = UMAP(n_neighbors=50, min_dist=0.3, metric='cosine', random_state=42)
    fit_umap = reducer.fit_transform(fit_emb.astype(np.float32))
    fit_bg_umap = np.asarray(fit_umap[:n_fit_bg])
    eval_umap = np.asarray(fit_umap[n_fit_bg:])
    print(f'Fit done ({time.time()-t0:.1f}s)')

    extra_bg_umap = np.asarray(reducer.transform(extra_bg_emb.astype(np.float32)))
    bg_umap = np.vstack([fit_bg_umap, extra_bg_umap])

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_file, bg_umap=bg_umap, eval_umap=eval_umap,
             fit_bg_idx=fit_bg_idx, extra_bg_idx=extra_bg_idx,
             eval_embeddings=eval_embeddings)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Load eval data for metadata
# ──────────────────────────────────────────────────────────────────────────────
with open(EVAL_JSON) as f:
    eval_data = json.load(f)
EVAL_TYPE_MAP_META = {'PROC': 'subject', 'FILE': 'file', 'SOCK': 'netflow', 'NETFLOW': 'netflow'}
eval_data = [e for e in eval_data
             if EVAL_TYPE_MAP_META.get(e['entity_type']) in ENTITY_TYPES]

# Auto-derive meta-clusters from class name prefixes.
# e.g. port_http, port_dns → "port"; config_cron, config_net → "config"; user_cache → "user"
cluster_names = sorted(set(e['cluster'] for e in eval_data))

def _auto_meta(cls_name):
    """Derive a visual meta-cluster from a class name."""
    for prefix in ('port_', 'config_', 'user_', 'data_', 'log_', 'browser_'):
        if cls_name.startswith(prefix):
            return prefix.rstrip('_')
    return cls_name
cluster_to_meta = {c: _auto_meta(c) for c in cluster_names}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Export binary data for Deck.gl viewer
# ──────────────────────────────────────────────────────────────────────────────
all_bg_idx = np.concatenate([fit_bg_idx, extra_bg_idx])
fit_bg_types = [label_types[i] for i in fit_bg_idx]
extra_bg_types = [label_types[i] for i in extra_bg_idx]
bg_all_types = fit_bg_types + extra_bg_types

# Type encoding: subject=0, file=1, netflow=2
TYPE_TO_ID = {'subject': 0, 'file': 1, 'netflow': 2}
EVAL_TYPE_TO_ID = {'PROC': 0, 'FILE': 1, 'SOCK': 2, 'NETFLOW': 2}

# Dataset encoding
ds_names_unique = list(DATASET_GRAPH_DIRS.keys())
DS_TO_ID = {name: i for i, name in enumerate(ds_names_unique)}

# Classify every background entity by entity class
from pidsmaker.spider.data.entity_classes import classify_entity as _classify
print('\nClassifying background entities...')
bg_all_classes = [_classify(bg_all_types[i], unique_labels[all_bg_idx[i]][1])
                  for i in range(len(all_bg_idx))]
class_names_sorted = sorted(set(bg_all_classes))
CLASS_TO_ID = {c: i for i, c in enumerate(class_names_sorted)}
bg_class_ids = [CLASS_TO_ID[c] for c in bg_all_classes]
print(f'  {len(class_names_sorted)} entity classes found')

# ── Export background points as binary (x, y, type_id, dataset_id, class_id) ──
# Each point: 2 float32 (x,y) + 1 uint8 (type) + 1 uint8 (dataset) + 1 uint8 (class) = 11 bytes
print(f'\nExporting {len(bg_umap):,} background points as binary...')
bg_bin_path = os.path.join(OUT_DIR, 'bg_points.bin')
with open(bg_bin_path, 'wb') as f:
    for i in range(len(bg_umap)):
        x, y = float(bg_umap[i, 0]), float(bg_umap[i, 1])
        type_id = TYPE_TO_ID.get(bg_all_types[i], 0)
        ds_id = DS_TO_ID.get(label_datasets[all_bg_idx[i]], 0)
        cls_id = bg_class_ids[i]
        f.write(struct.pack('<ff BBB', x, y, type_id, ds_id, cls_id))

print(f'  {bg_bin_path} ({os.path.getsize(bg_bin_path) / 1e6:.1f} MB)')

# ── Export background labels in chunks for lazy loading ──
CHUNK_SIZE = 10000
bg_all_labels = [unique_labels[i][1] for i in all_bg_idx]
bg_all_ds = [label_datasets[i] for i in all_bg_idx]
n_chunks = (len(bg_all_labels) + CHUNK_SIZE - 1) // CHUNK_SIZE

chunks_dir = os.path.join(OUT_DIR, 'label_chunks')
os.makedirs(chunks_dir, exist_ok=True)
print(f'Exporting {n_chunks} label chunks...')
for ci in range(n_chunks):
    start = ci * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(bg_all_labels))
    chunk_data = []
    for j in range(start, end):
        chunk_data.append(bg_all_labels[j])
    with open(os.path.join(chunks_dir, f'{ci}.json'), 'w') as f:
        json.dump(chunk_data, f, separators=(',', ':'))

# ── Export eval entities as JSON ──
eval_export = []
meta_names = sorted(set(cluster_to_meta.get(e['cluster'], e['cluster']) for e in eval_data))
META_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9', '#ffe119', '#e6beff', '#ff6961',
    '#77dd77', '#0095b6',
]
meta_color_map = {m: META_PALETTE[i % len(META_PALETTE)] for i, m in enumerate(meta_names)}

for i, e in enumerate(eval_data):
    meta = e.get('meta') or cluster_to_meta.get(e['cluster'], e['cluster'])
    seen = e.get('seen_during_training', True)
    eval_export.append({
        'x': float(eval_umap[i, 0]),
        'y': float(eval_umap[i, 1]),
        'text': e['entity_text'],
        'type': e['entity_type'],
        'cluster': e['cluster'],
        'sub_cluster': e.get('sub_cluster', ''),
        'meta': meta,
        'color': meta_color_map.get(meta, '#888888'),
        'seen_during_training': seen,
        'marker': 'circle' if seen else 'diamond',
    })

# ── Generate class color palette ──
# Distinct colors for up to ~70 classes via golden-angle hue spacing
import colorsys

def _class_palette(n):
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # golden angle
        # Convert HSL(hue, 65%, 55%) to RGB
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.55, 0.65)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors

class_colors = _class_palette(len(class_names_sorted))
class_color_map = {c: class_colors[i] for i, c in enumerate(class_names_sorted)}

# ── Export metadata JSON ──
metadata = {
    'n_background': len(bg_umap),
    'n_eval': len(eval_data),
    'n_chunks': n_chunks,
    'chunk_size': CHUNK_SIZE,
    'bytes_per_point': 11,
    'datasets': ds_names_unique,
    'entity_types': ENTITY_TYPES,
    'eval_entities': eval_export,
    'meta_clusters': {m: meta_color_map[m] for m in meta_names},
    'entity_classes': class_names_sorted,
    'entity_class_colors': class_color_map,
    'bg_type_colors': {
        'subject': [255, 153, 153],  # RGB for deck.gl
        'file':    [153, 204, 255],
        'netflow': [153, 230, 153],
    },
    'bg_type_display': {'subject': 'Process', 'file': 'File', 'netflow': 'Socket'},
}

meta_path = os.path.join(OUT_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'  {meta_path}')

print(f'\nDeck.gl data exported to {OUT_DIR}/')
print(f'  bg_points.bin  — {len(bg_umap):,} points ({os.path.getsize(bg_bin_path)/1e6:.1f} MB)')
print(f'  metadata.json  — config + {len(eval_export)} eval entities')
print(f'  label_chunks/  — {n_chunks} chunks for lazy hover text')
print(f'\nNext: open deckgl_viewer.html via a local HTTP server:')
print(f'  cd {OUT_DIR} && python -m http.server 8080')
