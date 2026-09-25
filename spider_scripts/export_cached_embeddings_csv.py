#!/usr/bin/env python3
"""
Export cached UMAP embeddings from deckgl pipeline as CSV for LaTeX/pgfplots.
Uses pre-computed UMAP coordinates — no model inference needed.
"""
import sys, os, json
sys.path.insert(0, '/home/pids')

import torch
import numpy as np

CACHE_DIR = '/home/pids/notebooks/cluster_plots/cache'
EVAL_JSON = '/home/pids/config/eval/eval_set_clustering.json'
OUT_DIR = '/home/pids/viz_scripts/figures'

DATASET_GRAPH_DIRS = {
    'CADETS_E3':       '/home/artifacts/preprocessing/CADETS_E3/build_graphs/caa79c683d91cf8cc35e12c28f0138e935069b84dcc28797bcb23d11c5217695',
    'CADETS_E5':       '/home/artifacts/preprocessing/CADETS_E5/build_graphs/7439d8ab235e9b5b0c5e9670ccbeaccb67f82eb6bf0abeef0799bea9042d7fb1',
    'CLEARSCOPE_E5':   '/home/artifacts/preprocessing/CLEARSCOPE_E5/build_graphs/537d373ee10384e57f5d42feaf5e6d98d1bed21753751e2ca2561335dd2c0ea5',
    'PROVENANCE_BENIGN': '/home/artifacts/preprocessing/PROVENANCE_BENIGN/build_graphs/4bde7dc110365008686f1e0ac91dd64ca94f420ffcc628aeb313b0ae38ae340d',
    'TRACE_E3':        '/home/artifacts/preprocessing/TRACE_E3/build_graphs/354e92a24a4aa94ceb53be366f0ccf60a23d4caa26438288d464da189c050dd3',
    'THEIA_E3':        '/home/artifacts/preprocessing/THEIA_E3/build_graphs/354e92a24a4aa94ceb53be366f0ccf60a23d4caa26438288d464da189c050dd3',
    'optc_h201':       '/home/artifacts/preprocessing/optc_h201/build_graphs/c22bd80871b9e27fccb3c878fafb5cbdf3ceed976b7e600833903c512f58bb05',
}

DS_TO_OS = {
    'CADETS_E3': 'FreeBSD', 'CADETS_E5': 'FreeBSD',
    'CLEARSCOPE_E5': 'Android',
    'PROVENANCE_BENIGN': 'Linux',
    'TRACE_E3': 'Linux', 'THEIA_E3': 'Linux',
    'optc_h201': 'Windows',
}

# ── 1. Rebuild unique_labels and label_datasets (same as deckgl script) ──
print('Loading entity metadata...')
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
    print(f'  {ds_name}: {len(unique_labels) - n_before:,} new unique labels')
    del indexid2msg

label_types = [t for t, _ in unique_labels]
print(f'Total unique labels: {len(unique_labels):,}')

# ── 2. Load cached UMAP coordinates ──
print('\nLoading cached UMAP...')
cached = np.load(os.path.join(CACHE_DIR, 'embeddings_umap.npz'), allow_pickle=True)
bg_umap = cached['bg_umap']
eval_umap = cached['eval_umap']
fit_bg_idx = cached['fit_bg_idx']
extra_bg_idx = cached['extra_bg_idx']

all_bg_idx = np.concatenate([fit_bg_idx, extra_bg_idx])
print(f'  Background: {len(bg_umap):,} points')
print(f'  Eval: {len(eval_umap):,} points')

# ── 3. Classify each background entity ──
from pidsmaker.spider.data.entity_classes import classify_entity as _classify

print('Classifying background entities...')
bg_classes = [_classify(label_types[i], unique_labels[i][1]) for i in all_bg_idx]
bg_oses = [DS_TO_OS.get(label_datasets[i], 'Unknown') for i in all_bg_idx]

class_counts = {}
for c in bg_classes:
    class_counts[c] = class_counts.get(c, 0) + 1
print(f'  {len(class_counts)} entity classes')
for c in sorted(class_counts, key=class_counts.get, reverse=True)[:15]:
    print(f'    {c:25s} {class_counts[c]:>8,}')
print(f'    ... ({len(class_counts) - 15} more)')

# ── 4. Export background CSV ──
bg_csv = os.path.join(OUT_DIR, 'cross_os_embedding_coords_large.csv')
print(f'\nExporting background to {bg_csv}...')
with open(bg_csv, 'w') as f:
    f.write('x,y,class,os\n')
    for i in range(len(bg_umap)):
        f.write(f'{bg_umap[i,0]:.4f},{bg_umap[i,1]:.4f},{bg_classes[i]},{bg_oses[i]}\n')
print(f'  {len(bg_umap):,} background points')

# ── 5. Export eval CSV ──
EVAL_TYPE_MAP = {'PROC': 'subject', 'FILE': 'file', 'SOCK': 'netflow', 'NETFLOW': 'netflow'}
with open(EVAL_JSON) as f:
    eval_data = json.load(f)

eval_csv = os.path.join(OUT_DIR, 'cross_os_embedding_eval_large.csv')
print(f'Exporting eval to {eval_csv}...')
with open(eval_csv, 'w') as f:
    f.write('x,y,class,cluster\n')
    for i, e in enumerate(eval_data):
        etype = EVAL_TYPE_MAP.get(e['entity_type'], 'file')
        cls = _classify(etype, e['entity_text'])
        f.write(f'{eval_umap[i,0]:.4f},{eval_umap[i,1]:.4f},{cls},{e["cluster"]}\n')
print(f'  {len(eval_data):,} eval points')

# ── 6. Export centroids ──
centroids = {}
for i in range(len(bg_umap)):
    c = bg_classes[i]
    if c not in centroids:
        centroids[c] = []
    centroids[c].append(bg_umap[i])

centroid_csv = os.path.join(OUT_DIR, 'cross_os_embedding_centroids_large.csv')
with open(centroid_csv, 'w') as f:
    f.write('x,y,class,count\n')
    for cls in sorted(centroids):
        arr = np.array(centroids[cls])
        f.write(f'{arr[:,0].mean():.4f},{arr[:,1].mean():.4f},{cls},{len(arr)}\n')
print(f'Exported {len(centroids)} centroids to {centroid_csv}')

print(f'\nDone. Files in {OUT_DIR}/')
