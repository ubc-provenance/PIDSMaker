# Pipeline

## Structure of the pipeline

```bash
config/                 # existing systems are defined by their own YML file
├── orthrus.yml
├── kairos.yml
└── ...
pidsmaker/
├── config/
│   ├── config.py       # available arguments to use in YML files
│   └── pipeline.py     # pipeline code
├── preprocessing/               
│   ├── build_graphs.py         # 1. feature extraction + graph TW construction
│   └── transformation.py       # 2. graph transformation
├── featurization/              
│   ├── feat_training.py        # 3. featurization (word2vec, doc2vec, ...) training
│   └── feat_inference.py       # 4. featurization inference
├── detection/                  
│   ├── graph_preprocessing.py  # 5. batch construction, neighbor sampling, etc
│   ├── gnn_training.py         # 6. GNN training + inference loop
│   └── evaluation.py           # 7. metrics calculation + plots
├── triage/       
│   └── tracing.py              # 8. optional post-processing attack tracing
```
