# Pipeline

Within PIDSMaker, the execution of each system is broken down into **7 stages**, each of which receives a set of input arguments defined in the system's YAML configuration files located in the `config/` directory.

<img src="../img/pipeline.svg" style="width: 100%"/>

```bash title="Main files associated with the pipeline"
config/                         # existing systems are defined by their own YML file
├── orthrus.yml
├── kairos.yml
└── ...
pidsmaker/
├── main.py                     # only entry point of the framework
│── config/
│   ├── config.py               # available arguments to use in YML files
│   └── pipeline.py             # pipeline code
├── tasks/                      
│   ├── construction.py         # stage 1. parse raw provenance + graph construction
│   ├── transformation.py       # stage 2. graph transformations
│   ├── featurization.py        # stage 3. text embedding featurization (Word2Vec, Doc2Vec, ...)
│   ├── batching.py             # stage 4. batch construction, neighbor sampling, etc
│   ├── training.py             # stage 5. GNN training + inference loop
│   ├── evaluation.py           # stage 6. metrics calculation + plots
│   └── triage.py               # stage 7. optional post-processing attack tracing
```

Under the hood, PIDSMaker generates a unique hash for each task based on its set of arguments. Once a task completes, its output files are saved to disk in a folder named after this hash. This mechanism allows the system to detect whether a task has already been executed by recomputing the hash from the current arguments and checking for the existence of the corresponding folder.

This approach prevents unnecessary recomputation of tasks.

!!! example
    For instance, if the same YAML configuration is run twice, the first execution will process all tasks (assuming no prior runs with the same arguments), while the second will skip all tasks—since they have already been computed and the results are assumed to be identical.
    However, if the second run introduces a change—such as modifying the text embedding size `emb_dim` in the `featurization` task—then the pipeline will resume execution starting from `featurization`, reusing earlier outputs as appropriate.

By reusing previously computed tasks, the pipeline significantly reduces redundant computations, enabling faster and more efficient experimentation.

## CLI arguments

While these YAML files provide the default configuration, they are not the only way to specify task arguments. CLI arguments can also be used. They take precedence over YAML-defined values, meaning that any argument provided via the CLI will override the corresponding value in the YAML file.

``` py
python pidsmaker/main.py orthrus CADETS_E3 \
    --featurization.emb_dim=64 \
    --training.lr=0.0001
```

The previous command is similar to the following YAML config:

``` yaml
featurization:
  emb_dim: 64
training:
  lr: 0.0001
```

## Forcing restart

During development or experimentation, you may need to restart the pipeline from specific tasks—even when using the same set of arguments. To achieve this, use the `--force_restart` flag.
For example, to restart from the `featurization` task, run.
``` py
python pidsmaker/main.py orthrus CADETS_E3 --force_restart=featurization
```

!!! note
    Forcing a restart will overwrite any previously generated data associated with the exact same tasks.

If you wish to restart the entire pipeline without overwriting previously generated data, you can use:

``` py
python pidsmaker/main.py orthrus CADETS_E3 --restart_from_scratch
```
This option instructs the pipeline to generate outputs in a new folder identified by a random hash, ensuring that the run is completely isolated from any prior executions.

!!! warning
    Since the hash is randomly generated, the outputs of this run cannot be retrieved later based on task arguments. As a result, all files produced when using `--restart_from_scratch` are deleted at the end of the pipeline to avoid unnecessary disk usage.
