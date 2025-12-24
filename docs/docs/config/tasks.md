Tasks are steps composing the pipeline, starting from graph construction (`construction`) to detection (`evaluation`) or optionally triage (`tracing`).
Each task takes as input the output from the previous task and write its output to the disk so that the next task can use it. This process enables "checkpointing" across the pipeline and avoids the duplication of compute. More information on tasks and the pipeline [here](../pipeline.md).

### Preprocessing

--8<-- "scripts/args/args_preprocessing.md"

### Featurization

--8<-- "scripts/args/args_featurization.md"

### Detection

--8<-- "scripts/args/args_detection.md"

### Triage

--8<-- "scripts/args/args_triage.md"

