Tasks are steps composing the pipeline, starting from graph construction (`construction`) to detection (`evaluation`) or optionally triage (`triage`).
Each task takes as input the output from the previous task and writes its output to the disk so that the next task can use it. This process enables "checkpointing" across the pipeline and avoids the duplication of compute. More information on tasks and the pipeline [here](../pipeline.md).

### Stage 1: Construction

--8<-- "scripts/args/args_construction.md"

### Stage 2: Transformation

--8<-- "scripts/args/args_transformation.md"

### Stage 3: Featurization

--8<-- "scripts/args/args_featurization.md"

### Stage 4: Batching

--8<-- "scripts/args/args_batching.md"

### Stage 5: Training

--8<-- "scripts/args/args_training.md"

### Stage 6: Evaluation

--8<-- "scripts/args/args_evaluation.md"

### Stage 7: Triage

--8<-- "scripts/args/args_triage.md"
