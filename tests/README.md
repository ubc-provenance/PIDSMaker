# Tests

We use functional tests to ensure future modifications of some components do not break others. Tests must be executed and updated upon modification of the existing code or integration of new components.

Tests run the pipeline in test mode (`cfg._test_mode`), which only builds the first time window graph of the first day of the train, val and test sets. By default, they need neither the full datasets nor a database server.

### Install

From the root of the repository, install PIDSMaker with the test dependencies (see the [installation guide](https://ubc-provenance.github.io/PIDSMaker/getting-started/installation/)):

``` sh
pip install -e ".[test]"
python -m nltk.downloader punkt   # once, otherwise each pytest-xdist worker downloads it at import
```

### Run the tests

``` sh
pytest tests -n 4            # in parallel with pytest-xdist (one worker per CPU core is a good default)
pytest tests -v              # sequentially
```

On CPU (as in the CI):

``` sh
pytest tests -n 4 --device cpu
```

### Datasets

By default (`--db sample`), the CLEARSCOPE_E3 tests run on a small sample of the dataset
(`fixtures/clearscope_e3_sample/`, ~0.6 MB), written as a Parquet dataset in a temporary directory at the beginning of the tests.
The sample holds the events read in test mode (the first events of each set; the test set starts at the attack, so
that the evaluation has malicious nodes), the nodes they touch, the ground truth nodes and the events between ground truth
nodes during the attack (the malicious edges of the edge-level evaluation).
It is extracted from the full database by `fixtures/make_clearscope_e3_sample.py`, which should be run again if the
test mode reads other events.

To run the CLEARSCOPE_E3 tests end to end on the full dataset, read from `$PIDSMAKER_DATASETS_DIR` (else `~/.cache/pidsmaker/datasets`), where it is downloaded first if it is missing:

``` sh
pytest tests -n 4 --db full
```

The LANL tests (`test_network_datasets.py`) run on a small synthetic dataset in the LANL format (`synthetic_lanl.py`),
written to a temporary Parquet dataset by the LANL preprocessing script. They only run with `--backend parquet`, and are skipped with `--backend postgres`.

`test_datasets.py` tests the downloads of datasets and the `pidsmaker` command against a fake Hugging Face repository, without network access.

### Legacy Postgres backend

With `--backend postgres`, the tests read the datasets from a Postgres server (`pip install -e ".[test,postgres]"`): the sample is loaded into a temporary database, dropped at the end of the tests, and `--db full` uses the full CLEARSCOPE_E3 database. The server is reached with the default connection settings of the framework (host `postgres`, port `5432`, user and password `postgres`), e.g. from the `pids` container with the Postgres container running.

``` sh
pytest tests -n 4 --backend postgres
```

### Options

- `--device cpu`: runs the tests that support it on CPU.
- `--db {sample,full}`: dataset of CLEARSCOPE_E3 (see above).
- `--backend {parquet,postgres}`: where the tests read the datasets from (default: `parquet`).
- `--artifact_dir DIR`: directory of the artifacts (default: a temporary directory), removed after the tests.

Evaluation figures are not saved (`evaluation.save_figures=False`), except by `TestEvaluationFigures`, as plotting dominates the evaluation time.

### CI

The tests run on CPU on each pull request with GitHub Actions (`.github/workflows/tests.yml`), on Python 3.9 and 3.11.
The package is installed as users install it, with `pip install ".[test]"` and the CPU build of PyTorch, without Postgres
and without the compiled extensions of PyTorch Geometric.

### Test coverage
```
pytest --cov=pidsmaker tests/
```
