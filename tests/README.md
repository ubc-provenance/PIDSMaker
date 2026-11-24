# Tests

We use functional tests to ensure future modifications of some components do not break others. Tests must be executed and updated upon modification of the existing code or integration of new components.

Tests run the pipeline in test mode (`cfg._test_mode`), which only builds the first time window graph of the first day of the train, val and test sets. They need a Postgres server (the one of the framework, see `--database_host`), but not the full datasets.

### Run the tests

``` sh
pytest -n 4            # in parallel with pytest-xdist (one worker per CPU core is a good default)
pytest -v              # sequentially
```

On CPU (as in the CI):

``` sh
pytest -n 4 --device cpu
```

### Datasets

By default (`--db sample`), the CLEARSCOPE_E3 tests run on a small sample of the dataset
(`fixtures/clearscope_e3_sample/`, ~0.6 MB), loaded into a temporary database at the beginning of the tests and dropped at the end.
The sample holds the events read in test mode (the first events of each set; the test set starts at the attack, so
that the evaluation has malicious nodes), the nodes they touch and the ground truth nodes.
It is extracted from the full database by `fixtures/make_clearscope_e3_sample.py`, which should be run again if the
test mode reads other events.

To run the tests end to end on the full CLEARSCOPE_E3 database:

``` sh
pytest -n 4 --db full
```

### Options

- `--device cpu`: runs the tests that support it on CPU.
- `--db {sample,full}`: database of CLEARSCOPE_E3 (see above).
- `--artifact_dir DIR`: directory of the artifacts (default: a temporary directory), removed after the tests.

Evaluation figures are not saved (`--no_figures`), except by `TestEvaluationFigures`, as plotting dominates the evaluation time.

### CI

The tests run on CPU on each pull request with GitHub Actions (`.github/workflows/tests.yml`),
with a Postgres service container and the dependencies of `requirements.txt`.

### Test coverage
```
pytest --cov=pidsmaker tests/
```
