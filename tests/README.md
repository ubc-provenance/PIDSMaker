# Tests

We use functional tests to ensure future modifications of some components do not break others. Tests must be executed and updated upon modification of the existing code or integration of new components.

### Tests on GPU

``` sh
pytest -v
```

### Tests on CPU

``` sh
pytest -v --device cpu -k "not (test_transformations or test_featurizations)"
```

### Test coverage
```
pytest --cov=pidsmaker tests/
```
