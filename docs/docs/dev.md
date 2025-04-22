# Development

### Format all files
```
pre-commit run --all-files
```

### Run tests
```
pytest -v tests/
```

### Test coverage
```
pytest --cov=pidsmaker tests/
```

### Build documentation

```
cd docs
mkdocs build
```