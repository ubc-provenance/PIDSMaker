# Docs

The documentation is built with [Zensical](https://zensical.org) and published at https://ubc-provenance.github.io/PIDSMaker/.

| What | Where |
|---|---|
| Pages | `docs/**/*.md` |
| Navigation, theme, redirects of old URLs | `zensical.toml` |
| Styles | `docs/stylesheets/extra.css` |
| Generated argument lists | `scripts/args/`, produced by `scripts/get_arguments.py` |

## Build locally

Zensical needs Python >= 3.10. In the `pids` container (whose `pids` env is Python 3.9), install it in the base conda env:

```shell
/opt/conda/bin/pip install zensical
```

Then, from this folder:

```shell
./build.sh                          # regenerate the argument reference, then build into site/
zensical serve -a 0.0.0.0:8000      # live preview on http://localhost:8000/PIDSMaker/
```

Zensical reports broken links and anchors at every build: please fix them before opening a pull request.

## Generated content

`build.sh` regenerates the argument lists of the Reference from `pidsmaker/config/config.py` (`scripts/get_arguments.py`). Run it after adding or changing arguments in the config, and commit the files of `scripts/args/`.

## Deployment

`.github/workflows/ci.yml` builds the site with Zensical and pushes it to the `gh-pages` branch on every push to `main`. Pull requests that touch the docs are built too, without deploying.
