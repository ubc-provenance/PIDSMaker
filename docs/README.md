# Docs

The documentation is built with [Zensical](https://zensical.org) and published at https://ubc-provenance.github.io/PIDSMaker/.

| What | Where |
|---|---|
| Pages | `docs/**/*.md` |
| Navigation, theme, redirects of old URLs | `zensical.toml` |
| Styles | `docs/stylesheets/extra.css` |
| Generated argument lists | `scripts/args/`, produced by `scripts/get_arguments.py` |
| Pipeline figure | `docs/img/pipeline.svg`, produced by `scripts/gen_pipeline_figure.py` |

## Build locally

Zensical needs Python >= 3.10. In the `pids` container (whose `pids` env is Python 3.9), install it in the base conda env:

```shell
/opt/conda/bin/pip install zensical
```

Then, from this folder:

```shell
./build.sh                          # regenerate the generated content, then build into site/
zensical serve -a 0.0.0.0:8000      # live preview on http://localhost:8000/PIDSMaker/
```

Zensical reports broken links and anchors at every build: please fix them before opening a pull request.

## Generated content

`build.sh` regenerates what is derived from `pidsmaker/config/config.py`: the argument lists and the pipeline figure (`docs/img/pipeline.svg`, also shown in the main README).
After adding a method to the config (encoder, objective, featurization, pretrained model...), the figure alone can be updated from anywhere, without any dependency:

```shell
python docs/scripts/gen_pipeline_figure.py           # rewrite the figure
python docs/scripts/gen_pipeline_figure.py --check   # exit 1 if the figure is out of date
```

Display names, hidden entries and the grouping of pretrained models are set at the top of the script.

## Deployment

`.github/workflows/ci.yml` builds the site with Zensical and pushes it to the `gh-pages` branch on every push to `main`. Pull requests that touch the docs are built too, without deploying.
