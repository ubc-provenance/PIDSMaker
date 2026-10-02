# Docs

If you update or add content to the docs, you can build it locally by running these comands from the pids container:

```shell
./build.sh
mkdocs serve --dev-addr=0.0.0.0:8000
```

`build.sh` also regenerates what is derived from `pidsmaker/config/config.py`: the argument lists and the pipeline figure (`docs/img/pipeline.svg`, also shown in the main README).
After adding a method to the config (encoder, objective, featurization, pretrained model...), the figure alone can be updated from anywhere, without any dependency:

```shell
python docs/scripts/gen_pipeline_figure.py           # rewrite the figure
python docs/scripts/gen_pipeline_figure.py --check   # exit 1 if the figure is out of date
```

Display names, hidden entries and the grouping of pretrained models are set at the top of the script.
