nohup sh -c "\
./run_same_corpus.sh orthrus CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh nodlink CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh kairos CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh flash CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training" \
> uncertainty_exps.log 2>&1 &

nohup sh -c "\
./run_same_corpus.sh magic CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh rcaid CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh sigl CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training && \
./run_same_corpus.sh threatrace CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=gnn_training" \
> uncertainty_exps.log 2>&1 &

nohup sh -c "\
./run_same_corpus.sh orthrus CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh nodlink CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh kairos CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh flash CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training" \
> uncertainty_exps.log 2>&1 &

nohup sh -c "\
./run_same_corpus.sh magic CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh rcaid CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh sigl CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training && \
./run_same_corpus.sh threatrace CLEARSCOPE_E3 --tuned --tuning_mode=featurization --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=feat_training" \
> uncertainty_exps.log 2>&1 &