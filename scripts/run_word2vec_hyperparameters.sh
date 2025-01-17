#!/bin/bash

# This script forwards all command-line arguments to the Python script

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

# Construct the argument string to pass to the Python script
args=""
for arg in "$@"; do
    args+="$arg "
done

nohup sh -c "\
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_3,min_count_1,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=3 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_7,min_count_1,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=7 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_9,min_count_1,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=9 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes" &

nohup sh -c "\
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_3,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=3 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_5,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=5 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_7,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=7 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_9,negative_5,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=9 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes" &

nohup sh -c "\
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_2,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=2 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_10,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=10 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_15,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=15 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_20,decline_rate_30 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=20 --featurization.embed_nodes.feature_word2vec.decline_rate=30 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes" &

nohup sh -c "\
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_0 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=0 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_10 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=10 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_20 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=20 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_40 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=40 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes && \
python ../src/benchmark.py $args --wandb --tags=window_size_5,min_count_1,negative_5,decline_rate_50 \
--featurization.embed_nodes.feature_word2vec.window_size=5 --featurization.embed_nodes.feature_word2vec.min_count=1 \
--featurization.embed_nodes.feature_word2vec.negative=5 --featurization.embed_nodes.feature_word2vec.decline_rate=50 \
--project=hyperparameter_amalgame --experiment=run_n_times --experiment.uncertainty.deep_ensemble.restart_from=embed_nodes" &
