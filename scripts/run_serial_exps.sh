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
./run_serial.sh $args --wandb --preprocessing.build_graphs.mimicry_edge_num=1000 --exp=mimicry_1000 --tags=$1,$2 && \
./run_serial.sh $args --wandb --preprocessing.build_graphs.mimicry_edge_num=2000 --exp=mimicry_2000 --tags=$1,$2 && \
./run_serial.sh $args --wandb --preprocessing.build_graphs.mimicry_edge_num=3000 --exp=mimicry_3000 --tags=$1,$2 && \
./run_serial.sh $args --wandb --preprocessing.build_graphs.mimicry_edge_num=4000 --exp=mimicry_4000 --tags=$1,$2 && \
./run_serial.sh $args --wandb --preprocessing.build_graphs.mimicry_edge_num=5000 --exp=mimicry_5000 --tags=$1,$2 " &
