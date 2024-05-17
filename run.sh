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

# Execute the Python script with the passed arguments
nohup python src/benchmark.py $args --wandb & 
