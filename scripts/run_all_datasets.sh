#!/bin/bash

# Hard-coded dataset names
DATASETS=("CLEARSCOPE_E3" "CADETS_E3" "THEIA_E3" "CLEARSCOPE_E5" "THEIA_E5" "CADETS_E5")

# Check if the minimum required arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <command> <arg1> [additional_args...]"
  exit 1
fi

# Extract the first argument
COMMAND=$1

# Get additional arguments passed to the script
ADDITIONAL_ARGS="${@:2}"

# Construct the nohup command
NOHUP_CMD=""

for DATASET in "${DATASETS[@]}"; do
  if [ "$DATASET" != "${DATASETS[-1]}" ]; then
    NOHUP_CMD+="./run_serial.sh $COMMAND $DATASET $ADDITIONAL_ARGS && "
  else
    NOHUP_CMD+="./run_serial.sh $COMMAND $DATASET $ADDITIONAL_ARGS"
  fi
done

nohup bash -c "$NOHUP_CMD" &

# Notify the user
echo "Command is running in the background with nohup."
echo "To monitor progress, check the nohup.out file."