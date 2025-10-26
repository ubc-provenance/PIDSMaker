#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide the directory of compress files"
  exit 1
fi

find "$1" -name "*.gz" -exec gunzip {} \;

echo "Finish extracting data from compressed files."