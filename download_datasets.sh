#!/usr/bin/env bash

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset1,dataset2,...|all> <ACCESS_TOKEN>"
  exit 1
fi

DATASETS_ARG="$1"
ACCESS_TOKEN="$2"

DATA_DIR="data"
mkdir -p "$DATA_DIR"

get_id() {
  case "$1" in
    cadets_e3) echo "1DGcGBhpavNmXTnCDd_s4NWBNh2n4-6nd" ;;
    clearscope_e3) echo "1v8vsgJxv4OPFVxx8UlYvixDq8FpkIqkd" ;;
    theia_e3) echo "1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6" ;;
    theia_e5) echo "1Q0rCCj2KnJeOqJN2NcmHnDBF1sl33g9x" ;;
    clearscope_e5) echo "1zfQBbpIRIqo5EugAT5Ou0354Ii_naINC" ;;
    optc_h051) echo "1zzwge346AdAdxldykZOai5FtUrIaWc6q" ;;
    optc_h201) echo "1OSZXCQrocFSRN7wkPM02p-BqE2WmgdLD" ;;
    optc_h501) echo "1046BVjpMql1bb5WHr9yQeB6Uq6RpngbM" ;;
    cadets_e5) echo "1Xiq7w0Ofz4jZG2PVFuNqi_i0fm28kRcT" ;;
    trace_e3) echo "1xZNBbhWQO0xGVBsg6ujdPh9UMUXiUQQd" ;;
    fivedirections_e3) echo "17YHqUMbuNwP05iaOaifxvcQc2oC9pJbZ" ;;
    fivedirections_e5) echo "1EkxlbReJgMHW4TwAypPggQBA-RTRVxs4" ;;
    atlasv2_edr) echo "1t7pW7i23Ry-Z7Tpu2B-E4gezT7Wu74ma" ;;
    carbanakv2_edr) echo "1rJN9F8JkmqMAV9zUjralcFB5wykAgLnd" ;;
    *) echo "" ;;
  esac
}

download_file() {
  NAME="$1"
  ID="$2"

  echo "Downloading ${NAME}..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/${ID}?alt=media" \
    -o "${DATA_DIR}/${NAME}.dump"
}

download_trace_e5() {
  echo "Downloading trace_e5 (part aa)..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/14asyR0arrxLS4uglx8pMiR2OuoQ4emq6?alt=media" \
    -o "${DATA_DIR}/trace_e5.dump.part_aa"

  echo "Downloading trace_e5 (part ab)..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/1SOo-McGOJJqQM4I_m7xWRF-PlFN9U7Wf?alt=media" \
    -o "${DATA_DIR}/trace_e5.dump.part_ab"

  echo "Merging trace_e5..."
  cat "${DATA_DIR}/trace_e5.dump.part_aa" \
      "${DATA_DIR}/trace_e5.dump.part_ab" \
      > "${DATA_DIR}/trace_e5.dump"
}

ALL_DATASETS="cadets_e3 clearscope_e3 theia_e3 theia_e5 clearscope_e5 \
optc_h051 optc_h201 optc_h501 cadets_e5 trace_e3 \
fivedirections_e3 fivedirections_e5 trace_e5"

if [ "$DATASETS_ARG" = "all" ]; then
  SELECTED="$ALL_DATASETS"
else
  SELECTED=$(echo "$DATASETS_ARG" | tr ',' ' ')
fi

for dataset in $SELECTED; do
  if [ "$dataset" = "trace_e5" ]; then
    download_trace_e5
  else
    ID=$(get_id "$dataset")
    if [ -z "$ID" ]; then
      echo "Unknown dataset: $dataset"
      exit 1
    fi
    download_file "$dataset" "$ID"
  fi
done

echo "All requested datasets processed in ${DATA_DIR}/"
