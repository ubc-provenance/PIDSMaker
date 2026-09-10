#!/bin/bash
# Benign workload for the streaming tutorial (docs/docs/features/streaming_tutorial.md):
# shells forking tools that read files, write output, and talk to the network. Every
# command is a real binary doing real syscalls - this is what the audit subsystem records.
#   usage: bash scripts/workload/benign.sh [work_dir] [rounds]
set -u
WORK="${1:-/tmp/spade_work}"
ROUNDS="${2:-12}"
mkdir -p "$WORK"

for i in $(seq 1 "$ROUNDS"); do
    # read a few system files, the way any tool would
    grep -c . /etc/passwd > "$WORK/passwd_lines_$i.txt" 2>/dev/null
    head -20 /etc/hosts > "$WORK/hosts_head_$i.txt" 2>/dev/null
    wc -l /etc/services > "$WORK/services_count_$i.txt" 2>/dev/null

    # a build-like step: python reads sources and writes an artifact
    python3 -c "
import json, os
data = {'round': $i, 'files': sorted(os.listdir('/etc'))[:20]}
open('$WORK/manifest_$i.json', 'w').write(json.dumps(data))
" 2>/dev/null

    # a text pipeline over the file just written
    sort "$WORK/manifest_$i.json" | uniq | tail -3 > "$WORK/sorted_$i.txt" 2>/dev/null
    cat "$WORK/sorted_$i.txt" >> "$WORK/all_rounds.log" 2>/dev/null

    # network: a real DNS lookup and a real HTTPS fetch
    if (( i % 3 == 0 )); then
        getent hosts example.com > "$WORK/dns_$i.txt" 2>/dev/null
        curl -s --max-time 8 -o "$WORK/page_$i.html" https://example.com/ 2>/dev/null
    fi

    # archive step: read many files (the point is the read syscalls). Write to
    # /dev/null, not a file in $WORK — a real archive would include the previous
    # rounds' archives and grow quadratically, filling the disk over a long run.
    if (( i % 4 == 0 )); then
        tar -cf /dev/null -C "$WORK" . 2>/dev/null
    fi
    sleep 0.4
done
echo "benign workload done: $ROUNDS rounds in $WORK"
