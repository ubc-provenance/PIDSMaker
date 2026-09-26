#!/bin/bash
# ============================================================================
# PROVENANCE WORKLOAD — Runner Framework
# ============================================================================
# Source this file from each domain script:
#   source ./prov_framework.sh
#
# Provides: run(), run_t(), logging, color output, summary tracking
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# State
export PROV_LOGDIR="${PROV_LOGDIR:-/tmp/prov_logs}"
export PROV_WORKDIR="${PROV_WORKDIR:-/tmp/prov_workdir}"
mkdir -p "$PROV_LOGDIR" "$PROV_WORKDIR"

DETAIL_LOG="$PROV_LOGDIR/detail_$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
TOTAL_SUCCESS=0
TOTAL_FAIL=0
SCRIPT_START=$(date +%s)

# Default timeout for all commands (seconds)
DEFAULT_TIMEOUT=30

# ── Core run function ─────────────────────────────────────────────────────
# Usage: run "description" command arg1 arg2 ...
# - Always wraps in timeout (DEFAULT_TIMEOUT seconds)
# - Always closes stdin (< /dev/null) to prevent interactive prompts hanging
# - Console: one-line green/red/yellow status
# - Detail log: command + full stdout/stderr output
run() {
    local desc="$1"
    shift
    local cmd="$*"

    # Detail log: record command
    echo "──────────────────────────────────────────────────" >> "$DETAIL_LOG"
    echo "CMD: $cmd" >> "$DETAIL_LOG"
    echo "TIME: $(date '+%H:%M:%S')" >> "$DETAIL_LOG"

    # Execute with timeout and closed stdin to prevent prompts
    local output
    output=$(timeout "$DEFAULT_TIMEOUT" bash -c "$cmd" < /dev/null 2>&1)
    local rc=$?

    # Detail log: record output (truncate huge outputs)
    if [ -n "$output" ]; then
        echo "$output" | head -200 >> "$DETAIL_LOG"
        local lines
        lines=$(echo "$output" | wc -l)
        if [ "$lines" -gt 200 ]; then
            echo "... [truncated, $lines total lines]" >> "$DETAIL_LOG"
        fi
    fi
    if [ $rc -eq 124 ]; then
        echo "EXIT: TIMEOUT (${DEFAULT_TIMEOUT}s)" >> "$DETAIL_LOG"
    else
        echo "EXIT: $rc" >> "$DETAIL_LOG"
    fi
    echo "" >> "$DETAIL_LOG"

    # Console: one-line colored status
    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $desc"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    elif [ $rc -eq 124 ]; then
        echo -e "  ${YELLOW}⏱${NC} $desc ${YELLOW}(timeout ${DEFAULT_TIMEOUT}s)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    else
        echo -e "  ${RED}✗${NC} $desc ${RED}(exit=$rc)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# ── Run with custom timeout ───────────────────────────────────────────────
# Usage: run_t SECONDS "description" command arg1 arg2 ...
run_t() {
    local secs="$1"
    local desc="$2"
    shift 2
    local cmd="$*"

    echo "──────────────────────────────────────────────────" >> "$DETAIL_LOG"
    echo "CMD: timeout $secs $cmd" >> "$DETAIL_LOG"
    echo "TIME: $(date '+%H:%M:%S')" >> "$DETAIL_LOG"

    local output
    output=$(timeout "$secs" bash -c "$cmd" < /dev/null 2>&1)
    local rc=$?

    if [ -n "$output" ]; then
        echo "$output" | head -200 >> "$DETAIL_LOG"
        local lines
        lines=$(echo "$output" | wc -l)
        if [ "$lines" -gt 200 ]; then
            echo "... [truncated, $lines total lines]" >> "$DETAIL_LOG"
        fi
    fi

    if [ $rc -eq 124 ]; then
        echo "EXIT: TIMEOUT (${secs}s)" >> "$DETAIL_LOG"
    else
        echo "EXIT: $rc" >> "$DETAIL_LOG"
    fi
    echo "" >> "$DETAIL_LOG"

    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $desc"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    elif [ $rc -eq 124 ]; then
        echo -e "  ${YELLOW}⏱${NC} $desc ${YELLOW}(timeout ${secs}s)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    else
        echo -e "  ${RED}✗${NC} $desc ${RED}(exit=$rc)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# ── Run many variants of a command ────────────────────────────────────────
# Usage: run_many "base description" cmd_array
# Where cmd_array is a newline-separated list of commands
run_many() {
    local desc_prefix="$1"
    shift
    local i=0
    while IFS= read -r cmd; do
        [ -z "$cmd" ] && continue
        [[ "$cmd" == \#* ]] && continue
        i=$((i + 1))
        run "$desc_prefix #$i" "$cmd"
    done <<< "$*"
}

# ── Section header ────────────────────────────────────────────────────────
section() {
    echo ""
    echo -e "${CYAN}${BOLD}── $1 ──${NC}"
    echo "" >> "$DETAIL_LOG"
    echo "═══════════════════════════════════════════════════" >> "$DETAIL_LOG"
    echo "SECTION: $1" >> "$DETAIL_LOG"
    echo "═══════════════════════════════════════════════════" >> "$DETAIL_LOG"
}

# ── Domain header/footer ──────────────────────────────────────────────────
domain_start() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  $1${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo "================================================================" >> "$DETAIL_LOG"
    echo "DOMAIN: $1" >> "$DETAIL_LOG"
    echo "START: $(date)" >> "$DETAIL_LOG"
    echo "================================================================" >> "$DETAIL_LOG"

    TOTAL_SUCCESS=0
    TOTAL_FAIL=0
}

domain_end() {
    local total=$((TOTAL_SUCCESS + TOTAL_FAIL))
    local pct=0
    [ $total -gt 0 ] && pct=$((TOTAL_SUCCESS * 100 / total))
    local elapsed=$(( $(date +%s) - SCRIPT_START ))

    echo ""
    if [ $pct -ge 80 ]; then
        echo -e "${GREEN}${BOLD}Summary: $TOTAL_SUCCESS/$total succeeded ($pct%) in ${elapsed}s${NC}"
    elif [ $pct -ge 50 ]; then
        echo -e "${YELLOW}${BOLD}Summary: $TOTAL_SUCCESS/$total succeeded ($pct%) in ${elapsed}s${NC}"
    else
        echo -e "${RED}${BOLD}Summary: $TOTAL_SUCCESS/$total succeeded ($pct%) in ${elapsed}s${NC}"
    fi
    echo -e "Detail log: ${CYAN}$DETAIL_LOG${NC}"
    echo ""

    echo "================================================================" >> "$DETAIL_LOG"
    echo "SUMMARY: $TOTAL_SUCCESS/$total ($pct%) in ${elapsed}s" >> "$DETAIL_LOG"
    echo "================================================================" >> "$DETAIL_LOG"
}

# ── Generate random data files ────────────────────────────────────────────
generate_sample_files() {
    local dir="$PROV_WORKDIR/samples"
    mkdir -p "$dir"

    # Text files of various sizes
    echo "Hello provenance" > "$dir/tiny.txt"
    seq 1 100 > "$dir/small.txt"
    seq 1 10000 > "$dir/medium.txt"
    for i in $(seq 1 50); do echo "line $i: $(head -c 60 /dev/urandom | base64 | head -c 80)"; done > "$dir/multiline.txt"

    # Binary files
    dd if=/dev/urandom of="$dir/random_1k.bin" bs=1024 count=1 2>/dev/null
    dd if=/dev/urandom of="$dir/random_10k.bin" bs=1024 count=10 2>/dev/null
    dd if=/dev/urandom of="$dir/random_100k.bin" bs=1024 count=100 2>/dev/null
    dd if=/dev/zero of="$dir/zeros_10k.bin" bs=1024 count=10 2>/dev/null

    # Structured files
    echo '{"key":"value","list":[1,2,3],"nested":{"a":"b"}}' > "$dir/data.json"
    echo '<?xml version="1.0"?><root><item id="1">hello</item></root>' > "$dir/data.xml"
    echo -e "name,age,role\nalice,30,admin\nbob,25,user\ncharlie,35,admin" > "$dir/data.csv"
    echo -e "[section]\nkey=value\nother=123" > "$dir/config.ini"
    echo -e "key: value\nlist:\n  - one\n  - two" > "$dir/config.yaml"

    # Directory tree
    mkdir -p "$dir/tree/a/b/c" "$dir/tree/d/e" "$dir/tree/f"
    for f in "$dir/tree/a/file1.txt" "$dir/tree/a/b/file2.log" "$dir/tree/a/b/c/file3.dat" \
             "$dir/tree/d/file4.conf" "$dir/tree/d/e/file5.sh" "$dir/tree/f/file6.py"; do
        echo "content of $(basename $f)" > "$f"
    done
    chmod +x "$dir/tree/d/e/file5.sh"

    # Copy system files for safe manipulation
    cp /etc/passwd "$dir/passwd_copy" 2>/dev/null
    cp /etc/hosts "$dir/hosts_copy" 2>/dev/null
    cp /etc/hostname "$dir/hostname_copy" 2>/dev/null

    echo "$dir"
}
