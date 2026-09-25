#!/bin/bash
# ============================================================================
# PROVENANCE WORKLOAD — Master Runner
# ============================================================================
# Runs all domain scripts in sequence. Each domain is independent.
#
# Usage:
#   sudo bash run_all.sh                    # Run all domains
#   sudo bash run_all.sh 04 05 20           # Run specific domains
#   sudo bash run_all.sh 2>&1 | tee run.log # Save console output
# ============================================================================

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PROV_LOGDIR="/tmp/prov_logs"
export PROV_WORKDIR="/tmp/prov_workdir"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

mkdir -p "$PROV_LOGDIR" "$PROV_WORKDIR"

echo -e "${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PROVENANCE WORKLOAD — COMPREHENSIVE GENERATOR        ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Started: $(date)"
echo "║  Host:    $(hostname)"
echo "║  User:    $(whoami)"
echo "║  Logs:    $PROV_LOGDIR/"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Install phase first
if [ -f "$SCRIPT_DIR/00_install.sh" ]; then
    echo -e "${BOLD}Phase 0: Installing tools...${NC}"
    source "$SCRIPT_DIR/00_install.sh"
    echo ""
fi

# Discover domain scripts (sorted)
SCRIPTS=()
if [ $# -gt 0 ]; then
    # Run specific domains
    for num in "$@"; do
        pattern="$SCRIPT_DIR/${num}_*.sh"
        for f in $pattern; do
            [ -f "$f" ] && SCRIPTS+=("$f")
        done
    done
else
    # Run all domains (skip framework, runner, and install)
    for f in "$SCRIPT_DIR"/[0-9][0-9]_*.sh; do
        [ -f "$f" ] && [ "$(basename "$f")" != "00_install.sh" ] && SCRIPTS+=("$f")
    done
fi

GLOBAL_SUCCESS=0
GLOBAL_FAIL=0
DOMAIN_RESULTS=()
MASTER_START=$(date +%s)

for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script" .sh)
    echo -e "\n${CYAN}${BOLD}Starting domain: $name${NC}"
    
    # Source the framework fresh for each domain
    source "$SCRIPT_DIR/prov_framework.sh"
    SCRIPT_START=$(date +%s)
    
    # Run the domain script
    source "$script"
    
    # Collect results
    total=$((TOTAL_SUCCESS + TOTAL_FAIL))
    GLOBAL_SUCCESS=$((GLOBAL_SUCCESS + TOTAL_SUCCESS))
    GLOBAL_FAIL=$((GLOBAL_FAIL + TOTAL_FAIL))
    DOMAIN_RESULTS+=("$name|$TOTAL_SUCCESS|$TOTAL_FAIL|$total")
done

# ── Final summary ─────────────────────────────────────────────────────────
GLOBAL_TOTAL=$((GLOBAL_SUCCESS + GLOBAL_FAIL))
GLOBAL_PCT=0
[ $GLOBAL_TOTAL -gt 0 ] && GLOBAL_PCT=$((GLOBAL_SUCCESS * 100 / GLOBAL_TOTAL))
ELAPSED=$(( $(date +%s) - MASTER_START ))

echo ""
echo -e "${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    FINAL SUMMARY                             ║"
echo "╠════════════════════════════════════════════════════════════════╣"
printf "║  Total commands:  %-6d                                      ║\n" $GLOBAL_TOTAL
printf "║  Succeeded:       %-6d (%d%%)                               \n" $GLOBAL_SUCCESS $GLOBAL_PCT
printf "║  Failed:          %-6d                                      \n" $GLOBAL_FAIL
printf "║  Elapsed:         %dm %ds                                   \n" $((ELAPSED/60)) $((ELAPSED%60))
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Per-domain breakdown:                                       ║"
echo "╟────────────────────────────────────────────────────────────────╢"

for entry in "${DOMAIN_RESULTS[@]}"; do
    IFS='|' read -r dname dsucc dfail dtotal <<< "$entry"
    dpct=0
    [ $dtotal -gt 0 ] && dpct=$((dsucc * 100 / dtotal))
    if [ $dpct -ge 80 ]; then
        color=$GREEN
    elif [ $dpct -ge 50 ]; then
        color=$YELLOW
    else
        color=$RED
    fi
    printf "║  ${color}%-30s %5d / %-5d (%3d%%)${NC}\n" "$dname" "$dsucc" "$dtotal" "$dpct"
done

echo -e "${BOLD}"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Logs directory: $PROV_LOGDIR/"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Save machine-readable summary
SUMMARY="$PROV_LOGDIR/summary_$(date +%Y%m%d_%H%M%S).txt"
echo "# Provenance Workload Summary — $(date)" > "$SUMMARY"
echo "total_commands=$GLOBAL_TOTAL" >> "$SUMMARY"
echo "succeeded=$GLOBAL_SUCCESS" >> "$SUMMARY"
echo "failed=$GLOBAL_FAIL" >> "$SUMMARY"
echo "elapsed_seconds=$ELAPSED" >> "$SUMMARY"
for entry in "${DOMAIN_RESULTS[@]}"; do
    echo "domain=$entry" >> "$SUMMARY"
done
echo -e "Summary saved to: ${CYAN}$SUMMARY${NC}"
