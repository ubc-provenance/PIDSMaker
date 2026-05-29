#!/bin/bash

# Dataset to test on (default is CADETS_E3 if not provided)
DATASET=${1:-"CADETS_E3"}

# List of all primary architectures in PIDSMaker
MODELS=("velox" "orthrus" "magic" "kairos" "flash" "nodlink" "rcaid" "threatrace")

# 80GB in Megabytes (80 * 1024)
MEM_LIMIT=81920

# Check if runlim is installed
if ! command -v runlim &> /dev/null; then
    echo "[*] 'runlim' is not installed. Installing it on the fly..."
    # Try apt-get first
    apt-get update -y && apt-get install -y runlim
    
    # If apt-get fails (e.g. package not found in distro), compile it manually
    if ! command -v runlim &> /dev/null; then
        echo "[*] apt-get failed to find runlim. Compiling from source..."
        wget http://fmv.jku.at/runlim/runlim-1.10.tar.gz -O /tmp/runlim.tar.gz
        cd /tmp && tar xzf runlim.tar.gz
        cd runlim-1.10 && ./configure && make
        cp runlim /usr/local/bin/
        cd - > /dev/null
        
        if ! command -v runlim &> /dev/null; then
            echo "[!] Fatal Error: Failed to install or compile runlim."
            exit 1
        fi
    fi
    echo "[+] 'runlim' installed successfully!"
fi
LOG_DIR="scripts/test_logs_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Starting PIDSMaker Architecture Test Suite"
echo "Dataset: $DATASET"
echo "Memory Limit: 80GB ($MEM_LIMIT MB)"
echo "Logs Output: $LOG_DIR/"
echo "=========================================================="

RESULTS=()

for model in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "-> Testing Architecture: $model"
    echo "----------------------------------------------------------"
    
    LOG_FILE="${LOG_DIR}/${model}.log"
    RUNLIM_LOG_FILE="${LOG_DIR}/${model}_runlim.log"
    
    # Run the model with runlim and restart from scratch, piping output and errors to the log file
    # We use PYTHONUNBUFFERED=1 and `python -u` to prevent stdout buffering when piping to tee!
    # -o directs the [runlim] sampling spam away from the terminal and into its own profile file.
    PYTHONUNBUFFERED=1 runlim -o "$RUNLIM_LOG_FILE" -s $MEM_LIMIT python -u pidsmaker/main.py "$model" "$DATASET" --restart_from_scratch 2>&1 | tee "$LOG_FILE"
    
    # We must use PIPESTATUS[0] to get the exit code of runlim rather than tee
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -ne 0 ]; then
        echo "[!] WARNING: Model '$model' threw an error or was killed (OOM). Exit Code: $exit_code"
        echo "-> Moving to the next architecture..."
        RESULTS+=("FAILED ($exit_code)")
    else
        echo "[+] SUCCESS: Model '$model' completed successfully."
        RESULTS+=("PASSED")
    fi
done

echo ""
echo "=========================================================="
echo "                   TEST RESULTS SUMMARY                   "
echo "=========================================================="
printf "%-15s | %-15s\n" "ARCHITECTURE" "STATUS"
echo "---------------------------------"
for i in "${!MODELS[@]}"; do
    printf "%-15s | %-15s\n" "${MODELS[$i]}" "${RESULTS[$i]}"
done
echo "=========================================================="
echo "Full logs for each architecture are saved in: $LOG_DIR/"
