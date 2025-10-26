#!/bin/bash

# PostgreSQL shutdown script for Singularity/Apptainer

# Detect which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping PostgreSQL...${NC}"

STOPPED=false

# Method 1: Stop instance if it exists
if $CONTAINER_CMD instance list | grep -q "postgres_instance"; then
    echo -e "${YELLOW}Stopping ${CONTAINER_CMD} instance: postgres_instance${NC}"
    $CONTAINER_CMD instance stop postgres_instance
    STOPPED=true
fi

# Method 2: Stop using PID file if it exists
if [ -f postgres.pid ]; then
    PID=$(cat postgres.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping PostgreSQL process (PID: $PID)${NC}"
        kill $PID
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo -e "${GREEN}PostgreSQL stopped gracefully${NC}"
                STOPPED=true
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Force killing PostgreSQL${NC}"
            kill -9 $PID
            STOPPED=true
        fi
    fi
    rm postgres.pid
fi

# Method 3: Fallback - kill any container postgres processes
if pkill -f "${CONTAINER_CMD}.*postgres"; then
    echo -e "${YELLOW}Killed remaining PostgreSQL processes${NC}"
    STOPPED=true
fi

if [ "$STOPPED" = true ]; then
    echo -e "${GREEN}PostgreSQL stopped${NC}"
else
    echo -e "${YELLOW}No PostgreSQL instances were found running${NC}"
fi