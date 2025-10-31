#!/bin/bash

# PostgreSQL startup script for Singularity/Apptainer

set -e

# Detect which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer-${USER}"
    export APPTAINER_CACHEDIR="${HOME}/.apptainer/cache"
    export APPTAINER_SESSIONDIR="${TMPDIR:-/tmp}/apptainer-sessions-${USER}"
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_SESSIONDIR"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/singularity-${USER}"
    export SINGULARITY_CACHEDIR="${HOME}/.singularity/cache"
    mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
else
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Configuration
POSTGRES_IMAGE="postgres.sif"
POSTGRES_INSTANCE="postgres_instance"
DATA_DIR="postgres_data"
RUN_DIR="postgres_run"
LOG_DIR="postgres_log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting PostgreSQL with ${CONTAINER_CMD}...${NC}"

# Check if postgres.sif exists
if [ ! -f "$POSTGRES_IMAGE" ]; then
    echo -e "${YELLOW}PostgreSQL image not found. Pulling from Docker Hub...${NC}"
    $CONTAINER_CMD pull $POSTGRES_IMAGE docker://postgres:17
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p $DATA_DIR $RUN_DIR $LOG_DIR

# Create INPUT_DIR if it doesn't exist
INPUT_DIR=${INPUT_DIR:-$(pwd)/data}
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${YELLOW}Creating INPUT_DIR: $INPUT_DIR${NC}"
    mkdir -p "$INPUT_DIR"
fi

# Check if instance already exists
if $CONTAINER_CMD instance list | grep -q "$POSTGRES_INSTANCE"; then
    echo -e "${YELLOW}PostgreSQL instance $POSTGRES_INSTANCE already exists${NC}"
    # Check if it's responsive
    if $CONTAINER_CMD exec instance://$POSTGRES_INSTANCE pg_isready -h localhost -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}PostgreSQL instance is already running and responsive${NC}"
        exit 0
    else
        echo -e "${YELLOW}Instance exists but not responsive, stopping it...${NC}"
        $CONTAINER_CMD instance stop $POSTGRES_INSTANCE
        sleep 2
    fi
fi

# Check if any other postgres processes are running
if pgrep -f "${CONTAINER_CMD}.*postgres" > /dev/null; then
    echo -e "${YELLOW}Other PostgreSQL processes detected, cleaning up...${NC}"
    pkill -f "${CONTAINER_CMD}.*postgres" || true
    sleep 2
fi

# Set environment variables (works for both singularity and apptainer)
if [ "$CONTAINER_CMD" = "apptainer" ]; then
    export APPTAINERENV_POSTGRES_PASSWORD=postgres
    export APPTAINERENV_POSTGRES_USER=postgres
    export APPTAINERENV_POSTGRES_DB=postgres
else
    export SINGULARITYENV_POSTGRES_PASSWORD=postgres
    export SINGULARITYENV_POSTGRES_USER=postgres
    export SINGULARITYENV_POSTGRES_DB=postgres
fi

# Prepare bind mounts - only bind if files/directories exist
BIND_MOUNTS="--bind $DATA_DIR:/var/lib/postgresql/data"
BIND_MOUNTS="$BIND_MOUNTS --bind $RUN_DIR:/var/run/postgresql"
BIND_MOUNTS="$BIND_MOUNTS --bind $LOG_DIR:/var/log"
BIND_MOUNTS="$BIND_MOUNTS --bind ./:/scripts"

# Always bind INPUT_DIR
BIND_MOUNTS="$BIND_MOUNTS --bind $INPUT_DIR:/data"

if [ -f "./postgres_config/postgresql.conf" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind ./postgres_config/postgresql.conf:/etc/postgresql/postgresql.conf"
fi

# Start PostgreSQL instance
echo -e "${YELLOW}Starting PostgreSQL instance...${NC}"
echo -e "${YELLOW}Using INPUT_DIR: $INPUT_DIR${NC}"

$CONTAINER_CMD instance start $BIND_MOUNTS $POSTGRES_IMAGE $POSTGRES_INSTANCE

# Start PostgreSQL inside the instance
echo -e "${YELLOW}Starting PostgreSQL server inside instance...${NC}"
$CONTAINER_CMD exec instance://$POSTGRES_INSTANCE bash -c "docker-entrypoint.sh postgres &"

# Get the PID of the instance (optional, for compatibility)
INSTANCE_PID=$(pgrep -f "${CONTAINER_CMD}.*$POSTGRES_INSTANCE" | head -1)
if [ -n "$INSTANCE_PID" ]; then
    echo $INSTANCE_PID > postgres.pid
fi

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}Waiting for PostgreSQL to start...${NC}"
for i in {1..30}; do
    if $CONTAINER_CMD exec instance://$POSTGRES_INSTANCE pg_isready -h localhost -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}PostgreSQL is ready!${NC}"
        echo -e "${GREEN}Connection: $CONTAINER_CMD exec instance://$POSTGRES_INSTANCE psql -h localhost -U postgres${NC}"
        echo -e "${GREEN}Instance: $POSTGRES_INSTANCE${NC}"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo -e "${RED}PostgreSQL failed to start within 60 seconds${NC}"
$CONTAINER_CMD instance stop $POSTGRES_INSTANCE 2>/dev/null || true
exit 1
