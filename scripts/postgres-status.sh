#!/bin/bash

# PostgreSQL status script for Singularity/Apptainer

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

echo -e "${YELLOW}PostgreSQL Status:${NC}"

# Check multiple ways to detect if PostgreSQL is running
POSTGRES_RUNNING=false

# Method 1: Check PID file
if [ -f postgres.pid ]; then
    PID=$(cat postgres.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}✓ PostgreSQL process is running (PID: $PID)${NC}"
        POSTGRES_RUNNING=true
    else
        echo -e "${YELLOW}! PID file exists but process is not running${NC}"
        rm postgres.pid
    fi
fi

# Method 2: Check if we can connect (most reliable test)
if [ ! -f postgres.sif ]; then
    echo -e "${RED}✗ postgres.sif not found${NC}"
    exit 1
fi

if $CONTAINER_CMD exec postgres.sif pg_isready -h localhost -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL is accepting connections${NC}"
    echo -e "${GREEN}  Connection: ${CONTAINER_CMD} exec postgres.sif psql -h localhost -U postgres${NC}"
    POSTGRES_RUNNING=true
    
    # Show database list
    echo -e "${YELLOW}Databases:${NC}"
    $CONTAINER_CMD exec postgres.sif psql -h localhost -U postgres -c "\l" 2>/dev/null | \
        grep -v template | grep -v "^-" | grep -v "^(" | grep -v "Name.*Owner" | \
        grep -v "^\s*$" | head -10
    
    # Show PostgreSQL version
    echo -e "${YELLOW}Version:${NC}"
    $CONTAINER_CMD exec postgres.sif psql -h localhost -U postgres -c "SELECT version();" -t 2>/dev/null | head -1
    
else
    echo -e "${RED}✗ PostgreSQL is not accepting connections${NC}"
fi

# Method 3: Check process list as fallback
if [ "$POSTGRES_RUNNING" = false ]; then
    # Check for any postgres-related processes with more flexible patterns
    if pgrep -f "postgres" > /dev/null || pgrep -f "${CONTAINER_CMD}.*postgres" > /dev/null; then
        echo -e "${YELLOW}! Found postgres-related process but cannot connect${NC}"
        echo -e "${YELLOW}  Process list:${NC}"
        ps aux | grep -E "(postgres|${CONTAINER_CMD})" | grep -v grep | head -5
    else
        echo -e "${RED}✗ No PostgreSQL processes found${NC}"
    fi
fi

# Method 4: Check if port 5432 is listening
if ss -tlnp 2>/dev/null | grep -q ":5432 " || netstat -tlnp 2>/dev/null | grep -q ":5432 "; then
    echo -e "${GREEN}✓ Port 5432 is listening${NC}"
    POSTGRES_RUNNING=true
else
    echo -e "${RED}✗ Port 5432 is not listening${NC}"
fi

# Final status
if [ "$POSTGRES_RUNNING" = true ]; then
    echo -e "${GREEN}Overall Status: PostgreSQL is running and accessible${NC}"
    exit 0
else
    echo -e "${RED}Overall Status: PostgreSQL is not running or not accessible${NC}"
    exit 1
fi