# Makefile for Singularity/Apptainer PostgreSQL management

.PHONY: up down status load-dumps full-setup logs clean help

# PostgreSQL management
up:
	@./postgres-start.sh

down:
	@./postgres-stop.sh

status:
	@./postgres-status.sh

logs:
	@echo "PostgreSQL logs:"
	@tail -f postgres_log/postgresql*.log 2>/dev/null || echo "No logs found"

clean: down
	@echo "Cleaning up PostgreSQL data..."
	@rm -rf postgres_data postgres_run postgres_log
	@echo "PostgreSQL data cleaned"

reset: clean up

app-build:
	@echo "Building PIDSMaker container..."
	@if command -v apptainer &> /dev/null; then \
		apptainer build pidsmaker.sif pidsmaker.def || echo "Build failed - check if you have fakeroot access"; \
	elif command -v singularity &> /dev/null; then \
		singularity build pidsmaker.sif pidsmaker.def || echo "Build failed - check if you have fakeroot access"; \
	else \
		echo "ERROR: Neither apptainer nor singularity found"; exit 1; \
	fi

app-run: up
	@echo "Running PIDSMaker application..."
	@CONTAINER_CMD=$$(command -v apptainer &> /dev/null && echo "apptainer" || echo "singularity"); \
	$$CONTAINER_CMD run --nv \
		--env DB_HOST=localhost \
		--env DOCKER_PORT=5432 \
		--env DB_USER=postgres \
		--env DB_PASSWORD=postgres \
		--bind ${PWD}:/workspace \
		pidsmaker.sif

load-dumps: up
	@echo "Loading database dumps from inside container..."
	@CONTAINER_CMD=$$(command -v apptainer &> /dev/null && echo "apptainer" || echo "singularity"); \
	if [ -f "./load_dumps.sh" ]; then \
		echo "Found load_dumps.sh, executing inside container..."; \
		$$CONTAINER_CMD exec instance://postgres_instance /scripts/load_dumps.sh; \
	    else \
		echo "Error: ./load_dumps.sh not found"; \
		exit 1; \
	    fi

full-setup: up load-dumps
	@echo "PostgreSQL setup complete with dumps loaded"

help:
	@echo "Available commands:"
	@echo "  up     	- Start PostgreSQL"
	@echo "  down   	- Stop PostgreSQL"
	@echo "  status 	- Check PostgreSQL status"
	@echo "  logs   	- Show PostgreSQL logs"
	@echo "  load-dumps 	- Load database dumps"
	@echo "  full-setup 	- Start PostgreSQL and load dumps"
	@echo "  clean  	- Stop and remove all data"
	@echo "  reset  	- Clean and restart PostgreSQL"
	@echo "  app-build	- Build PIDSMaker container"
	@echo "  app-run	- Run PIDSMaker with PostgreSQL"
