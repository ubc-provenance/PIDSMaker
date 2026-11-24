#!/bin/bash

set -e

echo "Starting database dump restoration..."

for dump_file in /data/*.dump; do
  if [ ! -f "$dump_file" ]; then
    echo "No .dump files found in /data/ directory"
    break
  fi
  
  db_name=$(basename "$dump_file" .dump)
  
  echo "Processing $dump_file -> database '$db_name'"
  
  if psql -U postgres -h localhost -p 5432 -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
    echo "Database '$db_name' already exists. Checking if it has data..."
    
    table_count=$(psql -U postgres -h localhost -p 5432 -d "$db_name" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null || echo "0")
    
    if [ "$table_count" -gt 0 ]; then
      echo "Database '$db_name' already has $table_count tables. Skipping restoration."
      continue
    else
      echo "Database '$db_name' exists but is empty. Proceeding with restoration..."
    fi
  else
    echo "Creating database '$db_name'..."
    psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE \"$db_name\";" 2>/dev/null || {
      echo "Warning: Could not create database '$db_name' (may already exist)"
    }
  fi
  
  echo "Restoring $dump_file into database '$db_name'..."
  
  # Use --clean --if-exists to handle existing objects gracefully
  if pg_restore -U postgres -h localhost -p 5432 --clean --if-exists --no-owner --no-privileges -d "$db_name" "$dump_file" 2>/dev/null; then
    echo "Successfully restored $dump_file"

    # Datasets with labelled events (e.g. LANL, OPTC_NET): index the malicious events, queried by
    # the evaluation (dumps made before this index was created by the loaders do not have it)
    has_label=$(psql -U postgres -h localhost -p 5432 -d "$db_name" -tAc "SELECT 1 FROM information_schema.columns WHERE table_name = 'event_table' AND column_name = 'label';" 2>/dev/null)
    if [ "$has_label" = "1" ]; then
      echo "  Indexing malicious events..."
      psql -U postgres -h localhost -p 5432 -d "$db_name" -c "CREATE INDEX IF NOT EXISTS event_label_idx ON event_table (timestamp_rec) WHERE label = 1;" >/dev/null
    fi

    final_table_count=$(psql -U postgres -h localhost -p 5432 -d "$db_name" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null || echo "0")
    echo "  Database '$db_name' now has $final_table_count tables"
  else
    echo "Warning: pg_restore reported errors for $dump_file (this may be normal for some dump formats)"
  fi
  
  echo ""
done

echo "Database dump restoration completed"

echo "Summary of available databases:"
psql -U postgres -h localhost -p 5432 -c "\l" | grep -E "^\s+[a-zA-Z]" | head -20
