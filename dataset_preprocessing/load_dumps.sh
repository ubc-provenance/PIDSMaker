#!/bin/bash

for dump_file in /data/*.dump; do
  db_name=$(basename "$dump_file" .dump)

  echo "Restoring $dump_file into database '$db_name'..."

  pg_restore -U postgres -h localhost -p 5432 -d "$db_name" "$dump_file"
done
