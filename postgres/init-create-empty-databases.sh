#!/bin/bash
set -e  # Exit on any error

echo "Starting database and table creation..."

for dataset in clearscope_e3 cadets_e3 theia_e3 clearscope_e5 cadets_e5 theia_e5 optc_h051 optc_h501 optc_h201
do
    DATASET_NAME=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    echo "Creating database and tables for: $DATASET_NAME"

    psql -U postgres <<EOF
CREATE DATABASE $DATASET_NAME;
EOF

    echo "Database '$DATASET_NAME' and tables created successfully!"
done

echo "All databases and tables created successfully!"
