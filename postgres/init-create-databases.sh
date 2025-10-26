#!/bin/bash
set -e  # Exit on any error

echo "Starting database and table creation..."

# Loop over datasets
for dataset in clearscope_e3 cadets_e3 theia_e3 clearscope_e5 cadets_e5 theia_e5 optc_051 optc_501 optc_201
do
    # Convert dataset name to lowercase
    DATASET_NAME=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    echo "Creating database and tables for: $DATASET_NAME"

    # PostgreSQL commands
    psql -U postgres <<EOF
CREATE DATABASE $DATASET_NAME;
\c $DATASET_NAME;

CREATE TABLE event_table (
    src_node VARCHAR,
    src_index_id VARCHAR,
    operation VARCHAR,
    dst_node VARCHAR,
    dst_index_id VARCHAR,
    event_uuid VARCHAR NOT NULL,
    timestamp_rec BIGINT,
    _id SERIAL PRIMARY KEY
);
ALTER TABLE event_table OWNER TO postgres;
CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO postgres;

CREATE TABLE file_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    path VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

CREATE TABLE netflow_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    src_addr VARCHAR,
    src_port VARCHAR,
    dst_addr VARCHAR,
    dst_port VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO postgres;

CREATE TABLE subject_node_table (
    node_uuid VARCHAR,
    hash_id VARCHAR,
    path VARCHAR,
    cmd VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE subject_node_table OWNER TO postgres;
EOF

    echo "Database '$DATASET_NAME' and tables created successfully!"
done

echo "All databases and tables created successfully!"