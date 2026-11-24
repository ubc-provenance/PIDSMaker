-- Tables of the CLEARSCOPE_E3 sample, as created by dataset_preprocessing/create_database.sh
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
CREATE INDEX event_table_timestamp_idx ON event_table (timestamp_rec);

CREATE TABLE file_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    path VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);

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

CREATE TABLE subject_node_table (
    node_uuid VARCHAR,
    hash_id VARCHAR,
    path VARCHAR,
    cmd VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
