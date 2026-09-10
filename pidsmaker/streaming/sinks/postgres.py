"""Postgres sink: materializes a stream as a regular PIDSMaker dataset.

Real-time detection needs a trained model, and training needs data. This sink
writes the canonical records of a stream into exactly the schema the offline
pipeline reads (`dataset_preprocessing/create_database.sh`), so a capture from
SPADE becomes a dataset indistinguishable from a pre-processed DARPA one: the
eight pipeline tasks then run on it unchanged.

Capture a benign period into a database, train any PIDS on it, then point the
detector at the live topic.
"""

from typing import Optional

import psycopg2

from pidsmaker.streaming.records import StreamEvent, StreamNode
from pidsmaker.utils.utils import log

NODE_TABLES = {
    "subject": "subject_node_table",
    "file": "file_node_table",
    "netflow": "netflow_node_table",
}

# Column order matters: `compute_indexid2msg()` reads these tables positionally.
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS event_table (
    src_node VARCHAR,
    src_index_id VARCHAR,
    operation VARCHAR,
    dst_node VARCHAR,
    dst_index_id VARCHAR,
    event_uuid VARCHAR NOT NULL,
    timestamp_rec BIGINT,
    _id SERIAL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    path VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);

CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    src_addr VARCHAR,
    src_port VARCHAR,
    dst_addr VARCHAR,
    dst_port VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);

CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid VARCHAR,
    hash_id VARCHAR,
    path VARCHAR,
    cmd VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
"""


class PostgresSink:
    """Writes streamed nodes and events into a PIDSMaker dataset database.

    Args:
        database: Database name (lowercase, as `dataset.database` expects).
        host, user, password, port: Connection settings.
        reset: Drop the existing rows and restart node indices from zero. When
            False, the sink resumes: it reloads the node keys already stored and
            keeps numbering from the highest index in the database.
        batch_size: Rows buffered before hitting the database.
    """

    def __init__(
        self,
        database: str,
        host: Optional[str] = None,
        user: str = "postgres",
        password: str = "postgres",
        port: int = 5432,
        reset: bool = True,
        batch_size: int = 5000,
    ):
        self.database = database.lower()
        self.batch_size = batch_size

        self._ensure_database_exists(host, user, password, port)
        self.connection = psycopg2.connect(
            database=self.database, host=host, user=user, password=password, port=port
        )
        self.cursor = self.connection.cursor()
        self.cursor.execute(SCHEMA_DDL)
        self.connection.commit()

        self._node_rows = {node_type: [] for node_type in NODE_TABLES}
        self._event_rows = []
        self.num_nodes = 0
        self.num_events = 0
        self.min_timestamp = None
        self.max_timestamp = None

        if reset:
            self._truncate()
            self.known_keys = set()
            self.max_existing_index = -1
        else:
            self.known_keys, self.max_existing_index = self._load_existing_nodes()
            log(
                f"Resuming ingestion: {len(self.known_keys)} nodes already stored, "
                f"next index {self.max_existing_index + 1}."
            )

    def _ensure_database_exists(self, host, user, password, port):
        """Creates the database if it does not exist yet."""
        connection = psycopg2.connect(
            database="postgres", host=host, user=user, password=password, port=port
        )
        connection.autocommit = True
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.database,))
            if cursor.fetchone() is None:
                log(f"Creating database {self.database}...")
                # template0 rather than the default template1: it is pristine, and
                # a template1 carrying a collation-version mismatch (common after a
                # glibc upgrade under an existing postgres data directory) would
                # otherwise make every CREATE DATABASE fail.
                cursor.execute(f'CREATE DATABASE "{self.database}" TEMPLATE template0')
        finally:
            connection.close()

    def _truncate(self):
        for table in list(NODE_TABLES.values()) + ["event_table"]:
            self.cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")
        self.connection.commit()

    def _load_existing_nodes(self):
        keys, max_index = set(), -1
        for table in NODE_TABLES.values():
            self.cursor.execute(f"SELECT node_uuid, index_id FROM {table}")
            for node_uuid, index_id in self.cursor.fetchall():
                keys.add(node_uuid)
                if index_id is not None:
                    max_index = max(max_index, int(index_id))
        return keys, max_index

    def write_node(self, node: StreamNode, index_id: str) -> None:
        """Buffers a node row.

        Args:
            node: The node to store.
            index_id: The index assigned by the graph builder.
        """
        if node.key in self.known_keys:
            return
        self.known_keys.add(node.key)

        attrs = node.attrs
        if node.node_type == "subject":
            row = (
                node.key,
                node.key,
                attrs.get("path", ""),
                attrs.get("cmd_line", ""),
                int(index_id),
            )
        elif node.node_type == "file":
            row = (node.key, node.key, attrs.get("path", ""), int(index_id))
        else:
            row = (
                node.key,
                node.key,
                attrs.get("local_ip", ""),
                attrs.get("local_port", ""),
                attrs.get("remote_ip", ""),
                attrs.get("remote_port", ""),
                int(index_id),
            )

        self._node_rows[node.node_type].append(row)
        self.num_nodes += 1
        if len(self._node_rows[node.node_type]) >= self.batch_size:
            self._flush_nodes(node.node_type)

    def write_event(self, event: StreamEvent, src_index: str, dst_index: str) -> None:
        """Buffers an event row.

        Args:
            event: The event to store.
            src_index: Index id of the source node.
            dst_index: Index id of the destination node.
        """
        self._event_rows.append(
            (
                event.src_key,
                src_index,
                event.operation,
                event.dst_key,
                dst_index,
                event.key,
                int(event.timestamp),
            )
        )
        self.num_events += 1
        self.min_timestamp = (
            event.timestamp
            if self.min_timestamp is None
            else min(self.min_timestamp, event.timestamp)
        )
        self.max_timestamp = (
            event.timestamp
            if self.max_timestamp is None
            else max(self.max_timestamp, event.timestamp)
        )
        if len(self._event_rows) >= self.batch_size:
            self._flush_events()

    def _flush_nodes(self, node_type: str):
        rows = self._node_rows[node_type]
        if not rows:
            return
        table = NODE_TABLES[node_type]
        placeholders = ",".join(["%s"] * len(rows[0]))
        columns = {
            "subject": "(node_uuid, hash_id, path, cmd, index_id)",
            "file": "(node_uuid, hash_id, path, index_id)",
            "netflow": "(node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id)",
        }[node_type]
        args = ",".join(
            self.cursor.mogrify(f"({placeholders})", row).decode("utf-8") for row in rows
        )
        self.cursor.execute(f"INSERT INTO {table} {columns} VALUES {args} ON CONFLICT DO NOTHING")
        rows.clear()

    def _flush_events(self):
        if not self._event_rows:
            return
        args = ",".join(
            self.cursor.mogrify("(%s,%s,%s,%s,%s,%s,%s)", row).decode("utf-8")
            for row in self._event_rows
        )
        self.cursor.execute(
            "INSERT INTO event_table (src_node, src_index_id, operation, dst_node, "
            "dst_index_id, event_uuid, timestamp_rec) "
            f"VALUES {args}"
        )
        self._event_rows.clear()

    def flush(self) -> None:
        """Writes everything buffered and commits."""
        for node_type in NODE_TABLES:
            self._flush_nodes(node_type)
        self._flush_events()
        self.connection.commit()

    def create_indices(self) -> None:
        """Adds the indices the construction task's queries rely on."""
        log("Creating database indices...")
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS event_table_timestamp_idx ON event_table (timestamp_rec)"
        )
        self.connection.commit()

    def close(self) -> None:
        """Flushes and closes the connection."""
        self.flush()
        self.cursor.close()
        self.connection.close()
