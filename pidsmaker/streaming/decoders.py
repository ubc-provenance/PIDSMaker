"""Wire-format decoders turning raw stream payloads into Python dicts.

A stream carries bytes; what those bytes mean is a property of the *producer*,
not of the transport. SPADE's Kafka storage serializes each record with Avro's
binary encoder and no schema envelope (`spade.storage.kafka.GenericContainerSerializer`),
its file writer with the JSON encoder, and a future source may publish plain
JSON. Decoding is therefore a separate, swappable step between the source and
the adapter.

Formats:
    - `avro`: schemaless Avro binary (what SPADE's Kafka *server* writer emits).
    - `avro_json`: Avro's JSON encoding, with its union/`map` wrappers
      (what SPADE's Kafka storage writes when `kafka.output.file` ends in `.json`).
    - `json`: plain JSON objects.
"""

import io
import json
import os
from abc import ABC, abstractmethod
from typing import Optional

SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")

FORMATS = ("avro", "avro_json", "json")


def _import_fastavro():
    """Imports fastavro, with an actionable message when it is missing."""
    try:
        import fastavro
    except ImportError as e:
        raise ImportError(
            "Decoding Avro records requires the `fastavro` package (`pip install fastavro`). "
            "Use `--stream_format=json` if your producer publishes plain JSON."
        ) from e
    return fastavro


def load_schema(path: str) -> dict:
    """Loads an Avro schema, resolving bare names against the bundled schemas.

    Args:
        path: Either a filesystem path to a `.avsc` file, or the name of one of
            the schemas shipped in `pidsmaker/streaming/schemas/`.

    Returns:
        dict: The parsed schema, ready for fastavro.
    """
    candidates = [path, os.path.join(SCHEMAS_DIR, path), os.path.join(SCHEMAS_DIR, f"{path}.avsc")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "r") as f:
                return _import_fastavro().parse_schema(json.load(f))
    raise FileNotFoundError(
        f"Avro schema {path!r} not found. Looked in {candidates}. Bundled schemas: "
        f"{sorted(os.listdir(SCHEMAS_DIR))}"
    )


class RecordDecoder(ABC):
    """Turns a single stream payload into a Python dict."""

    @abstractmethod
    def decode(self, payload: bytes) -> Optional[dict]:
        """Decodes one payload, or returns None if it carries no record."""


class AvroBinaryDecoder(RecordDecoder):
    """Schemaless Avro binary, decoded with the producer's writer schema.

    SPADE writes records with `SpecificDatumWriter` + `BinaryEncoder` and sends
    the resulting bytes as-is: there is no Confluent-style magic byte, no schema
    id and no embedded schema, so the reader must already know the schema.
    """

    def __init__(self, schema: dict):
        self.schema = schema
        self._reader = _import_fastavro().schemaless_reader

    def decode(self, payload: bytes) -> Optional[dict]:
        if not payload:
            return None
        return self._reader(io.BytesIO(payload), self.schema)


class AvroJsonDecoder(RecordDecoder):
    """Avro's JSON encoding: unions are tagged (`{"string": ...}`) and maps wrapped."""

    def __init__(self, schema: dict):
        self.schema = schema
        self._json_reader = _import_fastavro().json_reader

    def decode(self, payload: bytes) -> Optional[dict]:
        if not payload or not payload.strip():
            return None
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return next(iter(self._json_reader(io.StringIO(text), self.schema)))


class JsonDecoder(RecordDecoder):
    """Plain JSON, for producers that do not use Avro at all."""

    def decode(self, payload: bytes) -> Optional[dict]:
        if not payload or not payload.strip():
            return None
        return json.loads(payload)


def build_decoder(fmt: str, schema_path: Optional[str] = None) -> RecordDecoder:
    """Builds the decoder for a wire format.

    Args:
        fmt: One of `FORMATS`.
        schema_path: Avro schema, required by the `avro` and `avro_json` formats.
            Passed to `load_schema()`.

    Returns:
        RecordDecoder: The decoder to hand to a stream pipeline.
    """
    fmt = fmt.strip()
    if fmt == "json":
        return JsonDecoder()
    if fmt in ("avro", "avro_json"):
        if not schema_path:
            raise ValueError(f"Format {fmt!r} requires an Avro schema (`--stream_schema`).")
        schema = load_schema(schema_path)
        return AvroBinaryDecoder(schema) if fmt == "avro" else AvroJsonDecoder(schema)
    raise ValueError(f"Invalid stream format {fmt!r}. Expected one of {FORMATS}.")
