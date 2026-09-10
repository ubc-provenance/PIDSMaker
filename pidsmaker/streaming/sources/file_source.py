"""File stream source: replays provenance captured to disk.

SPADE's Kafka storage can write to a local file instead of (or as well as) a
broker, which is the easiest way to capture a workload once and replay it as
many times as needed. The same source also tails a growing file, so a capture in
progress can be consumed live without a broker at all.

Formats mirror what SPADE's Kafka storage produces:
    - `avro_container`: the binary output of its file writer (an Avro object
      container file, schema embedded).
    - `avro_json`: one Avro-JSON object per line, what the file writer produces
      when the configured path ends in `.json`.
    - `json`: one plain JSON object per line.
"""

import os
import time
from typing import Iterator, Optional

from pidsmaker.streaming.sources.base import StreamSource
from pidsmaker.utils.utils import log

FILE_FORMATS = ("avro_container", "avro_json", "json")


class FileSource(StreamSource):
    """Replays a capture file, optionally following it as it grows.

    Args:
        path: File to read.
        fmt: One of `FILE_FORMATS`.
        follow: Keep the file open and wait for new records once the end is
            reached, instead of stopping (`tail -f` semantics).
        poll_timeout: Seconds to wait before emitting an idle tick while following.
        rate: Replay speed in records/second (0 = as fast as possible). Useful to
            simulate a live host from a recorded capture.
        max_records: Stop after this many records (0 = unlimited).
        idle_ticks_before_stop: While following, stop after this many consecutive
            idle ticks (0 = never stop).
    """

    def __init__(
        self,
        path: str,
        fmt: str = "avro_json",
        follow: bool = False,
        poll_timeout: float = 1.0,
        rate: float = 0.0,
        max_records: int = 0,
        idle_ticks_before_stop: int = 0,
    ):
        if fmt not in FILE_FORMATS:
            raise ValueError(f"Invalid file format {fmt!r}. Expected one of {FILE_FORMATS}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Stream file not found: {path}")

        self.path = path
        self.fmt = fmt
        self.follow = follow
        self.poll_timeout = poll_timeout
        self.rate = rate
        self.max_records = max_records
        self.idle_ticks_before_stop = idle_ticks_before_stop
        self.yields_records = fmt == "avro_container"

        self._stopped = False
        self.num_records = 0
        log(f"File source: {path} (format={fmt}, follow={follow})")

    def __iter__(self) -> Iterator[Optional[bytes]]:
        if self.fmt == "avro_container":
            yield from self._iter_avro_container()
        else:
            yield from self._iter_lines()

    def _throttle(self):
        if self.rate > 0:
            time.sleep(1.0 / self.rate)

    def _iter_avro_container(self):
        from pidsmaker.streaming.decoders import _import_fastavro

        reader = _import_fastavro().reader
        with open(self.path, "rb") as f:
            for record in reader(f):
                if self._stopped:
                    return
                self.num_records += 1
                yield record
                self._throttle()
                if self.max_records and self.num_records >= self.max_records:
                    log(f"Reached max_records={self.max_records}, stopping.")
                    return

    def _iter_lines(self):
        idle_ticks = 0
        with open(self.path, "rb") as f:
            while not self._stopped:
                line = f.readline()

                if not line:
                    if not self.follow:
                        return
                    idle_ticks += 1
                    yield None
                    if self.idle_ticks_before_stop and idle_ticks >= self.idle_ticks_before_stop:
                        log(f"File source idle for {idle_ticks} polls, stopping.")
                        return
                    time.sleep(self.poll_timeout)
                    continue

                # A partially written last line is re-read on the next pass rather
                # than handed over incomplete.
                if self.follow and not line.endswith(b"\n"):
                    f.seek(-len(line), os.SEEK_CUR)
                    time.sleep(self.poll_timeout)
                    continue

                if not line.strip():
                    continue

                idle_ticks = 0
                self.num_records += 1
                yield line
                self._throttle()

                if self.max_records and self.num_records >= self.max_records:
                    log(f"Reached max_records={self.max_records}, stopping.")
                    return

    def stop(self) -> None:
        """Asks the iteration to end after the record being processed."""
        self._stopped = True
