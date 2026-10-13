"""Reads recorded sessions back in order."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_records(paths: Iterable[str | Path]) -> Iterator[dict]:
    """Yield records from one or more .jsonl.gz recordings, file by file in name order.

    A truncated final line (e.g. a recording killed mid-write) is skipped.
    """
    for path in sorted(Path(p) for p in paths):
        with gzip.open(path, "rt") as f:
            try:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            except EOFError:
                continue
