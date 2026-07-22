"""Small atomic filesystem write helpers shared by scheduler processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import tempfile


def atomic_json_dump(
    path: Path,
    payload: dict[str, Any],
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    """Atomically replace a JSON file without sharing a temporary pathname.

    Heartbeats can be emitted by both the scheduler runner and the elastic
    training subprocess. A fixed ``<name>.tmp`` path lets concurrent writers
    rename each other's temporary file, causing the losing writer to fail with
    ``ENOENT``. Each writer therefore creates its own temporary file in the
    destination directory before using an atomic ``os.replace``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            handle.flush()
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
