"""Framework-neutral metadata transport; no candidate code is imported here."""
from __future__ import annotations

import json
from pathlib import Path

from .released_artifacts import digest


def prediction_metadata(job):
    """Explicit job metadata wins; otherwise accept a source-bound JSON sidecar.

    Agents can write ``candidate.py.perfseer.json`` without modifying their job
    runner. The source digest makes an old recipe unusable after a mutation.
    """
    explicit = job.metadata.get("perfseer_model")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("perfseer_model must be a metadata object")
        return dict(explicit)
    source = (job.metadata.get("architecture_source") or job.metadata.get("source_path")
              or job.baseline_model_path)
    if not source:
        return {}
    source = Path(source).expanduser().resolve()
    sidecar = source.with_suffix(source.suffix + ".perfseer.json")
    if not sidecar.is_file():
        return {}
    if sidecar.stat().st_size > 256 * 1024:
        raise ValueError("prediction sidecar exceeds 256 KiB")
    document = json.loads(sidecar.read_text())
    if document.get("schema") != "perfseer-source-job-v1":
        raise ValueError("unsupported prediction sidecar schema")
    if document.get("source_sha256") != digest(source):
        raise ValueError("prediction sidecar source hash mismatch; regenerate the recipe after changing code")
    result = document.get("perfseer_model")
    if not isinstance(result, dict):
        raise ValueError("prediction sidecar requires perfseer_model metadata")
    result = dict(result)
    if result.get("source_path") and Path(result["source_path"]).expanduser().resolve() != source:
        raise ValueError("prediction sidecar must describe its own source file")
    result["source_path"] = str(source)
    return result


def write_prediction_sidecar(source_path, metadata):
    """Call after a code agent has finalized its source and actual training recipe."""
    source = Path(source_path).expanduser().resolve()
    destination = source.with_suffix(source.suffix + ".perfseer.json")
    document = {"schema": "perfseer-source-job-v1", "source_sha256": digest(source),
                "perfseer_model": metadata}
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if len(text.encode()) > 256 * 1024:
        raise ValueError("prediction sidecar exceeds 256 KiB")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(destination)
    return destination
