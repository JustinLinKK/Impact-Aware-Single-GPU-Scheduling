"""Agent-facing result receipts after successful worker exit and artifact parse.

The scheduler's COMPLETED state precedes process exit. These receipts deliberately
include polling, final artifact transport, parsing, and checksum costs in JCT.
Checkpoint/optimizer-state replay is a separate post-run validity audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True)
class Submission:
    job_id: str
    accepted_monotonic: float
    accepted_unix: float
    expected_updates: int
    expected_epochs: int
    result_path: str
    execution_path: str
    expected_runtime_signature: str
    expected_workload_sha256: str
    validation_size: int = 256

    def __post_init__(self):
        for value in (self.expected_updates, self.expected_epochs):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("positive integer training budget required")
        if not self.job_id or any(not math.isfinite(t) for t in
                                 (self.accepted_monotonic, self.accepted_unix)):
            raise ValueError("invalid accepted submission")
        for identity in (self.expected_runtime_signature, self.expected_workload_sha256):
            if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                raise ValueError("immutable expected signature/hash required")


def _artifact(path: str) -> tuple[dict[str, Any], str, int]:
    raw = Path(path).read_bytes()
    value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("result artifact must contain an object")
    return value, sha256(raw).hexdigest(), len(raw)


def receive(submission: Submission, job, observed_exit_code: int | None,
            *, monotonic=time.monotonic, wall_clock=time.time):
    """Return a durable-receipt payload, None while pending, or fail closed.

    ``job`` is the scheduler's latest TrainingJob (a small duck-typed object in
    tests). The caller persists the returned receipt before admitting dependent
    requests. No labels, score comparisons, or predictor output affect readiness.
    """
    status = getattr(job.status, "value", job.status)
    status = str(status).upper()
    if job.job_id != submission.job_id:
        raise ValueError("submission/job identity mismatch")
    if status in {"FAILED", "CANCELLED"} or observed_exit_code not in (None, 0):
        raise RuntimeError(f"required job {job.job_id} failed: {status}/{observed_exit_code}")
    if status != "COMPLETED" or observed_exit_code is None:
        return None
    metadata = job.metadata
    if (metadata.get("runtime_global_step") != submission.expected_updates
            or metadata.get("runtime_total_optimizer_updates") != submission.expected_updates
            or metadata.get("runtime_finalization_complete") is not True):
        raise ValueError("completed job lacks exact update/finalization proof")
    result, result_hash, result_bytes = _artifact(submission.result_path)
    execution, execution_hash, execution_bytes = _artifact(submission.execution_path)
    if (type(result.get("global_step")) is not int
            or result["global_step"] != submission.expected_updates
            or type(result.get("epochs")) is not int
            or result["epochs"] != submission.expected_epochs
            or result.get("signature") != submission.expected_runtime_signature
            or result["signature"] != metadata.get("runtime_completion_identity")):
        raise ValueError("returned training artifact identity or budget mismatch")
    if execution.get("exc_type") is not None:
        raise ValueError("returned execution artifact reports failure")
    history = result.get("history")
    if (result.get("metric_name") != "negative_validation_auc" or not isinstance(history, list)
            or len(history) != submission.expected_epochs):
        raise ValueError("AUC maximize selection history is required")
    for epoch in history:
        auc, bce = epoch.get("validation_auc"), epoch.get("validation_bce")
        counts = [epoch.get("validation_negative_count"), epoch.get("validation_positive_count")]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in [auc, bce, *counts]) or not 0 <= auc <= 1 or bce < 0
                or any(v <= 0 or not float(v).is_integer() for v in counts)
                or sum(counts) != submission.validation_size
                or not math.isclose(epoch.get("negative_validation_auc", float("nan")), -auc,
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError("finite AUC/BCE and both held-out classes are required")
    best = min(history, key=lambda row: row["negative_validation_auc"])
    if (not isinstance(result.get("selected_metric"), (int, float))
            or not math.isclose(result["selected_metric"], best["negative_validation_auc"],
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("selected checkpoint does not maximize held-out AUC")
    ready_mono = monotonic()
    if not math.isfinite(ready_mono) or ready_mono < submission.accepted_monotonic:
        raise ValueError("invalid receipt clock")
    return {**asdict(submission), "feedback_ready_monotonic": ready_mono,
            "feedback_ready_unix": wall_clock(),
            "jct_seconds": ready_mono - submission.accepted_monotonic,
            "worker_exit_code": observed_exit_code,
            "result_sha256": result_hash, "result_bytes": result_bytes,
            "execution_sha256": execution_hash, "execution_bytes": execution_bytes,
            "runtime_completion_identity": result["signature"],
            "selected_auc": best["validation_auc"], "auc_maximize": True,
            "selected_validation_bce": best["validation_bce"], "selected_epoch": best["epoch"],
            "returned_result": result}


def request_latency(mode: str, required_ids, receipts, submitted_monotonic: float):
    """Individual JCT or all-required bracket JCT; missing jobs are not dropped."""
    required_ids = tuple(required_ids)
    if (mode not in {"individual", "group"} or not required_ids
            or len(set(required_ids)) != len(required_ids)
            or (mode == "individual" and len(required_ids) != 1)):
        raise ValueError("invalid finalized request membership")
    if not set(required_ids) <= receipts.keys():
        return None
    ready = max(receipts[j]["feedback_ready_monotonic"] for j in required_ids)
    if not math.isfinite(submitted_monotonic) or ready < submitted_monotonic:
        raise ValueError("invalid request clock")
    return ready - submitted_monotonic
