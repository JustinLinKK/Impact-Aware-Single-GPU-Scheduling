"""Dedicated-solo template controls for scheduler mechanism experiments.

These are measured-template forecasts, not outputs or accuracy improvements of
S_A10. Evaluation outcomes never update the table. Original datasets and model
artifacts are not read or rewritten by this module.
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import mean

from ...prediction import JobPredictionError
from .workloads import digest, expected_signature, spec_digest, template_key

SOURCE = "dedicated_solo_template_proxy"
SCHEMA = "a10_resource_template_v1"


def positive(value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric measurement required")
    if value < 0 or (value == 0 and not zero):
        raise ValueError("positive measurement required")
    return float(value)


def fit_templates(rows, *, hardware_key, gpu_uuid, source_identity):
    """Pool nonoverlapping successful-update windows from dedicated replicas."""
    if not rows or not hardware_key or not gpu_uuid or not source_identity:
        raise ValueError("dedicated data and exact host/source identities required")
    grouped = {}
    seen_jobs = set()
    for row in rows:
        if (row.get("role") != "calibration" or row.get("complete") is not True
                or row.get("hardware_key") != hardware_key or row.get("gpu_uuid") != gpu_uuid
                or not row.get("job_signature") or row["job_signature"] in seen_jobs):
            raise ValueError("only unique complete same-host calibration replicas may fit templates")
        seen_jobs.add(row["job_signature"])
        if not row.get("solo_process_isolation_verified"):
            raise ValueError("solo process isolation was not verified")
        key = row.get("template_key")
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("explicit template identity required")
        windows = row.get("windows") or []
        if len(windows) < 2:
            raise ValueError("multiple independent steady windows required")
        previous_end = -1
        for window in sorted(windows, key=lambda w: w["start_step"]):
            a, b = window["start_step"], window["end_step"]
            if (type(a) is not int or type(b) is not int or a < 2 or b <= a
                    or a < previous_end or b > row["successful_updates"]
                    or window.get("steps") != b-a
                    or window.get("excludes_nonupdate_phases") is not True):
                raise ValueError("window must contain distinct successful updates after warm-up")
            positive(window["elapsed_seconds"])
            previous_end = b
        grouped.setdefault(key, []).append(deepcopy(row))
    entries = {}
    phase_names = ("startup", "resume", "pause", "validation", "checkpoint", "finalization", "delivery")
    for key, replicas in grouped.items():
        steps = sum(w["steps"] for row in replicas for w in row["windows"])
        seconds = sum(w["elapsed_seconds"] for row in replicas for w in row["windows"])
        reserved = max(positive(row["peak_reserved_mib"]) for row in replicas)
        allocated = max(positive(row["peak_allocated_mib"]) for row in replicas)
        device = max(positive(row["peak_device_used_mib"]) for row in replicas)
        if allocated > reserved + 1e-6:
            raise ValueError("allocated memory exceeds allocator reservation")
        allocated_samples = [positive(v) for row in replicas for v in row["steady_allocated_mib"]]
        if not allocated_samples or max(allocated_samples) > allocated + 1e-6:
            raise ValueError("valid steady allocated-memory samples required")
        busy = [positive(v, zero=True) for row in replicas for v in row["steady_gpu_busy_percent"]]
        if not busy or max(busy) > 100:
            raise ValueError("valid steady GPU-busy samples required")
        phases = {name: max(positive(row["phase_seconds"][name], zero=True) for row in replicas)
                  for name in phase_names}
        entries[key] = {"updates_per_second": steps / seconds,
            "measured_update_count": steps, "measured_window_seconds": seconds,
            "measurement_windows": sum(len(row["windows"]) for row in replicas),
            "warmup_updates_per_window": 2, "calibration_replicas": len(replicas),
            "peak_allocated_mib": allocated, "peak_reserved_mib": reserved,
            "mean_allocated_mib": mean(allocated_samples),
            "peak_device_used_mib": device,
            "admission_reservation_mib": max(reserved, device) + max(128., .05*reserved),
            "gpu_busy_percent": mean(busy), "phase_seconds": phases,
            "measurement_signatures": [row["job_signature"] for row in replicas],
            "observed_at": min(row["observed_at"] for row in replicas),
            "memory_note": "explicit admission reserve; allocated and reserved measurements remain separate",
            "busy_note": "sampled NVML kernel-active time, not SM occupancy or arithmetic capacity"}
    table = {"schema": SCHEMA, "source": SOURCE, "hardware_key": hardware_key,
        "gpu_uuid": gpu_uuid, "source_identity": deepcopy(source_identity), "entries": entries,
        "measurement_note": "successful updates divided by elapsed solo training-window wall time; warm-up/non-update phases excluded",
        "scope": "same-configuration dedicated measurement control; not held-out neural accuracy"}
    table["artifact_sha256"] = digest(table)
    return table


def duration_matched_budget(entry, target_seconds, *, epochs=2, maximum_updates=65536):
    """Choose the evaluation work budget before any policy outcome is observed."""
    rate, target = positive(entry["updates_per_second"]), positive(target_seconds)
    if type(epochs) is not int or epochs < 1 or type(maximum_updates) is not int or maximum_updates < 32:
        raise ValueError("valid epoch/budget bounds required")
    quantum = math.lcm(8, epochs)
    maximum = maximum_updates // quantum * quantum
    total = min(maximum, max(32, math.ceil(rate*target/quantum)*quantum))
    return {"total_updates": total, "epochs": epochs, "steps_per_epoch": total // epochs,
        "target_training_seconds": target, "predicted_training_seconds": total/rate,
        "clipped": rate*target > maximum,
        "selection_source": "dedicated calibration only; frozen before arm evaluation"}


class MeasuredTemplatePredictor:
    """Explicit common profile source; an unknown configuration stays unknown."""
    available = True
    unavailable_reason = None
    model_id = SCHEMA
    uses_predicted_profiles = True
    requires_measured_runtime = True
    requires_branch_memory = False
    profile_source = SOURCE

    def __init__(self, table, hardware, *, gpu_uuid):
        expected = digest({k:v for k,v in table.items() if k != "artifact_sha256"})
        if (table.get("schema") != SCHEMA or table.get("artifact_sha256") != expected
                or table.get("hardware_key") != hardware.hardware_key or table.get("gpu_uuid") != gpu_uuid):
            raise ValueError("template source or exact host identity mismatch")
        self.table, self.hardware = deepcopy(table), hardware
        self.artifact_hash = expected
        self.last_sources, self.last_errors = {}, {}

    def predict_profile_options(self, job, batch_sizes):
        try:
            spec, entry = compatible_template(self.table, job)
            if list(batch_sizes) != [spec["batch"]]:
                raise ValueError("template has no observation for another physical batch")
            wall = 1000. / positive(entry["updates_per_second"])
            profile = {"kind": "predicted_workload", "artifact_sha256": self.artifact_hash,
                "hardware": "A10", "batch_size": spec["batch"],
                "job_signature": job.packing.signature or job.job_id,
                "step_cost": wall, "epoch_cost": wall * spec["steps_per_epoch"],
                "busy_indicator": entry["gpu_busy_percent"],
                "avg_vram_mib": entry["mean_allocated_mib"],
                "peak_vram_mib": entry["admission_reservation_mib"],
                "profile_source": SOURCE, "calibration": SCHEMA,
                "requires_online_validation": True,
                "memory_semantics": "avg is a dedicated steady allocated-memory mean; peak is a conservative reserved/device admission bound, not a learned allocated-peak target",
                "derived_solo_slope": {"value": entry["updates_per_second"],
                    "unit": "successful_updates_per_second", "artifact_sha256": self.artifact_hash,
                    "source": SOURCE, "calibration": SCHEMA}}
            self.last_sources[job.job_id] = SOURCE
            self.last_errors.pop(job.job_id, None)
            return {spec["batch"]: profile}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            self.last_errors[job.job_id] = str(exc)
            raise JobPredictionError("missing compatible dedicated template: " + str(exc)) from exc


def compatible_template(table, job):
    """Bind a lookup to the sealed recipe and its actual code/runtime identity."""
    spec = job.metadata["resource_workload_spec"]
    if (spec.get("spec_sha256") != spec_digest(spec)
            or spec.get("expected_runtime_signature") != expected_signature(spec)):
        raise ValueError("resource workload identity mismatch")
    actual_key = template_key(spec)
    if (spec.get("template_key") != actual_key
            or job.metadata.get("resource_template_key") != actual_key):
        raise ValueError("resource template key does not match sealed workload")
    return spec, table["entries"][actual_key]


def phase_model(table, job, *, hardware_key, batch_size, backend="cuda_process"):
    if table["hardware_key"] != hardware_key:
        raise ValueError("phase template hardware mismatch")
    spec, entry = compatible_template(table, job)
    if type(batch_size) is not int or spec["batch"] != batch_size:
        raise ValueError("phase template batch mismatch")
    costs, epochs = entry["phase_seconds"], job.max_epochs or job.config.max_epochs
    return {"source": "measured_phase_model_v1", "covers_required_phases": True,
        "job_signature": job.packing.signature or job.job_id, "batch_size": batch_size,
        "hardware_key": hardware_key, "backend": backend, "observed_at": entry["observed_at"],
        "startup_seconds": costs["startup"], "resume_seconds": costs["resume"],
        "pause_seconds": costs["pause"], "delivery_seconds": costs["delivery"],
        "finish_seconds": epochs*costs["validation"]+(epochs+1)*costs["checkpoint"]+costs["finalization"],
        "model_role": "dedicated_same_configuration_template_proxy",
        "calibration_sha256": table["artifact_sha256"],
        "measurement_job_signatures": entry["measurement_signatures"]}
