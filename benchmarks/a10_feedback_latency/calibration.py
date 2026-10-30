"""Dedicated-calibration-only A10 correction and serialized phase-cost proxy."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from statistics import median


def fit_calibration(rows, *, hardware_key, predictor_artifact_sha256):
    """Freeze a two-scalar input-only correction from dedicated calibration rows.

    ``rows`` contain isolated successful-update windows and returned complete
    calibration jobs, never evaluation scores. Peak reserved memory is explicitly
    an admission observation, not relabeled as the predictor's allocated target.
    """
    if not rows or not hardware_key or not predictor_artifact_sha256:
        raise ValueError("calibration requires rows and hardware/artifact identity")
    logs, peak_residuals = [], []
    validation, checkpoint, finalization, startup, pause, resume, delivery = [], [], [], [], [], [], []
    for row in rows:
        if row.get("role") != "calibration" or row.get("complete") is not True:
            raise ValueError("only complete dedicated calibration rows may fit correction")
        profile = row["predicted_profile"]
        if profile["artifact_sha256"] != predictor_artifact_sha256:
            raise ValueError("mixed predictor artifact calibration")
        seconds = row["isolated_seconds_per_update"]
        values = (seconds, profile["step_cost"], profile["peak_vram_mib"], row["peak_reserved_mib"])
        if any(isinstance(v, bool) or not math.isfinite(float(v)) or v <= 0 for v in values):
            raise ValueError("invalid isolated calibration observation")
        logs.append(math.log(seconds / (profile["step_cost"] / 1000.)))
        peak_residuals.append(max(0., row["peak_reserved_mib"] - profile["peak_vram_mib"]))
        startup.append(row["startup_seconds"])
        pause.append(row["pause_seconds"])
        resume.append(row["resume_seconds"])
        delivery.append(row["delivery_seconds"])
        for event in row["phase_history"]:
            if event["phase"] == "validation": validation.append(event["seconds"])
            if event["phase"] == "checkpoint": checkpoint.append(event["seconds"])
            if event["phase"] == "finalization": finalization.append(event["seconds"])
    groups = (validation, checkpoint, finalization, startup, pause, resume, delivery)
    if any(not group for group in groups) or any(
            not math.isfinite(float(value)) or value < 0 for group in groups for value in group):
        raise ValueError("complete measured non-update phases are required")
    result = {"schema": "a10_dedicated_calibration_v1", "hardware_key": hardware_key,
        "predictor_artifact_sha256": predictor_artifact_sha256,
        "wall_multiplier": math.exp(median(logs)),
        "peak_add_mib": max(peak_residuals),
        "wall_fit": "exp(median(log(measured_isolated_seconds / predicted_wall_seconds)))",
        "peak_fit": "max(0, maximum(calibration_peak_reserved_mib - predicted_peak_mib))",
        "observations": len(rows), "calibration_names": [r["name"] for r in rows],
        "calibration_parameter_count_max": max(r["parameter_count"] for r in rows),
        "source_identity": [r["source_identity"] for r in rows],
        "measurement_job_signatures": [r["runtime_completion_identity"] for r in rows],
        "observed_at": min(r["observed_at"] for r in rows),
        "measurement_observed_at": [r["observed_at"] for r in rows],
        "costs": dict(zip(("validation", "checkpoint", "finalization", "startup", "pause", "resume", "delivery"),
                          (max(g) for g in groups))),
        "isolated_seconds_per_update_max": max(r["isolated_seconds_per_update"] for r in rows),
        "calibration_ratios": [math.exp(value) for value in logs],
        "memory_target_note": "reserved memory admission proxy; no allocated-VRAM accuracy claim",
        "cost_note": "serialized measured cross-workload maxima, an estimate not a proven upper bound"}
    result["calibration_sha256"] = sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def phase_model(calibration, job, *, hardware_key, batch_size, backend="cuda_process"):
    if calibration["hardware_key"] != hardware_key:
        raise ValueError("calibration hardware mismatch")
    c = calibration["costs"]
    epochs = job.max_epochs or job.config.max_epochs
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("known required epoch count needed for non-update reserve")
    parameters = job.metadata.get("pilot_parameter_count")
    if not isinstance(parameters, int) or parameters <= 0:
        raise ValueError("known input-derived parameter count required for phase proxy")
    scale = max(1., parameters / calibration["calibration_parameter_count_max"])
    return {"source": "measured_phase_model_v1", "covers_required_phases": True,
        "job_signature": job.packing.signature or job.job_id, "batch_size": batch_size,
        "hardware_key": hardware_key, "backend": backend, "observed_at": calibration["observed_at"],
        "startup_seconds": c["startup"],
        "finish_seconds": epochs * c["validation"] * scale + ((epochs + 1) * c["checkpoint"] + c["finalization"]) * scale,
        "delivery_seconds": c["delivery"], "pause_seconds": c["pause"] * scale,
        "resume_seconds": c["resume"] * scale,
        "calibration_sha256": calibration["calibration_sha256"],
        "model_role": "calibrated_cross_workload_phase_proxy_v1",
        "observation_source": "dedicated_disjoint_calibration_not_this_target_job",
        "parameter_scale": scale,
        "parameter_scale_rule": "max(1,target_parameter_count/maximum_calibration_parameter_count); model estimate, not a bound",
        "measurement_job_signatures": calibration["measurement_job_signatures"],
        "calibration_transfer": "legacy source tag denotes model compatibility; target identity is NOT a direct measurement"}


def trial_reserve(calibration):
    c = calibration["costs"]
    # Two warmup updates + one 8-update matching window per phase. This reserves
    # measured resident time conservatively even though updates advance the job.
    return c["startup"] + c["pause"] + c["resume"] + 20 * calibration["isolated_seconds_per_update_max"]


class CalibratedPredictor:
    """Wrap the original released A10 predictor; never fit during evaluation."""
    def __init__(self, predictor, calibration, hardware_key):
        if (predictor.artifact_hash != calibration["predictor_artifact_sha256"]
                or hardware_key != calibration["hardware_key"]):
            raise ValueError("calibrated predictor artifact/hardware mismatch")
        self.predictor, self.calibration = predictor, deepcopy(calibration)
        self.artifact_hash = sha256((predictor.artifact_hash + calibration["calibration_sha256"]).encode()).hexdigest()
        self.model_id = predictor.model_id + "+dedicated_a10_calibration_v1"

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def predict_profile_options(self, job, batch_sizes):
        outputs = deepcopy(self.predictor.predict_profile_options(job, batch_sizes))
        for output in outputs.values():
            if output["hardware"] != "A10":
                raise ValueError("this pilot requires the A10 release")
            output["step_cost"] *= self.calibration["wall_multiplier"]
            output["epoch_cost"] *= self.calibration["wall_multiplier"]
            output["peak_vram_mib"] = max(output["avg_vram_mib"], output["peak_vram_mib"] + self.calibration["peak_add_mib"])
            output["artifact_sha256"] = self.artifact_hash
            output["calibration"] = "frozen_dedicated_a10_calibration_v1"
            output["calibration_sha256"] = self.calibration["calibration_sha256"]
            output["derived_solo_slope"] = {**output.get("derived_solo_slope", {}),
                "value": 1000. / output["step_cost"], "artifact_sha256": self.artifact_hash,
                "calibration": output["calibration"], "source": "dedicated_calibrated_wall_reciprocal"}
        return outputs


class RecordingPredictor:
    """Record profiles from actual preparation calls; never perform extra inference."""
    def __init__(self, predictor, on_profile):
        self.predictor, self.on_profile, self.records = predictor, on_profile, {}
        self.failures, self.preparations_finished = {}, set()

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def predict_profile_options(self, job, batch_sizes):
        try:
            outputs = self.predictor.predict_profile_options(job, batch_sizes)
            for batch in batch_sizes:
                profile = outputs.get(batch)
                if (not isinstance(profile, dict) or profile.get("kind") != "predicted_workload"
                        or profile.get("hardware") != "A10"
                        or profile.get("job_signature") != (job.packing.signature or job.job_id)
                        or profile.get("batch_size") != batch or not profile.get("artifact_sha256")):
                    raise ValueError("missing or incompatible prepared A10 profile")
                for field in ("step_cost", "epoch_cost", "avg_vram_mib", "peak_vram_mib"):
                    if not math.isfinite(float(profile[field])) or profile[field] <= 0:
                        raise ValueError("invalid prepared numeric profile:" + field)
            previous = self.records.setdefault(job.job_id, {})
            if any(previous.get(batch) != value for batch, value in outputs.items()):
                previous.update(deepcopy(outputs))
                self.on_profile(job.job_id, deepcopy(previous))
            return outputs
        except Exception as exc:
            self.failures[job.job_id] = {"type": type(exc).__name__, "error": str(exc)}
            raise
        finally:
            self.preparations_finished.add(job.job_id)
