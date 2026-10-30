"""Adapter between v4's pure objective and the existing add-only lifecycle."""
from copy import deepcopy
import time

from ..domain import BatchResolution, JobStatus
from .completion_objective import remaining_optimizer_updates
from .feedback_objective import (
    CompletionCosts, FEEDBACK_POLICY, compare_feedback_admission, derived_solo_slope,
    number, requests_from_jobs, timestamp,
)


def forecast_cohort(service):
    all_jobs = service.store.list_jobs()
    outstanding = [j for j in all_jobs if not j.status.is_terminal]
    referenced = set()
    for job in outstanding:
        raw = job.metadata.get("feedback_request")
        if isinstance(raw, dict) and isinstance(raw.get("required_job_ids"), (list, tuple)):
            referenced.update(raw["required_job_ids"])
    jobs = [j for j in all_jobs if not j.status.is_terminal or j.job_id in referenced]
    requests, completed = requests_from_jobs(jobs)
    if any(j.hold for j in outstanding):
        raise ValueError("held job has no forecast release time")
    return outstanding, requests, completed


def profile_snapshot(service, jobs):
    profiles = {j.job_id: service.planner.estimator.predicted_profile(j) for j in jobs}
    if not all(profiles.values()):
        raise ValueError("every outstanding job requires a valid compatible prediction")
    if len({(p["artifact_sha256"], p["hardware"]) for p in profiles.values()}) != 1:
        raise ValueError("mixed predictor artifact or hardware scale")
    for profile in profiles.values():
        derived_solo_slope(profile["step_cost"])
    return deepcopy(profiles)


def freeze_feedback_trial(service, trial):
    jobs, requests, _ = forecast_cohort(service)
    return {"profiles": profile_snapshot(service, jobs),
            "settings": service.settings.gpu_scheduler.feedback_latency.to_dict(),
            "requests": [r.to_dict() for r in requests],
            "active_job_ids": sorted(service.supervisor.active_job_ids()),
            "job_ids": sorted(j.job_id for j in jobs)}


def remaining_for_forecast(job):
    step = job.metadata.get("runtime_global_step")
    if step is None:
        if job.started_at or job.latest_checkpoint_path or job.resume_from_checkpoint:
            raise ValueError("resumed job has no exact update counter")
        step = 0
    steps = job.metadata.get("runtime_steps_per_epoch")
    if steps is None:
        from ..prediction.released_student import steps_for_job
        try:
            steps = steps_for_job(job, BatchResolution.resolved_batch_size(job))
        except (ValueError, KeyError, TypeError):
            steps = None
    remaining = remaining_optimizer_updates(job, {
        "global_step": step, "steps_per_epoch": steps,
        "total_optimizer_updates": job.metadata.get("runtime_total_optimizer_updates"),
        "planned_epochs": job.metadata.get("runtime_planned_epochs"),
    })
    if remaining is None or remaining <= 0:
        raise ValueError("unknown or exhausted remaining update budget")
    return remaining


def phase_costs(service, job, now):
    settings = service.settings.gpu_scheduler.feedback_latency
    if settings.allow_zero_overhead:
        return CompletionCosts(), "explicit_research_zero_overhead"
    raw = job.metadata.get("feedback_phase_model") or {}
    if not isinstance(raw, dict):
        raise ValueError("completion phase model must be an object")
    # An external timing collector must supply a matching, complete model.
    # Absence is unknown, never an implicit zero or a v3 fallback.
    if (raw.get("source") != "measured_phase_model_v1"
            or raw.get("covers_required_phases") is not True
            or raw.get("job_signature") != (job.packing.signature or job.job_id)
            or raw.get("batch_size") != BatchResolution.resolved_batch_size(job)
            or raw.get("hardware_key") != service.store.hardware_key()
            or raw.get("backend") != service.settings.gpu_scheduler.packing_backend
            or not 0 <= now - timestamp(raw.get("observed_at")) <= settings.phase_model_ttl_seconds):
        raise ValueError("missing, incomplete, or incompatible completion phase model")
    values = {name: number(raw[name + "_seconds"]) for name in ("startup", "finish", "delivery", "pause", "resume")}
    if job.metadata.get("runtime_global_step", 0) > 0 or job.latest_checkpoint_path:
        values["startup"] = values["resume"]
    return CompletionCosts(**values), "measured_serialized_phase_proxy"


def evaluate_feedback_trial(service, trial, jobs, windows, *, compare=compare_feedback_admission):
    result = {"admission_policy": FEEDBACK_POLICY, "decision": "inconclusive", "error": None}
    try:
        frozen = trial.cold_start["add_only"]["feedback"]
        cohort, requests, completed = forecast_cohort(service)
        current = freeze_feedback_trial(service, trial)
        if current != frozen:
            raise ValueError("forecast cohort, request, runtime membership, or predictor changed; fresh trial required")
        active_ids = set(service.supervisor.active_job_ids())
        if active_ids != {j.job_id for j in jobs}:
            raise ValueError("unmodeled active worker")
        now = time.time()
        remaining = {j.job_id: remaining_for_forecast(j) for j in cohort}
        profiles = frozen["profiles"]
        solo = {j: derived_solo_slope(profiles[j]["step_cost"]) for j in remaining}
        phase_models = {j.job_id: phase_costs(service, j, now) for j in cohort}
        queue = [j.job_id for j in sorted(cohort, key=lambda j: (timestamp(j.submitted_at), j.job_id))
                 if j.job_id not in trial.preexisting_job_ids]
        pre = trial.cold_start["add_only"]["pre_add"]
        result = compare(
            remaining=remaining, solo=solo, incumbents=trial.preexisting_job_ids,
            candidate=trial.candidate_job_id, queue=queue,
            pre_halves={j: [1 / number(s, positive=True) for s in halves] for j, halves in pre.items()},
            packed_halves={j: [1 / number(s, positive=True) for s in w["halves"]] for j, w in windows.items()},
            requests=requests, completed=completed, now=now,
            costs={j: values[0] for j, values in phase_models.items()},
            min_gain_seconds=service.settings.gpu_scheduler.feedback_latency.min_gain_seconds,
            bracket_admission_mode=service.settings.gpu_scheduler.feedback_latency.bracket_admission_mode)
        result.update(remaining_updates=remaining, predicted_solo_updates_per_second=solo,
                      queue_order=queue, frozen_profiles=profiles,
                      phase_cost_sources={j: v[1] for j, v in phase_models.items()},
                      phase_model="serialized remaining GPU phases; delivery does not hold GPU",
                      min_gain_seconds=service.settings.gpu_scheduler.feedback_latency.min_gain_seconds)
    except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        result["error"] = str(exc)
    return result
