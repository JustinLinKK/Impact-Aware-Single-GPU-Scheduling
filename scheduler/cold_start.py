"""Incremental trials which acquire their own timing references during training."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
import time

from ..domain import BatchResolution, JobStatus, RuntimeProfile, build_colocation_profile_key, utc_now
from .planner_types import DispatchPlan


def reference_identity(job):
    return {
        "signature": job.packing.signature,
        "batch": BatchResolution.resolved_batch_size(job),
        "backend": job.metadata.get("placement_backend"),
        "config": job.metadata.get("placement_backend_config") or {},
        "precision": job.metadata.get("precision_mode"),
    }


def cold_start_plan(planner, jobs, active, available, used_mb, *, mandatory, now):
    settings = planner.settings.gpu_scheduler
    backend = settings.packing_backend
    if not settings.colocation.live_trial_enabled or not available.get(backend):
        return None
    if any(
        not job.metadata.get("cooperative_trial")
        or job.metadata.get("placement_backend") != backend
        for job in active
    ):
        return None
    budget = min(31 * 1024.0, planner.estimator.safe_budget_mb())
    allowance = budget - used_mb - max(256.0, budget * 0.05)
    if allowance <= 0:
        return None
    ordered = []
    for job in jobs:
        if not job.metadata.get("cooperative_trial"):
            continue
        if not planner.compatibility.pack_eligible(job, backend_name=backend):
            continue
        if not planner.compatibility.compatible_group([*active, job], backend_name=backend):
            continue
        batch = BatchResolution.resolved_batch_size(job)
        memory = planner.estimator.estimate_peak_vram_mb(job, batch, backend)
        if job.metadata.get("trial_allocator_supported") is False and not job.resource_requirements.estimated_vram_mb:
            continue
        if memory > allowance:
            continue
        allocations = (job.metadata.get("placement_backend_config") or {}).get("allocation_percentages") or [100]
        ceiling = max(1, min(100, int(job.metadata.get("mps_active_thread_pct", allocations[0]))))
        config = {"allocation_percentages": [ceiling]} if backend == "mps_process" else {}
        descriptors = [planner.time_objective.member_descriptor(
            member, backend_name=backend,
            batch_size=BatchResolution.resolved_batch_size(member),
            backend_config=config if member.job_id == job.job_id else member.metadata.get("placement_backend_config"),
        ) for member in [*active, job]]
        key = build_colocation_profile_key(planner.repository.hardware_key(), descriptors)
        if key in job.metadata.get("cold_start_blocked_groups", []):
            continue
        remaining = planner.predicted_remaining_runtime_seconds(job, backend_name=backend)
        ordered.append(((
            remaining is None, remaining if remaining is not None else 0.0,
            -planner._effective_priority(job, now=now), job.queue_sequence, job.job_id,
        ), job, key, config))
    if not ordered:
        return None
    _, job, key, config = min(ordered, key=lambda item: item[0])
    return DispatchPlan(
        mode="concurrent_group" if active else "stack_anchor",
        backend_name=backend,
        job_ids=(job.job_id,),
        reason="cold-start addition trial" if active else "cooperative stack anchor",
        backend_config=config,
        trial_metadata={
            "cooperative_trial": True,
            "cold_start": bool(active),
            "requires_live_trial": bool(active),
            "preexisting_job_ids": [member.job_id for member in active],
            "profile_key": key,
            "memory_limit_mb": allowance,
        },
        objective_breakdown={"cold_start": bool(active), "gain": None},
        mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
        objective_version=settings.objective.objective_version,
    )


def normalized_throughput(references, pre_add, packed, candidate_id, margin):
    """Use both independent halves; unknown/invalid measurements are not zero."""
    gains = []
    try:
        for half in range(2):
            before = sum(
                references[job_id][half] / rates[half]
                for job_id, rates in pre_add.items()
            )
            after = sum(references[job_id][half] / rates[half] for job_id, rates in packed.items())
            if candidate_id not in packed or before <= 0:
                return "inconclusive", []
            if any(not isfinite(value) or value <= 0 for rates in [*references.values(), *pre_add.values(), *packed.values()] for value in rates):
                return "inconclusive", []
            gains.append(after / before)
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return "inconclusive", []
    if min(gains) >= 1.0 + margin:
        return "accepted", gains
    if max(gains) < 1.0:
        return "rejected", gains
    return "inconclusive", gains


class ColdStartTrialMixin:
    def _save_cold_reference(self, job, halves):
        self.store.update_job(job.job_id, metadata_updates={"trial_solo_reference": {
            "identity": reference_identity(job), "halves": halves, "at": time.time(),
        }})
        steps = job.metadata.get("runtime_steps_per_epoch")
        rate = sum(halves) / 2
        epochs = job.max_epochs or job.config.max_epochs
        self.store.upsert_runtime_profile(RuntimeProfile.create(
            signature=job.packing.signature or job.job_id, hardware_key=self.store.hardware_key(),
            backend_name=str(job.metadata.get("placement_backend")),
            resolved_batch_size=BatchResolution.resolved_batch_size(job), strategy="step_window",
            avg_step_time_ms=rate * 1000, steps_per_epoch=steps,
            epoch_1_seconds=rate * steps if steps else None,
            estimated_total_runtime_seconds=rate * steps * epochs if steps and epochs else None,
            confidence=0.6, observations=1, source="cold_start_solo_window", last_job_id=job.job_id,
            metadata={"reference_identity": reference_identity(job), "reference_halves": halves},
        ))

    def _cold_command(self, trial, job, action, kind, *, reset=False):
        previous = dict(job.metadata.get("trial_step_command") or {})
        if not reset and previous.get("owner") == trial.trial_id and previous.get("action") == action and previous.get("kind") == kind:
            if float(previous.get("expires_at", 0)) > time.time() + 1.0:
                return
            command = {**previous, "expires_at": time.time() + 2.0}
        else:
            command = {
                "owner": trial.trial_id, "token": f"{trial.trial_id}-{time.monotonic_ns()}",
                "action": action, "kind": kind, "expires_at": time.time() + 2.0,
                "candidate": job.job_id == trial.candidate_job_id,
                "target_epoch": trial.target_epoch,
            }
        self.store.update_job(job.job_id, metadata_updates={"trial_step_command": command})

    def _cold_phase(self, trial, phase, jobs, *, measuring=None):
        trial.phase = phase
        trial.cold_start["phase_started_at"] = time.time()
        trial.cold_start["measuring"] = measuring
        trial.cold_start.setdefault("phases", []).append({
            "phase": phase, "started_at": utc_now(), "measuring": measuring,
            "members": {job.job_id: job.metadata.get("runtime_global_step") for job in jobs},
        })
        for job in jobs:
            if phase == "pre_add":
                action = "yield" if job.job_id == trial.candidate_job_id else "measure"
            elif phase == "reference":
                action = "measure" if job.job_id == measuring else "yield"
            elif phase == "measure_overlap":
                action = "measure"
            else:
                action = "yield"
            self._cold_command(trial, job, action, phase, reset=True)
        self.store.update_job(trial.candidate_job_id, metadata_updates={
            "colocation_trial": {**trial.to_dict(), "decision": "pending"},
        })
        self._persist_scheduler_decision_state()
        self.event_logger.emit("cold_start_phase", job_id=trial.candidate_job_id, payload={
            "trial_id": trial.trial_id, "phase": phase, "measuring": measuring,
            "member_ids": [job.job_id for job in jobs],
        })

    @staticmethod
    def _cold_window(job):
        window = job.metadata.get("trial_step_window") or {}
        command = job.metadata.get("trial_step_command") or {}
        if not command.get("token") or window.get("token") != command.get("token") or not window.get("complete"):
            return None
        try:
            if len(window["halves"]) != 2 or any(not isfinite(value) or value <= 0 for value in window["halves"]):
                return None
        except (KeyError, TypeError):
            return None
        return window

    def _finish_cold_start_trial(self, trial, decision, reason, result=None):
        candidate = self.store.get_job(trial.candidate_job_id)
        result = result or {}
        for job_id in [*trial.preexisting_job_ids, trial.candidate_job_id]:
            job = self.store.get_job(job_id)
            if job is None:
                continue
            command = job.metadata.get("trial_step_command") or {}
            if command.get("owner") == trial.trial_id:
                self.store.update_job(job_id, metadata_updates={"trial_step_command": {}})
        if candidate is not None:
            blocked = list(candidate.metadata.get("cold_start_blocked_groups") or [])
            if decision != "accepted" and trial.profile_key not in blocked:
                blocked.append(trial.profile_key)
            self.store.update_job(candidate.job_id, metadata_updates={
                "cold_start_blocked_groups": blocked,
                "colocation_trial": {**trial.to_dict(), "phase": "decide", "decision": decision, "reason": reason, "result": result},
                "cold_trial_completed": True,
            })
            if decision != "accepted" and not candidate.status.is_terminal:
                if self.supervisor.request_pause(candidate.job_id, reason=reason, hold=False):
                    self.store.update_job(candidate.job_id, status=JobStatus.PAUSING, reason=reason, hold=False)
        self.event_logger.emit("cold_start_trial_decided", job_id=trial.candidate_job_id, payload={
            "trial_id": trial.trial_id, "decision": decision, "reason": reason,
            "wall_seconds": max(0.0, time.time() - datetime.fromisoformat(trial.started_at).timestamp()),
            **result,
        })
        self._colocation_trial = None
        self._persist_scheduler_decision_state()

    def _evaluate_cold_start_trial(self, trial):
        candidate = self.store.get_job(trial.candidate_job_id)
        if candidate is None:
            self._finish_cold_start_trial(trial, "inconclusive", "candidate disappeared")
            return
        active = self._active_jobs()
        expected = {*trial.preexisting_job_ids, trial.candidate_job_id}
        jobs = [job for job in active if job.job_id in expected]
        if candidate.status.is_terminal or candidate.metadata.get("trial_lease_expired") == trial.trial_id:
            self._finish_cold_start_trial(trial, "inconclusive", "candidate completed or measurement lease expired")
            return
        if set(job.job_id for job in jobs) != expected:
            if trial.phase == "prepare" and candidate.job_id not in {job.job_id for job in active}:
                return
            remaining = [job for job in jobs if job.job_id != candidate.job_id and not job.status.is_terminal]
            if candidate.job_id in {job.job_id for job in jobs} and remaining and int(candidate.metadata.get("last_completed_epoch", 0)) < trial.target_epoch:
                trial.preexisting_job_ids = tuple(job.job_id for job in remaining)
                descriptors = [self._member_descriptor(job, trial.backend_name) for job in [*remaining, candidate]]
                trial.profile_key = build_colocation_profile_key(self.store.hardware_key(), descriptors)
                self._cold_phase(trial, "pre_add", [*remaining, candidate])
                return
            self._finish_cold_start_trial(trial, "accepted" if not remaining and candidate.job_id in {job.job_id for job in jobs} else "inconclusive", "pack membership changed")
            return
        if self._device_samples and self._device_samples[-1].memory_used_mb > min(31 * 1024, self.planner.estimator.safe_budget_mb()):
            self._finish_cold_start_trial(trial, "inconclusive", "live memory budget exceeded")
            return
        if trial.phase == "prepare":
            self._cold_command(trial, candidate, "yield", "prepare")
            if not all(job.metadata.get("trial_runner_ready") for job in jobs):
                if time.time() - datetime.fromisoformat(trial.started_at).timestamp() > 30:
                    self._finish_cold_start_trial(trial, "inconclusive", "runner preparation timed out")
                return
            self._cold_phase(trial, "pre_add", jobs)
            return
        phase_elapsed = time.time() - float(trial.cold_start.get("phase_started_at", time.time()))
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            self._cold_command(trial, job, command.get("action", "yield"), command.get("kind", trial.phase))
        if any(job.metadata.get("trial_lease_expired") == trial.trial_id for job in jobs):
            self._finish_cold_start_trial(trial, "inconclusive", "measurement lease expired")
            return
        windows = {job.job_id: self._cold_window(job) for job in jobs}
        if trial.phase != "measure_overlap" and int(candidate.metadata.get("last_completed_epoch", 0)) >= trial.target_epoch:
            self._finish_cold_start_trial(trial, "inconclusive", "one-epoch reference budget exhausted")
            return
        if trial.phase == "pre_add":
            if not all(windows[job_id] for job_id in trial.preexisting_job_ids):
                if phase_elapsed > 2:
                    self._finish_cold_start_trial(trial, "inconclusive", "pre-add reference timed out")
                return
            pre_add = {job_id: windows[job_id]["halves"] for job_id in trial.preexisting_job_ids}
            trial.cold_start["pre_add"] = pre_add
            references = trial.cold_start["references"]
            for job in jobs:
                reference = job.metadata.get("trial_solo_reference") or {}
                ttl = self.settings.gpu_scheduler.colocation.profile_rejection_ttl_seconds
                if reference.get("identity") == reference_identity(job) and 0 <= time.time() - float(reference.get("at", 0)) < ttl:
                    references[job.job_id] = reference["halves"]
                else:
                    profile = self.store.get_runtime_profile(job.packing.signature or job.job_id,
                        resolved_batch_size=BatchResolution.resolved_batch_size(job), backend_name=trial.backend_name)
                    if profile is not None and self.planner.time_objective.profile_is_fresh(profile) and profile.metadata.get("reference_identity") == reference_identity(job) and profile.metadata.get("reference_halves"):
                        references[job.job_id] = profile.metadata["reference_halves"]
            if len(trial.preexisting_job_ids) == 1:
                job = next(job for job in jobs if job.job_id != candidate.job_id)
                references[job.job_id] = pre_add[job.job_id]
                self._save_cold_reference(job, pre_add[job.job_id])
            queue = [job.job_id for job in jobs if job.job_id not in references]
            trial.cold_start["reference_queue"] = queue
            self._cold_phase(trial, "reference_barrier" if queue else "measure_overlap", jobs, measuring=queue[0] if queue else None)
            return
        if trial.phase == "reference_barrier":
            if all(job.metadata.get("trial_yield_ack") == (job.metadata.get("trial_step_command") or {}).get("token") for job in jobs):
                self._cold_phase(trial, "reference", jobs, measuring=trial.cold_start["measuring"])
            elif phase_elapsed > 2:
                self._finish_cold_start_trial(trial, "inconclusive", "safe-step yield timed out")
            return
        if trial.phase == "reference":
            job_id = trial.cold_start["measuring"]
            window = windows[job_id]
            if window is None:
                if phase_elapsed > 2:
                    self._finish_cold_start_trial(trial, "inconclusive", "solo reference timed out")
                return
            job = next(job for job in jobs if job.job_id == job_id)
            trial.cold_start["references"][job_id] = window["halves"]
            self._save_cold_reference(job, window["halves"])
            queue = trial.cold_start["reference_queue"]
            queue.remove(job_id)
            self._cold_phase(trial, "reference_barrier" if queue else "measure_overlap", jobs, measuring=queue[0] if queue else None)
            return
        if trial.phase != "measure_overlap":
            self._finish_cold_start_trial(trial, "inconclusive", "unknown trial phase")
            return
        if not all(windows.values()):
            if int(candidate.metadata.get("last_completed_epoch", 0)) >= trial.target_epoch or phase_elapsed > 2:
                self._finish_cold_start_trial(trial, "inconclusive", "one-epoch measurement budget exhausted")
            return
        packed = {job_id: window["halves"] for job_id, window in windows.items()}
        old = [job for job in jobs if job.job_id != candidate.job_id]
        pre_add = trial.cold_start["pre_add"]
        references = trial.cold_start["references"]
        active_rates = {job.job_id: sum(pre_add[job.job_id]) / 2 * int(job.metadata.get("runtime_steps_per_epoch") or 0) for job in old}
        packed_rates = {job.job_id: sum(packed[job.job_id]) / 2 * int(job.metadata.get("runtime_steps_per_epoch") or 0) for job in jobs}
        solo = sum(references[candidate.job_id]) / 2 * int(candidate.metadata.get("runtime_steps_per_epoch") or 0)
        gain = self.planner.time_objective.estimate_gain(
            old, candidate, backend_name=trial.backend_name,
            active_epoch_seconds=active_rates, candidate_solo_epoch_seconds=solo,
            packed_epoch_seconds=packed_rates,
        ) if all(value > 0 for value in [solo, *active_rates.values(), *packed_rates.values()]) else None
        if gain is not None and any(phase.inherited_parent_rates for phase in [*gain.sequential_phases, *gain.packed_phases]):
            gain = None
        decision, halves = normalized_throughput(references, pre_add, packed, candidate.job_id, self.settings.gpu_scheduler.colocation.cold_start_gain_margin)
        result = {"objective": "normalized_throughput", "half_gains": halves, "references": references, "pre_add": pre_add, "packed": packed, "measurement_windows": windows}
        if gain is not None:
            result.update({"objective": "projected_drain", "gain": gain.gain, "sequential_seconds": gain.sequential_drain_seconds, "packed_seconds": gain.packed_drain_seconds})
            decision = "accepted" if gain.gain >= self.settings.gpu_scheduler.colocation.min_gain and decision == "accepted" else "rejected" if gain.gain < self.settings.gpu_scheduler.colocation.min_gain and decision == "rejected" else "inconclusive"
        if decision != "inconclusive" and all(packed_rates.values()):
            self._persist_colocation_timing_profile(jobs, packed_rates, trial, sources={job.job_id: "fresh_step_window" for job in jobs}, gain=gain.gain if gain is not None else None, decision=decision if gain is not None else None)
        self._finish_cold_start_trial(trial, decision, "measured cold-start " + decision, result)
