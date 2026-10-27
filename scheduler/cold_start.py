"""Incremental trials which acquire their own timing references during training."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
import time

from ..domain import BatchResolution, JobStatus, RuntimeProfile, build_colocation_profile_key, utc_now
from .planner_types import DispatchPlan
from .resource_estimator import COOPERATIVE_BUSY_GROUP_WIDTH_PP


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
    # Supervisor membership includes workers while their checkpoint/exit drains.
    # Admitting against that transient group used to bypass the rejection key.
    if any(job.status in {JobStatus.PAUSING, JobStatus.PAUSED, JobStatus.RECOVERABLE} or job.status.is_terminal
           or job.metadata.get("cold_start_solo_required") for job in active):
        return None
    if any(
        not job.metadata.get("cooperative_trial")
        or job.metadata.get("placement_backend") != backend
        for job in active
    ):
        return None
    budget = planner.estimator.safe_budget_mb()
    allowance = budget - used_mb
    if allowance <= 0:
        return None
    candidates = []
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
        if active and cold_group_blocked(job, key, settings.colocation, now.timestamp()):
            continue
        candidates.append((job, key, config, batch))
    if not candidates:
        return None
    work_ranks = planner.estimator.cooperative_work_ranks(
        [item[0] for item in candidates], prefer_high=not active, required_jobs=active)
    ordered = []
    for job, key, config, batch in candidates:
        if work_ranks is not None:
            rank = work_ranks[job.job_id]
        else:
            remaining = planner.predicted_remaining_runtime_seconds(job, backend_name=backend)
            rank = (planner.estimator.workload_rank(job, prefer_high=not active, batch_size=batch),
                    remaining is None, remaining if remaining is not None else 0.0)
        ordered.append(((not (not active and job.metadata.get("cold_start_solo_required")),
                         *rank, -planner._effective_priority(job, now=now), job.job_id),
                        job, key, config))
    _, job, key, config = min(ordered, key=lambda item: item[0])
    profiles = {member.job_id: planner.estimator.predicted_profile(member)
                for member in [*active, job]}
    if (not all(profiles.values()) or
            len({(p["artifact_sha256"], p["hardware"]) for p in profiles.values() if p}) != 1):
        profiles = {}  # Never mix measured seconds and predicted work units.
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
            "predicted_work_profiles": profiles,
        },
        objective_breakdown={"cold_start": bool(active), "gain": None,
                             "profile_origin": "predicted_workload" if profiles else "measurement_fallback",
                             "candidate_ranking_policy": "cooperative_busy_groups_relative_work_v1" if work_ranks is not None else "legacy_workload_rank",
                             **({"busy_group_width_pp": COOPERATIVE_BUSY_GROUP_WIDTH_PP,
                                 "busy_group_basis": "heuristic; not predictive confidence or capacity",
                                 "eligible_work_ranks": {name: list(rank) for name, rank in work_ranks.items()}}
                                if work_ranks is not None else {}),
                             "candidate_memory_mib": planner.estimator.estimate_peak_vram_mb(job, BatchResolution.resolved_batch_size(job), backend)},
        mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
        objective_version=settings.objective.objective_version,
    )


def cold_group_blocked(job, key, settings, now):
    """Use the same admission identity at planning and after membership changes."""
    attempt = (job.metadata.get("cold_start_group_attempts") or {}).get(key) or {}
    return bool(
        job.metadata.get("cold_start_solo_required")
        or int(job.metadata.get("cold_start_failed_attempts", 0)) >= settings.cold_start_max_attempts
        or key in job.metadata.get("cold_start_blocked_groups", [])
        or int(attempt.get("failures", 0)) >= settings.cold_start_max_group_attempts
        or float(job.metadata.get("cold_start_retry_after", 0)) > now
        or float(attempt.get("retry_after", 0)) > now
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
        settings = self.settings.gpu_scheduler.colocation
        hard_deadline = datetime.fromisoformat(trial.started_at).timestamp() + settings.cold_start_trial_timeout_seconds
        lease = max(5.0, 4 * self.settings.scheduler_poll_interval_seconds)
        expires_at = min(hard_deadline, time.time() + lease)
        previous = dict(job.metadata.get("trial_step_command") or {})
        if not reset and previous.get("owner") == trial.trial_id and previous.get("action") == action and previous.get("kind") == kind:
            if float(previous.get("expires_at", 0)) > time.time() + lease / 2:
                return
            command = {**previous, "expires_at": expires_at}
        else:
            command = {
                "owner": trial.trial_id, "token": f"{trial.trial_id}-{time.monotonic_ns()}",
                "action": action, "kind": kind, "expires_at": expires_at,
                "candidate": job.job_id == trial.candidate_job_id,
                "target_epoch": trial.target_epoch,
                "target_step": trial.cold_start.get("target_step"),
                "hard_deadline_at": hard_deadline,
                "window_min_seconds": settings.cold_start_window_min_seconds,
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
            if phase == "prepare":
                if job.job_id != trial.candidate_job_id:
                    self.store.update_job(job.job_id, metadata_updates={"trial_step_command": {}})
                    continue
                action = "yield"
            elif phase == "pre_add":
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

    def _cold_measurement_timed_out(self, trial, jobs):
        """Start a bounded timer only after this command made optimizer progress.

        The model's relative work units never enter this wall-clock calculation.
        A hung first step is bounded separately by the absolute trial deadline.
        """
        settings = self.settings.gpu_scheduler.colocation
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            progress = job.metadata.get("trial_measurement_progress") or {}
            if progress.get("token") != command.get("token") or not progress.get("started_at"):
                continue
            rate = progress.get("seconds_per_step")
            pre_add = (trial.cold_start.get("pre_add") or {}).get(job.job_id) or []
            measured = [float(value) for value in [rate, *pre_add]
                        if isinstance(value, (int, float)) and isfinite(value) and value > 0]
            # Eight measured updates, with room for input/step variance and two
            # warmup updates. Unknown timing gets the conservative upper bound.
            budget = min(settings.cold_start_window_timeout_max_seconds, max(
                settings.cold_start_window_timeout_min_seconds,
                4 * (10 * max(measured) + settings.cold_start_window_min_seconds)
                if measured else settings.cold_start_window_timeout_max_seconds,
            ))
            if time.time() - float(progress["started_at"]) > budget:
                return True
        return False

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
            attempts = dict(candidate.metadata.get("cold_start_group_attempts") or {})
            failures = int(candidate.metadata.get("cold_start_failed_attempts", 0))
            settings = self.settings.gpu_scheduler.colocation
            retry_after = float(candidate.metadata.get("cold_start_retry_after", 0))
            if decision != "accepted":
                failures += 1
                retry_after = time.time() + settings.cold_start_retry_cooldown_seconds
                # Keep both the planned and actual group identities. A worker
                # draining mid-trial must not make its original key retryable.
                for key in set([trial.profile_key, *trial.cold_start.get("admission_profile_keys", [])]):
                    count = int((attempts.get(key) or {}).get("failures", 0)) + 1
                    attempts[key] = {"failures": count, "retry_after": retry_after,
                                     "decision": decision, "reason": reason}
                    if (decision == "rejected" or count >= settings.cold_start_max_group_attempts) and key not in blocked:
                        blocked.append(key)
            self.store.update_job(candidate.job_id, metadata_updates={
                "cold_start_blocked_groups": blocked,
                "cold_start_group_attempts": attempts,
                "cold_start_failed_attempts": failures,
                "cold_start_retry_after": retry_after,
                "cold_start_solo_required": failures >= settings.cold_start_max_attempts,
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
        settings = self.settings.gpu_scheduler.colocation
        elapsed = time.time() - datetime.fromisoformat(trial.started_at).timestamp()
        if elapsed >= settings.cold_start_trial_timeout_seconds:
            self._finish_cold_start_trial(trial, "inconclusive", "absolute cold-start trial deadline exceeded")
            return
        active = self._active_jobs()
        expected = {*trial.preexisting_job_ids, trial.candidate_job_id}
        jobs = [job for job in active if job.job_id in expected]
        if candidate.status.is_terminal or candidate.metadata.get("trial_lease_expired") == trial.trial_id:
            self._finish_cold_start_trial(trial, "inconclusive", "candidate completed or measurement lease expired")
            return
        if any(job.status in {JobStatus.PAUSING, JobStatus.PAUSED, JobStatus.RECOVERABLE} or job.status.is_terminal for job in jobs):
            # Checkpoint serialization and process teardown can exceed a lease.
            # Keep healthy members' commands alive while the group settles;
            # the absolute deadline above still bounds the entire trial.
            for job in jobs:
                command = job.metadata.get("trial_step_command") or {}
                if job.status == JobStatus.RUNNING and command.get("owner") == trial.trial_id:
                    self._cold_command(trial, job, command["action"], command["kind"])
            return  # Wait for supervisor/process cleanup before choosing a group.
        if set(job.job_id for job in jobs) != expected:
            if trial.phase == "prepare" and candidate.job_id not in {job.job_id for job in active}:
                return
            remaining = [job for job in jobs if job.job_id != candidate.job_id and not job.status.is_terminal]
            if candidate.job_id in {job.job_id for job in jobs} and remaining:
                trial.preexisting_job_ids = tuple(job.job_id for job in remaining)
                descriptors = [self._member_descriptor(job, trial.backend_name) for job in [*remaining, candidate]]
                trial.profile_key = build_colocation_profile_key(self.store.hardware_key(), descriptors)
                if cold_group_blocked(candidate, trial.profile_key, settings, time.time()):
                    self._finish_cold_start_trial(trial, "inconclusive", "changed admission group is blocked")
                    return
                keys = trial.cold_start.setdefault("admission_profile_keys", [])
                if trial.profile_key not in keys:
                    keys.append(trial.profile_key)
                self._cold_phase(trial, "prepare", [*remaining, candidate])
                return
            self._finish_cold_start_trial(trial, "accepted" if not remaining and candidate.job_id in {job.job_id for job in jobs} else "inconclusive", "pack membership changed")
            return
        if self._device_samples and self._device_samples[-1].memory_used_mb > self.planner.estimator.safe_budget_mb():
            self._finish_cold_start_trial(trial, "inconclusive", "live memory budget exceeded")
            return
        predicted = trial.cold_start.get("predicted_work_profiles") or {}
        if predicted:
            # Freeze one shared model/hardware scale throughout the trial.
            try:
                valid = (len({(predicted[j.job_id]["artifact_sha256"], predicted[j.job_id]["hardware"])
                              for j in jobs}) == 1 and all(
                    predicted[j.job_id]["batch_size"] == BatchResolution.resolved_batch_size(j)
                    and predicted[j.job_id]["job_signature"] == (j.packing.signature or j.job_id)
                    and isfinite(float(predicted[j.job_id]["step_cost"]))
                    and float(predicted[j.job_id]["step_cost"]) > 0 for j in jobs))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                self._finish_cold_start_trial(trial, "inconclusive", "predicted profile identity changed")
                return
        if trial.phase == "prepare":
            self._cold_command(trial, candidate, "yield", "prepare")
            command = candidate.metadata.get("trial_step_command") or {}
            if (not all(job.metadata.get("trial_runner_ready") for job in jobs)
                    or candidate.metadata.get("trial_yield_ack") != command.get("token")):
                if elapsed > settings.cold_start_prepare_timeout_seconds:
                    self._finish_cold_start_trial(trial, "inconclusive", "runner preparation timed out")
                return
            self._cold_phase(trial, "pre_add", jobs)
            return
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            self._cold_command(trial, job, command.get("action", "yield"), command.get("kind", trial.phase))
        if any(job.metadata.get("trial_lease_expired") == trial.trial_id for job in jobs):
            self._finish_cold_start_trial(trial, "inconclusive", "measurement lease expired")
            return
        windows = {job.job_id: self._cold_window(job) for job in jobs}
        step_budget_exhausted = int(candidate.metadata.get("runtime_global_step", 0)) >= int(trial.cold_start.get("target_step", 2**63))
        if trial.phase != "measure_overlap" and step_budget_exhausted:
            self._finish_cold_start_trial(trial, "inconclusive", "reference optimizer-step budget exhausted")
            return
        if trial.phase == "pre_add":
            if not all(windows[job_id] for job_id in trial.preexisting_job_ids):
                if self._cold_measurement_timed_out(trial, [job for job in jobs if job.job_id in trial.preexisting_job_ids and not windows[job.job_id]]):
                    self._finish_cold_start_trial(trial, "inconclusive", "pre-add reference timed out")
                return
            pre_add = {job_id: windows[job_id]["halves"] for job_id in trial.preexisting_job_ids}
            trial.cold_start["pre_add"] = pre_add
            if predicted:
                # Progress is measured online, weighted by fixed predicted work.
                # No per-job solo reference measurement or synthetic time profile.
                trial.cold_start["references"] = {
                    j.job_id: [float(predicted[j.job_id]["step_cost"])] * 2 for j in jobs}
                trial.cold_start["reference_queue"] = []
                self._cold_phase(trial, "measure_overlap", jobs)
                return
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
            elif time.time() - float(trial.cold_start["phase_started_at"]) > settings.cold_start_prepare_timeout_seconds:
                self._finish_cold_start_trial(trial, "inconclusive", "safe-step yield timed out")
            return
        if trial.phase == "reference":
            job_id = trial.cold_start["measuring"]
            window = windows[job_id]
            if window is None:
                if self._cold_measurement_timed_out(trial, [job for job in jobs if job.job_id == job_id]):
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
            if step_budget_exhausted:
                self._finish_cold_start_trial(trial, "inconclusive", "measurement optimizer-step budget exhausted")
            elif self._cold_measurement_timed_out(trial, [job for job in jobs if not windows[job.job_id]]):
                self._finish_cold_start_trial(trial, "inconclusive", "overlap measurement timed out after optimizer progress")
            return
        packed = {job_id: window["halves"] for job_id, window in windows.items()}
        old = [job for job in jobs if job.job_id != candidate.job_id]
        pre_add = trial.cold_start["pre_add"]
        references = trial.cold_start["references"]
        active_rates = {job.job_id: sum(pre_add[job.job_id]) / 2 * int(job.metadata.get("runtime_steps_per_epoch") or 0) for job in old}
        packed_rates = {job.job_id: sum(packed[job.job_id]) / 2 * int(job.metadata.get("runtime_steps_per_epoch") or 0) for job in jobs}
        solo = (sum(references[candidate.job_id]) / 2 * int(candidate.metadata.get("runtime_steps_per_epoch") or 0)
                if not predicted else 0.0)
        gain = self.planner.time_objective.estimate_gain(
            old, candidate, backend_name=trial.backend_name,
            active_epoch_seconds=active_rates, candidate_solo_epoch_seconds=solo,
            packed_epoch_seconds=packed_rates,
        ) if not predicted and all(value > 0 for value in [solo, *active_rates.values(), *packed_rates.values()]) else None
        if gain is not None and any(phase.inherited_parent_rates for phase in [*gain.sequential_phases, *gain.packed_phases]):
            gain = None
        decision, halves = normalized_throughput(references, pre_add, packed, candidate.job_id, self.settings.gpu_scheduler.colocation.cold_start_gain_margin)
        result = {"objective": "normalized_throughput", "half_gains": halves, "references": references, "pre_add": pre_add, "packed": packed, "measurement_windows": windows}
        if predicted:
            result.update({"objective": "predicted_work_throughput", "reference_unit": "model_step_cost",
                           "measurement_unit": "seconds_per_optimizer_step",
                           "predicted_work_profiles": predicted, "solo_profile_measurements": 0})
        if gain is not None:
            result.update({"objective": "projected_drain", "gain": gain.gain, "sequential_seconds": gain.sequential_drain_seconds, "packed_seconds": gain.packed_drain_seconds})
            decision = "accepted" if gain.gain >= self.settings.gpu_scheduler.colocation.min_gain and decision == "accepted" else "rejected" if gain.gain < self.settings.gpu_scheduler.colocation.min_gain and decision == "rejected" else "inconclusive"
        if decision != "inconclusive" and all(packed_rates.values()):
            self._persist_colocation_timing_profile(jobs, packed_rates, trial, sources={job.job_id: "fresh_step_window" for job in jobs}, gain=gain.gain if gain is not None else None, decision=decision if gain is not None else None)
        self._finish_cold_start_trial(trial, decision, "measured cold-start " + decision, result)
