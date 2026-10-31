"""Opt-in idle-GPU bootstrap for a sealed, prepared bracket of two/three jobs.

This lifecycle is deliberately separate from newcomer/pre-add admission. It
measures the actual full group and never invents incumbent or subgroup rates.
Unknown alternatives use bounded checkpoint/drain/solo cleanup, preserving all
productive updates and the ordinary worker checkpoint/exit/receipt machinery.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from uuid import uuid4

from ..domain import BatchResolution, JobStatus
from .completion_objective import remaining_optimizer_updates
from .feedback_objective import derived_solo_slope, number, timestamp
from .feedback_trials import profile_snapshot
from .v5_lifecycle import forecast_cohort, snapshot as lifecycle_snapshot
from .planner_types import DispatchPlan


POLICY = "sealed_bracket_bootstrap_v5"
_META = "v5_bracket_bootstrap"


@dataclass
class BracketBootstrapState:
    trial_id: str
    generation: str
    job_ids: tuple[str, ...]
    profiles: dict
    started_at: float
    deadline_at: float
    backend: str
    cleanup_deadline_at: float = 0.
    live_identities: dict = field(default_factory=dict)
    phase: str = "prepare"
    phase_started_at: float = 0.
    start_steps: dict = field(default_factory=dict)
    warmup_start_steps: dict = field(default_factory=dict)
    settings_snapshot: dict = field(default_factory=dict)
    runtime_identity: dict = field(default_factory=dict)
    phase_counters: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)
    finishing_ids: list = field(default_factory=list)
    result: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _remaining(job, snapshot=None):
    """Zero is a valid finishing state; missing/inconsistent counters are unknown."""
    metadata = job.metadata
    if snapshot is None:
        steps = metadata.get("runtime_steps_per_epoch")
        if steps is None:
            from ..prediction.released_student import steps_for_job
            try:
                steps = steps_for_job(job, BatchResolution.resolved_batch_size(job))
            except (ValueError, KeyError, TypeError):
                return None
        step = metadata.get("runtime_global_step")
        if step is None:
            if job.started_at or job.latest_checkpoint_path or job.resume_from_checkpoint:
                return None
            step = 0
        snapshot = {"global_step": step, "steps_per_epoch": steps,
            "total_optimizer_updates": metadata.get("runtime_total_optimizer_updates"),
            "planned_epochs": metadata.get("runtime_planned_epochs")}
    return remaining_optimizer_updates(job, snapshot)


def bracket_generation(jobs, profiles, hardware_key, backend):
    """Bind retry/evidence identity to sealed membership and frozen recipes."""
    members = []
    for job in sorted(jobs, key=lambda item: item.job_id):
        members.append({"job_id": job.job_id, "signature": job.packing.signature or job.job_id,
            "batch": BatchResolution.resolved_batch_size(job), "profile": profiles[job.job_id],
            "recipe": job.metadata.get("perfseer_model"),
            "workload_sha256": job.metadata.get("pilot_frozen_workload_sha256"),
            "precision": job.metadata.get("precision_mode"),
            "request": job.metadata.get("feedback_request"),
            "backend_config": job.metadata.get("placement_backend_config") or {}})
    return sha256(json.dumps({"members": members, "hardware": hardware_key, "backend": backend},
        sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class BracketBootstrapMixin:
    """Compose before the experiment service/dispatch and preparation mixins.

    Parent hooks:
      _v5_bracket_startup_envelope(jobs, profiles) -> {peak_mib, source}
      _v5_bracket_reservation_allows(jobs) -> bool (optional)
      _v5_bracket_compare(jobs, windows, profiles, remaining) -> result (optional)
      _v5_bracket_exit_observed(job_id, original_launch_identity) -> bool (optional)

    Restart absence is not exit proof. The original persisted cleanup deadline
    is never renewed; exceeding it records v5_failure and aborts the episode.

    With no startup envelope the ordinary safe launch path remains available.
    No predictor/capture work runs here: estimator profiles must already exist.
    """
    _bracket_bootstrap = None
    _bracket_bootstrap_restored = False

    def _bracket_enabled(self):
        return bool(getattr(getattr(self, "v5_options", None), "bracket", False))

    def _bracket_emit(self, event, **payload):
        state = self._bracket_bootstrap
        self.event_logger.emit(event, payload={"admission_policy": POLICY,
            **({"trial_id": state.trial_id, "generation": state.generation,
                "job_ids": list(state.job_ids), "phase": state.phase} if state else {}), **payload})

    def _bracket_save(self):
        state = self._bracket_bootstrap
        if state:
            for name in state.job_ids:
                self.store.update_job(name, metadata_updates={_META: state.to_dict()})

    def _bracket_launch_identity(self, name):
        hook = getattr(self, "_v5_bracket_launch_identity", None)
        if callable(hook):
            return deepcopy(hook(name))
        for run in self._active_runs.values():
            if name in run.job_ids:
                return {"group_id": run.group_id, "opened_at": run.opened_at, "job_id": name}
        job = self.store.get_job(name)
        if job is not None and job.metadata.get("placement_group_id"):
            return {"group_id": job.metadata["placement_group_id"], "job_id": name}
        return None

    def _bracket_capture_live(self):
        state = self._bracket_bootstrap
        active = set(self._supervisor_active_job_ids())
        for name in state.job_ids:
            identity = self._bracket_launch_identity(name)
            job = self.store.get_job(name)
            if identity is not None:
                state.live_identities.setdefault(name, identity)
            elif name in active or (job is not None and job.status in {JobStatus.RUNNING, JobStatus.PAUSING, JobStatus.RECOVERABLE}):
                # Missing launch proof remains unresolved after a restart. It
                # never becomes permission to assume an old process exited.
                state.live_identities.setdefault(name, {"job_id": name, "unverified_launch": True})

    def _bracket_cleanup_pending(self):
        state = self._bracket_bootstrap
        active = set(self._supervisor_active_job_ids())
        observed = getattr(self, "_v5_bracket_exit_observed", None)
        reaped = getattr(self.supervisor, "has_reaped_process", None)
        live = getattr(self.supervisor, "has_live_process", None)
        pending = []
        for name, identity in state.live_identities.items():
            if callable(observed):
                released = observed(name, deepcopy(identity)) is True
            else:
                released = callable(reaped) and callable(live) and reaped(name) is True and live(name) is False
            if name in active or not released:
                pending.append(name)
        return sorted(pending)

    def _bracket_cleanup_budget(self):
        return min(number(self.settings.gpu_scheduler.colocation.cold_start_prepare_timeout_seconds, positive=True),
            number(self.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds, positive=True))

    def _bracket_restore(self):
        if self._bracket_bootstrap_restored:
            return
        self._bracket_bootstrap_restored = True
        saved_states = [j.metadata[_META] for j in self.store.list_jobs()
            if isinstance(j.metadata.get(_META), dict) and j.metadata[_META].get("phase") != "closed"]
        if not saved_states:
            return
        if len({s.get("trial_id") for s in saved_states}) != 1:
            raise RuntimeError("owned bracket episode has ambiguous persisted live trials")
        saved = max(saved_states, key=lambda s: s.get("phase_started_at", 0))
        allowed = {f.name for f in fields(BracketBootstrapState)}
        state = BracketBootstrapState(**{k: deepcopy(v) for k, v in saved.items() if k in allowed})
        state.job_ids = tuple(state.job_ids)
        state.deadline_at = min(number(s["deadline_at"], positive=True) for s in saved_states)
        deadlines = [number(s["cleanup_deadline_at"], positive=True) for s in saved_states if s.get("cleanup_deadline_at")]
        state.cleanup_deadline_at = min(deadlines) if deadlines else state.deadline_at+self._bracket_cleanup_budget()
        for old in saved_states:
            state.live_identities.update(deepcopy(old.get("live_identities") or {}))
        self._bracket_bootstrap = state
        self._bracket_capture_live()
        previous = deepcopy(state.result)
        self._bracket_close("inconclusive", "controller restart requires observed original-group exits",
            cleanup=True, result={"restart_previous_result": previous})
        self._bracket_emit("bracket_bootstrap_restart_cleanup", live_identities=state.live_identities,
            cleanup_deadline_at=state.cleanup_deadline_at)

    def _bracket_readiness_event(self, request, required, by_id):
        key = (request.get("workflow_id"), request.get("request_id"), tuple(sorted(required)),
               tuple(sorted(name for name in required if name in by_id)))
        previous = getattr(self, "_bracket_readiness_logged", set())
        if key not in previous:
            previous.add(key)
            self._bracket_readiness_logged = previous
            self._bracket_emit("bracket_bootstrap_readiness", request_id=request.get("request_id"),
                ready_job_ids=sorted(name for name in required if name in by_id),
                unready_or_not_runnable_ids=sorted(name for name in required if name not in by_id),
                observed_at=time.time(), action="normal_execution_available_no_sibling_barrier")

    def _bracket_ready_group(self):
        if (self._supervisor_active_job_ids() or self._active_runs
                or self._colocation_trial is not None or not self._admission_gate.is_open):
            return None
        if not self.settings.gpu_scheduler.colocation.live_trial_enabled:
            return None
        backend = self.settings.gpu_scheduler.packing_backend
        if backend != "cuda_process" or not self.supervisor.available_backends().get(backend):
            return None  # Concurrent MPS percentages need a separate coordinated contract.
        runnable = self._runnable_jobs()
        by_id = {job.job_id: job for job in runnable}
        for first in sorted(runnable, key=lambda j: (timestamp(j.submitted_at), j.job_id)):
            request = first.metadata.get("feedback_request") or {}
            if not isinstance(request, dict):
                continue
            required = request.get("required_job_ids")
            if (request.get("completion_policy") != "group" or request.get("membership_finalized") is not True
                    or not isinstance(required, (list, tuple)) or not 2 <= len(required) <= 3
                    or any(not isinstance(name, str) or not name for name in required)
                    or len(set(required)) != len(required)):
                continue
            self._bracket_readiness_event(request, required, by_id)
            if any(name not in by_id for name in required):
                continue
            cap = self.settings.gpu_scheduler.parallel_job_cap
            if cap is not None and len(required) > cap:
                continue
            jobs = [by_id[name] for name in sorted(required)]
            if any(job.hold or job.status.is_terminal or job.metadata.get("feedback_request") != request
                   or not job.metadata.get("cooperative_trial") or job.metadata.get("cold_start_solo_required")
                   for job in jobs):
                continue
            guard = getattr(self, "_v5_bracket_reservation_allows", None)
            if callable(guard) and not guard(jobs):
                continue
            if not callable(guard) and any(j.metadata.get("v5_deferral", {}).get("protected")
                                           and j.job_id not in required for j in runnable):
                continue
            compatibility = self.planner.compatibility
            if (not all(compatibility.pack_eligible(j, backend_name=backend) for j in jobs)
                    or not compatibility.compatible_group(jobs, backend_name=backend)):
                continue
            try:
                # Every outstanding submitted request remains scoreable; this
                # also rejects held/malformed/missing sealed request members.
                outstanding, _, _ = forecast_cohort(self)
                all_profiles = profile_snapshot(self, outstanding)
                profiles = {j.job_id: all_profiles[j.job_id] for j in jobs}
                generation = bracket_generation(jobs, profiles, self.store.hardware_key(), backend)
                job_limit = self.settings.gpu_scheduler.colocation.cold_start_max_attempts
                if any(int(j.metadata.get("v5_bracket_probe_attempts", 0)) >= job_limit
                       or generation in (j.metadata.get("v5_bracket_generations") or [])
                       or int(j.metadata.get("v5_bracket_probe_attempts", 0)) + int(j.metadata.get("cold_start_failed_attempts", 0)) >= job_limit for j in jobs):
                    continue
                if any(_remaining(j) is None or _remaining(j) < 12 for j in jobs):
                    continue
                envelope_hook = getattr(self, "_v5_bracket_startup_envelope", None)
                envelope = envelope_hook(jobs, profiles) if callable(envelope_hook) else None
                if (not isinstance(envelope, dict) or not envelope.get("source")
                        or number(envelope["peak_mib"], positive=True) < sum(number(p["peak_vram_mib"], positive=True) for p in profiles.values())
                        or number(envelope["peak_mib"], positive=True) > self.planner.estimator.safe_budget_mb()):
                    continue
                if self._device_samples and (self._device_samples[-1].memory_used_mb + number(envelope["peak_mib"]) > self.planner.estimator.safe_budget_mb()):
                    continue
                return jobs, profiles, generation, envelope
            except (ValueError, KeyError, TypeError, OverflowError):
                continue
        return None

    def _bracket_commands(self, phase, *, reset=True):
        state = self._bracket_bootstrap
        state.phase, state.phase_started_at = phase, time.time() if reset else state.phase_started_at
        for name in state.job_ids:
            job = self.store.get_job(name)
            if job is None or job.status.is_terminal or _remaining(job) == 0:
                continue
            previous = job.metadata.get("trial_step_command") or {}
            lease = min(state.deadline_at, time.time() + max(5., 4*self.settings.scheduler_poll_interval_seconds))
            if not reset and previous.get("owner") == state.trial_id:
                command = {**previous, "expires_at": lease}
            else:
                command = {"owner": state.trial_id, "token": uuid4().hex,
                    "action": "yield" if phase in {"prepare", "barrier"} else "measure",
                    "kind": "bracket_"+phase, "candidate": False, "expires_at": lease,
                    "hard_deadline_at": state.deadline_at,
                    "window_min_seconds": self.settings.gpu_scheduler.colocation.cold_start_window_min_seconds}
            self.store.update_job(name, metadata_updates={"trial_step_command": command})
        self._bracket_save()
        if reset:
            self._bracket_emit("bracket_bootstrap_phase")

    def _bracket_launch(self, selected):
        jobs, profiles, generation, envelope = selected
        cap = self.settings.gpu_scheduler.parallel_job_cap
        if ((cap is not None and len(jobs) > cap) or self._supervisor_active_job_ids() or self._active_runs):
            return False
        now = time.time()
        options = self.v5_options
        budget = getattr(options, "bracket_probe_seconds", self.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds)
        budget = number(budget, positive=True)
        if budget > self.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds:
            budget = self.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds
        state = BracketBootstrapState("bracket-"+uuid4().hex, generation,
            tuple(j.job_id for j in jobs), deepcopy(profiles), now, now+budget,
            self.settings.gpu_scheduler.packing_backend,
            cleanup_deadline_at=now+budget+self._bracket_cleanup_budget(), phase_started_at=now,
            start_steps={j.job_id: int(j.metadata.get("runtime_global_step", 0)) for j in jobs},
            settings_snapshot={"feedback": self.settings.gpu_scheduler.feedback_latency.to_dict(),
                "colocation": self.settings.gpu_scheduler.colocation.to_dict()})
        self._bracket_bootstrap = state
        for job in jobs:
            generations = list(job.metadata.get("v5_bracket_generations") or [])
            generations.append(generation)
            self.store.update_job(job.job_id, metadata_updates={"v5_bracket_generations": generations,
                "v5_bracket_probe_attempts": int(job.metadata.get("v5_bracket_probe_attempts", 0))+1})
        self._bracket_commands("prepare")
        plan = DispatchPlan(mode="concurrent_group", backend_name=state.backend,
            job_ids=state.job_ids, reason="sealed idle-GPU bracket bootstrap exploration",
            trial_metadata={"cooperative_trial": True, "bracket_bootstrap": True,
                "requires_live_trial": False, "memory_limit_mb": self.planner.estimator.safe_budget_mb(),
                "startup_envelope": envelope}, objective_breakdown={"admission_policy": POLICY,
                "exploration": True, "generation": generation})
        self._bracket_emit("bracket_bootstrap_started", startup_envelope=envelope,
            frozen_profiles=profiles, probe_allowance_seconds=budget,
            note="finite exploration; no pre-add or empty-incumbent evidence")
        try:
            dispatched = self._dispatch_plan(plan)
        except Exception:
            self._bracket_close("inconclusive", "coordinated dispatch raised", cleanup=True)
            raise
        self._bracket_capture_live()
        self._bracket_save()
        if not dispatched or set(self._supervisor_active_job_ids()) != set(state.job_ids):
            self._bracket_close("inconclusive", "coordinated full-group dispatch unavailable", cleanup=True)
        return True

    def _dispatch_pending_work(self):
        if not self._bracket_enabled():
            return super()._dispatch_pending_work()
        self._bracket_restore()
        if self._bracket_bootstrap is not None:
            return
        selected = self._bracket_ready_group()
        if selected is not None and self._bracket_launch(selected):
            return
        return super()._dispatch_pending_work()

    def _evaluate_colocation_trial(self):
        if self._bracket_enabled():
            self._bracket_restore()
            if self._bracket_bootstrap is not None:
                self._evaluate_bracket_bootstrap()
                return
        return super()._evaluate_colocation_trial()

    def _bracket_clear_commands(self, jobs):
        state = self._bracket_bootstrap
        for job in jobs:
            if (job.metadata.get("trial_step_command") or {}).get("owner") == state.trial_id:
                self.store.update_job(job.job_id, metadata_updates={"trial_step_command": {}})

    def _bracket_close(self, decision, reason, *, cleanup=False, result=None):
        state = self._bracket_bootstrap
        jobs = [self.store.get_job(name) for name in state.job_ids]
        jobs = [j for j in jobs if j is not None]
        reference = getattr(self, "_v5_reference_begin_bootstrap", None)
        if callable(reference) and decision in {"accepted", "rejected", "inconclusive"}:
            reference_result = {**(result or {}), "decision": decision}
            try:
                cohort, _, _ = forecast_cohort(self)
                installed = reference(cohort, reference_result, state)
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                installed = False
                self._bracket_emit("bracket_bootstrap_reference_unavailable", reason=str(exc),
                    action="bounded_checkpoint_then_solo_fallback")
            if not installed:
                self._bracket_emit("bracket_bootstrap_reference_unavailable", reason="full-cohort reference installation unavailable",
                    action="bounded_checkpoint_then_solo_fallback")
                if decision == "accepted":
                    decision, cleanup, reason = "inconclusive", True, "accepted forecast lacks executable complete-cohort reference"
                    result = {**(result or {}), "decision": decision, "error": reason}
        self._bracket_clear_commands(jobs)
        self._bracket_capture_live()
        cleanup_ids = []
        for job in jobs:
            # Never turn a completed update budget into a pause/failure. Its
            # worker still owns memory until ordinary supervisor exit reaping.
            if cleanup and not job.status.is_terminal and _remaining(job) != 0:
                cleanup_ids.append(job.job_id)
                self.store.update_job(job.job_id, metadata_updates={"cold_start_solo_required": True})
                if job.job_id in self._supervisor_active_job_ids():
                    if self.supervisor.request_pause(job.job_id, reason=reason, hold=False):
                        self.store.update_job(job.job_id, status=JobStatus.PAUSING, hold=False)
        state.result = {"decision": decision, "reason": reason, "cleanup_job_ids": cleanup_ids,
            "measured_full_group": bool(state.windows), "measurement_windows": deepcopy(state.windows),
            "productive_updates": {j.job_id: max(0, int(j.metadata.get("runtime_global_step", 0))-state.start_steps[j.job_id]) for j in jobs},
            **(result or {})}
        needs_drain = cleanup and (cleanup_ids or self._bracket_cleanup_pending())
        state.phase = "cleanup" if needs_drain else "closed"
        if needs_drain and not state.cleanup_deadline_at:
            state.cleanup_deadline_at = state.deadline_at+self._bracket_cleanup_budget()
        self._bracket_save()
        self._bracket_emit("bracket_bootstrap_decided", **state.result,
            wall_seconds=max(0., time.time()-state.started_at))
        if not needs_drain:
            self._bracket_bootstrap = None

    def _bracket_windows(self, jobs):
        state = self._bracket_bootstrap
        windows = {}
        for job in jobs:
            window = deepcopy(job.metadata.get("trial_step_window") or {})
            command = job.metadata.get("trial_step_command") or {}
            try:
                valid = (command.get("owner") == state.trial_id and window.get("token") == command.get("token")
                    and window.get("owner") == state.trial_id and window.get("complete") is True
                    and window.get("kind") == command.get("kind") == "bracket_measure"
                    and window.get("excludes_nonupdate_phases") is True
                    and all(type(window[key]) is int for key in ("steps", "start_step", "end_step"))
                    and window["steps"] >= 8 and window["start_step"] >= 0
                    and window["end_step"]-window["start_step"] == window["steps"]
                    and (number(window["elapsed_seconds"]) >= self.settings.gpu_scheduler.colocation.cold_start_window_min_seconds
                         or int(window["steps"]) >= 64)
                    and len(window["halves"]) == 2 and all(number(v, positive=True) for v in window["halves"])
                    and window["completion_identity"] == state.runtime_identity[job.job_id]
                    and window["nonupdate_phase_counter"] == window["completed_nonupdate_phase_counter"] == state.phase_counters[job.job_id]
                    and number(window["started_at"]) >= state.phase_started_at
                    and timestamp(window["finished_at"]) > number(window["started_at"]))
                if valid:
                    windows[job.job_id] = window
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        return windows

    def _evaluate_bracket_bootstrap(self):
        state = self._bracket_bootstrap
        jobs = [self.store.get_job(name) for name in state.job_ids]
        active = set(self._supervisor_active_job_ids())
        if state.phase == "cleanup":
            pending = self._bracket_cleanup_pending()
            if not pending:
                state.phase = "closed"
                self._bracket_save()
                self._bracket_bootstrap = None
            elif time.time() >= state.cleanup_deadline_at:
                failure = {"kind": "bracket_cleanup_timeout", "pending_job_ids": pending,
                    "trial_id": state.trial_id, "cleanup_deadline_at": state.cleanup_deadline_at,
                    "live_identities": deepcopy(state.live_identities),
                    "reason": "owned workers ignored bounded checkpoint/drain; no resource release is claimed"}
                self.v5_failure = failure
                state.result["owned_episode_failure"] = failure
                self._bracket_save()
                self._bracket_emit("bracket_bootstrap_cleanup_timeout", **failure)
                raise TimeoutError(failure["reason"])
            return
        if any(j is None for j in jobs):
            self._bracket_close("inconclusive", "bracket member disappeared", cleanup=True)
            return
        failed = [j for j in jobs if j.status.is_terminal and j.status != JobStatus.COMPLETED]
        if failed:
            self._bracket_close("inconclusive", "bracket member failed or was cancelled", cleanup=True,
                result={"failed_members": {j.job_id: {"status": j.status.value,
                    "reason": j.status_reason} for j in failed}})
            return
        if self._device_samples and self._device_samples[-1].memory_used_mb > self.planner.estimator.safe_budget_mb():
            self._bracket_close("inconclusive", "live memory budget exceeded", cleanup=True)
            return
        finished = [j.job_id for j in jobs if j.status == JobStatus.COMPLETED or _remaining(j) == 0]
        if finished:
            self._bracket_clear_commands(jobs)
            if state.phase != "finishing":
                state.phase, state.finishing_ids = "finishing", finished
                self._bracket_save()
                self._bracket_emit("bracket_bootstrap_training_finished", finishing_job_ids=finished,
                    note="zero remaining is valid; release waits for observed worker exit")
            survivors = [j for j in jobs if j.job_id not in finished]
            pending_finished = set(self._bracket_cleanup_pending()).intersection(finished)
            if len(survivors) <= 1 and not pending_finished:
                self._bracket_close("completed_unverified", "normal finishing/release; surviving singleton continues")
            elif time.time() >= state.deadline_at:
                self._bracket_close("completed_unverified" if not survivors else "inconclusive",
                    "finished workers await bounded observed release" if not survivors else
                    "bounded changed-membership continuation lacks full-group evidence", cleanup=True)
            return
        if time.time() >= state.deadline_at:
            self._bracket_close("inconclusive", "absolute bracket probe allowance exhausted", cleanup=True)
            return
        try:
            if state.settings_snapshot != {"feedback": self.settings.gpu_scheduler.feedback_latency.to_dict(),
                    "colocation": self.settings.gpu_scheduler.colocation.to_dict()}:
                raise ValueError("frozen trial settings changed")
            current = profile_snapshot(self, jobs)
            if bracket_generation(jobs, current, self.store.hardware_key(), state.backend) != state.generation:
                raise ValueError("frozen group/profile/recipe identity changed")
            if active != set(state.job_ids):
                raise ValueError("actual full-group membership changed")
            if any(j.hold or j.status in {JobStatus.PAUSING, JobStatus.PAUSED, JobStatus.RECOVERABLE}
                   or j.metadata.get("trial_lease_expired") == state.trial_id for j in jobs):
                raise ValueError("member held, paused, recovering, or lease expired")
            if state.phase == "prepare":
                self._bracket_commands("prepare", reset=False)
                ready = all(j.metadata.get("trial_runner_ready") and j.metadata.get("trial_yield_ack") ==
                    (j.metadata.get("trial_step_command") or {}).get("token") for j in jobs)
                if not ready:
                    if time.time()-state.started_at >= self.settings.gpu_scheduler.colocation.cold_start_prepare_timeout_seconds:
                        raise ValueError("coordinated runner preparation timed out")
                    return
                for j in jobs:
                    identity, counter = j.metadata.get("runtime_completion_identity"), j.metadata.get("runtime_nonupdate_phase_counter")
                    if not isinstance(identity, str) or not identity or type(counter) is not int or counter < 0:
                        raise ValueError("missing shared-runner phase identity")
                    state.runtime_identity[j.job_id] = identity
                    state.warmup_start_steps[j.job_id] = int(j.metadata["runtime_global_step"])
                self._bracket_commands("warmup")
                return
            if any(j.metadata.get("runtime_completion_identity") != state.runtime_identity[j.job_id] for j in jobs):
                raise ValueError("runtime completion identity changed")
            if state.phase == "warmup":
                self._bracket_commands("warmup", reset=False)
                for j in jobs:
                    command = j.metadata.get("trial_step_command") or {}
                    progress = j.metadata.get("trial_measurement_progress") or {}
                    if (progress.get("token") != command.get("token")
                            or number(progress.get("started_at", 0)) < state.phase_started_at
                            or int(progress.get("global_step", 0)) < state.warmup_start_steps[j.job_id]+2):
                        return
                state.phase_counters = {j.job_id: j.metadata["runtime_nonupdate_phase_counter"] for j in jobs}
                self._bracket_commands("measure")
                return
            if state.phase == "measure":
                self._bracket_commands("measure", reset=False)
                if any(j.metadata.get("runtime_nonupdate_phase_counter") != state.phase_counters[j.job_id] for j in jobs):
                    raise ValueError("non-update phase crossed full-group measurement")
                windows = self._bracket_windows(jobs)
                if len(windows) != len(jobs):
                    return
                if max(number(w["started_at"]) for w in windows.values()) >= min(timestamp(w["finished_at"]) for w in windows.values()):
                    raise ValueError("full-group measurement intervals do not overlap")
                state.windows = windows
                self._bracket_emit("bracket_bootstrap_full_group_evidence", measurement_windows=windows)
                self._bracket_commands("barrier")
                return
            if state.phase == "barrier":
                self._bracket_commands("barrier", reset=False)
                remaining = {}
                for j in jobs:
                    command = j.metadata.get("trial_step_command") or {}
                    snapshot = j.metadata.get("trial_yield_snapshot") or {}
                    if (j.metadata.get("trial_yield_ack") != command.get("token") or snapshot.get("token") != command.get("token")
                            or snapshot.get("owner") != state.trial_id):
                        return
                    if snapshot.get("completion_identity") != state.runtime_identity[j.job_id]:
                        raise ValueError("yield snapshot runtime identity changed")
                    remaining[j.job_id] = _remaining(j, snapshot)
                    if remaining[j.job_id] is None:
                        raise ValueError("post-probe successful-update budget is unknown")
                if any(value == 0 for value in remaining.values()):
                    self._bracket_clear_commands(jobs)
                    state.phase = "finishing"
                    self._bracket_save()
                    return
                compare = getattr(self, "_v5_bracket_compare", self._bracket_compare)
                result = compare(jobs, state.windows, state.profiles, remaining)
                decision = result.get("decision", "inconclusive")
                if decision not in {"accepted", "rejected", "inconclusive"}:
                    raise ValueError("invalid bracket comparison decision")
                self._bracket_close(decision, result.get("error") or "post-probe request continuation "+decision,
                    cleanup=decision != "accepted", result=result)
                return
            raise ValueError("unsupported bracket bootstrap phase")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self._bracket_close("inconclusive", str(exc), cleanup=True)

    def _bracket_compare(self, jobs, windows, profiles, remaining):
        from .continuation_forecast import compare_continuations
        cohort, requests, completed = forecast_cohort(self)
        frozen_profiles = profile_snapshot(self, cohort)
        now = time.time()
        snapshot = lifecycle_snapshot(self, cohort, now=now)
        all_remaining = {**snapshot["remaining"], **remaining}
        costs = snapshot["costs"]
        members = tuple(sorted(j.job_id for j in jobs))
        passive = snapshot["passive"]
        initial = (*members, *passive)
        queue = [j.job_id for j in sorted(cohort, key=lambda j: (timestamp(j.submitted_at), j.job_id)) if j.job_id not in initial]
        order = [j.job_id for j in sorted(cohort, key=lambda j: (timestamp(j.submitted_at), j.job_id))]
        context_hook = getattr(self, "_v5_snapshot_context", None)
        context = context_hook(frozen_profiles) if callable(context_hook) else {
            "evidence_identity": self._bracket_bootstrap.generation,
            "membership_generation": self._bracket_bootstrap.trial_id,
            "snapshot_at": now}
        ttl = getattr(self.v5_options, "evidence_ttl_seconds", 300.)
        observed = min(timestamp(w["finished_at"]) for w in windows.values())
        record_halves = [{"rates": {name: 1/number(windows[name]["halves"][half], positive=True) for name in members},
            "identity": context["evidence_identity"], "membership_generation": context["membership_generation"],
            "observed_at": observed, "max_age_seconds": ttl} for half in range(2)]
        evidence = [{members: record} for record in record_halves]
        cache = getattr(self, "_v5_rate_cache", None)
        if isinstance(cache, dict):
            cache[members] = {"context": context, "halves": record_halves}
            for key, saved in cache.items():
                old = saved.get("context") or {}
                if (old.get("evidence_identity") == context["evidence_identity"]
                        and old.get("membership_generation") == context["membership_generation"]
                        and 0 <= now-old.get("snapshot_at", now+1) <= ttl):
                    for half in range(2):
                        evidence[half][key] = saved["halves"][half]
        protected = []
        if getattr(self.v5_options, "deferral", False):
            from .deferral_reservations import choose_protected
            chosen = choose_protected(self._runnable_jobs(), **self._v5_deferral_options())
            if chosen:
                protected = [chosen.job_id]
        placement = getattr(self, "_v5_placement_allowed", None)
        allowed = (lambda names: placement(names, context=context)) if callable(placement) else (lambda names: len(names) == 1)
        result = compare_continuations(remaining=all_remaining,
            solo={name: derived_solo_slope(p["step_cost"]) for name, p in frozen_profiles.items()},
            requests=requests, completed=completed, now=now, costs=costs,
            alternatives={"admit": {"initial": initial, "queue": queue, "phase_state": snapshot["phase_state"]},
                "defer": {"initial": initial, "queue": queue, "resumed": members,
                    "queue_order": order, "concurrency_limit": 1,
                    "phase_state": {**snapshot["phase_state"], **{name: {"phase": "pausing", "pause": costs[name].pause} for name in members}}}},
            evidence_halves=evidence, context=context, protected=protected, placement_allowed=allowed,
            candidate_count=len(members), concurrency_limit=self.settings.gpu_scheduler.parallel_job_cap,
            cpu_budget_ms=getattr(self.v5_options, "forecast_cpu_ms", 50.),
            max_events=getattr(self.v5_options, "forecast_max_events", 128),
            memory={name: number(p["peak_vram_mib"], positive=True) for name, p in frozen_profiles.items()},
            memory_capacity=self.planner.estimator.safe_budget_mb(),
            compatible=lambda names: self.planner.compatibility.compatible_group(
                [self.store.get_job(name) for name in names], backend_name=self._bracket_bootstrap.backend),
            min_gain_seconds=self.settings.gpu_scheduler.feedback_latency.min_gain_seconds)
        result.update(remaining_updates=deepcopy(all_remaining), frozen_profiles=deepcopy(frozen_profiles),
            context=deepcopy(context), queue_order=order,
            bootstrap_origin={"trial_id": self._bracket_bootstrap.trial_id,
                "generation": self._bracket_bootstrap.generation, "group_members": list(members),
                "cleanup_members": list(members) if result["decision"] != "accepted" else []})
        return result
