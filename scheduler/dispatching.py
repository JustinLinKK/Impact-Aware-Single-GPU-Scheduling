"""Dispatch plans and maintain the boundary between planning and execution.

Planning is intentionally side-effect free. The methods in this module own
the side effects that follow a selected DispatchPlan: applying batch sizes,
preloading models, starting workers, recording placement metadata, and
opening the run-group log.
"""

from __future__ import annotations

import time

from ..domain import (
    BatchResolution,
    JobStatus,
    SchedulingClass,
    TrainingJob,
    build_group_signature,
)
from .planner_types import DispatchPlan
from .queue import RunnableJobQueue
from .service_state import ActiveRun


class DispatchMixin:
    """Execute plans chosen by the placement planner."""

    def _next_job(self) -> TrainingJob | None:
        """Peek at the highest-priority runnable job without mutating the queue."""
        queue = RunnableJobQueue(policy=self.policy, jobs=self._runnable_jobs())
        return queue.peek()

    def _resolved_batch_size_for_job_id(self, job_id: str) -> int:
        """Return the persisted effective batch size for one job."""
        job = self.store.get_job(job_id)
        if job is None:
            return 1
        return BatchResolution.resolved_batch_size(job)

    def _supervisor_active_job_ids(self) -> list[str]:
        """Normalize the supervisor API into active job identifiers."""
        if hasattr(self.supervisor, "active_job_ids"):
            return list(self.supervisor.active_job_ids())
        active_group = getattr(self.supervisor, "active_group", lambda: None)()
        if active_group is None:
            return []
        if hasattr(active_group, "active_job_ids"):
            return list(active_group.active_job_ids())
        workers = getattr(active_group, "workers", {}) or {}
        return list(workers.keys())

    def _supervisor_active_job_ids_by_group(self) -> dict[str, list[str]]:
        """Group active supervisor jobs by scheduler run identifier."""
        if hasattr(self.supervisor, "active_job_ids_by_group"):
            return {
                str(group_id): list(job_ids)
                for group_id, job_ids in self.supervisor.active_job_ids_by_group().items()
            }
        active_group = getattr(self.supervisor, "active_group", lambda: None)()
        if active_group is None:
            return {}
        group_id = str(getattr(active_group, "group_id", "untracked-active-group"))
        if hasattr(active_group, "active_job_ids"):
            return {group_id: list(active_group.active_job_ids())}
        workers = getattr(active_group, "workers", {}) or {}
        return {group_id: list(workers.keys())}

    def _apply_batch_override(self, job: TrainingJob, batch_size: int) -> TrainingJob:
        """Persist a planner-selected batch size before worker launch."""
        updated_job = BatchResolution.apply(job, batch_size)
        self.store.save_job(updated_job)
        return updated_job

    def _maybe_preempt(self) -> None:
        """Time-aware placement is non-preemptive and changes at drain boundaries."""
        return

    def _preload_job_baseline(self, job: TrainingJob) -> None:
        """Best-effort preload the model needed by an imminent dispatch."""
        target = self._resolve_preload_target(job)
        try:
            self.cache.preload(
                target.model_id,
                target.model_path,
                loader_target=target.loader_target,
                metadata={"source": "dispatch", "job_id": job.job_id},
            )
        except Exception as exc:
            self.logger.warning(
                "Baseline preload failed for job %s (%s): %s",
                job.job_id,
                target.model_id,
                exc,
            )

    def _active_vram_occupancy(self) -> float:
        """Estimate VRAM currently occupied by materialized run groups."""
        active_vram_mb = 0.0
        for group_id, run in self._active_runs.items():
            jobs = [
                self.store.get_job(job_id)
                for job_id in self._supervisor_active_job_ids_by_group().get(
                    group_id, []
                )
            ]
            materialized = [job for job in jobs if job is not None]
            if materialized:
                active_vram_mb += self.planner.predicted_group_vram_mb(
                    materialized, backend_name=run.backend_name
                )
        return active_vram_mb

    def _active_jobs(self) -> list[TrainingJob]:
        """Materialize the jobs that the supervisor still considers active."""
        active_ids = self._supervisor_active_job_ids()
        jobs = [self.store.get_job(job_id) for job_id in active_ids]
        return [job for job in jobs if job is not None]

    def _prediction_metadata(self, job_id: str) -> dict[str, str | None]:
        """Return the estimator source and error recorded for a job."""
        estimator = getattr(self.planner, "estimator", None)
        method = getattr(estimator, "prediction_metadata", None)
        if callable(method):
            result = method(job_id)
            if isinstance(result, dict):
                return result
        return {
            "vram_prediction_source": "branch_profile",
            "vram_prediction_error": None,
        }

    def _dispatch_plan(self, plan: DispatchPlan) -> bool:
        """Apply a chosen plan and register its workers as one active run.

        This is the planner/executor transaction boundary. Preparation that
        affects persisted state is rolled back when worker dispatch fails.
        """
        if self._known_colocation_rejection(plan):
            return False
        selected_jobs = []
        for job_id in plan.job_ids:
            job = self.store.get_job(job_id)
            if job is None:
                return False
            selected_jobs.append(job)
        if plan.batch_overrides:
            selected_jobs = [
                self._apply_batch_override(
                    job,
                    plan.batch_overrides.get(
                        job.job_id, self._resolved_batch_size_for_job_id(job.job_id)
                    ),
                )
                for job in selected_jobs
            ]
        if plan.backend_config or plan.trial_metadata.get(
            "start_delay_seconds_by_job"
        ):
            delay_by_job = dict(
                plan.trial_metadata.get("start_delay_seconds_by_job") or {}
            )
            source_signatures = dict(
                plan.trial_metadata.get("source_fingerprint_signatures") or {}
            )
            configured_jobs: list[TrainingJob] = []
            for job in selected_jobs:
                updated = job.copy(
                    metadata={
                        **job.metadata,
                        "placement_backend_config": dict(plan.backend_config),
                        "placement_start_delay_seconds": float(
                            delay_by_job.get(job.job_id, 0.0)
                        ),
                        "placement_source_fingerprint_signature": source_signatures.get(
                            job.job_id
                        ),
                    }
                )
                self.store.save_job(updated)
                configured_jobs.append(updated)
            selected_jobs = configured_jobs
        replayed = bool(plan.objective_breakdown.get("placement_replay"))
        if replayed:
            replay_slot = int(plan.objective_breakdown.get("placement_replay_slot", 0))
            replay_width = int(
                plan.objective_breakdown.get("placement_replay_target_width", 1)
            )
            persisted_jobs: list[TrainingJob] = []
            for job in selected_jobs:
                updated = job.copy(
                    metadata={
                        **job.metadata,
                        "placement_replay": True,
                        "placement_replay_slot": replay_slot,
                        "placement_replay_target_width": replay_width,
                        "skip_active_scheduler_probes": True,
                    }
                )
                self.store.save_job(updated)
                persisted_jobs.append(updated)
            selected_jobs = persisted_jobs
        for job in selected_jobs:
            self._preload_job_baseline(job)

        candidate_job_id = str(
            plan.trial_metadata.get("candidate_job_id")
            or (selected_jobs[0].job_id if selected_jobs else "")
        )
        candidate_job = next(
            (job for job in selected_jobs if job.job_id == candidate_job_id),
            selected_jobs[0] if selected_jobs else None,
        )
        prepared_trial = (
            self._prepare_colocation_trial(plan, candidate_job)
            if candidate_job is not None
            else None
        )

        try:
            dispatched = self.supervisor.dispatch(
                selected_jobs,
                mode=plan.mode,
                backend_name=plan.backend_name,
                batch_overrides=plan.batch_overrides,
                fallback_order=plan.fallback_order,
            )
        except Exception as exc:
            self._cancel_prepared_colocation_trial(
                prepared_trial, reason="trial dispatch failed"
            )
            if replayed:
                self._invalidate_placement_replay(
                    reason="replayed backend dispatch failed",
                    job=selected_jobs[0] if selected_jobs else None,
                    details={"backend_name": plan.backend_name, "error": str(exc)},
                )
                self.logger.warning(
                    "Replayed dispatch failed for job %s: %s", plan.job_ids[0], exc
                )
                return False
            self.logger.warning(
                "Dispatch failed for jobs %s: %s", ",".join(plan.job_ids), exc
            )
            if (
                plan.backend_name != "exclusive"
                and selected_jobs
                and not self._active_runs
            ):
                fallback_job = selected_jobs[0]
                self.logger.warning(
                    "Falling back to exclusive dispatch for %s after backend %s failed",
                    fallback_job.job_id,
                    plan.backend_name,
                )
                try:
                    fallback_decision = self.supervisor.dispatch(
                        [fallback_job], mode="exclusive", backend_name="exclusive"
                    )
                    if fallback_decision.can_run:
                        group_id = (
                            fallback_decision.group_id
                            or f"fallback-{fallback_job.job_id}-{time.monotonic_ns()}"
                        )
                        self._active_runs[group_id] = ActiveRun(
                            group_id=group_id,
                            mode="exclusive",
                            backend_name="exclusive",
                            job_ids=(fallback_job.job_id,),
                            batch_overrides={
                                fallback_job.job_id: self._resolved_batch_size_for_job_id(
                                    fallback_job.job_id
                                )
                            },
                            hardware_key=self.store.hardware_key(),
                            group_signature=build_group_signature(
                                [fallback_job.packing.signature or fallback_job.job_id]
                            ),
                        )
                        self._log_run_group_open(
                            self._active_runs[group_id],
                            [fallback_job],
                            reason="backend_fallback_dispatch",
                        )
                        self._last_telemetry_poll_at = 0.0
                        self.store.update_job(
                            fallback_job.job_id,
                            status=JobStatus.RUNNING,
                            reason="dispatched to worker after backend fallback",
                            hold=False,
                            metadata_updates={
                                "placement_mode": "exclusive",
                                "placement_backend": "exclusive",
                                "placement_role": "solo",
                                **self._prediction_metadata(fallback_job.job_id),
                            },
                        )
                        return True
                except Exception as fallback_exc:
                    self.logger.warning(
                        "Exclusive fallback dispatch also failed for %s: %s",
                        fallback_job.job_id,
                        fallback_exc,
                    )
            return False
        if not dispatched.can_run:
            self._cancel_prepared_colocation_trial(
                prepared_trial, reason=dispatched.reason
            )
            if replayed:
                self._invalidate_placement_replay(
                    reason="replayed dispatch was rejected",
                    job=selected_jobs[0] if selected_jobs else None,
                    details={
                        "backend_name": plan.backend_name,
                        "error": dispatched.reason,
                    },
                )
            self.logger.info(
                "Skipping dispatch for %s: %s",
                ",".join(plan.job_ids),
                dispatched.reason,
            )
            return False
        group_id = (
            dispatched.group_id or f"dispatch-{plan.job_ids[0]}-{time.monotonic_ns()}"
        )

        signatures = [job.packing.signature or job.job_id for job in selected_jobs]
        self._active_runs[group_id] = ActiveRun(
            group_id=group_id,
            mode=plan.mode,
            backend_name=plan.backend_name,
            job_ids=plan.job_ids,
            batch_overrides=dict(plan.batch_overrides),
            fallback_order=list(plan.fallback_order),
            hardware_key=self.store.hardware_key(),
            group_signature=build_group_signature(signatures),
            objective_breakdown=dict(plan.objective_breakdown),
            objective_version=plan.objective_version,
            mandatory_anchor_job_id=plan.mandatory_anchor_job_id,
        )
        self._log_run_group_open(
            self._active_runs[group_id], selected_jobs, reason=plan.reason
        )
        if len(self._active_runs) > 1:
            for run in self._active_runs.values():
                run.overlapped = True
        self._last_telemetry_poll_at = 0.0

        if prepared_trial is not None:
            self.event_logger.emit(
                "colocation_trial_started",
                job_id=prepared_trial.candidate_job_id,
                payload=prepared_trial.to_dict(),
            )

        if replayed:
            self._placement_replay.suppressed_probes += 1
            self._placement_replay.suppressed_trials += 1
            self._placement_replay.suppressed_decisions += 1
            self._persist_scheduler_decision_state()
            replay_job = selected_jobs[0] if selected_jobs else None
            self.event_logger.emit(
                "placement_replayed",
                job_id=replay_job.job_id if replay_job else None,
                payload={
                    "backend_name": plan.backend_name,
                    "mode": plan.mode,
                    "slot": plan.objective_breakdown.get("placement_replay_slot"),
                    "target_width": plan.objective_breakdown.get(
                        "placement_replay_target_width"
                    ),
                    "batch_size": (
                        plan.batch_overrides.get(replay_job.job_id)
                        if replay_job is not None
                        else None
                    ),
                    "skipped": [
                        "batch_probe",
                        "runtime_probe",
                        "colocation_trial",
                        "gain_scoring",
                    ],
                },
            )

        for index, job in enumerate(selected_jobs):
            if (
                prepared_trial is not None
                and job.job_id == prepared_trial.candidate_job_id
            ):
                role = "trial_newcomer"
            elif len(plan.job_ids) == 1:
                role = "solo"
            elif len(plan.job_ids) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            self.store.update_job(
                job.job_id,
                status=JobStatus.RUNNING,
                reason="dispatched to worker",
                hold=False,
                metadata_updates={
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "placement_role": role,
                    "placement_batch_size": plan.batch_overrides.get(job.job_id),
                    "placement_group_id": group_id,
                    "placement_objective_version": plan.objective_version,
                    "placement_objective_breakdown": plan.objective_breakdown,
                    "placement_trial_metadata": plan.trial_metadata,
                    "placement_backend_config": dict(plan.backend_config),
                    "placement_mandatory_anchor_job_id": plan.mandatory_anchor_job_id,
                    **self._prediction_metadata(job.job_id),
                },
            )
            self.event_logger.emit(
                "job_dispatched",
                job_id=job.job_id,
                payload={
                    "priority": job.priority,
                    "placement_mode": plan.mode,
                    "placement_backend": plan.backend_name,
                    "group_id": group_id,
                    "job_ids": list(plan.job_ids),
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                    "objective_version": plan.objective_version,
                    "objective_breakdown": plan.objective_breakdown,
                    "trial_metadata": plan.trial_metadata,
                    "mandatory_anchor_job_id": plan.mandatory_anchor_job_id,
                },
            )
        if len(plan.job_ids) == 2:
            self.event_logger.emit(
                "packed_pair_dispatched",
                payload={
                    "job_ids": list(plan.job_ids),
                    "group_id": group_id,
                    "backend_name": plan.backend_name,
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                },
            )
        elif len(plan.job_ids) > 2:
            self.event_logger.emit(
                "packed_group_dispatched",
                payload={
                    "job_ids": list(plan.job_ids),
                    "group_id": group_id,
                    "backend_name": plan.backend_name,
                    "batch_overrides": dict(plan.batch_overrides),
                    "reason": plan.reason,
                },
            )
        return True

    def _log_run_group_open(
        self, run: ActiveRun, jobs: list[TrainingJob], *, reason: str
    ) -> None:
        """Persist the membership and decision data for a new run group."""
        self.log_store.open_run_group(
            group_id=run.group_id,
            mode=run.mode,
            backend_name=run.backend_name,
            hardware_key=run.hardware_key or self.store.hardware_key(),
            group_signature=run.group_signature,
            opened_at=run.opened_at,
            overlapped=run.overlapped,
            metadata={"job_ids": list(run.job_ids), "reason": reason},
        )
        for index, job in enumerate(jobs):
            if len(jobs) == 1:
                role = "solo"
            elif len(jobs) == 2:
                role = "primary" if index == 0 else "secondary"
            else:
                role = f"slot-{index}"
            self.log_store.upsert_run_group_member(
                group_id=run.group_id,
                job_id=job.job_id,
                role=role,
                batch_size=run.batch_overrides.get(job.job_id),
                joined_at=run.opened_at,
                metadata={
                    "task_type": job.task_type,
                    "probe_task": bool(
                        job.batch_probe.enabled or job.runtime_probe.enabled
                    ),
                },
            )

    def _dispatch_pending_work(self) -> None:
        """Fill available capacity until policy or resource state says stop.

        Placement replay gets first refusal because it represents a previously
        verified decision. Otherwise the planner chooses from the live queue;
        concurrent modes may repeat the loop to build a packed stack.
        """
        concurrent_mode = bool(self.settings.gpu_scheduler.enabled)
        if not concurrent_mode and self._active_runs:
            return

        while True:
            active_job_ids = set(self._supervisor_active_job_ids())
            runnable = [
                job for job in self._runnable_jobs() if job.job_id not in active_job_ids
            ]
            if not runnable:
                return
            active_jobs = self._active_jobs()
            backend_available = self.supervisor.available_backends()
            replay_handled, replay_plan = self._choose_placement_replay_plan(
                runnable,
                active_jobs=active_jobs,
                backend_available=backend_available,
            )
            if replay_handled:
                if replay_plan is None:
                    return
                replay_dispatched = self._dispatch_plan(replay_plan)
                if not replay_dispatched:
                    if self._placement_replay.template is None:
                        continue
                    return
                if replay_plan.backend_name == "exclusive":
                    return
                continue
            if self.settings.gpu_scheduler.exclusive_probe.enabled:
                probes = [
                    job
                    for job in runnable
                    if job.scheduling_class == SchedulingClass.EXCLUSIVE_PROBE
                ]
                if self._exclusive_probe_job_id is None and probes:
                    reserved = sorted(
                        probes, key=lambda job: self.policy.sort_key(job)
                    )[0]
                    self._exclusive_probe_job_id = reserved.job_id
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "exclusive_probe_drain_requested",
                        job_id=reserved.job_id,
                        payload={
                            "active_job_ids": sorted(active_job_ids),
                            "draining": bool(active_job_ids),
                        },
                    )
                if self._exclusive_probe_job_id is not None:
                    if active_job_ids:
                        return
                    reserved = next(
                        (
                            job
                            for job in runnable
                            if job.job_id == self._exclusive_probe_job_id
                        ),
                        None,
                    )
                    if reserved is None:
                        self._exclusive_probe_job_id = None
                        self._persist_scheduler_decision_state()
                        continue
                    runnable = [reserved]
            active_vram_mb = self._active_vram_occupancy()
            plan = self.planner.choose_plan(
                runnable,
                backend_available=backend_available,
                active_vram_mb=active_vram_mb,
                active_jobs=active_jobs,
                admission_open=self._admission_gate.is_open,
                exclusive_drain_requested=bool(
                    self._exclusive_probe_job_id and active_jobs
                ),
                packing_admission_stalled=self._colocation_stall is not None,
                trial_pending=self._colocation_trial is not None,
            )
            if plan is None:
                return
            if (
                concurrent_mode
                and self._active_runs
                and plan.backend_name == "exclusive"
            ):
                return
            dispatched = self._dispatch_plan(plan)
            if (
                not dispatched
                or not concurrent_mode
                or plan.backend_name == "exclusive"
            ):
                return
