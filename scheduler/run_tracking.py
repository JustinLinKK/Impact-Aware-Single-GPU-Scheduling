"""Track active worker groups and turn completed runs into scheduler evidence.

The service loop delegates worker exits here. A completed exclusive run
updates a solo profile; a completed packed run updates combination and pair
profiles. Keeping this accounting together prevents dispatch code from also
having to understand how historical evidence is stored.
"""

from __future__ import annotations

from typing import Any

from ..domain import (
    CombinationProfile,
    JobStatus,
    PairProfile,
    SoloProfile,
    TrainingJob,
    build_group_signature,
    utc_now,
)
from .service_state import ActiveRun
from .supervisor import WorkerSnapshot
from .telemetry import GpuTelemetrySummary


class RunTrackingMixin:
    """Maintain active-run state and persist evidence after workers exit."""

    def _poll_active_workers(self) -> None:
        """Reconcile worker exits and close groups that no longer have members."""
        snapshots = self.supervisor.poll()
        if not snapshots:
            return
        for snapshot in snapshots:
            run = self._active_runs.get(snapshot.group_id)
            self._handle_worker_exit(snapshot, run_context=run)

        remaining_by_group = self._supervisor_active_job_ids_by_group()
        for group_id, run in list(self._active_runs.items()):
            remaining_job_ids = remaining_by_group.get(group_id, [])
            if len(run.job_ids) > 1 and len(remaining_job_ids) < len(run.job_ids):
                self._record_combination_profiles(run)
            elif len(run.job_ids) == 1 and not remaining_job_ids:
                self._record_solo_profiles(run)

            if not remaining_job_ids:
                self.log_store.close_run_group(
                    group_id=group_id,
                    closed_at=utc_now(),
                    overlapped=run.overlapped,
                    fallback_triggered=run.fallback_triggered,
                    fallback_reason=run.fallback_reason,
                    exit_reason=run.fallback_reason or "group_complete",
                )
                self._active_runs.pop(group_id, None)
                if self._exclusive_probe_job_id in run.job_ids:
                    completed_probe = self._exclusive_probe_job_id
                    self._exclusive_probe_job_id = None
                    self._persist_scheduler_decision_state()
                    self.event_logger.emit(
                        "exclusive_probe_drain_cleared",
                        job_id=completed_probe,
                        payload={"reason": "probe finished"},
                    )
                continue
            if tuple(remaining_job_ids) != run.job_ids:
                removed_job_ids = [
                    job_id for job_id in run.job_ids if job_id not in remaining_job_ids
                ]
                for removed_job_id in removed_job_ids:
                    self.log_store.mark_run_group_member_left(
                        group_id=group_id, job_id=removed_job_id, left_at=utc_now()
                    )
                if len(remaining_job_ids) == 1:
                    run.mode = "exclusive"
                run.job_ids = tuple(remaining_job_ids)
                run.fallback_order = [
                    job_id
                    for job_id in run.fallback_order
                    if job_id in remaining_job_ids
                ]
                run.group_signature = build_group_signature(
                    [
                        (self.store.get_job(job_id).packing.signature or job_id)
                        for job_id in remaining_job_ids
                        if self.store.get_job(job_id) is not None
                    ]
                )
        self._finalize_pending_pattern()

    def _handle_worker_exit(
        self, snapshot: WorkerSnapshot, *, run_context: ActiveRun | None
    ) -> None:
        """Translate one worker exit into job state, metrics, and evidence."""
        job = self.store.get_job(snapshot.job_id)
        if job is None:
            return
        if snapshot.reported_by == "store":
            if (
                run_context is not None
                and len(run_context.job_ids) > 1
                and job.status == JobStatus.FAILED
            ):
                self._register_packed_fallback(
                    run_context,
                    job.status_reason or "packed-backend worker failed",
                    payload={"failed_job_id": snapshot.job_id},
                )
            return
        if snapshot.returncode == 0:
            if job.status in {
                JobStatus.COMPLETED,
                JobStatus.PAUSED,
                JobStatus.CANCELLED,
                JobStatus.READY,
            }:
                return
            self.store.set_job_status(
                job.job_id,
                JobStatus.FAILED,
                reason="worker exited without terminal status update",
                hold=True,
            )
            self.event_logger.emit(
                "job_failed",
                job_id=job.job_id,
                payload={"reason": "worker exited cleanly without terminal status"},
            )
            return

        if not job.status.is_terminal:
            reason = f"worker exited with code {snapshot.returncode}"
            self.store.set_job_status(
                job.job_id, JobStatus.FAILED, reason=reason, hold=True
            )
            self.event_logger.emit(
                "job_failed",
                job_id=job.job_id,
                payload={"returncode": snapshot.returncode},
            )
            if run_context is not None and len(run_context.job_ids) > 1:
                self._register_packed_fallback(
                    run_context,
                    reason,
                    payload={
                        "failed_job_id": snapshot.job_id,
                        "returncode": snapshot.returncode,
                    },
                )
            return

        if (
            run_context is not None
            and len(run_context.job_ids) > 1
            and job.status == JobStatus.FAILED
        ):
            reason = (
                job.status_reason or f"worker exited with code {snapshot.returncode}"
            )
            self._register_packed_fallback(
                run_context,
                reason,
                payload={
                    "failed_job_id": snapshot.job_id,
                    "returncode": snapshot.returncode,
                },
            )

    def _register_packed_fallback(
        self, run: ActiveRun, reason: str, *, payload: dict[str, Any]
    ) -> None:
        """Record why a packed run fell back to safer execution."""
        if len(run.job_ids) < 2 or run.fallback_triggered:
            return
        run.fallback_triggered = True
        run.fallback_reason = reason
        for job_id in run.job_ids:
            replayed_job = self.store.get_job(job_id)
            if replayed_job is not None and replayed_job.metadata.get(
                "skip_active_scheduler_probes"
            ):
                self._clear_replay_job_metadata(replayed_job)
        self._invalidate_placement_replay(
            reason="packed backend failure",
            details={"backend_name": run.backend_name, "failure_reason": reason},
        )
        self.event_logger.emit(
            "packed_group_fallback",
            payload={"job_ids": list(run.job_ids), "reason": reason, **payload},
        )

    def _record_solo_profiles(self, run: ActiveRun) -> None:
        """Persist exclusive-run timing and memory measurements."""
        if run.overlapped:
            return
        summary = GpuTelemetrySummary.from_samples(run.samples)
        for job_id in run.job_ids:
            job = self.store.get_job(job_id)
            if job is None or not job.packing.signature:
                continue
            if not job.packing.eligible:
                continue
            if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                continue
            peak_vram_mb = summary.peak_vram_mb
            if peak_vram_mb is None:
                peak_vram_mb = job.resource_requirements.estimated_vram_mb
            avg_vram_mb = summary.avg_vram_mb
            if avg_vram_mb is None:
                avg_vram_mb = job.resource_requirements.estimated_avg_vram_mb
            self.store.upsert_solo_profile(
                SoloProfile(
                    signature=job.packing.signature,
                    hardware_key=run.hardware_key or self.store.hardware_key(),
                    family=job.packing.family,
                    peak_vram_mb=peak_vram_mb,
                    avg_vram_mb=avg_vram_mb,
                    avg_gpu_utilization=(
                        summary.avg_gpu_utilization
                        if summary.avg_gpu_utilization is not None
                        else 0.0
                    ),
                    avg_memory_utilization=(
                        summary.avg_memory_utilization
                        if summary.avg_memory_utilization is not None
                        else 0.0
                    ),
                    sample_count=summary.sample_count,
                    last_job_id=job.job_id,
                    metadata={
                        "source": "exclusive_run",
                        "backend_name": run.backend_name,
                    },
                )
            )

    def _record_combination_profiles(self, run: ActiveRun) -> None:
        """Persist packed-run compatibility without inventing slowdown data.

        Slowdown is retained only when it came from clean per-epoch evidence;
        whole-run elapsed time includes startup and drain effects and is not a
        valid member-level slowdown measurement.
        """
        if len(run.job_ids) < 2 or run.overlapped:
            return
        jobs = [self.store.get_job(job_id) for job_id in run.job_ids]
        if any(job is None for job in jobs):
            return
        summary = GpuTelemetrySummary.from_samples(run.samples)
        materialized_jobs = [job for job in jobs if job is not None]
        group_signature = run.group_signature or build_group_signature(
            [job.packing.signature or job.job_id for job in materialized_jobs]
        )
        existing = self.store.best_combination_profile(
            group_signature=group_signature,
            hardware_key=run.hardware_key or self.store.hardware_key(),
            backend_name=run.backend_name,
            scheduler_mode=self.settings.gpu_scheduler.mode,
        )
        compatible = not run.fallback_triggered and all(
            job.status != JobStatus.FAILED for job in materialized_jobs
        )
        objective_score = run.objective_breakdown.get("score")
        numeric_objective_score = (
            float(objective_score)
            if isinstance(objective_score, (int, float))
            else None
        )
        self.store.upsert_combination_profile(
            CombinationProfile.create(
                group_signature=group_signature,
                hardware_key=run.hardware_key or self.store.hardware_key(),
                backend_name=run.backend_name,
                scheduler_mode=self.settings.gpu_scheduler.mode,
                batch_vector=run.batch_overrides,
                compatible=compatible,
                observations=(existing.observations + 1) if existing else 1,
                peak_vram_mb=summary.peak_vram_mb,
                avg_vram_mb=summary.avg_vram_mb,
                memory_total_mb=(
                    run.samples[-1].memory_total_mb if run.samples else None
                ),
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                avg_step_time_ms=None,
                objective_score=numeric_objective_score,
                resolved_optimal=False,
                last_failure_reason=run.fallback_reason,
                fallback_order=run.fallback_order,
                metadata={
                    "backend_name": run.backend_name,
                    "job_ids": list(run.job_ids),
                    "objective_version": run.objective_version,
                    "objective_breakdown": run.objective_breakdown,
                    "mandatory_anchor_job_id": run.mandatory_anchor_job_id,
                },
            )
        )
        if len(materialized_jobs) != 2:
            return
        left_job, right_job = materialized_jobs
        if not left_job.packing.signature or not right_job.packing.signature:
            return
        if not compatible:
            self.store.mark_pair_incompatible(
                left_job.packing.signature,
                right_job.packing.signature,
                backend_name=run.backend_name,
                reason=run.fallback_reason or "packed group failed",
                cooldown_seconds=self.settings.gpu_scheduler.fallback_cooldown_seconds,
                peak_vram_mb=summary.peak_vram_mb,
                avg_vram_mb=summary.avg_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                metadata={"backend_name": run.backend_name},
            )
            return
        existing_pair = self.store.get_pair_profile(
            left_job.packing.signature,
            right_job.packing.signature,
            backend_name=run.backend_name,
        )
        existing_metadata = dict(existing_pair.metadata or {}) if existing_pair else {}
        existing_sources = existing_metadata.get("slowdown_sources")
        has_epoch_evidence = isinstance(existing_sources, dict) and any(
            source == "measured_epoch_against_exclusive_profile"
            for source in existing_sources.values()
        )
        # Combination completion still records compatibility and telemetry,
        # but it must never infer slowdown from whole-run elapsed time. Preserve
        # only slowdown evidence produced from clean epoch intervals.
        recorded_slowdown = (
            existing_pair.slowdown_ratio
            if existing_pair and has_epoch_evidence
            else None
        )
        per_member_slowdown = (
            existing_metadata.get("per_member_slowdown", {})
            if has_epoch_evidence
            else {}
        )
        per_signature_slowdown = (
            existing_metadata.get("per_signature_slowdown", {})
            if has_epoch_evidence
            else {}
        )
        slowdown_sources = existing_sources if has_epoch_evidence else {}
        slowdown_batch_vector = (
            existing_metadata.get("batch_vector", {}) if has_epoch_evidence else {}
        )
        self.store.upsert_pair_profile(
            PairProfile.create(
                left_job.packing.signature,
                right_job.packing.signature,
                backend_name=run.backend_name,
                hardware_key=run.hardware_key or self.store.hardware_key(),
                compatible=True,
                observations=(existing_pair.observations + 1) if existing_pair else 1,
                peak_vram_mb=summary.peak_vram_mb,
                avg_vram_mb=summary.avg_vram_mb,
                avg_gpu_utilization=summary.avg_gpu_utilization,
                avg_memory_utilization=summary.avg_memory_utilization,
                slowdown_ratio=recorded_slowdown,
                cooldown_until=None,
                last_failure_reason=None,
                metadata={
                    "backend_name": run.backend_name,
                    "batch_vector": slowdown_batch_vector,
                    "per_member_slowdown": per_member_slowdown,
                    "per_signature_slowdown": per_signature_slowdown,
                    "slowdown_sources": slowdown_sources,
                },
            )
        )
