"""Incremental admission that never stops incumbent optimizer progress."""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

from ..domain import BatchResolution
from .completion_objective import remaining_optimizer_updates


ADD_ONLY_POLICY = "add_only_bounded_interference_v1"


class AddOnlyTrialMixin:
    def _admission_phase_supported(self, trial):
        return trial.phase in {"prepare", "pre_add", "overlap_warmup", "measure_overlap"}

    def _admission_commands_valid(self, trial, jobs):
        if trial.phase == "prepare":
            for job in jobs:
                command = job.metadata.get("trial_step_command") or {}
                if job.job_id != trial.candidate_job_id:
                    if command.get("owner") == trial.trial_id:
                        return False  # Prepare never owns an incumbent command.
                elif (command.get("owner") != trial.trial_id or command.get("action") != "yield"
                      or command.get("kind") != "prepare" or not command.get("token")):
                    return False
            return True
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            action = "yield" if trial.phase == "pre_add" and job.job_id == trial.candidate_job_id else "measure"
            if (command.get("owner") != trial.trial_id or command.get("kind") != trial.phase
                    or command.get("action") != action):
                return False
        return True

    def _add_only_identity(self, jobs):
        for job in jobs:
            identity = job.metadata.get("runtime_completion_identity")
            counter = job.metadata.get("runtime_nonupdate_phase_counter")
            if (not isinstance(identity, str) or not identity.strip()
                    or not isinstance(counter, int) or isinstance(counter, bool) or counter < 0):
                return None
        return {"hardware_key": self.store.hardware_key(), "members": [
            {"job_id": job.job_id, "signature": job.packing.signature,
             "runtime_identity": job.metadata.get("runtime_completion_identity"),
             "batch_size": BatchResolution.resolved_batch_size(job),
             "backend": job.metadata.get("placement_backend"),
             "config": job.metadata.get("placement_backend_config") or {},
             "precision": job.metadata.get("precision_mode")}
            for job in sorted(jobs, key=lambda item: item.job_id)]}

    @staticmethod
    def _add_only_window_valid(window):
        try:
            return (window is not None and window.get("excludes_nonupdate_phases") is True
                    and int(window["steps"]) >= 8 and int(window["end_step"]) - int(window["start_step"]) >= 8)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _add_only_interval(window):
        try:
            start = float(window["started_at"])
            finished = datetime.fromisoformat(window["finished_at"])
            if finished.tzinfo is None:
                finished = finished.replace(tzinfo=timezone.utc)
            end = finished.timestamp()
            return (start, end) if isfinite(start) and isfinite(end) and end > start else None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _refresh_disjoint_overlap_windows(self, trial, jobs, windows):
        """Refresh early samples while every resident continues training.

        A slow member's first measured update may occur after a fast member has
        finished its whole window. Start the latter again as soon as that slow
        start becomes visible, even before the slow member finishes its window.
        """
        starts = []
        intervals = {}
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            progress = job.metadata.get("trial_measurement_progress") or {}
            try:
                started = float(progress["started_at"])
                if (progress.get("token") == command.get("token") and isfinite(started)
                        and started >= trial.cold_start["phase_started_at"]):
                    starts.append(started)
            except (KeyError, TypeError, ValueError):
                pass
            interval = self._add_only_interval(windows.get(job.job_id) or {})
            if interval and interval[0] >= trial.cold_start["phase_started_at"]:
                intervals[job.job_id] = interval
                starts.append(interval[0])
        if not starts:
            return False, intervals
        latest_start = max(starts)
        refresh = [job for job in jobs if job.job_id in intervals and intervals[job.job_id][1] <= latest_start]
        if refresh:
            if len(intervals) == len(jobs):
                # Every saved sample is already in the past. Alternating
                # refreshes now could chase completed windows indefinitely.
                self._finish_cold_start_trial(trial, "inconclusive", "completed optimizer measurement intervals do not overlap")
                return True, intervals
            for job in refresh:
                self._cold_command(trial, job, "measure", "measure_overlap", reset=True)
            state = trial.cold_start["add_only"]
            state["overlap_window_refreshes"] = state.get("overlap_window_refreshes", 0) + len(refresh)
            self._persist_scheduler_decision_state()
            self.event_logger.emit("add_only_overlap_windows_refreshed", job_id=trial.candidate_job_id, payload={
                "trial_id": trial.trial_id, "admission_policy": ADD_ONLY_POLICY,
                "job_ids": [job.job_id for job in refresh], "latest_measurement_start": latest_start,
                "reason": "completed windows ended before another member's measured updates began",
            })
        return bool(refresh), intervals

    def _begin_cold_admission(self, trial, jobs, windows):
        identity = self._add_only_identity(jobs)
        if identity is None:
            self._finish_cold_start_trial(trial, "inconclusive", "shared-runner identity or non-update phase proof is missing")
            return
        old = [job for job in jobs if job.job_id != trial.candidate_job_id]
        if not all(self._add_only_window_valid(windows.get(job.job_id)) for job in old):
            self._finish_cold_start_trial(trial, "inconclusive", "pre-add evidence lacks complete phase-clean optimizer windows")
            return
        candidate = next(job for job in jobs if job.job_id == trial.candidate_job_id)
        start = int(candidate.metadata.get("runtime_global_step", 0))
        steps = candidate.metadata.get("runtime_steps_per_epoch")
        remaining = remaining_optimizer_updates(candidate, {
            "global_step": start, "steps_per_epoch": steps,
            "total_optimizer_updates": candidate.metadata.get("runtime_total_optimizer_updates"),
            "planned_epochs": candidate.metadata.get("runtime_planned_epochs"),
        })
        if remaining is None or not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            self._finish_cold_start_trial(trial, "inconclusive", "add-only optimizer-update budget is unknown")
            return
        trial.cold_start["target_step"] = min(start + remaining,
            start + self.settings.gpu_scheduler.colocation.cold_start_measurement_epoch_limit * steps)
        trial.cold_start["admission_policy"] = ADD_ONLY_POLICY
        trial.cold_start["add_only"] = {
            "identity": identity,
            "pre_add": {job.job_id: list(windows[job.job_id]["halves"]) for job in old},
            "pre_add_windows": {job.job_id: windows[job.job_id] for job in old},
            "candidate_start_step": start,
            "warmup_start_steps": {job.job_id: int(job.metadata.get("runtime_global_step", 0)) for job in jobs},
        }
        # Candidate release is the only behavior change. Incumbents have been
        # running throughout prepare/pre_add and continue while measuring.
        self._cold_phase(trial, "overlap_warmup", jobs)

    def _evaluate_admission_phase(self, trial, jobs, windows):
        if trial.phase not in {"overlap_warmup", "measure_overlap"}:
            return False
        state = trial.cold_start.get("add_only")
        if not state or state.get("identity") != self._add_only_identity(jobs):
            self._finish_cold_start_trial(trial, "inconclusive", "add-only measurement identity changed or is missing")
            return True
        candidate = next(job for job in jobs if job.job_id == trial.candidate_job_id)
        if int(candidate.metadata.get("runtime_global_step", 0)) >= int(trial.cold_start["target_step"]):
            self._finish_cold_start_trial(trial, "inconclusive", "add-only measurement optimizer-update budget exhausted")
            return True
        if trial.phase == "overlap_warmup":
            missing = []
            for job in jobs:
                command = job.metadata.get("trial_step_command") or {}
                progress = job.metadata.get("trial_measurement_progress") or {}
                try:
                    ready = (progress.get("token") == command.get("token")
                             and isfinite(float(progress["started_at"]))
                             and float(progress["started_at"]) >= trial.cold_start["phase_started_at"]
                             and int(progress["global_step"]) >= state["warmup_start_steps"][job.job_id] + 2)
                except (KeyError, TypeError, ValueError):
                    ready = False
                if not ready:
                    missing.append(job)
            if not missing:
                # Both sides have made fresh optimizer progress before any
                # accepted overlap window. Reset tokens without yielding.
                self._cold_phase(trial, "measure_overlap", jobs)
            elif self._cold_measurement_timed_out(trial, missing):
                self._finish_cold_start_trial(trial, "inconclusive", "add-only overlap warmup timed out")
            return True
        refreshed, intervals = self._refresh_disjoint_overlap_windows(trial, jobs, windows)
        if refreshed:
            return True
        missing = [job for job in jobs if not self._add_only_window_valid(windows.get(job.job_id))]
        if missing:
            if self._cold_measurement_timed_out(trial, missing):
                self._finish_cold_start_trial(trial, "inconclusive", "add-only overlap window timed out after optimizer progress")
            return True
        if len(intervals) != len(jobs):
            self._finish_cold_start_trial(trial, "inconclusive", "overlap windows lack fresh measured-time intervals")
            return True
        common_start = max(interval[0] for interval in intervals.values())
        common_end = min(interval[1] for interval in intervals.values())
        if common_start >= common_end:
            # Normally the early-window refresh above handles this. Keep this
            # explicit guard so missing or malformed interval evidence cannot
            # silently become a simultaneous-compute claim.
            self._finish_cold_start_trial(trial, "inconclusive", "optimizer measurement intervals do not overlap")
            return True
        candidate_progress = int(windows[candidate.job_id]["end_step"]) - state["candidate_start_step"]
        if candidate_progress < 8:
            self._finish_cold_start_trial(trial, "inconclusive", "newcomer did not make enough optimizer progress")
            return True
        old = [job for job in jobs if job.job_id != candidate.job_id]
        ratios = {job.job_id: [float(windows[job.job_id]["halves"][half]) / state["pre_add"][job.job_id][half]
                              for half in range(2)] for job in old}
        tolerance = self.settings.gpu_scheduler.colocation.cold_start_max_incumbent_slowdown_fraction
        threshold = 1 + tolerance
        decision = ("accepted" if all(value <= threshold + 1e-12 for values in ratios.values() for value in values)
                    else "rejected" if any(all(value > threshold + 1e-12 for value in values) for values in ratios.values())
                    else "inconclusive")
        delays = {}
        for job in old:
            remaining = remaining_optimizer_updates(job, {
                "global_step": windows[job.job_id]["end_step"],
                "steps_per_epoch": job.metadata.get("runtime_steps_per_epoch"),
                "total_optimizer_updates": job.metadata.get("runtime_total_optimizer_updates"),
                "planned_epochs": job.metadata.get("runtime_planned_epochs"),
            })
            delays[job.job_id] = ([remaining * (float(windows[job.job_id]["halves"][half]) - state["pre_add"][job.job_id][half])
                                  for half in range(2)] if remaining is not None else None)
        result = {"objective": "bounded_incumbent_interference", "admission_policy": ADD_ONLY_POLICY,
                  "max_incumbent_slowdown_fraction": tolerance, "incumbent_step_time_ratios": ratios,
                  "reference": "current incumbent pack before this addition; not original solo speed",
                  "candidate_optimizer_progress": candidate_progress,
                  "pre_add": state["pre_add"], "packed": {name: window["halves"] for name, window in windows.items()},
                  "measurement_windows": windows,
                  "common_measurement_interval": {"started_at": common_start, "finished_at": common_end},
                  "overlap_window_refreshes": state.get("overlap_window_refreshes", 0),
                  "remaining_update_delay_if_rates_persist_seconds": delays,
                  "completion_gain": None, "candidate_solo_seconds": None,
                  "completion_note": "delay diagnostics assume current step rates persist; validation/finalization and unknown solo/tail rates prevent a makespan proof",
                  "predictor_role": "VRAM feasibility and busy ordering only"}
        self._finish_cold_start_trial(trial, decision, "add-only bounded interference " + decision, result)
        return True
