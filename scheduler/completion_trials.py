"""Selected-only, productive evidence for the measured completion proxy."""
from __future__ import annotations

from datetime import datetime
from math import isfinite
import json
import time

from ..domain import BatchResolution
from .completion_objective import compare_update_completion, observed_phase_reserve, remaining_optimizer_updates


def _group_key(member_ids):
    return json.dumps(sorted(member_ids), separators=(",", ":"))


class CompletionTrialMixin:
    """Research-only mixin: no production SchedulerService inherits this class."""

    def _admission_phase_supported(self, trial):
        return trial.phase in {"prepare", "pre_add"} or trial.phase.startswith("completion_")

    def _admission_commands_valid(self, trial, jobs):
        return trial.phase == "prepare" or all(
            (job.metadata.get("trial_step_command") or {}).get("owner") == trial.trial_id for job in jobs)

    def _begin_cold_admission(self, trial, jobs, windows):
        trial.cold_start["admission_policy"] = "research_completion_proxy_v1"
        self._completion_begin(trial, jobs, windows)

    def _evaluate_admission_phase(self, trial, jobs, windows):
        if not trial.phase.startswith("completion_"):
            return False
        self._evaluate_completion_phase(trial, jobs, windows)
        return True

    def _completion_identity(self, jobs):
        return {"hardware_key": self.store.hardware_key(), "members": [
            {"job_id": job.job_id, "signature": job.packing.signature,
             "completion_identity": job.metadata.get("runtime_completion_identity"),
             "batch_size": BatchResolution.resolved_batch_size(job),
             "backend": job.metadata.get("placement_backend"),
             "config": job.metadata.get("placement_backend_config") or {},
             "precision": job.metadata.get("precision_mode")}
            for job in sorted(jobs, key=lambda item: item.job_id)]}

    def _completion_valid_observation(self, observation, jobs):
        try:
            return (observation["identity"] == self._completion_identity(jobs)
                    and all(job.metadata.get("runtime_completion_identity") for job in jobs)
                    and observation["source"] == "phase_clean_step_windows_v1"
                    and 0 <= time.time() - float(observation["at"]) <= self.settings.gpu_scheduler.colocation.profile_rejection_ttl_seconds
                    and set(observation["rates"]) == {job.job_id for job in jobs}
                    and all(len(values) == 2 and all(isfinite(float(value)) and float(value) > 0 for value in values)
                            for values in observation["rates"].values()))
        except (KeyError, TypeError, ValueError):
            return False

    def _completion_load_observations(self, trial, jobs):
        state = trial.cold_start.setdefault("completion", {"groups": {}})
        by_id = {job.job_id: job for job in jobs}
        for job in jobs:
            for key, observation in (job.metadata.get("completion_group_observations") or {}).items():
                try:
                    members = json.loads(key)
                    if not set(members) <= set(by_id):
                        continue
                    selected = [by_id[name] for name in members]
                    if self._completion_valid_observation(observation, selected):
                        previous = state["groups"].get(key)
                        if previous is None or observation["at"] > previous["at"]:
                            state["groups"][key] = observation
                except (KeyError, TypeError, ValueError):
                    continue
        # A runtime/source/membership change never turns old rates into new proof.
        state["groups"] = {key: value for key, value in state["groups"].items()
                           if set(json.loads(key)) <= set(by_id)
                           and self._completion_valid_observation(value, [by_id[name] for name in json.loads(key)])}
        return state

    def _completion_record_group(self, trial, jobs, windows):
        if any(not windows.get(job.job_id) or windows[job.job_id].get("excludes_nonupdate_phases") is not True for job in jobs):
            return False
        observation = {"identity": self._completion_identity(jobs), "at": time.time(),
                       "rates": {job.job_id: list(windows[job.job_id]["halves"]) for job in jobs},
                       "source": "phase_clean_step_windows_v1", "trial_id": trial.trial_id}
        if not self._completion_valid_observation(observation, jobs):
            return False
        key = _group_key([job.job_id for job in jobs])
        trial.cold_start.setdefault("completion", {"groups": {}})["groups"][key] = observation
        for job in jobs:
            current = self.store.get_job(job.job_id)
            cached = dict(current.metadata.get("completion_group_observations") or {})
            cached[key] = observation
            cached = dict(sorted(cached.items(), key=lambda item: item[1].get("at", 0))[-32:])
            self.store.update_job(job.job_id, metadata_updates={"completion_group_observations": cached})
        return True

    def _completion_snapshot(self, trial, jobs):
        snapshots = {}
        for job in jobs:
            command = job.metadata.get("trial_step_command") or {}
            snapshot = job.metadata.get("trial_yield_snapshot") or {}
            if (command.get("owner") != trial.trial_id or not command.get("token")
                    or job.metadata.get("trial_yield_ack") != command["token"]
                    or snapshot.get("token") != command["token"] or snapshot.get("owner") != trial.trial_id):
                return None
            if remaining_optimizer_updates(job, snapshot) is None:
                return None
            try:
                at = float(snapshot["observed_at"])
                if not isfinite(at) or not float(trial.cold_start["phase_started_at"]) <= at <= time.time():
                    return None
            except (KeyError, TypeError, ValueError):
                return None
            snapshots[job.job_id] = dict(snapshot)
        return snapshots

    def _completion_request_group(self, trial, jobs, member_ids):
        trial.cold_start["completion"]["measuring"] = sorted(member_ids)
        self._cold_phase(trial, "completion_measurement_barrier", jobs, measuring=sorted(member_ids))

    def _completion_begin(self, trial, jobs, windows):
        self._completion_load_observations(trial, jobs)
        old = [job for job in jobs if job.job_id != trial.candidate_job_id]
        if not self._completion_record_group(trial, old, windows):
            self._finish_cold_start_trial(trial, "inconclusive", "pre-add window lacks phase-clean runtime identity")
            return
        self._cold_phase(trial, "completion_initial_barrier", jobs)

    def _completion_phase_reserves(self, jobs, snapshots):
        values, diagnostics = {}, {}
        for job in jobs:
            value, detail = observed_phase_reserve(job, snapshots[job.job_id], hardware_key=self.store.hardware_key(),
                now=time.time(), ttl_seconds=self.settings.gpu_scheduler.colocation.profile_rejection_ttl_seconds)
            values[job.job_id], diagnostics[job.job_id] = value, detail
        return values, diagnostics

    def _completion_score(self, trial, jobs, snapshots):
        state = self._completion_load_observations(trial, jobs)
        initial = state["initial_snapshots"]
        initial_remaining = {job.job_id: remaining_optimizer_updates(job, initial[job.job_id]) for job in jobs}
        remaining = {job.job_id: remaining_optimizer_updates(job, snapshots[job.job_id]) for job in jobs}
        observed_at = max(float(snapshot["observed_at"]) for snapshot in snapshots.values())
        elapsed = observed_at - state["comparison_started_at"]
        rates = {tuple(json.loads(key)): value["rates"] for key, value in state["groups"].items()}
        comparison = compare_update_completion(initial_remaining, remaining, trial.candidate_job_id, rates, elapsed)
        if comparison["error"]:
            self._finish_cold_start_trial(trial, "inconclusive", comparison["error"])
            return
        if comparison["missing_memberships"]:
            self._completion_request_group(trial, jobs, comparison["missing_memberships"][0])
            return
        initial_reserves, initial_details = self._completion_phase_reserves(jobs, initial)
        current_reserves, current_details = self._completion_phase_reserves(jobs, snapshots)
        if any(detail.get("error") for detail in [*initial_details.values(), *current_details.values()]):
            self._finish_cold_start_trial(trial, "inconclusive", "invalid completion phase evidence",
                                          {"initial_phases": initial_details, "current_phases": current_details})
            return
        if any(value is None for value in [*initial_reserves.values(), *current_reserves.values()]):
            state["missing_phase_evidence"] = {"initial": initial_details, "current": current_details}
            self._cold_phase(trial, "completion_provisional", jobs)
            return
        halves = []
        for half in comparison["halves"]:
            initial_defer = half["initial_defer_seconds"] + sum(initial_reserves.values())
            keep = half["current_keep_seconds"] + sum(current_reserves.values())
            defer = half["current_defer_seconds"] + sum(current_reserves.values())
            actual_keep = elapsed + keep
            halves.append({**half, "initial_defer_proxy_seconds": initial_defer,
                           "current_keep_proxy_seconds": keep, "current_defer_proxy_seconds": defer,
                           "attempt_keep_proxy_seconds": actual_keep,
                           "amortized_gain": initial_defer / actual_keep if actual_keep > 0 else None,
                           "continuation_gain": defer / keep if keep > 0 else None})
        gains = [half["amortized_gain"] for half in halves]
        forward = [half["continuation_gain"] for half in halves]
        margin = max(self.settings.gpu_scheduler.colocation.min_gain,
                     1 + self.settings.gpu_scheduler.colocation.cold_start_gain_margin)
        decision = ("accepted" if all(value is not None and value >= margin for value in [*gains, *forward])
                    else "rejected" if all(value is not None and value < 1 for value in forward)
                    else "inconclusive")
        result = {"objective": "modeled_completion_proxy_v1", "halves": halves,
                  "required_gain": margin,
                  "initial_remaining_updates": initial_remaining, "remaining_updates": remaining,
                  "initial_phase_reserves": initial_details, "current_phase_reserves": current_details,
                  "unknown_finalization_job_ids": [name for name, detail in current_details.items() if detail["unknown_finalization"]],
                  "excluded_precomparison_seconds": state["excluded_precomparison_seconds"],
                  "excluded_precomparison_basis": "startup and pre-add evidence are sunk before the shared initial snapshot",
                  "defer_transition_overhead": {"seconds": 0.0,
                      "basis": "unmeasured pause/checkpoint/relaunch costs omitted for an optimistic DEFER comparator"},
                  "comparison_started_at": state["comparison_started_at"],
                  "scope": "selected active cohort plus candidate; unknown pending queue durations are not invented",
                  "reserve_model": "serialized empirical phase maxima; not a guaranteed completion bound",
                  "predictor_role": "ordering and VRAM guard only", "measurement_groups": list(state["groups"])}
        self._finish_cold_start_trial(trial, decision, "modeled completion proxy " + decision, result)

    def _evaluate_completion_phase(self, trial, jobs, windows):
        state = trial.cold_start["completion"]
        candidate = next(job for job in jobs if job.job_id == trial.candidate_job_id)
        settings = self.settings.gpu_scheduler.colocation
        if state.get("initial_snapshots") and (
                state.get("hardware_key") != self.store.hardware_key()
                or any(state["initial_snapshots"].get(job.job_id, {}).get("completion_identity")
                       != job.metadata.get("runtime_completion_identity") for job in jobs)):
            self._finish_cold_start_trial(trial, "inconclusive", "runtime completion identity changed")
            return
        exhausted = int(candidate.metadata.get("runtime_global_step", 0)) >= int(trial.cold_start["target_step"])
        if trial.phase in {"completion_initial_barrier", "completion_measurement_barrier", "completion_decision_barrier"}:
            snapshots = self._completion_snapshot(trial, jobs)
            if snapshots is None:
                if time.time() - trial.cold_start["phase_started_at"] > settings.cold_start_prepare_timeout_seconds:
                    self._finish_cold_start_trial(trial, "inconclusive", "exact completion snapshot timed out")
                return
            if trial.phase == "completion_initial_barrier":
                state["initial_snapshots"] = snapshots
                state["hardware_key"] = self.store.hardware_key()
                state["comparison_started_at"] = max(float(value["observed_at"]) for value in snapshots.values())
                state["excluded_precomparison_seconds"] = max(0.0, state["comparison_started_at"] - datetime.fromisoformat(trial.started_at).timestamp())
                base = snapshots[candidate.job_id]
                trial.cold_start["target_step"] = int(base["global_step"]) + settings.cold_start_measurement_epoch_limit * int(base["steps_per_epoch"])
                group = ([candidate.job_id] if _group_key([candidate.job_id]) not in state["groups"]
                         else [job.job_id for job in jobs])
                self._completion_request_group(trial, jobs, group)
            elif trial.phase == "completion_measurement_barrier":
                if exhausted:
                    self._finish_cold_start_trial(trial, "inconclusive", "completion evidence optimizer-update budget exhausted")
                else:
                    self._cold_phase(trial, "completion_measure", jobs, measuring=state["measuring"])
            else:
                self._completion_score(trial, jobs, snapshots)
            return
        if trial.phase == "completion_measure":
            selected = [job for job in jobs if job.job_id in state["measuring"]]
            if exhausted:
                # A fast candidate may already have yielded at its budget while
                # a slow member completed a window under reduced contention.
                self._finish_cold_start_trial(trial, "inconclusive", "completion subgroup evidence optimizer-update budget exhausted")
                return
            if not all(windows.get(job.job_id) for job in selected):
                if self._cold_measurement_timed_out(trial, [job for job in selected if not windows.get(job.job_id)]):
                    self._finish_cold_start_trial(trial, "inconclusive", "completion subgroup evidence budget exhausted")
                return
            if not self._completion_record_group(trial, selected, windows):
                self._finish_cold_start_trial(trial, "inconclusive", "completion subgroup window lacks phase-clean identity")
                return
            if len(selected) == len(jobs):
                state["packed_measured"] = True
            if not state.get("packed_measured"):
                self._completion_request_group(trial, jobs, [job.job_id for job in jobs])
            else:
                self._cold_phase(trial, "completion_decision_barrier", jobs)
            return
        if trial.phase == "completion_provisional":
            reserves, _ = self._completion_phase_reserves(jobs, state["initial_snapshots"])
            if all(value is not None for value in reserves.values()):
                self._cold_phase(trial, "completion_decision_barrier", jobs)
            elif exhausted:
                self._finish_cold_start_trial(trial, "inconclusive", "completion phase evidence optimizer-update budget exhausted")
            return
        self._finish_cold_start_trial(trial, "inconclusive", "unknown completion phase")
