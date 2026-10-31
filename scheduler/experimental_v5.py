"""Opt-in V5 integration. Unchanged SchedulerService remains the V4 comparator.

The three flags change one mechanism each. Shadow forecasts have no authority
and record only predictions; realized errors are evaluated on executed paths.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
import time
from pathlib import Path

from ..domain import BatchResolution, JobStatus
from ..benchmarks.a10_feedback_latency.async_preparation import AsyncV4Service
from .bracket_bootstrap import BracketBootstrapMixin
from .reference_execution import ReferenceExecutionMixin
from .completion_objective import remaining_optimizer_updates
from .continuation_forecast import compare_feedback_admission, reference_continuation
from .deferral_reservations import DeferralReservationMixin, choose_protected, KEY
from .feedback_objective import derived_solo_slope, timestamp
from .feedback_trials import (freeze_feedback_trial, phase_costs, profile_snapshot)
from .v5_lifecycle import forecast_cohort, snapshot as lifecycle_snapshot, freeze_trial


@dataclass(frozen=True)
class V5Options:
    continuation: bool = False
    deferral: bool = False
    bracket: bool = False
    shadow: bool = False
    deferral_seconds: float = 30.0
    max_younger_bypasses: int = 2
    evidence_ttl_seconds: float = 30.0
    bracket_probe_seconds: float = 30.0
    forecast_cpu_ms: float = 50.0
    forecast_max_events: int = 128

    def __post_init__(self):
        for key in ('continuation', 'deferral', 'bracket', 'shadow'):
            if type(getattr(self, key)) is not bool:
                raise ValueError('feature switches must be booleans')
        for key in ('deferral_seconds', 'evidence_ttl_seconds', 'bracket_probe_seconds', 'forecast_cpu_ms'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive bound required: ' + key)
        if (type(self.max_younger_bypasses) is not int or self.max_younger_bypasses < 1
                or type(self.forecast_max_events) is not int or not 1 <= self.forecast_max_events <= 128
                or self.forecast_cpu_ms > 50):
            raise ValueError('invalid bounded reference rollout options')


def update_state(job):
    """Zero is FINISHING, never UNKNOWN. Strict runner budget checks are retained."""
    step = job.metadata.get('runtime_global_step')
    if step is None:
        if job.started_at or job.latest_checkpoint_path or job.resume_from_checkpoint:
            return 'UNKNOWN', None
        step = 0
    steps = job.metadata.get('runtime_steps_per_epoch')
    if steps is None:
        from ..prediction.released_student import steps_for_job
        try:
            steps = steps_for_job(job, BatchResolution.resolved_batch_size(job))
        except (ValueError, KeyError, TypeError):
            pass
    remaining = remaining_optimizer_updates(job, {
        'global_step': step, 'steps_per_epoch': steps,
        'total_optimizer_updates': job.metadata.get('runtime_total_optimizer_updates'),
        'planned_epochs': job.metadata.get('runtime_planned_epochs')})
    if remaining is None:
        return 'UNKNOWN', None
    if remaining == 0:
        return ('DONE' if job.metadata.get('feedback_ready_at') else 'FINISHING'), 0
    return 'TRAINING', remaining


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class V5Service(DeferralReservationMixin, ReferenceExecutionMixin, BracketBootstrapMixin, AsyncV4Service):
    def __init__(self, settings, *, v5_options=None, **kwargs):
        self.v5_options = v5_options or V5Options()
        self._v5_rate_cache = {}
        self._v5_finish_transition = None
        self.v5_failure = None
        path = Path(settings.runtime_root) / 'v5-finish-transition.json'
        if path.exists():
            saved = json.loads(path.read_text())
            self._v5_finish_transition = saved.get('transition')
        super().__init__(settings, **kwargs)
        self.event_logger.emit('v5_options', payload=asdict(self.v5_options))

    def _v5_deferral_block_reason(self, job):
        if job.hold:
            return 'user_hold'
        preparation = getattr(self, 'predictor_preparation', None)
        if preparation is not None and not preparation.ready(job, BatchResolution.resolved_batch_size(job)):
            return 'prediction_unready'
        try:
            size = self.planner.estimator.estimate_peak_vram_mb(job,
                BatchResolution.resolved_batch_size(job), self.settings.gpu_scheduler.packing_backend)
            if not math.isfinite(size) or size <= 0:
                return 'solo_memory_unknown'
            if size > self.planner.estimator.safe_budget_mb():
                return 'solo_memory_infeasible'
        except (ValueError, KeyError, TypeError):
            return 'solo_memory_unknown'
        if not self.planner.compatibility.compatible_group([job],
                backend_name=self.settings.gpu_scheduler.packing_backend):
            return 'solo_backend_incompatible'
        return None

    def _v5_deferral_legal_placement(self, job, plan):
        # Initial implementation reserves a solo anchor. Do not turn cached
        # timing evidence into permission to bypass failed-group/retry gates.
        return not self.supervisor.active_job_ids() and plan.job_ids == (job.job_id,)

    def _v5_bracket_reservation_allows(self, jobs):
        if not self.v5_options.deferral:
            return True
        selected = choose_protected(self._runnable_jobs(), **self._v5_deferral_options())
        return selected is None  # Protected solo service precedes exploration.

    def _v5_bracket_startup_envelope(self, jobs, profiles):
        peaks = []
        for job in jobs:
            raw = job.metadata.get('v5_startup_envelope') or {}
            profile = profiles[job.job_id]
            if (raw.get('source') != 'dedicated_same_host_peak_device_plus_headroom'
                    or raw.get('hardware_key') != self.store.hardware_key()
                    or raw.get('artifact_sha256') != profile.get('artifact_sha256')
                    or raw.get('job_signature') != (job.packing.signature or job.job_id)
                    or raw.get('batch_size') != BatchResolution.resolved_batch_size(job)):
                return None
            value = raw.get('peak_mib')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                return None
            peaks.append(value)
        # Per-process whole-device peaks include context and startup. Summing
        # deliberately overcounts shared overhead. This is a measured proxy,
        # never a guarantee for arbitrary unmeasured concurrent transients.
        return {'peak_mib': sum(peaks), 'source': 'sum_same_host_solo_device_peaks_with_headroom'}

    def _v5_snapshot_context(self, profiles):
        return {'evidence_identity': fingerprint({
            'profiles': profiles, 'hardware': self.store.hardware_key(),
            'backend': self.settings.gpu_scheduler.packing_backend}),
            'membership_generation': sorted((run.group_id, run.opened_at, list(run.job_ids))
                for run in self._active_runs.values()),
            'snapshot_at': time.time(), 'trial_max_age_seconds': self.v5_options.evidence_ttl_seconds}

    def _v5_forecast_trial(self, trial, jobs, windows):
        result = {'admission_policy': 'event_continuation_v5', 'decision': 'inconclusive', 'error': None}
        try:
            frozen = trial.cold_start['add_only']['feedback']
            cohort, requests, completed = forecast_cohort(self)
            if freeze_trial(self, trial) != frozen:
                raise ValueError('forecast cohort, request, runtime membership, or predictor changed')
            if set(self.supervisor.active_job_ids()) != {j.job_id for j in jobs}:
                raise ValueError('unmodeled active worker')
            lifecycle = lifecycle_snapshot(self, cohort)
            remaining = lifecycle['remaining']
            profiles = frozen['profiles']
            now = time.time()
            context = self._v5_snapshot_context(profiles)
            costs = {j.job_id: phase_costs(self, j, now)[0] for j in cohort}
            solo = {j: derived_solo_slope(profiles[j]['step_cost']) for j in remaining}
            memory = {j: profiles[j]['peak_vram_mib'] for j in remaining}
            queue = [j.job_id for j in sorted(cohort, key=lambda j: (timestamp(j.submitted_at), j.job_id))
                     if j.job_id not in trial.preexisting_job_ids and j.job_id not in lifecycle['passive']]
            pre = {j: [1. / v for v in h] for j, h in trial.cold_start['add_only']['pre_add'].items()}
            packed = {j: [1. / v for v in w['halves']] for j, w in windows.items()}
            # Cache only complete clean trial windows. Context binds recipe,
            # hardware, predictor and every active process generation; unrelated
            # changes invalidate reuse conservatively, never extend validity.
            for vector, source_windows, label in ((pre, trial.cold_start['add_only']['pre_add_windows'], 'pre'), (packed, windows, 'packed')):
                members = tuple(sorted(vector))
                intervals = [self._add_only_interval(source_windows[j]) for j in members]
                if any(interval is None for interval in intervals):
                    raise ValueError('measurement timestamp proof missing')
                observed_at = min(interval[1] for interval in intervals)
                if not 0 <= now - observed_at <= self.v5_options.evidence_ttl_seconds:
                    raise ValueError('measured rate evidence is stale')
                context[label + '_observed_at'] = observed_at
                self._v5_rate_cache[members] = {'context': context, 'halves': [
                    {'rates': {j: vector[j][h] for j in members},
                     'identity': context['evidence_identity'], 'membership_generation': context['membership_generation'],
                     'observed_at': observed_at, 'max_age_seconds': self.v5_options.evidence_ttl_seconds}
                    for h in range(2)]}
            cache = {k: {'halves': v['halves']} for k, v in self._v5_rate_cache.items()
                if v['context']['evidence_identity'] == context['evidence_identity']
                and v['context']['membership_generation'] == context['membership_generation']
                and 0 <= now - v['context']['snapshot_at'] <= self.v5_options.evidence_ttl_seconds}
            by_id = {j.job_id: j for j in cohort}
            compatible = lambda group: self.planner.compatibility.compatible_group(
                [by_id[j] for j in group], backend_name=self.settings.gpu_scheduler.packing_backend)
            protected = []
            if self.v5_options.deferral:
                chosen = choose_protected(self._runnable_jobs(), **self._v5_deferral_options())
                if chosen: protected = [chosen.job_id]
            result = compare_feedback_admission(remaining=remaining, solo=solo,
                incumbents=trial.preexisting_job_ids, candidate=trial.candidate_job_id,
                queue=queue, pre_halves=pre, packed_halves=packed, requests=requests,
                completed=completed, now=now, costs=costs, evidence=cache, context=context,
                phase_state=lifecycle['phase_state'], passive=lifecycle['passive'],
                memory=memory, memory_capacity=self.planner.estimator.safe_budget_mb(),
                concurrency_limit=self.settings.gpu_scheduler.parallel_job_cap,
                protected=protected, compatible=compatible,
                placement_allowed=lambda group: self._v5_placement_allowed(group, context=context),
                min_gain_seconds=self.settings.gpu_scheduler.feedback_latency.min_gain_seconds,
                bracket_admission_mode=self.settings.gpu_scheduler.feedback_latency.bracket_admission_mode,
                cpu_budget_ms=self.v5_options.forecast_cpu_ms, max_events=self.v5_options.forecast_max_events)
            result.update(remaining_updates=remaining, predicted_solo_updates_per_second=solo,
                queue_order=queue, frozen_profiles=profiles, context=context,
                phase_model='remaining resource-holding finish then resource-free delivery; phase interference unmodeled',
                future_policy='reference memory/compatibility FIFO; only known measured packed refills; online remeasurement can differ')
        except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
            result['error'] = str(exc)
        return result

    def _score_feedback_admission(self, trial, jobs, windows):
        if not (self.v5_options.continuation or self.v5_options.shadow):
            return super()._score_feedback_admission(trial, jobs, windows)
        result = self._v5_forecast_trial(trial, jobs, windows)
        self.event_logger.emit('v5_continuation_forecast', job_id=trial.candidate_job_id,
            payload={'trial_id': trial.trial_id, 'authoritative': self.v5_options.continuation, **result})
        if self.v5_options.continuation:
            try:
                cohort, _, _ = forecast_cohort(self)
                if self._v5_reference_begin(cohort, result, trial) is not True:
                    raise ValueError('cohort or profiles changed before reference installation')
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                result = {**result, 'decision': 'inconclusive', 'error': 'reference continuation unavailable: ' + str(exc)}
                self.event_logger.emit('v5_reference_installation_unavailable', job_id=trial.candidate_job_id, payload={'trial_id': trial.trial_id, 'error': str(exc), 'action': 'existing_safe_trial_cleanup'})
            return result
        return super()._score_feedback_admission(trial, jobs, windows)

    def _evaluate_cold_start_trial(self, trial):
        # Lifecycle correction belongs to C; A-only is a pure forecaster change.
        if self.v5_options.bracket:
            ids = {*trial.preexisting_job_ids, trial.candidate_job_id}
            members = [self.store.get_job(j) for j in ids]
            finished = [j for j in members if j is not None and update_state(j)[0] in {'FINISHING', 'DONE'}]
            if finished:
                now = time.time()
                deadline = timestamp(trial.started_at) + self.settings.gpu_scheduler.colocation.cold_start_trial_timeout_seconds
                active_members = [j for j in members if j and j.job_id in self.supervisor.active_job_ids()]
                unsafe = (any(j is None or j.status in {JobStatus.FAILED, JobStatus.CANCELLED} for j in members)
                    or now >= deadline or not self._admission_phase_supported(trial)
                    or not self._admission_commands_valid(trial, active_members)
                    or any(j.metadata.get('trial_lease_expired') == trial.trial_id for j in active_members)
                    or any((j.metadata.get('trial_step_command') or {}).get('owner') == trial.trial_id
                           and float(j.metadata['trial_step_command'].get('expires_at', 0)) <= now for j in active_members)
                    or bool(self._device_samples and self._device_samples[-1].memory_used_mb > self.planner.estimator.safe_budget_mb()))
                if unsafe:
                    return super()._evaluate_cold_start_trial(trial)
                for job in members:
                    if job is not None and (job.metadata.get('trial_step_command') or {}).get('owner') == trial.trial_id:
                        self.store.update_job(job.job_id, metadata_updates={'trial_step_command': {}})
                self.event_logger.emit('v5_training_complete_pending_feedback', job_id=trial.candidate_job_id,
                    payload={'trial_id': trial.trial_id, 'decision': 'completed_unverified',
                        'finished_update_ids': [j.job_id for j in finished],
                        'remaining_updates': {j.job_id: update_state(j)[1] for j in members if j},
                        'reason': 'measurement closed at training completion; no artificial rejection/pause'})
                self._v5_finish_transition = {'ids': [j.job_id for j in finished],
                    'started_at': now, 'trial_id': trial.trial_id, 'phase': 'finishing',
                    'members': sorted(ids), 'deadline': deadline,
                    'cleanup_deadline': deadline + self.settings.gpu_scheduler.colocation.cold_start_prepare_timeout_seconds}
                self._v5_save_finish_transition()
                self._colocation_trial = None
                self._v5_rate_cache.clear()
                self._persist_scheduler_decision_state()
                return
        return super()._evaluate_cold_start_trial(trial)

    def _dispatch_pending_work(self):
        transition = self._v5_finish_transition
        if transition:
            active = set(self.supervisor.active_job_ids())
            ids = transition.get('members', transition['ids'])
            pending = sorted(j for j in ids if j in active or not self.supervisor.has_reaped_process(j) or self.supervisor.has_live_process(j))
            jobs = [self.store.get_job(j) for j in pending]
            if any(j is None or j.status in {JobStatus.FAILED, JobStatus.CANCELLED} for j in jobs):
                self.v5_failure = {'reason': 'member_failed_during_finish_transition', 'pending_ids': pending}
                raise RuntimeError(str(self.v5_failure))
            live_finishers = [j.job_id for j in jobs if update_state(j)[1] == 0]
            survivors = [j for j in jobs if update_state(j)[1] != 0]
            now = time.time()
            if self._device_samples and self._device_samples[-1].memory_used_mb > self.planner.estimator.safe_budget_mb():
                self.v5_failure = {'reason': 'finish_transition_memory_guard', 'pending_ids': pending}
                raise RuntimeError(str(self.v5_failure))
            if transition.get('phase') == 'cleanup':
                if pending and now > transition['cleanup_deadline']:
                    self.v5_failure = {'reason': 'v5_finish_cleanup_timeout', 'pending_ids': pending}
                    self.event_logger.emit('v5_finish_cleanup_timeout', payload=self.v5_failure)
                    raise TimeoutError('finishing cleanup resource release exceeded finite bound')
                if pending: return
            elif live_finishers:
                if now > transition['deadline']:
                    self.v5_failure = {'reason': 'v5_finish_transition_timeout', 'pending_ids': live_finishers}
                    self.event_logger.emit('v5_finish_transition_timeout', payload=self.v5_failure)
                    raise TimeoutError('finishing resource release exceeded finite transition bound')
                return
            elif len(survivors) > 1:
                # No inherited triple-to-pair rate. Preserve productive work
                # only inside the ORIGINAL trial bound, then checkpoint the
                # unfinished survivors and drain to normal solo execution.
                if now < transition['deadline']: return
                transition['phase'] = 'cleanup'
                self._v5_save_finish_transition()
                for job in survivors:
                    self.store.update_job(job.job_id, metadata_updates={'cold_start_solo_required': True,
                        'v5_completion_cleanup_reason': 'unmeasured_surviving_group'})
                    if self.supervisor.request_pause(job.job_id, reason='bounded unmeasured surviving group', hold=False):
                        self.store.update_job(job.job_id, status=JobStatus.PAUSING, hold=False)
                self.event_logger.emit('v5_surviving_group_cleanup', payload=transition)
                return
            # A singleton can continue solo; all completed members must still
            # have observed process release before new admission is considered.
            self.event_logger.emit('v5_finish_resources_released', payload=transition)
            self._v5_finish_transition = None
            self._v5_save_finish_transition()
        return super()._dispatch_pending_work()

    def _v5_save_finish_transition(self):
        path = Path(self.settings.runtime_root) / 'v5-finish-transition.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'schema': 'v5_finish_transition_v1',
            'transition': self._v5_finish_transition}, sort_keys=True, allow_nan=False))
        tmp.replace(path)

    def _persist_scheduler_decision_state(self):
        super()._persist_scheduler_decision_state()
        self._v5_save_finish_transition()

    def _finish_cold_start_trial(self, trial, decision, reason, result=None):
        if reason == 'controller restarted' and self._v5_finish_transition:
            # Crash may occur after durable barrier write, before clearing the
            # old trial file. Do not create a new pause/rejection from that gap.
            self._colocation_trial = None
            return
        return super()._finish_cold_start_trial(trial, decision, reason, result)

    def _cold_phase(self, trial, phase, jobs, *, measuring=None):
        if phase == 'overlap_warmup' and (self.v5_options.continuation or self.v5_options.shadow):
            try:
                trial.cold_start['add_only']['feedback'] = freeze_trial(self, trial)
            except (ValueError, KeyError, TypeError) as exc:
                self._finish_cold_start_trial(trial, 'inconclusive', 'v5 receipt-aware freeze: ' + str(exc))
                return
        return super()._cold_phase(trial, phase, jobs, measuring=measuring)
