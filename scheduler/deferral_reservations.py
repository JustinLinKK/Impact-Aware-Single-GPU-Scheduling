"""Opt-in persistent next-service reservations, not preemption or a wait SLA.

Put DeferralReservationMixin before SchedulerService in V5Service's MRO.
Required hook `_v5_deferral_block_reason(job)` returns None only for globally
ready, solo-feasible work; current-pack incompatibility is NOT a global blocker.
Optional `_v5_deferral_legal_placement(job, plan)` may authorize an already legal,
measured placement. Default is a solo anchor after all incumbents drain. Neither
hook may relax existing compatibility, memory, failed-group or retry guards;
normal super()._dispatch_plan remains the final execution gate.

Launch identity hook `_v5_deferral_launch_identity(job_id)` may supply an actual
run ID or PID+birth token. Otherwise actual ActiveRun group_id/opened_at is used.
No job ID/time-only synthetic launch token is generated. Observe runnable/hold
transitions through `_v5_deferral_sync` so held/unready time is not accrued.
Debt is NEVER reset by updates or trial launches. Genuine protected normal
service clears pending, retaining lifetime debt, bypass IDs and reasons.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import math
import time

KEY = 'v5_deferral_reservation'
SCHEMA = 'deferral_reservation_v1'


def submission_seconds(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def new_record(submitted_at, now):
    submission_seconds(submitted_at)
    return dict(schema=SCHEMA, original_submitted_at=submitted_at,
        deferred_seconds=0., bypass_launch_ids=[], reasons=[], failed_group_ids=[],
        pending=False, last_observed_unix=float(now), last_runnable=False,
        blocked_reason=None, defer_count=0, served_launch_ids=[])


def advance_record(record, *, now, runnable, blocked_reason=None):
    """Charge only intervals bracketed by runnable deferred observations.

    A restart with an unchanged runnable record retains/charges its elapsed debt.
    A newly observed hold/unready transition conservatively charges neither edge.
    Controllers must observe actual eligibility transitions, not infer hold time.
    """
    value = deepcopy(record)
    if (value.get('schema') != SCHEMA or not isinstance(runnable, bool)
            or not math.isfinite(now) or now < value['last_observed_unix']
            or not math.isfinite(value['deferred_seconds']) or value['deferred_seconds'] < 0):
        raise ValueError('invalid deferral record or regressed observation clock')
    if value['pending'] and runnable and value['last_runnable']:
        value['deferred_seconds'] += now - value['last_observed_unix']
    value.update(last_observed_unix=float(now), last_runnable=runnable,
                 blocked_reason=blocked_reason)
    return value


def mark_deferred(record, *, now, runnable, reason, failed_group_id=None):
    value = advance_record(record, now=now, runnable=runnable)
    value['pending'] = True
    value['defer_count'] += 1
    if reason not in value['reasons']:
        value['reasons'].append(reason)
    if failed_group_id and failed_group_id not in value['failed_group_ids']:
        value['failed_group_ids'].append(failed_group_id)
    return value


def record_bypass(record, *, launched_submitted_at, launch_id):
    value = deepcopy(record)
    if not isinstance(launch_id, str) or not launch_id:
        raise ValueError('actual launch identity required')
    if (value['pending'] and value['last_runnable']
            and submission_seconds(launched_submitted_at) > submission_seconds(value['original_submitted_at'])
            and launch_id not in value['bypass_launch_ids']):
        value['bypass_launch_ids'].append(launch_id)
    return value


def protected(record, *, seconds=30., bypasses=2):
    return bool(record['pending'] and record['last_runnable'] and not record['blocked_reason']
        and (record.get('outer_age_protected') is True or record['deferred_seconds'] >= seconds or len(record['bypass_launch_ids']) >= bypasses))


def choose_protected(jobs, *, seconds=30., bypasses=2):
    eligible = [job for job in jobs if job.metadata.get(KEY)
        and protected(job.metadata[KEY], seconds=seconds, bypasses=bypasses)]
    return min(eligible, key=lambda job: (submission_seconds(job.metadata[KEY]['original_submitted_at']), job.job_id), default=None)


class DeferralReservationMixin:
    def _v5_deferral_enabled(self):
        return getattr(getattr(self, 'v5_options', None), 'deferral', False) is True

    def _v5_deferral_now(self):
        return time.time()

    def _v5_deferral_event(self, name, job_id=None, **payload):
        logger = getattr(self, 'event_logger', None)
        if logger is not None:
            logger.emit(name, job_id=job_id, payload=payload)

    def _v5_deferral_options(self):
        options = self.v5_options
        seconds = getattr(options, 'deferral_seconds', 30.)
        bypasses = getattr(options, 'max_younger_bypasses', 2)
        if (isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0
                or type(bypasses) is not int or bypasses <= 0):
            raise ValueError('finite positive deferral thresholds required')
        return dict(seconds=seconds, bypasses=bypasses)

    def _v5_deferral_reason(self, job, runnable_ids):
        if job.hold:
            return 'user_hold'
        if job.status.is_terminal:
            return 'terminal'
        if job.job_id not in runnable_ids:
            return 'not_runnable_or_preparation_unready'
        hook = getattr(self, '_v5_deferral_block_reason', None)
        if hook is None:
            return 'eligibility_hook_missing'
        reason = hook(job)
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError('eligibility hook must return None or explicit reason')
        return reason

    def _v5_outer_age_eligible(self, job, now, reason):
        settings = getattr(self, 'settings', self.store.settings).gpu_scheduler
        feedback = settings.feedback_latency
        if (reason is not None or not feedback.enabled or not job.metadata.get('cooperative_trial')
                or now - submission_seconds(job.submitted_at) < feedback.max_queued_age_seconds):
            return False
        planner = getattr(self, 'planner', None)
        if planner is None:
            return False  # Never manufacture compatibility/memory permission.
        backend = settings.packing_backend
        from ..domain import BatchResolution
        return (planner.compatibility.pack_eligible(job, backend_name=backend)
            and planner.estimator.estimate_peak_vram_mb(job, BatchResolution.resolved_batch_size(job), backend)
                <= planner.estimator.safe_budget_mb())

    def _v5_deferral_sync(self, runnable_jobs):
        """Persist debt without overwriting other metadata written by workers."""
        if not self._v5_deferral_enabled():
            return list(runnable_jobs)
        now = self._v5_deferral_now()
        ids = {job.job_id for job in runnable_jobs}
        for job in self.store.list_jobs():
            record = job.metadata.get(KEY)
            solo_required = job.metadata.get('cold_start_solo_required') is True
            reason = self._v5_deferral_reason(job, ids)
            outer_age = self._v5_outer_age_eligible(job, now, reason)
            if record is None and not solo_required and not outer_age:
                continue
            record = record or new_record(job.submitted_at, now)
            if outer_age:
                if not record['pending']:
                    record = mark_deferred(record, now=now, runnable=True, reason='outer_age_guard')
                if 'outer_age_guard' not in record['reasons']:
                    record['reasons'].append('outer_age_guard')
                record['outer_age_protected'] = True
            if solo_required and 'solo_required' not in record['reasons']:
                record['reasons'].append('solo_required')
            reason = self._v5_deferral_reason(job, ids)
            if solo_required and not record['pending'] and not record['served_launch_ids']:
                record = mark_deferred(record, now=now, runnable=reason is None, reason='solo_required')
            record = advance_record(record, now=now, runnable=reason is None, blocked_reason=reason)
            self.store.update_job(job.job_id, metadata_updates={KEY: record})
        result = [self.store.get_job(job.job_id) for job in runnable_jobs]
        selected = choose_protected(result, **self._v5_deferral_options())
        return sorted(result, key=lambda job: (job.job_id != selected.job_id if selected else False,
            submission_seconds(job.metadata.get(KEY, {}).get('original_submitted_at', job.submitted_at)), job.job_id))

    def _runnable_jobs(self):
        jobs = super()._runnable_jobs()
        if not self._v5_deferral_enabled():
            return jobs
        jobs = self._v5_deferral_sync(jobs)
        # Sorting alone is insufficient: PlacementPlanner reorders candidates.
        # Restrict only its admission view, never store.list_jobs()/the complete
        # forecast cohort or current incumbents. A protected legal slot must not
        # become an endlessly vetoed younger/ordinary selection on an idle GPU.
        jobs = [job for job in jobs if not (job.metadata.get(KEY, {}).get('pending')
            and job.metadata[KEY].get('blocked_reason'))]
        reserved = choose_protected(jobs, **self._v5_deferral_options())
        return [reserved] if reserved is not None else jobs

    def _v5_deferral_can_dispatch(self, plan, jobs):
        reservation = choose_protected(jobs, **self._v5_deferral_options())
        if reservation is None:
            return True
        active = set(self.supervisor.active_job_ids())
        measured = getattr(self, '_v5_deferral_legal_placement', None)
        no_trial = not (plan.trial_metadata.get('requires_live_trial') or plan.trial_metadata.get('bracket_bootstrap'))
        solo = not active and plan.job_ids == (reservation.job_id,)
        allowed = (reservation.job_id in plan.job_ids and no_trial
            and (solo or (not reservation.metadata.get('cold_start_solo_required')
                          and measured is not None and measured(reservation, plan))))
        if not allowed:
            self._v5_deferral_event('v5_deferral_slot_reserved', reservation.job_id,
                reason='protected_slot_reserved', proposed_job_ids=list(plan.job_ids),
                active_job_ids=sorted(active), action='drain_incumbents_then_legal_service')
        return bool(allowed)

    def _v5_deferral_actual_launch_id(self, job_id):
        hook = getattr(self, '_v5_deferral_launch_identity', None)
        if hook is not None:
            token = hook(job_id)
            if token is not None:
                return token
        for run in reversed(tuple(getattr(self, '_active_runs', {}).values())):
            if job_id in run.job_ids and run.group_id and run.opened_at:
                return f'run:{run.group_id}:{run.opened_at}:{job_id}'
        return None

    def _dispatch_plan(self, plan):
        if not self._v5_deferral_enabled():
            return super()._dispatch_plan(plan)
        jobs = self._v5_deferral_sync(super()._runnable_jobs())
        if not self._v5_deferral_can_dispatch(plan, jobs):
            return False
        # Save currently waiting, actually submitted jobs; never imagined children.
        waiting = [job.job_id for job in jobs if job.metadata.get(KEY, {}).get('pending')]
        was_protected = choose_protected(jobs, **self._v5_deferral_options())
        accepted = super()._dispatch_plan(plan)
        if not accepted:
            return accepted
        now = self._v5_deferral_now()
        for launched_id in plan.job_ids:
            launched = self.store.get_job(launched_id)
            if launched is not None and launched_id in waiting:
                # Stop waiting accrual immediately, even for a productive trial.
                # Debt/counters persist; a restart before the next tick must not
                # mischarge RUNNING time as runnable-but-deferred time.
                stopped = advance_record(launched.metadata[KEY], now=now,
                    runnable=False, blocked_reason='worker_launch_in_progress')
                launched = self.store.update_job(launched_id, metadata_updates={KEY: stopped})
            token = self._v5_deferral_actual_launch_id(launched_id)
            if launched is None or not token:
                self._v5_deferral_event('v5_deferral_launch_identity_missing', launched_id)
                continue
            for waiting_id in waiting:
                if waiting_id == launched_id:
                    continue
                job = self.store.get_job(waiting_id)
                value = record_bypass(job.metadata[KEY], launched_submitted_at=launched.submitted_at, launch_id=token)
                self.store.update_job(waiting_id, metadata_updates={KEY: value})
            if launched_id in waiting and not (plan.trial_metadata.get('requires_live_trial') or plan.trial_metadata.get('bracket_bootstrap')):
                job = self.store.get_job(launched_id)
                value = deepcopy(job.metadata[KEY]); value.update(pending=False, last_runnable=False, last_observed_unix=now)
                if token not in value['served_launch_ids']:
                    value['served_launch_ids'].append(token)
                self.store.update_job(launched_id, metadata_updates={KEY: value})
                self._v5_deferral_event('v5_deferral_service_started', launched_id, launch_id=token,
                    deferred_seconds=value['deferred_seconds'], bypasses=len(value['bypass_launch_ids']),
                    protected_service=bool(was_protected and launched_id == was_protected.job_id))
        return accepted

    def _finish_cold_start_trial(self, trial, decision, reason, result=None):
        value = super()._finish_cold_start_trial(trial, decision, reason, result)
        if self._v5_deferral_enabled() and decision != 'accepted':
            job = self.store.get_job(trial.candidate_job_id)
            if job is not None and not job.status.is_terminal:
                now = self._v5_deferral_now(); record = job.metadata.get(KEY) or new_record(job.submitted_at, now)
                runnable_ids = {j.job_id for j in super()._runnable_jobs()}
                block = self._v5_deferral_reason(job, runnable_ids)
                record = mark_deferred(record, now=now, runnable=block is None,
                    reason=decision + ':' + reason, failed_group_id=trial.profile_key)
                record['blocked_reason'] = block
                self.store.update_job(job.job_id, metadata_updates={KEY: record})
                self._v5_deferral_event('v5_deferral_recorded', job.job_id, decision=decision,
                    reason=reason, failed_group_id=trial.profile_key)
        return value
