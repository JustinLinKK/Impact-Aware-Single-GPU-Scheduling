"""Receipt-aware opt-in cohort; worker completion is not usable feedback."""
from copy import deepcopy
import time

from ..domain import JobStatus
from .feedback_objective import requests_from_jobs, timestamp
from .feedback_trials import phase_costs


def forecast_cohort(service):
    all_jobs = service.store.list_jobs()
    copies = []
    for job in all_jobs:
        copy = deepcopy(job)
        if job.status == JobStatus.COMPLETED and not job.metadata.get('feedback_ready_at'):
            copy.status = JobStatus.RUNNING  # retain its unresolved request
        if job.status.is_terminal and job.status != JobStatus.COMPLETED:
            raw = job.metadata.get('feedback_request') or {}
            if raw.get('completion_condition', 'successful') == 'successful':
                raise ValueError('required successful result failed or was cancelled: ' + job.job_id)
        if job.metadata.get('feedback_ready_at'):
            if job.status != JobStatus.COMPLETED:
                raise ValueError('receipt published before successful terminal worker state')
            if timestamp(job.metadata['feedback_ready_at']) < timestamp(job.submitted_at):
                raise ValueError('receipt precedes submission')
        copies.append(copy)
    requests, completed = requests_from_jobs(copies)
    referenced = {j for request in requests for j in request.required_job_ids}
    outstanding = [j for j in all_jobs if j.job_id in referenced and j.job_id not in completed]
    if any(j.hold for j in outstanding):
        raise ValueError('held job has no forecast release time')
    return outstanding, requests, completed


def snapshot(service, jobs, now=None):
    # Import lazily to avoid the service/bootstrap adapter cycle.
    from .experimental_v5 import update_state
    now = time.time() if now is None else now
    active = set(service.supervisor.active_job_ids())
    remaining, states, costs = {}, {}, {}
    passive = []
    for job in jobs:
        state, left = update_state(job)
        if state == 'UNKNOWN':
            raise ValueError('unknown exact update budget: ' + job.job_id)
        remaining[job.job_id] = left
        costs[job.job_id] = phase_costs(service, job, now)[0]
        if job.job_id in active:
            phase = 'finishing' if left == 0 else ('training' if job.metadata.get('trial_runner_ready') else 'starting')
            states[job.job_id] = {'phase': phase}
        elif job.status == JobStatus.COMPLETED:
            # observed_exit_code is the real supervisor process-poll receipt,
            # unlike the training result's earlier finished_at timestamp.
            code = service.supervisor.observed_exit_code(job.job_id)
            if left != 0 or code != 0:
                raise ValueError('terminal job lacks exact successful update/exit proof: ' + job.job_id)
            states[job.job_id] = {'phase': 'delivering'}
            passive.append(job.job_id)
        elif job.status in {JobStatus.PAUSING, JobStatus.RUNNING}:
            raise ValueError('running lifecycle lacks observed active worker: ' + job.job_id)
        else:
            states[job.job_id] = {'phase': 'paused'}
    return {'remaining': remaining, 'phase_state': states, 'costs': costs, 'passive': tuple(passive)}


def freeze_trial(service, trial):
    from .feedback_trials import profile_snapshot
    jobs, requests, _ = forecast_cohort(service)
    return {'profiles': profile_snapshot(service, jobs),
        'settings': service.settings.gpu_scheduler.feedback_latency.to_dict(),
        'requests': [r.to_dict() for r in requests],
        'active_job_ids': sorted(service.supervisor.active_job_ids()),
        'job_ids': sorted(j.job_id for j in jobs)}
