"""Optional scheduler context for generated scripts; inactive for standalone runs."""

from __future__ import annotations

import os

from .trial_control import configure_trial_memory


def script_scheduler_context():
    runtime_root = os.environ.get("MLEVOLVE_SCHEDULER_RUNTIME_ROOT")
    job_id = os.environ.get("MLEVOLVE_SCHEDULER_JOB_ID")
    if not runtime_root or not job_id:
        return None
    from .worker_runtime import create_runner_context, load_runtime_settings
    from ..observability.events import EventLogger
    from ..storage.state_store import StateStore

    settings = load_runtime_settings(runtime_root)
    store = StateStore(settings)
    events = EventLogger(store, settings.events_jsonl_path)
    context, job = create_runner_context(settings, store, events, job_id)
    if context is None:
        raise KeyError(f"Unknown scheduled script: {job_id}")
    configure_trial_memory(context)
    return context
