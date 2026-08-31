"""Optional admission guard that protects long-running jobs from contention.

This is the online-implementable half of the critical-path scheduler evaluated
offline in scheduler_benchmark_test/compare_dag_schedulers.py. That policy had
two parts:

    ordering    rank by subtree cost, the total work in a node's descendants
    admission   veto co-runners while a critical-path job is running

Only the admission half can run live. Subtree cost needs the DAG of a node's
children, and in MLEvolve a child does not exist until its parent's execution
has produced a result, so at dispatch time the future shape of the tree is
unknown. Ordering therefore stays as the planner already has it.

The guard itself needs no future knowledge. A job already running for a long
time has a long remaining time, and every co-runner admitted alongside it
stretches that job by the pair slowdown, which pushes out the completion that
in turn gates the next generation. The guard caps how many co-runners may
share the device while such a job is active.

Disabled unless LOCALML_CONTENTION_GUARD is set, so the default production path
is unchanged.

Environment:
    LOCALML_CONTENTION_GUARD          "1" to enable
    LOCALML_CONTENTION_PROTECT_RATIO  float, a running job counts as protected
                                      when its predicted remaining runtime is
                                      at least this share of the longest
                                      running job's. Default 0.35.
    LOCALML_CONTENTION_MAX_CORUNNERS  int, co-runners tolerated alongside a
                                      protected job. Default 1.
"""

from __future__ import annotations

import os

from ..domain import TrainingJob

ENV_ENABLED = "LOCALML_CONTENTION_GUARD"
ENV_PROTECT_RATIO = "LOCALML_CONTENTION_PROTECT_RATIO"
ENV_MAX_CORUNNERS = "LOCALML_CONTENTION_MAX_CORUNNERS"

DEFAULT_PROTECT_RATIO = 0.35
DEFAULT_MAX_CORUNNERS = 1


def guard_enabled() -> bool:
    """Return whether the contention guard should be consulted at all."""
    return str(os.environ.get(ENV_ENABLED, "")).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def admits(
    active_jobs: list[TrainingJob],
    remaining_seconds: dict[str, float],
) -> bool:
    """Decide whether one more job may join the jobs already running.

    Args:
        active_jobs       : list[TrainingJob], jobs currently holding the GPU
        remaining_seconds : dict[str, float], job_id -> predicted remaining
                            runtime in seconds, taken from the profile-based
                            estimator. Jobs missing from this mapping are
                            treated as having no known remaining time and
                            cannot trigger protection.

    Returns:
        bool. True when the candidate may be admitted. An empty device always
        admits, so the guard can never deadlock the queue.
    """
    if not active_jobs:
        return True

    known = [
        float(remaining_seconds[job.job_id])
        for job in active_jobs
        if remaining_seconds.get(job.job_id) is not None
    ]
    if not known:
        return True

    longest = max(known)
    if longest <= 0:
        return True

    ratio = _float_env(ENV_PROTECT_RATIO, DEFAULT_PROTECT_RATIO)
    threshold = longest * ratio
    protected = [value for value in known if value >= threshold]
    if not protected:
        return True

    # One co-runner alongside a protected job is tolerated: the pair slowdown
    # measured on this hardware was about 1.20, so a second job still nets more
    # aggregate throughput than it costs. Beyond that the protected job's
    # stretch outweighs the extra parallelism.
    max_corunners = _int_env(ENV_MAX_CORUNNERS, DEFAULT_MAX_CORUNNERS)
    return len(active_jobs) <= max_corunners
