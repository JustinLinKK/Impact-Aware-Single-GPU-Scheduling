"""Cache warming heuristics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from ..domain import PreloadSource, TrainingJob


def _default_target(job: TrainingJob) -> PreloadSource:
    return PreloadSource(
        model_id=job.baseline_model_id,
        model_path=job.baseline_model_path,
        loader_target=job.config.loader_target,
    )


def _default_size_estimate(target: PreloadSource) -> int | None:
    try:
        return max(0, int(Path(target.model_path).stat().st_size))
    except Exception:
        return None


def select_models_to_warm(
    jobs: list[TrainingJob],
    *,
    top_k: int | None = 2,
    selection_policy: str = "top_k",
    available_budget_bytes: int | None = None,
    cached_model_ids: set[str] | None = None,
    resolve_target: Callable[[TrainingJob], PreloadSource] | None = None,
    estimate_target_bytes: Callable[[PreloadSource], int | None] | None = None,
) -> list[PreloadSource]:
    """Choose baseline models worth preloading from the pending queue."""
    if not jobs:
        return []
    selection_policy = str(selection_policy or "top_k").strip().lower()
    target_resolver = resolve_target or _default_target
    size_estimator = estimate_target_bytes or _default_size_estimate
    resolved_targets = {job.job_id: target_resolver(job) for job in jobs}
    counts = Counter(target.model_id for target in resolved_targets.values())
    ranked = sorted(
        jobs,
        key=lambda job: (-job.priority, -counts[resolved_targets[job.job_id].model_id], job.queue_sequence),
    )
    cached_ids = set(cached_model_ids or set())
    remaining_budget = None if available_budget_bytes is None else max(0, int(available_budget_bytes))
    seen: set[str] = set()
    selected: list[PreloadSource] = []
    for job in ranked:
        target = resolved_targets[job.job_id]
        if target.model_id in seen:
            continue
        if selection_policy == "budget_only":
            if target.model_id not in cached_ids and remaining_budget is not None:
                estimated_size = size_estimator(target)
                estimated_size = max(0, int(estimated_size or 0))
                if estimated_size > remaining_budget:
                    continue
                remaining_budget -= estimated_size
            selected.append(target)
            seen.add(target.model_id)
            continue
        selected.append(target)
        seen.add(target.model_id)
        if top_k is not None and len(selected) >= top_k:
            break
    return selected
