"""Cache warming heuristics."""

from __future__ import annotations

from collections import Counter

from ..schemas import TrainingJob


def select_models_to_warm(jobs: list[TrainingJob], *, top_k: int = 2) -> list[tuple[str, str, str | None]]:
    """Choose baseline models worth preloading from the pending queue."""
    if not jobs:
        return []
    counts = Counter(job.baseline_model_id for job in jobs)
    ranked = sorted(
        jobs,
        key=lambda job: (-job.priority, -counts[job.baseline_model_id], job.queue_sequence),
    )
    seen: set[str] = set()
    selected: list[tuple[str, str, str | None]] = []
    for job in ranked:
        if job.baseline_model_id in seen:
            continue
        selected.append((job.baseline_model_id, job.baseline_model_path, job.config.loader_target))
        seen.add(job.baseline_model_id)
        if len(selected) >= top_k:
            break
    return selected
