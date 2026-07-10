"""Metric-plateau early-stop analysis for scheduler-managed jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..domain import JobMetricSample


MAXIMIZE_KEYWORDS = ("accuracy", "acc", "auc", "f1", "score", "precision", "recall", "iou", "map")
MINIMIZE_KEYWORDS = ("loss", "error", "rmse", "mae", "mse", "logloss", "cross_entropy")
LR_KEYS = {"lr", "learning_rate", "learn_rate"}


@dataclass(slots=True)
class EarlyStopDecision:
    should_stop: bool
    reason: str = ""
    metric_key: str | None = None
    direction: str | None = None
    best_value: float | None = None
    best_global_step: int | None = None
    best_epoch: int | None = None
    latest_value: float | None = None
    latest_global_step: int | None = None
    latest_epoch: int | None = None
    sample_count: int = 0
    objective_sample_count: int = 0
    samples_since_best: int = 0
    min_delta: float = 0.0
    patience_samples: int = 0
    warmup_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "metric_key": self.metric_key,
            "direction": self.direction,
            "best_value": self.best_value,
            "best_global_step": self.best_global_step,
            "best_epoch": self.best_epoch,
            "latest_value": self.latest_value,
            "latest_global_step": self.latest_global_step,
            "latest_epoch": self.latest_epoch,
            "sample_count": self.sample_count,
            "objective_sample_count": self.objective_sample_count,
            "samples_since_best": self.samples_since_best,
            "min_delta": self.min_delta,
            "patience_samples": self.patience_samples,
            "warmup_samples": self.warmup_samples,
            "metadata": dict(self.metadata),
        }


def _normalize_key(key: str) -> str:
    return str(key or "").strip().lower().replace("-", "_").replace("/", "_")


def is_learning_rate_key(key: str) -> bool:
    normalized = _normalize_key(key)
    return normalized in LR_KEYS or normalized.endswith("_lr") or normalized.endswith("_learning_rate")


def infer_metric_direction(metric_key: str, configured_direction: str | None = None) -> str | None:
    configured = str(configured_direction or "auto").strip().lower()
    if configured in {"maximize", "minimize"}:
        return configured
    key = _normalize_key(metric_key)
    if is_learning_rate_key(key):
        return None
    if any(token in key for token in MAXIMIZE_KEYWORDS):
        return "maximize"
    if any(token in key for token in MINIMIZE_KEYWORDS):
        return "minimize"
    return None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _metric_keys(samples: list[JobMetricSample]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sample in samples:
        for key, value in sample.metrics.items():
            normalized = _normalize_key(key)
            if normalized in seen or _finite_float(value) is None:
                continue
            seen.add(normalized)
            ordered.append(str(key))
    return ordered


def select_objective_metric(
    samples: list[JobMetricSample],
    *,
    metric_key: str | None = None,
    direction: str | None = "auto",
) -> tuple[str | None, str | None]:
    keys = _metric_keys(samples)
    if metric_key:
        requested = _normalize_key(metric_key)
        for key in keys:
            if _normalize_key(key) == requested:
                inferred = infer_metric_direction(key, direction)
                return (key, inferred)
        return (None, None)

    def candidates(predicate) -> list[str]:
        return [key for key in keys if predicate(_normalize_key(key))]

    priority = (
        candidates(lambda key: key.startswith("val_") and infer_metric_direction(key, direction) == "maximize"),
        candidates(lambda key: key.startswith("val_") and infer_metric_direction(key, direction) == "minimize"),
        candidates(lambda key: infer_metric_direction(key, direction) == "maximize"),
        candidates(lambda key: infer_metric_direction(key, direction) == "minimize"),
    )
    for group in priority:
        for key in group:
            if not is_learning_rate_key(key):
                return (key, infer_metric_direction(key, direction))
    return (None, None)


def analyze_metric_plateau(
    samples: list[JobMetricSample],
    *,
    metric_key: str | None = None,
    direction: str = "auto",
    warmup_samples: int = 5,
    patience_samples: int = 5,
    min_delta: float = 1e-4,
) -> EarlyStopDecision:
    warmup = max(0, int(warmup_samples or 0))
    patience = max(1, int(patience_samples or 1))
    min_delta = max(0.0, float(min_delta or 0.0))
    selected_key, selected_direction = select_objective_metric(samples, metric_key=metric_key, direction=direction)
    if selected_key is None or selected_direction is None:
        return EarlyStopDecision(
            should_stop=False,
            reason="no objective metric available",
            sample_count=len(samples),
            warmup_samples=warmup,
            patience_samples=patience,
            min_delta=min_delta,
            metadata={"available_metric_keys": _metric_keys(samples)},
        )

    objective: list[tuple[JobMetricSample, float]] = []
    selected_normalized = _normalize_key(selected_key)
    for sample in samples:
        matched_value = None
        for key, value in sample.metrics.items():
            if _normalize_key(key) == selected_normalized:
                matched_value = _finite_float(value)
                break
        if matched_value is not None:
            objective.append((sample, matched_value))

    if len(objective) < warmup + patience:
        latest_sample, latest_value = objective[-1] if objective else (None, None)
        return EarlyStopDecision(
            should_stop=False,
            reason="not enough objective samples",
            metric_key=selected_key,
            direction=selected_direction,
            latest_value=latest_value,
            latest_global_step=latest_sample.global_step if latest_sample else None,
            latest_epoch=latest_sample.epoch if latest_sample else None,
            sample_count=len(samples),
            objective_sample_count=len(objective),
            warmup_samples=warmup,
            patience_samples=patience,
            min_delta=min_delta,
        )

    best_index = 0
    best_sample, best_value = objective[0]
    for index, (sample, value) in enumerate(objective[1:], start=1):
        if selected_direction == "maximize":
            improved = value > best_value + min_delta
        else:
            improved = value < best_value - min_delta
        if improved:
            best_index = index
            best_sample = sample
            best_value = value

    latest_sample, latest_value = objective[-1]
    samples_since_best = len(objective) - best_index - 1
    should_stop = samples_since_best >= patience
    reason = (
        f"{selected_key} plateaued for {samples_since_best} samples"
        if should_stop
        else f"{selected_key} has not plateaued"
    )
    return EarlyStopDecision(
        should_stop=should_stop,
        reason=reason,
        metric_key=selected_key,
        direction=selected_direction,
        best_value=best_value,
        best_global_step=best_sample.global_step,
        best_epoch=best_sample.epoch,
        latest_value=latest_value,
        latest_global_step=latest_sample.global_step,
        latest_epoch=latest_sample.epoch,
        sample_count=len(samples),
        objective_sample_count=len(objective),
        samples_since_best=samples_since_best,
        warmup_samples=warmup,
        patience_samples=patience,
        min_delta=min_delta,
        metadata={"available_metric_keys": _metric_keys(samples)},
    )
