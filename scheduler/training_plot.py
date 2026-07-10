"""Render training metric timelines for scheduler-managed jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

from ..domain import JobMetricSample, parse_timestamp
from .early_stop import EarlyStopDecision, is_learning_rate_key


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _relative_times(samples: list[JobMetricSample]) -> list[float]:
    parsed = [parse_timestamp(sample.created_at) for sample in samples]
    first = next((item for item in parsed if item is not None), None)
    if first is None:
        return [float(index) for index, _sample in enumerate(samples)]
    times: list[float] = []
    for index, item in enumerate(parsed):
        if item is None:
            times.append(float(index))
        else:
            times.append(max(0.0, (item - first).total_seconds()))
    return times


def _metric_group(key: str) -> str:
    normalized = str(key or "").lower().replace("-", "_")
    if is_learning_rate_key(normalized):
        return "learning_rate"
    if any(token in normalized for token in ("acc", "accuracy", "auc", "f1", "score", "precision", "recall", "iou", "map")):
        return "accuracy"
    if any(token in normalized for token in ("loss", "error", "rmse", "mae", "mse", "logloss")):
        return "loss"
    return "other"


def _series(samples: list[JobMetricSample], times: list[float]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {
        "learning_rate": {},
        "accuracy": {},
        "loss": {},
        "other": {},
    }
    for sample, time_value in zip(samples, times, strict=False):
        for key, raw_value in sample.metrics.items():
            value = _safe_float(raw_value)
            if value is None:
                continue
            group = _metric_group(key)
            bucket = grouped.setdefault(group, {}).setdefault(str(key), [[], []])
            bucket[0].append(time_value)
            bucket[1].append(value)
    return grouped


def render_training_process(
    samples: list[JobMetricSample],
    output_dir: str | Path,
    *,
    decision: EarlyStopDecision | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "training_metrics_summary.json"
    plot_path = output_path / "training_process.png"

    times = _relative_times(samples)
    grouped = _series(samples, times)
    latest_metrics = dict(samples[-1].metrics) if samples else {}
    summary = {
        "sample_count": len(samples),
        "time_axis": "relative_seconds",
        "metric_groups": {group: sorted(metrics.keys()) for group, metrics in grouped.items() if metrics},
        "latest_metrics": latest_metrics,
        "early_stop_decision": decision.to_dict() if decision is not None else None,
        "plot_path": str(plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if not samples or not any(grouped.values()):
        return {"plot_path": None, "summary_path": str(summary_path), "summary": summary}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visible_groups = [group for group in ("learning_rate", "accuracy", "loss", "other") if grouped.get(group)]
    fig, axes = plt.subplots(len(visible_groups), 1, figsize=(9, max(3, 2.6 * len(visible_groups))), sharex=True)
    if len(visible_groups) == 1:
        axes = [axes]
    titles = {
        "learning_rate": "Learning rate vs. time",
        "accuracy": "Accuracy/score vs. time",
        "loss": "Loss/error vs. time",
        "other": "Other metrics vs. time",
    }
    for axis, group in zip(axes, visible_groups, strict=True):
        for key, (x_values, y_values) in grouped[group].items():
            axis.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=3, label=key)
        axis.set_title(titles[group])
        axis.set_ylabel(group.replace("_", " "))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    if decision is not None and decision.should_stop:
        fig.suptitle(f"Early stopped: {decision.reason}", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return {"plot_path": str(plot_path), "summary_path": str(summary_path), "summary": summary}
