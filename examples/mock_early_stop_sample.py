"""Mock metric samples for checking scheduler early-stop decisions.

Run with:
    python -m localml_scheduler.examples.mock_early_stop_sample
"""

from __future__ import annotations

import json
from typing import Any

from ..domain import JobMetricSample
from ..scheduler.early_stop import analyze_metric_plateau


def _sample(job_id: str, step: int, loss: float) -> JobMetricSample:
    return JobMetricSample(
        job_id=job_id,
        created_at=f"2026-01-01T00:00:{step:02d}+00:00",
        epoch=max(0, step // 2),
        global_step=step,
        metrics={"val_loss": loss, "lr": 0.001},
    )


def build_plateau_samples() -> list[JobMetricSample]:
    losses = [1.0, 0.82, 0.70, 0.7002, 0.7001, 0.7003]
    return [_sample("mock-plateau-job", step, loss) for step, loss in enumerate(losses, start=1)]


def build_improving_samples() -> list[JobMetricSample]:
    losses = [1.0, 0.82, 0.70, 0.62, 0.55, 0.49]
    return [_sample("mock-improving-job", step, loss) for step, loss in enumerate(losses, start=1)]


def run_mock_early_stop_check() -> dict[str, dict[str, Any]]:
    settings = {
        "metric_key": "val_loss",
        "direction": "minimize",
        "warmup_samples": 2,
        "patience_samples": 3,
        "min_delta": 1e-3,
    }
    plateau = analyze_metric_plateau(build_plateau_samples(), **settings)
    improving = analyze_metric_plateau(build_improving_samples(), **settings)
    return {
        "plateau": plateau.to_dict(),
        "improving": improving.to_dict(),
    }


def main() -> int:
    print(json.dumps(run_mock_early_stop_check(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
