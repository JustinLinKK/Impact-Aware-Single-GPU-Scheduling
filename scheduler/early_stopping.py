"""Pure validation-metric patience state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ..config import EarlyStoppingSettings


@dataclass(frozen=True, slots=True)
class EarlyStoppingState:
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epoch_count: int = 0
    last_evaluated_epoch: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "EarlyStoppingState":
        raw = payload or {}
        best_metric = raw.get("best_metric")
        best_epoch = raw.get("best_epoch")
        return cls(
            best_metric=float(str(best_metric)) if best_metric is not None else None,
            best_epoch=int(str(best_epoch)) if best_epoch is not None else None,
            bad_epoch_count=max(0, int(str(raw.get("bad_epoch_count", 0)))),
            last_evaluated_epoch=max(0, int(str(raw.get("last_evaluated_epoch", 0)))),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EarlyStoppingDecision:
    state: EarlyStoppingState
    evaluated: bool
    improved: bool
    should_stop: bool
    warning: str | None = None


class EarlyStoppingWatchdog:
    def __init__(self, settings: EarlyStoppingSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        state: EarlyStoppingState,
    ) -> EarlyStoppingDecision:
        if epoch <= state.last_evaluated_epoch:
            return EarlyStoppingDecision(state, False, False, False)
        raw_metric = metrics.get(self.settings.metric_name)
        try:
            metric = float(raw_metric) if raw_metric is not None else None
        except (TypeError, ValueError):
            metric = None
        if metric is None or not math.isfinite(metric):
            warning = f"missing or non-finite early-stopping metric: {self.settings.metric_name}"
            if self.settings.missing_metric_policy == "error":
                raise ValueError(warning)
            updated = EarlyStoppingState(
                best_metric=state.best_metric,
                best_epoch=state.best_epoch,
                bad_epoch_count=state.bad_epoch_count,
                last_evaluated_epoch=epoch,
            )
            return EarlyStoppingDecision(updated, True, False, False, warning)

        improved = state.best_metric is None
        if state.best_metric is not None:
            if self.settings.mode == "max":
                improved = metric > state.best_metric + self.settings.min_delta
            else:
                improved = metric < state.best_metric - self.settings.min_delta
        updated = EarlyStoppingState(
            best_metric=metric if improved else state.best_metric,
            best_epoch=epoch if improved else state.best_epoch,
            bad_epoch_count=0 if improved else state.bad_epoch_count + 1,
            last_evaluated_epoch=epoch,
        )
        should_stop = epoch >= self.settings.min_epochs and updated.bad_epoch_count >= self.settings.patience_epochs
        return EarlyStoppingDecision(updated, True, improved, should_stop)
