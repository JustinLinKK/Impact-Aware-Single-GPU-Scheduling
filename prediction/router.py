"""Prediction router and rollout policy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import PredictionRequest, PredictionSource, ResourcePrediction
from .branch_adapter import BranchPredictionAdapter
from .ml_adapter import PerfSeerMLAdapter
from .providers import ResourcePredictionProvider
from ..config import PREDICTION_MODE_BRANCH_PROFILE, PREDICTION_MODE_ML_PREDICTOR


@dataclass(frozen=True, slots=True)
class PredictionRouterResult:
    selected: ResourcePrediction | None
    candidates: tuple[ResourcePrediction, ...] = ()
    shadow_predictions: tuple[ResourcePrediction, ...] = ()
    failures: tuple[str, ...] = ()
    mode: str = PREDICTION_MODE_BRANCH_PROFILE
    selection_reason: str = "no_prediction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "selection_reason": self.selection_reason,
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "shadow_predictions": [prediction.to_dict() for prediction in self.shadow_predictions],
            "failures": list(self.failures),
        }


@dataclass(slots=True)
class PredictionRouter:
    mode: str = PREDICTION_MODE_BRANCH_PROFILE
    branch_provider: ResourcePredictionProvider | None = None
    ml_provider: ResourcePredictionProvider | None = None
    fallback_to_exclusive: bool = True
    request_timeout_ms: int = 1000
    last_result: PredictionRouterResult | None = field(default=None, init=False)

    @classmethod
    def from_settings(cls, settings: Any) -> "PredictionRouter":
        prediction_settings = getattr(settings, "prediction", None)
        mode = str(getattr(prediction_settings, "mode", PREDICTION_MODE_BRANCH_PROFILE) or PREDICTION_MODE_BRANCH_PROFILE)
        branch_settings = getattr(prediction_settings, "branch", None)
        ml_settings = getattr(prediction_settings, "ml", None)
        branch_enabled = bool(getattr(branch_settings, "enabled", True))
        ml_enabled = bool(getattr(ml_settings, "enabled", False))
        branch_provider = (
            BranchPredictionAdapter(
                fixed_confidence_if_uncalibrated=float(getattr(branch_settings, "fixed_confidence_if_uncalibrated", 0.55))
            )
            if branch_enabled
            else None
        )
        ml_provider = PerfSeerMLAdapter(
            enabled=ml_enabled,
            hardware_key=getattr(ml_settings, "hardware_key", None),
            checkpoint_path=getattr(ml_settings, "checkpoint_path", None),
            calibration_path=getattr(ml_settings, "calibration_path", None),
            device=str(getattr(ml_settings, "device", "cpu") or "cpu"),
            cache_size=int(getattr(ml_settings, "cache_size", 1024) or 0),
        )
        return cls(
            mode=mode,
            branch_provider=branch_provider,
            ml_provider=ml_provider,
            fallback_to_exclusive=bool(getattr(prediction_settings, "fallback_to_exclusive", True)),
            request_timeout_ms=int(getattr(prediction_settings, "timeout_ms", 1000) or 1000),
        )

    def predict(self, request: PredictionRequest) -> PredictionRouterResult:
        mode = self._normalized_mode()
        provider = self.branch_provider if mode == PREDICTION_MODE_BRANCH_PROFILE else self.ml_provider
        selected, failure = self._try_provider(provider, request)
        failures = (failure,) if failure else ()
        candidates = (selected,) if selected is not None else ()
        shadow: tuple[ResourcePrediction, ...] = ()
        reason = f"{mode}_selected" if selected is not None else f"{mode}_unavailable"

        result = PredictionRouterResult(
            selected=selected,
            candidates=candidates,
            shadow_predictions=shadow,
            failures=failures,
            mode=mode,
            selection_reason=reason,
        )
        self.last_result = result
        return result

    def _normalized_mode(self) -> str:
        mode = str(self.mode or PREDICTION_MODE_BRANCH_PROFILE).strip().lower().replace("-", "_")
        if mode not in {PREDICTION_MODE_BRANCH_PROFILE, PREDICTION_MODE_ML_PREDICTOR}:
            raise ValueError(f"Unsupported prediction mode: {self.mode}")
        return mode

    def _try_provider(
        self,
        provider: ResourcePredictionProvider | None,
        request: PredictionRequest,
    ) -> tuple[ResourcePrediction | None, str | None]:
        if provider is None:
            return None, None
        try:
            if not provider.available(request.hardware_key):
                return None, f"{provider.name}:unavailable"
            prediction = provider.predict(request)
        except Exception as exc:
            return None, f"{provider.name}:{type(exc).__name__}:{exc}"
        if prediction is not None and prediction.source == PredictionSource.UNKNOWN:
            return None, f"{provider.name}:unknown"
        return prediction, None
