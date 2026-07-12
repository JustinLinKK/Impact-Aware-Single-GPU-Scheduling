"""Prediction router and rollout policy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import PredictionRequest, PredictionSource, ResourcePrediction
from .branch_adapter import BranchPredictionAdapter
from .ml_adapter import PerfSeerMLAdapter
from .providers import ResourcePredictionProvider


@dataclass(frozen=True, slots=True)
class PredictionRouterResult:
    selected: ResourcePrediction | None
    candidates: tuple[ResourcePrediction, ...] = ()
    shadow_predictions: tuple[ResourcePrediction, ...] = ()
    failures: tuple[str, ...] = ()
    mode: str = "branch_only"
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
    mode: str = "branch_only"
    branch_provider: ResourcePredictionProvider | None = None
    ml_provider: ResourcePredictionProvider | None = None
    fallback_to_exclusive: bool = True
    request_timeout_ms: int = 1000
    last_result: PredictionRouterResult | None = field(default=None, init=False)

    @classmethod
    def from_settings(cls, settings: Any) -> "PredictionRouter":
        prediction_settings = getattr(settings, "prediction", None)
        mode = str(getattr(prediction_settings, "mode", "branch_only") or "branch_only")
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
        branch, branch_failure = self._try_provider(self.branch_provider, request)
        ml, ml_failure = self._try_provider(self.ml_provider, request)
        failures = tuple(item for item in (branch_failure, ml_failure) if item)
        candidates = tuple(candidate for candidate in (branch, ml) if candidate is not None)
        shadow: tuple[ResourcePrediction, ...] = ()
        selected: ResourcePrediction | None = None
        reason = "no_prediction"

        if mode == "branch_only":
            selected = branch
            reason = "branch_selected" if selected is not None else "branch_unavailable"
        elif mode == "ml_shadow":
            selected = branch
            shadow = (ml,) if ml is not None else ()
            reason = "branch_selected_ml_shadowed" if selected is not None else "branch_unavailable"
        elif mode == "ml_primary":
            selected = ml or branch
            if selected is ml and ml is not None:
                reason = "ml_selected"
            elif selected is branch and branch is not None:
                reason = "ml_unavailable_branch_fallback"
            else:
                reason = "all_providers_unavailable"
        elif mode == "confidence_first":
            selected = max(candidates, key=lambda item: item.confidence, default=None)
            reason = f"{selected.source.value}_highest_confidence" if selected is not None else "all_providers_unavailable"

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
        mode = str(self.mode or "branch_only").strip().lower().replace("-", "_")
        if mode not in {"branch_only", "ml_shadow", "confidence_first", "ml_primary"}:
            return "branch_only"
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
