"""Deterministic trial filtering, Pareto ranking, and decision explanations."""

from __future__ import annotations

from .backend_compatibility import BackendCompatibilityPolicy
from .pareto import pareto_fronts
from .trial_candidate import BackendTrialConfig, TrialCandidate


class TrialPriorityPlanner:
    def __init__(self) -> None:
        self.compatibility = BackendCompatibilityPolicy()

    def backend_configs(
        self,
        backend_name: str,
        *,
        mps_templates: list[list[int]],
        stream_offsets: list[float],
        active_config: dict[str, object] | None = None,
    ) -> tuple[BackendTrialConfig, ...]:
        normalized = str(backend_name).lower().replace("-", "_")
        if active_config:
            return (
                BackendTrialConfig(
                    allocation_percentages=tuple(
                        int(value)
                        for value in active_config.get("allocation_percentages", [])
                    ),
                    stream_offset_steps=(
                        float(active_config["stream_offset_steps"])
                        if active_config.get("stream_offset_steps") is not None
                        else None
                    ),
                    mps_clients=(
                        int(active_config["mps_clients"])
                        if active_config.get("mps_clients") is not None
                        else None
                    ),
                    streams_per_client=(
                        int(active_config["streams_per_client"])
                        if active_config.get("streams_per_client") is not None
                        else None
                    ),
                ),
            )
        if normalized in {"mps", "mps_process"}:
            return tuple(
                BackendTrialConfig(
                    allocation_percentages=tuple(template),
                    mps_clients=2,
                    streams_per_client=1,
                )
                for template in mps_templates
            )
        if normalized in {"stream", "cuda_stream"}:
            return tuple(
                BackendTrialConfig(
                    stream_offset_steps=float(offset),
                    mps_clients=1,
                    streams_per_client=2,
                )
                for offset in stream_offsets
            )
        if normalized == "mps_stream":
            return tuple(
                BackendTrialConfig(
                    allocation_percentages=tuple(template),
                    stream_offset_steps=float(offset),
                    mps_clients=2,
                    streams_per_client=1,
                )
                for template in mps_templates
                for offset in stream_offsets
            )
        return (BackendTrialConfig(),)

    def rank(self, candidates: list[TrialCandidate]) -> list[TrialCandidate]:
        eligible: list[TrialCandidate] = []
        for candidate in candidates:
            assessment = self.compatibility.evaluate(
                candidate.fingerprints,
                backend_name=candidate.backend_name,
                backend_config=candidate.backend_config,
            )
            candidate.compatibility = assessment
            if assessment.hard_rejection_reasons:
                continue
            if candidate.exact_profile_status == "bad":
                continue
            eligible.append(candidate)
        by_backend: dict[str, list[TrialCandidate]] = {}
        for candidate in eligible:
            by_backend.setdefault(candidate.backend_name, []).append(candidate)
        for backend_candidates in by_backend.values():
            fronts = pareto_fronts(
                backend_candidates,
                lambda item: (
                    item.compatibility.risk_components
                    if item.compatibility is not None
                    else {}
                ),
                stable_key=lambda item: item.stable_candidate_id,
            )
            for candidate in backend_candidates:
                candidate.pareto_front = fronts[candidate.stable_candidate_id]
        profile_class = {"good": 0, "unknown": 1, "bad": 2}
        confidence_class = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        ranked = sorted(
            eligible,
            key=lambda candidate: (
                profile_class.get(candidate.exact_profile_status, 1),
                candidate.pareto_front,
                -candidate.optimistic_makespan_gain_seconds,
                confidence_class.get(
                    (
                        candidate.compatibility.analysis_confidence
                        if candidate.compatibility is not None
                        else "LOW"
                    ),
                    2,
                ),
                candidate.uncertainty,
                -candidate.vram_headroom_bytes,
                candidate.priority_key,
                candidate.stable_candidate_id,
            ),
        )
        for index, candidate in enumerate(ranked):
            candidate.final_rank = index
        return ranked
