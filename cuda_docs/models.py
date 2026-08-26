"""Typed contracts for local-first NVIDIA CUDA documentation enrichment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..backend_mode import RUNNER_CONTRACT_SUBPROCESS_V1

CUDA_DOCS_SCHEMA_VERSION = "cuda_docs_context_v2"
CUDA_DOCS_QUERY_TEMPLATE_VERSION = "cuda_docs_query_v2"
CUDA_DOCS_CACHE_PREFIX = "localml:cuda_docs:v2"
CUDA_DOCS_ROLLOUT_MODES = frozenset(
    {"off", "shadow", "prefetch_only", "debug_cached", "debug_live", "improve_live"}
)


class RouteOutcome(str, Enum):
    ELIGIBLE = "eligible"
    NOT_APPLICABLE = "not_applicable"
    FEATURE_DISABLED = "feature_disabled"


class CapabilitySupport(str, Enum):
    FUNCTIONALLY_SUPPORTED = "functionally_supported"
    NATIVELY_ACCELERATED = "natively_accelerated"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown_pending_local_verification"


@dataclass(slots=True)
class CudaDocsSettings:
    """Runtime policy. Defaults keep the integration completely inert."""

    enabled: bool = False
    rollout_mode: str = "off"
    endpoint: str = "https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs"
    auth_token_env: str = "NVIDIA_CUDA_MCP_TOKEN"
    remote_roles: list[str] = field(default_factory=lambda: ["debug"])
    blocking_roles: list[str] = field(default_factory=lambda: ["debug"])
    local_roles: list[str] = field(
        default_factory=lambda: [
            "draft",
            "improve",
            "debug",
            "code_review",
            "evolution",
            "fusion",
            "aggregation",
        ]
    )
    soft_timeout_seconds: float = 6.0
    hard_timeout_seconds: float = 8.0
    total_enrichment_deadline_seconds: float = 10.0
    max_remote_calls_per_action: int = 1
    prompt_max_chars: int = 2000
    prompt_max_chunks: int = 3
    ram_cache_max_entries: int = 512
    ram_cache_ttl_seconds: int = 21600
    positive_ttl_seconds: int = 604800
    stale_ttl_seconds: int = 2592000
    negative_ttl_seconds: int = 600
    transient_failure_ttl_seconds: int = 60
    auth_failure_ttl_seconds: int = 600
    ttl_jitter_fraction: float = 0.1
    async_prewarm: bool = True
    prewarm_concurrency: int = 2
    persist_raw_chunks: bool = True
    synthesize_recipes_async: bool = True
    send_source_code: bool = False
    remote_rate_per_minute: float = 12.0
    remote_burst: int = 2
    circuit_failure_threshold: int = 3
    circuit_window_seconds: int = 60
    circuit_cooldown_seconds: int = 60
    singleflight_wait_seconds: float = 0.25
    redis_namespace_capacity: int = 512
    raw_response_max_chars: int = 32000
    normalized_chunk_max_chars: int = 4000

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.rollout_mode = str(self.rollout_mode or "off").strip().lower()
        if self.rollout_mode not in CUDA_DOCS_ROLLOUT_MODES:
            raise ValueError(
                "cuda_docs.rollout_mode must be one of: "
                + ", ".join(sorted(CUDA_DOCS_ROLLOUT_MODES))
            )
        if not self.enabled:
            self.rollout_mode = "off"
        self.endpoint = str(self.endpoint or "").strip()
        self.auth_token_env = str(
            self.auth_token_env or "NVIDIA_CUDA_MCP_TOKEN"
        ).strip()
        if self.enabled and not self.endpoint.startswith("https://"):
            raise ValueError("cuda_docs.endpoint must use https")
        self.remote_roles = _role_list(self.remote_roles)
        self.blocking_roles = _role_list(self.blocking_roles)
        self.local_roles = _role_list(self.local_roles)
        self.hard_timeout_seconds = max(0.1, float(self.hard_timeout_seconds))
        self.soft_timeout_seconds = min(
            self.hard_timeout_seconds,
            max(0.1, float(self.soft_timeout_seconds)),
        )
        self.total_enrichment_deadline_seconds = max(
            self.hard_timeout_seconds, float(self.total_enrichment_deadline_seconds)
        )
        self.max_remote_calls_per_action = max(0, int(self.max_remote_calls_per_action))
        self.prompt_max_chars = max(0, int(self.prompt_max_chars))
        self.prompt_max_chunks = max(0, int(self.prompt_max_chunks))
        self.ram_cache_max_entries = max(0, int(self.ram_cache_max_entries))
        self.ram_cache_ttl_seconds = max(1, int(self.ram_cache_ttl_seconds))
        self.positive_ttl_seconds = max(1, int(self.positive_ttl_seconds))
        self.stale_ttl_seconds = max(
            self.positive_ttl_seconds, int(self.stale_ttl_seconds)
        )
        self.negative_ttl_seconds = max(1, int(self.negative_ttl_seconds))
        self.transient_failure_ttl_seconds = max(
            1, int(self.transient_failure_ttl_seconds)
        )
        self.auth_failure_ttl_seconds = max(1, int(self.auth_failure_ttl_seconds))
        self.ttl_jitter_fraction = min(0.5, max(0.0, float(self.ttl_jitter_fraction)))
        self.prewarm_concurrency = max(1, min(2, int(self.prewarm_concurrency)))
        self.remote_rate_per_minute = max(0.0, float(self.remote_rate_per_minute))
        self.remote_burst = max(1, int(self.remote_burst))
        self.circuit_failure_threshold = max(1, int(self.circuit_failure_threshold))
        self.circuit_window_seconds = max(1, int(self.circuit_window_seconds))
        self.circuit_cooldown_seconds = max(1, int(self.circuit_cooldown_seconds))
        self.singleflight_wait_seconds = min(
            0.25, max(0.0, float(self.singleflight_wait_seconds))
        )
        self.redis_namespace_capacity = max(0, int(self.redis_namespace_capacity))
        self.raw_response_max_chars = max(1000, int(self.raw_response_max_chars))
        self.normalized_chunk_max_chars = max(256, int(self.normalized_chunk_max_chars))
        # The plan deliberately prohibits source-code transmission. Keep this
        # setting parseable for config compatibility but never permit it.
        self.send_source_code = False

    @classmethod
    def from_any(cls, value: Any) -> "CudaDocsSettings":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**value)
        fields = cls.__dataclass_fields__
        return cls(
            **{name: getattr(value, name) for name in fields if hasattr(value, name)}
        )


def _role_list(values: Any) -> list[str]:
    if values is None:
        return []
    return list(
        dict.fromkeys(
            str(value).strip().lower().replace("-", "_")
            for value in values
            if str(value).strip()
        )
    )


@dataclass(frozen=True, slots=True)
class CudaDocsApplicability:
    gpu_architecture: str = ""
    compute_capability: str = ""
    driver_major_minor: str = ""
    cuda_major_minor: str = ""
    framework: str = "pytorch"
    framework_major_minor: str = ""
    backend_mode: str = ""
    backend_config_hash: str = ""
    runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1
    remote_tool_schema_hash: str = "unknown"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CudaDocsApplicability":
        value = value or {}
        return cls(
            **{name: str(value.get(name) or "") for name in cls.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class SourceRef:
    title: str
    url: str
    source_version: str = ""
    retrieved_or_verified_date: str = ""
    source_type: str = "vendor_documentation"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRef":
        return cls(
            title=str(value.get("title") or "NVIDIA CUDA documentation").strip(),
            url=str(value.get("url") or "").strip(),
            source_version=str(
                value.get("source_version") or value.get("version") or ""
            ).strip(),
            retrieved_or_verified_date=str(
                value.get("retrieved_or_verified_date")
                or value.get("retrieved_at")
                or value.get("last_verified")
                or ""
            ).strip(),
            source_type=str(value.get("source_type") or "vendor_documentation").strip(),
        )


@dataclass(frozen=True, slots=True)
class DocChunk:
    chunk_id: str
    text: str
    title: str
    source_url: str
    source_version: str = ""
    retrieved_or_verified_date: str = ""
    support_status: str = CapabilitySupport.UNKNOWN.value
    applicability_support: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["applicability_support"] = dict(self.applicability_support)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocChunk":
        support = value.get("applicability_support") or {}
        if isinstance(support, dict):
            support_items = tuple(sorted((str(k), str(v)) for k, v in support.items()))
        else:
            support_items = tuple((str(k), str(v)) for k, v in support)
        return cls(
            chunk_id=str(value.get("chunk_id") or value.get("record_id") or ""),
            text=str(value.get("text") or value.get("summary_text") or ""),
            title=str(value.get("title") or "NVIDIA CUDA documentation"),
            source_url=str(value.get("source_url") or ""),
            source_version=str(value.get("source_version") or ""),
            retrieved_or_verified_date=str(
                value.get("retrieved_or_verified_date")
                or value.get("retrieved_at")
                or ""
            ),
            support_status=str(
                value.get("support_status") or CapabilitySupport.UNKNOWN.value
            ),
            applicability_support=support_items,
        )


@dataclass(frozen=True, slots=True)
class CudaDocsContext:
    applicable: bool
    topic: str | None
    cache_tier: str
    freshness: str
    evidence_chunks: tuple[DocChunk, ...] = ()
    source_refs: tuple[SourceRef, ...] = ()
    remote_latency_ms: float | None = None
    reason: str | None = None
    cache_key: str | None = None

    @classmethod
    def unavailable(
        cls,
        *,
        topic: str | None = None,
        reason: str,
        applicable: bool = False,
        cache_tier: str = "none",
        freshness: str = "unavailable",
        cache_key: str | None = None,
    ) -> "CudaDocsContext":
        return cls(
            applicable=applicable,
            topic=topic,
            cache_tier=cache_tier,
            freshness=freshness,
            reason=reason,
            cache_key=cache_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "topic": self.topic,
            "cache_tier": self.cache_tier,
            "freshness": self.freshness,
            "evidence_chunks": [item.to_dict() for item in self.evidence_chunks],
            "source_refs": [item.to_dict() for item in self.source_refs],
            "remote_latency_ms": self.remote_latency_ms,
            "reason": self.reason,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CudaDocsContext":
        return cls(
            applicable=bool(value.get("applicable")),
            topic=value.get("topic"),
            cache_tier=str(value.get("cache_tier") or "none"),
            freshness=str(value.get("freshness") or "unavailable"),
            evidence_chunks=tuple(
                DocChunk.from_dict(item)
                for item in value.get("evidence_chunks") or []
                if isinstance(item, dict)
            ),
            source_refs=tuple(
                SourceRef.from_dict(item)
                for item in value.get("source_refs") or []
                if isinstance(item, dict)
            ),
            remote_latency_ms=(
                float(value["remote_latency_ms"])
                if value.get("remote_latency_ms") is not None
                else None
            ),
            reason=value.get("reason"),
            cache_key=value.get("cache_key"),
        )


@dataclass(frozen=True, slots=True)
class CudaDocsRequest:
    role: str
    topic: str
    error_signature_class: str
    sanitized_error_excerpt: str
    applicability: CudaDocsApplicability
    canonical_key: str
    query: str


@dataclass(frozen=True, slots=True)
class RouteDecision:
    outcome: RouteOutcome
    role: str
    topic: str | None = None
    error_signature_class: str = ""
    sanitized_error_excerpt: str = ""
    reason: str | None = None
