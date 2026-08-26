"""Local-first role-gated orchestration for NVIDIA CUDA documentation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit
import logging
import random
import threading
import time

from ..backend_mode import RUNNER_CONTRACT_SUBPROCESS_V1
from ..code_knowledge.records import validate_code_knowledge_record
from ..cuda_mcp_bridge import (
    HardwareFacts,
    facts_from_knowledge_base,
    to_records,
)
from ..observability.metrics import CudaDocsMetrics
from ..observability.events import sanitize_cuda_docs_event_payload
from ..redis_cache import RedisLRUCache
from .cache import CudaDocsCache
from .client import CudaDocsMCPClient
from .curator import synthesize_structured_recipe_records
from .models import (
    CapabilitySupport,
    CudaDocsContext,
    CudaDocsRequest,
    CudaDocsSettings,
    DocChunk,
    RouteDecision,
    RouteOutcome,
    SourceRef,
)
from .normalizer import normalize_mcp_result
from .router import applicability_from_facts, build_request, route_request

LOGGER = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        window_seconds: float,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        self.window_seconds = max(1.0, float(window_seconds))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.clock = clock
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None
        self._half_open_probe = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if self.clock() - self._opened_at >= self.cooldown_seconds:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self.clock() - self._opened_at < self.cooldown_seconds:
                return False
            if self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def success(self) -> None:
        with self._lock:
            self._failures.clear()
            self._opened_at = None
            self._half_open_probe = False

    def failure(self) -> None:
        now = self.clock()
        with self._lock:
            self._half_open_probe = False
            while self._failures and now - self._failures[0] > self.window_seconds:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= self.failure_threshold:
                self._opened_at = now


class TokenBucket:
    def __init__(
        self,
        *,
        rate_per_minute: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rate_per_second = max(0.0, float(rate_per_minute)) / 60.0
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self.updated = clock()
        self.clock = clock
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            now = self.clock()
            self.tokens = min(
                self.capacity,
                self.tokens + max(0.0, now - self.updated) * self.rate_per_second,
            )
            self.updated = now
            if self.tokens < 1.0:
                return False
            self.tokens -= 1.0
            return True


class CudaDocsService:
    """Enrich agent prompts without entering scheduler or LLM execution paths."""

    def __init__(
        self,
        settings: CudaDocsSettings | Any,
        *,
        scheduler_client: Any,
        mcp_client: CudaDocsMCPClient | None = None,
        cache: CudaDocsCache | None = None,
        metrics: CudaDocsMetrics | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        facts: HardwareFacts | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = CudaDocsSettings.from_any(settings)
        self.scheduler_client = scheduler_client
        self.client = mcp_client or CudaDocsMCPClient(self.settings)
        redis_cache = RedisLRUCache.from_settings(
            getattr(scheduler_client, "settings", None)
        )
        self.cache = cache or CudaDocsCache(self.settings, redis_cache=redis_cache)
        self.metrics = metrics or CudaDocsMetrics()
        self.event_sink = event_sink
        self.facts = facts or facts_from_knowledge_base(scheduler_client)
        gpu = getattr(
            getattr(scheduler_client, "settings", None), "gpu_scheduler", None
        )
        self.backend_mode = str(getattr(gpu, "packing_backend", "") or "cuda_process")
        self.runner_contract = RUNNER_CONTRACT_SUBPROCESS_V1
        self.clock = clock
        self.breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_failure_threshold,
            window_seconds=self.settings.circuit_window_seconds,
            cooldown_seconds=self.settings.circuit_cooldown_seconds,
            clock=clock,
        )
        self.rate_limiter = TokenBucket(
            rate_per_minute=self.settings.remote_rate_per_minute,
            burst=self.settings.remote_burst,
            clock=clock,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.prewarm_concurrency,
            thread_name_prefix="cuda-docs-curator",
        )
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and self.settings.rollout_mode != "off"

    def start(self) -> None:
        """Warm authentication/schema and canonical topics asynchronously."""

        if not self.enabled or self.settings.rollout_mode == "shadow":
            return
        self.client.preconnect()
        if self.settings.async_prewarm and self.settings.rollout_mode not in {
            "shadow",
            "off",
        }:
            # Tool discovery affects the canonical key. Wait only on a
            # background curator thread so startup and prompt assembly remain
            # non-blocking.
            self._executor.submit(self._prewarm_after_discovery)

    def _prewarm_after_discovery(self) -> None:
        wait_until_ready = getattr(self.client, "wait_until_ready", None)
        if callable(wait_until_ready):
            wait_until_ready(self.settings.hard_timeout_seconds)
        if not self._closed and not bool(
            getattr(self.client, "auth_unavailable", False)
        ):
            self.prewarm()

    def get_context(
        self,
        *,
        role: str,
        error_text: str = "",
        question: str = "",
        topic: str | None = None,
        profile_symptoms: list[str] | tuple[str, ...] = (),
    ) -> CudaDocsContext:
        request_started = self.clock()
        if not self.enabled:
            self.metrics.increment(
                "cuda_docs_context_requests_total",
                labels={
                    "role": str(role).strip().lower(),
                    "route": "feature_gate",
                    "outcome": "feature_disabled",
                },
            )
            return CudaDocsContext.unavailable(reason="feature_disabled")
        decision = route_request(
            role=role,
            error_text=error_text,
            question=question,
            topic=topic,
            profile_symptoms=profile_symptoms,
        )
        self.metrics.increment(
            "cuda_docs_context_requests_total",
            labels={
                "role": decision.role,
                "route": decision.reason or "none",
                "outcome": decision.outcome.value,
            },
        )
        if decision.outcome != RouteOutcome.ELIGIBLE:
            return CudaDocsContext.unavailable(
                topic=decision.topic,
                reason=decision.reason or "not_applicable",
            )
        if self.settings.rollout_mode == "shadow":
            self._emit(
                "cuda_docs_shadow_route",
                {
                    "role": decision.role,
                    "topic": decision.topic,
                    "status": "hypothetical",
                },
            )
            return CudaDocsContext.unavailable(
                topic=decision.topic,
                reason="shadow_mode",
                applicable=True,
            )
        if decision.role not in self.settings.local_roles:
            return CudaDocsContext.unavailable(
                topic=decision.topic,
                reason="role_not_local_enabled",
                applicable=False,
            )
        applicability = applicability_from_facts(
            self.facts,
            backend_mode=self.backend_mode,
            runner_contract=self.runner_contract,
            remote_tool_schema_hash=self.client.tool_schema_hash,
        )
        if not _applicability_complete(applicability, allow_unknown_schema=True):
            self._observe_total_latency(request_started, role=decision.role)
            return CudaDocsContext.unavailable(
                topic=decision.topic,
                reason="incomplete_installed_stack_applicability",
                applicable=True,
            )
        request = build_request(decision, applicability)
        if error_text and decision.sanitized_error_excerpt != str(error_text).strip():
            self.metrics.increment(
                "cuda_docs_redactions_total",
                labels={"kind": "sensitive_or_workload_identity"},
            )

        cached = self._cache_lookup(request)
        if cached is not None:
            if cached.freshness == "stale" and self._may_background_refresh():
                self.queue_refresh(request)
                self.metrics.increment(
                    "cuda_docs_stale_served_total", labels={"role": decision.role}
                )
            self._observe_total_latency(request_started, role=decision.role)
            return cached

        l3 = self._l3_lookup(request)
        if l3 is not None:
            if l3.freshness == "fresh":
                self.cache.set_context(request.canonical_key, l3)
            else:
                self.cache.memo.set(request.canonical_key, l3)
                if self._may_background_refresh():
                    self.queue_refresh(request)
                    self.metrics.increment(
                        "cuda_docs_stale_served_total",
                        labels={"role": decision.role},
                    )
            self._observe_total_latency(request_started, role=decision.role)
            return l3

        if self._may_block(decision.role):
            remaining = self.settings.total_enrichment_deadline_seconds - (
                self.clock() - request_started
            )
            if remaining <= 0:
                context = CudaDocsContext.unavailable(
                    topic=request.topic,
                    reason="enrichment_deadline_exhausted",
                    applicable=True,
                    cache_key=request.canonical_key,
                )
            else:
                request = self._request_after_schema_discovery(
                    request,
                    timeout_seconds=remaining,
                )
                if not _applicability_complete(request.applicability):
                    context = CudaDocsContext.unavailable(
                        topic=request.topic,
                        reason=(
                            "auth_unavailable"
                            if bool(getattr(self.client, "auth_unavailable", False))
                            else "remote_tool_schema_unavailable"
                        ),
                        applicable=True,
                        cache_key=request.canonical_key,
                    )
                    self._observe_total_latency(request_started, role=decision.role)
                    return context
                # Discovery can change the canonical key. Recheck every local
                # tier before spending the one hosted call.
                discovered_local = self._cache_lookup(request) or self._l3_lookup(
                    request
                )
                if discovered_local is not None:
                    self._observe_total_latency(request_started, role=decision.role)
                    return discovered_local
                remaining = self.settings.total_enrichment_deadline_seconds - (
                    self.clock() - request_started
                )
                if remaining <= 0:
                    context = CudaDocsContext.unavailable(
                        topic=request.topic,
                        reason="enrichment_deadline_exhausted",
                        applicable=True,
                        cache_key=request.canonical_key,
                    )
                else:
                    context = self._singleflight_remote(
                        request,
                        background=False,
                        timeout_seconds=min(
                            self.settings.hard_timeout_seconds, remaining
                        ),
                    )
            self._observe_total_latency(request_started, role=decision.role)
            return context
        if self._may_background_refresh():
            self.queue_refresh(request)
        self._observe_total_latency(request_started, role=decision.role)
        return CudaDocsContext.unavailable(
            topic=request.topic,
            reason=(
                "local_miss_refresh_queued"
                if self._may_background_refresh()
                else "local_miss"
            ),
            applicable=True,
            cache_key=request.canonical_key,
        )

    def get_run_backend_brief(self, *, role: str = "draft") -> CudaDocsContext:
        topic = (
            "NVIDIA MPS process controls and limitations for independent PyTorch jobs"
            if self.backend_mode == "mps_process"
            else "ordinary independent PyTorch CUDA process memory and execution behavior"
        )
        return self.get_context(role=role, topic=topic)

    def prewarm(self) -> None:
        topics = [
            "PyTorch CUDA out of memory allocator fragmentation behavior",
            "mixed precision capabilities for the installed NVIDIA GPU",
            "cuDNN and cuBLAS compatibility for installed CUDA and PyTorch versions",
        ]
        topics.append(
            "NVIDIA MPS process controls and limitations for independent PyTorch jobs"
            if self.backend_mode == "mps_process"
            else "ordinary independent PyTorch CUDA process memory and execution behavior"
        )
        for topic in topics:
            decision = route_request(role="draft", topic=topic)
            applicability = applicability_from_facts(
                self.facts,
                backend_mode=self.backend_mode,
                runner_contract=self.runner_contract,
                remote_tool_schema_hash=self.client.tool_schema_hash,
            )
            self.queue_refresh(build_request(decision, applicability))

    def queue_refresh(self, request: CudaDocsRequest) -> bool:
        if self._closed or not self._may_background_refresh():
            return False
        with self._pending_lock:
            if request.canonical_key in self._pending:
                return False
            self._pending.add(request.canonical_key)
        future = self._executor.submit(self._background_refresh, request)
        future.add_done_callback(
            lambda _future, key=request.canonical_key: self._finish_pending(key)
        )
        return True

    def _background_refresh(self, request: CudaDocsRequest) -> None:
        try:
            request = self._request_after_schema_discovery(
                request,
                timeout_seconds=self.settings.hard_timeout_seconds,
            )
            if not _applicability_complete(request.applicability):
                return
            if self._cache_lookup(request, allow_stale=False) is not None:
                return
            if self._l3_lookup(request, allow_stale=False) is not None:
                return
            result = self._singleflight_remote(request, background=True)
            if not result.evidence_chunks and result.reason in {
                "timeout",
                "error",
                "rate_limited",
            }:
                time.sleep(random.uniform(0.02, 0.2))
                self._singleflight_remote(request, background=True)
        except Exception as exc:
            self.metrics.increment(
                "cuda_docs_remote_calls_total",
                labels={
                    "topic": _topic_label(request.topic),
                    "outcome": "background_failure",
                },
            )
            LOGGER.info(
                "CUDA docs background refresh failed open: %s", exc.__class__.__name__
            )

    def _request_after_schema_discovery(
        self, request: CudaDocsRequest, *, timeout_seconds: float
    ) -> CudaDocsRequest:
        if request.applicability.remote_tool_schema_hash != "unknown":
            return request
        preconnect = getattr(self.client, "preconnect", None)
        if callable(preconnect):
            preconnect()
        wait_until_ready = getattr(self.client, "wait_until_ready", None)
        if callable(wait_until_ready):
            wait_until_ready(
                min(self.settings.hard_timeout_seconds, max(0.0, timeout_seconds))
            )
        schema_hash = str(
            getattr(self.client, "tool_schema_hash", "unknown") or "unknown"
        )
        if schema_hash == "unknown":
            return request
        applicability = replace(
            request.applicability,
            remote_tool_schema_hash=schema_hash,
        )
        decision = RouteDecision(
            RouteOutcome.ELIGIBLE,
            request.role,
            topic=request.topic,
            error_signature_class=request.error_signature_class,
            sanitized_error_excerpt=request.sanitized_error_excerpt,
        )
        return build_request(decision, applicability)

    def _finish_pending(self, key: str) -> None:
        with self._pending_lock:
            self._pending.discard(key)

    def _cache_lookup(
        self, request: CudaDocsRequest, *, allow_stale: bool = True
    ) -> CudaDocsContext | None:
        started = self.clock()
        lookup = self.cache.get(request.canonical_key)
        elapsed = self.clock() - started
        self.metrics.observe(
            "cuda_docs_latency_seconds",
            elapsed,
            labels={"tier": lookup.tier},
        )
        if not lookup.hit or (lookup.freshness == "stale" and not allow_stale):
            return None
        self.metrics.increment(
            "cuda_docs_cache_hits_total", labels={"tier": lookup.tier}
        )
        self._emit(
            "cuda_docs_cache_hit",
            {
                "role": request.role,
                "topic": request.topic,
                "cache_key_hash": request.canonical_key.rsplit(":", 1)[-1],
                "tier": lookup.tier,
                "timing_ms": elapsed * 1000.0,
                "status": lookup.negative_reason or lookup.freshness,
                "source_domains": _source_domains(
                    lookup.context.source_refs if lookup.context else ()
                ),
            },
        )
        if lookup.context is None:
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason=lookup.negative_reason or "negative_cache_hit",
                applicable=True,
                cache_tier=lookup.tier,
                freshness=lookup.freshness,
                cache_key=request.canonical_key,
            )
        return replace(
            lookup.context,
            cache_tier=lookup.tier,
            freshness=lookup.freshness,
            remote_latency_ms=None,
        )

    def _l3_lookup(
        self, request: CudaDocsRequest, *, allow_stale: bool = True
    ) -> CudaDocsContext | None:
        started = self.clock()
        store = self._code_store()
        if store is None or not hasattr(store, "get_cuda_doc_chunks"):
            return None
        try:
            rows = store.get_cuda_doc_chunks(
                cache_key=request.canonical_key,
                limit=self.settings.prompt_max_chunks,
            )
        except Exception:
            rows = []
        elapsed = self.clock() - started
        self.metrics.observe(
            "cuda_docs_latency_seconds",
            elapsed,
            labels={"tier": "qdrant"},
        )
        if not rows:
            return None
        chunks = tuple(
            DocChunk.from_dict(row)
            for row in rows
            if row.get("verified_source")
            and row.get("support_status") != CapabilitySupport.UNSUPPORTED.value
        )
        if not chunks:
            return None
        refs = _source_refs_from_rows(rows)
        freshness = _freshness_from_rows(rows, self.settings)
        if freshness == "expired" or (freshness == "stale" and not allow_stale):
            return None
        self.metrics.increment("cuda_docs_cache_hits_total", labels={"tier": "qdrant"})
        context = CudaDocsContext(
            applicable=True,
            topic=request.topic,
            cache_tier="qdrant",
            freshness=freshness,
            evidence_chunks=chunks,
            source_refs=refs,
            reason="persistent_exact_key_hit",
            cache_key=request.canonical_key,
        )
        self._emit(
            "cuda_docs_cache_hit",
            {
                "role": request.role,
                "topic": request.topic,
                "cache_key_hash": request.canonical_key.rsplit(":", 1)[-1],
                "tier": "qdrant",
                "timing_ms": elapsed * 1000.0,
                "status": freshness,
                "source_domains": _source_domains(refs),
            },
        )
        return context

    def _singleflight_remote(
        self,
        request: CudaDocsRequest,
        *,
        background: bool,
        timeout_seconds: float | None = None,
    ) -> CudaDocsContext:
        leader, event = self.cache.begin_singleflight(request.canonical_key)
        if not leader:
            self.metrics.increment("cuda_docs_singleflight_waiters")
            event.wait(
                timeout=min(
                    self.settings.singleflight_wait_seconds,
                    (
                        timeout_seconds
                        if timeout_seconds is not None
                        else self.settings.singleflight_wait_seconds
                    ),
                )
            )
            return self._cache_lookup(request) or CudaDocsContext.unavailable(
                topic=request.topic,
                reason="singleflight_wait_budget_exhausted",
                applicable=True,
                cache_key=request.canonical_key,
            )
        token: str | None = None
        try:
            token = self.cache.acquire_distributed_lock(request.canonical_key)
            if token is None:
                time.sleep(
                    min(
                        self.settings.singleflight_wait_seconds,
                        (
                            timeout_seconds
                            if timeout_seconds is not None
                            else self.settings.singleflight_wait_seconds
                        ),
                    )
                )
                return self._cache_lookup(request) or CudaDocsContext.unavailable(
                    topic=request.topic,
                    reason="distributed_singleflight_loser",
                    applicable=True,
                    cache_key=request.canonical_key,
                )
            return self._remote_lookup(
                request,
                background=background,
                timeout_seconds=timeout_seconds,
            )
        finally:
            self.cache.release_distributed_lock(request.canonical_key, token)
            self.cache.finish_singleflight(request.canonical_key)

    def _remote_lookup(
        self,
        request: CudaDocsRequest,
        *,
        background: bool,
        timeout_seconds: float | None = None,
    ) -> CudaDocsContext:
        if not self.breaker.allow():
            self.metrics.gauge("cuda_docs_circuit_state", 1.0)
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason="circuit_open",
                applicable=True,
                cache_key=request.canonical_key,
            )
        if not self.rate_limiter.take():
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason="local_rate_limit",
                applicable=True,
                cache_key=request.canonical_key,
            )
        started = self.clock()
        remote = self.client.search(
            request.query,
            timeout_seconds=min(
                self.settings.hard_timeout_seconds,
                (
                    timeout_seconds
                    if timeout_seconds is not None
                    else self.settings.hard_timeout_seconds
                ),
            ),
        )
        self.metrics.observe(
            "cuda_docs_latency_seconds",
            self.clock() - started,
            labels={"tier": "remote"},
        )
        self.metrics.increment(
            "cuda_docs_remote_calls_total",
            labels={"topic": _topic_label(request.topic), "outcome": remote.outcome},
        )
        if (
            remote.latency_ms is not None
            and remote.latency_ms > self.settings.soft_timeout_seconds * 1000.0
        ):
            self.metrics.increment(
                "cuda_docs_remote_soft_overruns_total",
                labels={"topic": _topic_label(request.topic)},
            )
        if not remote.ok:
            if remote.outcome in {"timeout", "rate_limited", "error"}:
                self.breaker.failure()
            ttl = (
                self.settings.auth_failure_ttl_seconds
                if remote.outcome == "auth_unavailable"
                else self.settings.transient_failure_ttl_seconds
            )
            self.cache.set_negative(
                request.canonical_key, remote.outcome, ttl_seconds=ttl
            )
            self.metrics.gauge(
                "cuda_docs_circuit_state",
                1.0 if self.breaker.state == "open" else 0.0,
            )
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason=remote.outcome,
                applicable=True,
                cache_tier="remote",
                cache_key=request.canonical_key,
            )
        normalized = normalize_mcp_result(
            remote.result,
            retrieved_date=date.today().isoformat(),
            max_raw_chars=self.settings.raw_response_max_chars,
            max_chunk_chars=self.settings.normalized_chunk_max_chars,
        )
        if not normalized.valid:
            self.breaker.failure()
            self.cache.set_negative(
                request.canonical_key,
                normalized.rejected_reason or "malformed_mcp_result",
            )
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason=normalized.rejected_reason or "malformed_mcp_result",
                applicable=True,
                cache_tier="remote",
                cache_key=request.canonical_key,
            )
        records: list[dict[str, Any]] = []
        prompt_chunks: list[DocChunk] = []
        for chunk in normalized.chunks:
            matching_refs = [
                ref.to_dict()
                for ref in normalized.source_refs
                if ref.url == chunk.source_url
            ]
            try:
                shaped = to_records(
                    topic=request.topic,
                    answer=chunk.text,
                    facts=self.facts,
                    source_refs=matching_refs,
                    verified_date=chunk.retrieved_or_verified_date,
                    effective_backend=self.backend_mode,
                    runner_contract=self.runner_contract,
                    cache_key=request.canonical_key,
                    remote_tool_schema_hash=request.applicability.remote_tool_schema_hash,
                )
            except Exception:
                shaped = []
            for item in shaped:
                try:
                    validated = validate_code_knowledge_record(item)
                except Exception:
                    continue
                records.append(validated)
                if validated["support_status"] != CapabilitySupport.UNSUPPORTED.value:
                    prompt_chunks.append(DocChunk.from_dict(validated))
        if not records:
            self.cache.set_negative(request.canonical_key, "no_valid_verified_chunks")
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason="no_valid_verified_chunks",
                applicable=True,
                cache_tier="remote",
                cache_key=request.canonical_key,
            )
        if self.settings.persist_raw_chunks or self.settings.synthesize_recipes_async:
            self._executor.submit(self._persist_and_curate, records, remote.result)
        if not prompt_chunks:
            self.cache.set_negative(request.canonical_key, "unsupported_on_current_gpu")
            return CudaDocsContext.unavailable(
                topic=request.topic,
                reason="unsupported_on_current_gpu",
                applicable=True,
                cache_tier="remote",
                cache_key=request.canonical_key,
            )
        self.breaker.success()
        self.metrics.gauge("cuda_docs_circuit_state", 0.0)
        context = CudaDocsContext(
            applicable=True,
            topic=request.topic,
            cache_tier="remote",
            freshness="fresh",
            evidence_chunks=tuple(prompt_chunks),
            source_refs=normalized.source_refs,
            remote_latency_ms=remote.latency_ms,
            reason="verified_nvidia_evidence",
            cache_key=request.canonical_key,
        )
        self.cache.set_context(request.canonical_key, context)
        self._emit(
            "cuda_docs_context_ready",
            {
                "role": request.role,
                "topic": request.topic,
                "cache_key_hash": request.canonical_key.rsplit(":", 1)[-1],
                "tier": "remote",
                "status": "success",
                "latency_ms": remote.latency_ms,
                "source_domains": sorted(
                    {urlsplit(ref.url).hostname or "" for ref in normalized.source_refs}
                ),
            },
        )
        return context

    def _persist_records(self, records: list[dict[str, Any]]) -> None:
        store = self._code_store()
        if store is None:
            return
        try:
            store.ingest_records(records)
        except Exception as exc:
            LOGGER.info("CUDA docs persistence failed open: %s", exc.__class__.__name__)

    def _persist_and_curate(
        self, records: list[dict[str, Any]], remote_result: Any
    ) -> None:
        """Persist raw chunks, then publish only schema-valid JSON recipes."""

        if self.settings.persist_raw_chunks:
            self._persist_records(records)
        if not self.settings.synthesize_recipes_async:
            return
        try:
            recipes = synthesize_structured_recipe_records(remote_result, records)
            if recipes:
                self._persist_records(recipes)
                self.metrics.increment(
                    "cuda_docs_recipes_published_total", value=len(recipes)
                )
        except Exception as exc:
            self.metrics.increment(
                "cuda_docs_remote_calls_total",
                labels={"topic": "curation", "outcome": "background_failure"},
            )
            LOGGER.info(
                "CUDA docs recipe curation failed open: %s", exc.__class__.__name__
            )

    def _code_store(self) -> Any | None:
        accessor = getattr(self.scheduler_client, "_code_store", None)
        if callable(accessor):
            try:
                return accessor()
            except Exception:
                return None
        return getattr(self.scheduler_client, "code_store", None)

    def _may_block(self, role: str) -> bool:
        if (
            role not in self.settings.blocking_roles
            or role not in self.settings.remote_roles
        ):
            return False
        if self.settings.max_remote_calls_per_action <= 0:
            return False
        if self.settings.rollout_mode == "debug_live":
            return role == "debug"
        if self.settings.rollout_mode == "improve_live":
            return role in {"debug", "improve"}
        return False

    def _may_background_refresh(self) -> bool:
        return self.settings.rollout_mode in {
            "prefetch_only",
            "debug_cached",
            "debug_live",
            "improve_live",
        }

    def _observe_total_latency(self, started: float, *, role: str) -> None:
        self.metrics.observe(
            "cuda_docs_latency_seconds",
            self.clock() - started,
            labels={"tier": "enrichment_total", "role": role},
        )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(event_type, sanitize_cuda_docs_event_payload(payload))
        except Exception:
            pass

    def close(self) -> None:
        """Cancel future work without waiting for background prefetch."""

        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.client.close()


def _source_refs_from_rows(rows: list[dict[str, Any]]) -> tuple[SourceRef, ...]:
    seen: set[str] = set()
    refs: list[SourceRef] = []
    for row in rows:
        for raw in row.get("source_refs") or []:
            if not isinstance(raw, dict):
                continue
            ref = SourceRef.from_dict(raw)
            if ref.url and ref.url not in seen:
                refs.append(ref)
                seen.add(ref.url)
    return tuple(refs)


def _freshness_from_rows(rows: list[dict[str, Any]], settings: CudaDocsSettings) -> str:
    newest: datetime | None = None
    for row in rows:
        value = str(row.get("retrieved_or_verified_date") or "")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            newest = max(newest, parsed) if newest is not None else parsed
        except ValueError:
            continue
    if newest is None:
        return "stale"
    age = (datetime.now(timezone.utc) - newest).total_seconds()
    if age <= settings.positive_ttl_seconds:
        return "fresh"
    if age <= settings.stale_ttl_seconds:
        return "stale"
    return "expired"


def _topic_label(topic: str) -> str:
    """Bound metric cardinality without exposing query contents."""

    lowered = topic.lower()
    for label in ("memory", "precision", "cudnn", "cublas", "mps", "architecture"):
        if label in lowered:
            return label
    return "cuda_api"


def _source_domains(refs: Any) -> list[str]:
    return sorted(
        {
            urlsplit(str(getattr(ref, "url", "") or "")).hostname or ""
            for ref in refs or ()
            if str(getattr(ref, "url", "") or "").strip()
        }
    )


def _applicability_complete(
    applicability: Any, *, allow_unknown_schema: bool = False
) -> bool:
    fields = (
        "gpu_architecture",
        "compute_capability",
        "driver_major_minor",
        "cuda_major_minor",
        "framework",
        "framework_major_minor",
        "backend_mode",
        "backend_config_hash",
        "runner_contract",
    )
    values = [
        str(getattr(applicability, field, "") or "").strip().lower() for field in fields
    ]
    if any(not value or value == "unknown" for value in values):
        return False
    schema_hash = (
        str(getattr(applicability, "remote_tool_schema_hash", "") or "").strip().lower()
    )
    return bool(schema_hash and (allow_unknown_schema or schema_hash != "unknown"))
