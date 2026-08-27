"""Local stdio MCP server for scheduler graph access."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import SchedulerClient
from .config import SchedulerConfig
from .cuda_mcp_bridge import (
    CUDA_MCP_ENDPOINT,
    build_query,
    facts_from_knowledge_base,
    to_records,
    topic_for_error,
)


def build_mcp_server(settings: SchedulerConfig | None = None) -> FastMCP:
    # Keep the MCP registration layer intentionally thin: the public contract
    # lives here, but all storage/backend logic stays inside SchedulerClient and
    # SchedulerKnowledgeBase so the tool surface remains stable.
    client = SchedulerClient(settings)
    server = FastMCP(
        name="localml_scheduler_graph",
        instructions=(
            "Query scheduler graph knowledge, inspect job and hardware context, and "
            "record curated tuning outcomes for batch size and epoch recommendations."
        ),
    )

    @server.tool()
    def get_job_graph_context(job_id: str) -> dict[str, Any]:
        return client.get_job_graph_context(job_id)

    @server.tool()
    def search_hardware(query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        return client.search_hardware(query=query, limit=limit)

    @server.tool()
    def get_hardware_context(
        hardware_key: str = "current",
        include_scheduler_limits: bool = True,
    ) -> dict[str, Any]:
        return client.get_hardware_context(
            hardware_key=hardware_key,
            include_scheduler_limits=include_scheduler_limits,
        )

    @server.tool()
    def get_job_design_context(candidate: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        return client.get_job_design_context(candidate=candidate, limit=limit)

    @server.tool()
    def get_profile_evidence(candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        return client.get_profile_evidence(candidate=candidate, limit=limit)

    @server.tool()
    def search_profiles(
        model_name: str | None = None,
        hardware: str | None = None,
        backend: str | None = None,
        toolkit: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return client.search_profiles(
            model_name=model_name,
            hardware=hardware,
            backend=backend,
            toolkit=toolkit,
            limit=limit,
        )

    @server.tool()
    def get_runtime_estimate(
        model_or_signature: str,
        batch_size: int,
        hardware: str | None = None,
        backend: str = "exclusive",
    ) -> dict[str, Any]:
        return client.get_runtime_estimate(
            model_or_signature=model_or_signature,
            batch_size=batch_size,
            hardware=hardware,
            backend=backend,
        )

    @server.tool()
    def recommend_batch_size(
        model_or_signature: str,
        hardware: str | None = None,
        toolkit: str | None = None,
        shape_signature: str | None = None,
        current_batch_size: int | None = None,
        candidate_batch_sizes: list[int] | None = None,
        planned_epochs: int | None = None,
        quality_tolerance: float = 0.0,
        metric_maximize: bool | None = None,
        baseline_metric: float | None = None,
        max_effective_batch_size: int | None = None,
        max_seed_variance: float | None = None,
    ) -> dict[str, Any]:
        return client.recommend_batch_size(
            model_or_signature=model_or_signature,
            hardware=hardware,
            toolkit=toolkit,
            shape_signature=shape_signature,
            current_batch_size=current_batch_size,
            candidate_batch_sizes=candidate_batch_sizes,
            planned_epochs=planned_epochs,
            quality_tolerance=quality_tolerance,
            metric_maximize=metric_maximize,
            baseline_metric=baseline_metric,
            max_effective_batch_size=max_effective_batch_size,
            max_seed_variance=max_seed_variance,
        )

    @server.tool()
    def recommend_epochs(
        model_or_signature: str,
        hardware: str | None = None,
        toolkit: str | None = None,
        current_epochs: int | None = None,
    ) -> dict[str, Any]:
        return client.recommend_epochs(
            model_or_signature=model_or_signature,
            hardware=hardware,
            toolkit=toolkit,
            current_epochs=current_epochs,
        )

    @server.tool()
    def get_packet_compatibility(
        model_a: str,
        model_b: str,
        hardware: str | None = None,
        backend: str = "exclusive",
    ) -> dict[str, Any]:
        return client.get_packet_compatibility(
            model_a=model_a,
            model_b=model_b,
            hardware=hardware,
            backend=backend,
        )

    @server.tool()
    def search_profile_summaries(query: str, limit: int = 20) -> list[dict[str, Any]]:
        return client.search_profile_summaries(query=query, limit=limit)

    @server.tool()
    def search_code_knowledge(
        query: str,
        filters: dict[str, Any] | None = None,
        record_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return client.search_code_knowledge(
            query=query,
            filters=filters or {},
            record_types=record_types,
            limit=limit,
        )

    @server.tool()
    def get_code_optimization_context(
        candidate: dict[str, Any],
        graph_context: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        return client.get_code_optimization_context(
            candidate=candidate,
            graph_context=graph_context,
            limit=limit,
        )

    @server.tool()
    def get_optimization_context(candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        return client.get_optimization_context(candidate=candidate, limit=limit)

    @server.tool()
    def record_tuning_outcome(
        job_id: str,
        chosen_batch_size: int | None = None,
        chosen_epochs: int | None = None,
        recommendation_source: str = "agent",
        outcome_metrics: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return client.record_tuning_outcome(
            job_id=job_id,
            chosen_batch_size=chosen_batch_size,
            chosen_epochs=chosen_epochs,
            recommendation_source=recommendation_source,
            outcome_metrics=outcome_metrics or {},
            notes=notes,
        )

    @server.tool()
    def get_cuda_docs_query(error_text: str, signature: str | None = None) -> dict[str, Any]:
        """Turn a training failure into a hardware-constrained CUDA docs query.

        Put the returned `query` to the nvidia-cuda-docs MCP server, then pass
        its answer back to ingest_cuda_docs_answer so the knowledge is kept.

        Returns applicable=False for failures CUDA documentation cannot fix,
        such as syntax errors and missing packages. Those dominate the observed
        failure mix, so querying for them would only add noise.
        """
        topic = topic_for_error(error_text)
        if topic is None:
            return {
                "applicable": False,
                "reason": "failure is not CUDA-related; CUDA docs will not help",
                "endpoint": CUDA_MCP_ENDPOINT,
            }
        facts = facts_from_knowledge_base(client, signature=signature)
        backend = str(
            getattr(client.settings.gpu_scheduler, "packing_backend", "cuda_process")
        )
        return {
            "applicable": True,
            "topic": topic,
            "query": build_query(topic, facts, effective_backend=backend),
            "endpoint": CUDA_MCP_ENDPOINT,
            "hardware": {
                "gpu_name": facts.gpu_name,
                "compute_capability": facts.capability_str,
                "total_vram_mb": facts.total_vram_mb,
                "residual_group_budget_mb": facts.residual_group_budget_mb,
                "active_group_usage_mb": facts.active_group_usage_mb,
                "safety_reserve_mb": facts.safety_reserve_mb,
                "backend_overhead_mb": facts.backend_overhead_mb,
                "measured_peak_vram_mb": facts.measured_peak_vram_mb,
                "measured_samples": facts.measured_samples,
            },
            "excluded_techniques": facts.excluded_techniques(),
        }

    @server.tool()
    def ingest_cuda_docs_answer(
        topic: str,
        answer: str,
        verified_date: str,
        signature: str | None = None,
        source_refs: list[dict[str, Any]] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Store a CUDA docs answer in HWKD, gated on this card's capability.

        Guidance the installed GPU cannot execute is dropped here rather than at
        read time, so the store stays honest for every later query. Pass
        source_refs through from the documentation answer so a reader can check
        the claim.
        """
        facts = facts_from_knowledge_base(client, signature=signature)
        backend = str(
            getattr(client.settings.gpu_scheduler, "packing_backend", "cuda_process")
        )
        records = to_records(
            topic=topic,
            answer=answer,
            facts=facts,
            source_refs=source_refs,
            verified_date=verified_date,
            effective_backend=backend,
        )
        if not records:
            return {
                "ingested": 0,
                "reason": "answer lacked verified NVIDIA source-labelled context or was quarantined",
            }
        result = client._code_store().ingest_records(records, dry_run=dry_run)
        return {
            "ingested": len(records),
            "dry_run": dry_run,
            "chunk_ids": [r.get("chunk_id") for r in records],
            "excluded_techniques": facts.excluded_techniques(),
            "store_result": result,
        }

    return server


def run_stdio(settings_path: str | None = None) -> None:
    settings = SchedulerConfig.from_file(settings_path) if settings_path else SchedulerConfig()
    build_mcp_server(settings).run("stdio")
