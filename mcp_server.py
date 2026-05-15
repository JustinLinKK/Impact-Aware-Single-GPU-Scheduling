"""Local stdio MCP server for scheduler graph access."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import SchedulerClient
from .config import SchedulerConfig


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
    ) -> dict[str, Any]:
        return client.recommend_batch_size(
            model_or_signature=model_or_signature,
            hardware=hardware,
            toolkit=toolkit,
            shape_signature=shape_signature,
            current_batch_size=current_batch_size,
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
    def search_hardware_features(
        query: str,
        hardware_key: str = "current",
        architecture: str | None = None,
        vendor: str | None = None,
        workload_type: str | None = None,
        framework: str = "pytorch",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return client.search_hardware_features(
            query=query,
            hardware_key=hardware_key,
            architecture=architecture,
            vendor=vendor,
            workload_type=workload_type,
            framework=framework,
            limit=limit,
        )

    @server.tool()
    def get_hardware_feature_context(
        hardware_key: str = "current",
        workload_type: str | None = None,
        model_family: str | None = None,
        framework: str = "pytorch",
        limit: int = 8,
    ) -> dict[str, Any]:
        return client.get_hardware_feature_context(
            hardware_key=hardware_key,
            workload_type=workload_type,
            model_family=model_family,
            framework=framework,
            limit=limit,
        )

    @server.tool()
    def get_hardware_optimization_context(candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        return client.get_hardware_optimization_context(candidate=candidate, limit=limit)

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

    return server


def run_stdio(settings_path: str | None = None) -> None:
    settings = SchedulerConfig.from_file(settings_path) if settings_path else SchedulerConfig()
    build_mcp_server(settings).run("stdio")
