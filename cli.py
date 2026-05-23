"""Typer CLI for localml_scheduler."""

from __future__ import annotations

from pathlib import Path
import json
import os

import typer
import yaml

from .client import SchedulerClient
from .config import SchedulerConfig
from .dto import JobCommandRequest, PreloadRequest
from .domain import CommandType
from .mcp_server import run_stdio as run_mcp_stdio


app = typer.Typer(help="Local single-GPU ML job scheduler")
scheduler_app = typer.Typer(help="Scheduler process commands")
hardware_features_app = typer.Typer(help="Hardware feature vector database commands")
code_knowledge_app = typer.Typer(help="Code-knowledge vector database commands")
knowledge_app = typer.Typer(help="Combined knowledge ingestion commands")
app.add_typer(scheduler_app, name="scheduler")
app.add_typer(hardware_features_app, name="hardware-features")
app.add_typer(code_knowledge_app, name="code-knowledge")
app.add_typer(knowledge_app, name="knowledge")


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_unified_config_path(config_path: str | None = None) -> Path | None:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("MLEVOLVE_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    root_config = _REPO_ROOT / "config.yaml"
    if root_config.exists():
        return root_config
    root_example = _REPO_ROOT / "config.example.yaml"
    if root_example.exists():
        return root_example
    return None


def _settings_from_unified_config(config_path: str | None = None) -> SchedulerConfig | None:
    path = _resolve_unified_config_path(config_path)
    if path is None:
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scheduler = payload.get("scheduler") if isinstance(payload, dict) else None
    if not isinstance(scheduler, dict):
        return None
    settings_payload = scheduler.get("settings")
    if not isinstance(settings_payload, dict):
        return None
    return SchedulerConfig.from_dict(settings_payload)


def _build_scheduler_config(settings_path: str | None = None, config_path: str | None = None) -> SchedulerConfig:
    if settings_path:
        typer.echo(
            "Warning: --settings is deprecated; prefer --config with root config.yaml.",
            err=True,
        )
        return SchedulerConfig.from_file(settings_path)
    return _settings_from_unified_config(config_path) or SchedulerConfig()


def _build_client(settings_path: str | None = None, config_path: str | None = None) -> SchedulerClient:
    settings = _build_scheduler_config(settings_path, config_path)
    return SchedulerClient(settings)


@scheduler_app.command("start")
def scheduler_start(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    engine = client.create_engine()
    try:
        engine.start(background=False)
    except KeyboardInterrupt:
        engine.stop()


@scheduler_app.command("mcp")
def scheduler_mcp(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    run_mcp_stdio(settings_path=settings and str(settings), settings=_build_scheduler_config(settings, config_path))


@scheduler_app.command("rebuild-evidence-graph")
def scheduler_rebuild_evidence_graph(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Preview writes by default; use --execute to write to Neo4j"),
    wipe: bool = typer.Option(False, "--wipe/--no-wipe", help="Explicitly wipe Neo4j before rebuilding; ignored during dry-run"),
) -> None:
    client = _build_client(settings, config_path)
    result = client.rebuild_evidence_graph(dry_run=dry_run, wipe=(wipe and not dry_run))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@hardware_features_app.command("ingest")
def hardware_features_ingest(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
    source: Path | None = typer.Option(None, "--source", help="YAML feature corpus to ingest; defaults to repo seed records"),
    recreate: bool = typer.Option(False, "--recreate/--no-recreate", help="Recreate the Qdrant collection before ingesting"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Validate and summarize records without writing to Qdrant"),
) -> None:
    client = _build_client(settings, config_path)
    result = client.ingest_hardware_features(source=source, recreate=recreate, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@code_knowledge_app.command("ingest")
def code_knowledge_ingest(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
    source: Path | None = typer.Option(None, "--source", help="YAML code-knowledge corpus to ingest; defaults to converted hardware-feature seed records"),
    recreate: bool = typer.Option(False, "--recreate/--no-recreate", help="Recreate Qdrant collections before ingesting"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Validate and summarize records without writing to Qdrant"),
) -> None:
    client = _build_client(settings, config_path)
    result = client.ingest_code_knowledge(source=source, recreate=recreate, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@knowledge_app.command("ingest-schema")
def knowledge_ingest_schema(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
    schema_root: Path = typer.Option(Path("schema"), "--schema-root", help="Schema root containing vector YAML directories"),
    recreate: bool = typer.Option(False, "--recreate/--no-recreate", help="Recreate Qdrant collections before ingesting"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Validate and summarize records without writing to Qdrant"),
) -> None:
    client = _build_client(settings, config_path)
    result = client.ingest_schema_knowledge(schema_root=schema_root, recreate=recreate, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("submit")
def submit(
    job_spec: Path,
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    job = client.submit_from_file(job_spec)
    typer.echo(job.job_id)


@app.command("list")
def list_jobs(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    typer.echo(json.dumps([job.to_dict() for job in client.list_jobs()], indent=2, sort_keys=True))


@app.command("status")
def status(
    job_id: str,
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    job = client.inspect(job_id)
    if job is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(job.to_dict(), indent=2, sort_keys=True))


@app.command("pause")
def pause(
    job_id: str,
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.PAUSE))
    typer.echo(job_id)


@app.command("resume")
def resume(
    job_id: str,
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.RESUME))
    typer.echo(job_id)


@app.command("cancel")
def cancel(
    job_id: str,
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.CANCEL))
    typer.echo(job_id)


@app.command("preload")
def preload(
    baseline_model_id: str,
    baseline_model_path: Path,
    loader_target: str | None = typer.Option(None, "--loader-target"),
    pin: bool = typer.Option(False, "--pin"),
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    client.preload(
        PreloadRequest(
            model_id=baseline_model_id,
            model_path=str(baseline_model_path),
            loader_target=loader_target,
            pin=pin,
        )
    )
    typer.echo(baseline_model_id)


@app.command("cache-stats")
def cache_stats(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    typer.echo(json.dumps(client.cache_stats(), indent=2, sort_keys=True))


@app.command("report")
def report(
    settings: str | None = typer.Option(None, "--settings", help="Deprecated path to scheduler YAML config"),
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
) -> None:
    client = _build_client(settings, config_path)
    typer.echo(json.dumps(client.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
