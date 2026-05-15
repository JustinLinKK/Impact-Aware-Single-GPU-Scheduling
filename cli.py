"""Typer CLI for localml_scheduler."""

from __future__ import annotations

from pathlib import Path
import json

import typer

from .client import SchedulerClient
from .config import SchedulerConfig
from .dto import JobCommandRequest, PreloadRequest
from .domain import CommandType
from .mcp_server import run_stdio as run_mcp_stdio


app = typer.Typer(help="Local single-GPU ML job scheduler")
scheduler_app = typer.Typer(help="Scheduler process commands")
hardware_features_app = typer.Typer(help="Hardware feature vector database commands")
app.add_typer(scheduler_app, name="scheduler")
app.add_typer(hardware_features_app, name="hardware-features")


def _build_client(settings_path: str | None) -> SchedulerClient:
    settings = SchedulerConfig.from_file(settings_path) if settings_path else SchedulerConfig()
    return SchedulerClient(settings)


@scheduler_app.command("start")
def scheduler_start(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    engine = client.create_engine()
    try:
        engine.start(background=False)
    except KeyboardInterrupt:
        engine.stop()


@scheduler_app.command("mcp")
def scheduler_mcp(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    run_mcp_stdio(settings)


@hardware_features_app.command("ingest")
def hardware_features_ingest(
    settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config"),
    source: Path | None = typer.Option(None, "--source", help="YAML feature corpus to ingest; defaults to repo seed records"),
    recreate: bool = typer.Option(False, "--recreate/--no-recreate", help="Recreate the Qdrant collection before ingesting"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Validate and summarize records without writing to Qdrant"),
) -> None:
    client = _build_client(settings)
    result = client.ingest_hardware_features(source=source, recreate=recreate, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("submit")
def submit(job_spec: Path, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    job = client.submit_from_file(job_spec)
    typer.echo(job.job_id)


@app.command("list")
def list_jobs(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    typer.echo(json.dumps([job.to_dict() for job in client.list_jobs()], indent=2, sort_keys=True))


@app.command("status")
def status(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    job = client.inspect(job_id)
    if job is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(job.to_dict(), indent=2, sort_keys=True))


@app.command("pause")
def pause(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.PAUSE))
    typer.echo(job_id)


@app.command("resume")
def resume(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.RESUME))
    typer.echo(job_id)


@app.command("cancel")
def cancel(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    client.command(JobCommandRequest(job_id=job_id, command_type=CommandType.CANCEL))
    typer.echo(job_id)


@app.command("preload")
def preload(
    baseline_model_id: str,
    baseline_model_path: Path,
    loader_target: str | None = typer.Option(None, "--loader-target"),
    pin: bool = typer.Option(False, "--pin"),
    settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config"),
) -> None:
    client = _build_client(settings)
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
def cache_stats(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    typer.echo(json.dumps(client.cache_stats(), indent=2, sort_keys=True))


@app.command("report")
def report(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    client = _build_client(settings)
    typer.echo(json.dumps(client.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
