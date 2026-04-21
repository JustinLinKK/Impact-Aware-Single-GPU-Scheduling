"""Typer CLI for localml_scheduler."""

from __future__ import annotations

from pathlib import Path
import json

import typer

from .api import LocalMLSchedulerAPI
from .settings import SchedulerSettings


app = typer.Typer(help="Local single-GPU ML job scheduler")
scheduler_app = typer.Typer(help="Scheduler process commands")
app.add_typer(scheduler_app, name="scheduler")


def _build_api(settings_path: str | None) -> LocalMLSchedulerAPI:
    settings = SchedulerSettings.from_file(settings_path) if settings_path else SchedulerSettings()
    return LocalMLSchedulerAPI(settings)


@scheduler_app.command("start")
def scheduler_start(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    service = api.create_scheduler_service()
    try:
        service.start(background=False)
    except KeyboardInterrupt:
        service.stop()


@app.command("submit")
def submit(job_spec: Path, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    job = api.submit_job_from_file(job_spec)
    typer.echo(job.job_id)


@app.command("list")
def list_jobs(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    typer.echo(json.dumps([job.to_dict() for job in api.list_jobs()], indent=2, sort_keys=True))


@app.command("status")
def status(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    job = api.get_job(job_id)
    if job is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(job.to_dict(), indent=2, sort_keys=True))


@app.command("pause")
def pause(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    api.pause_job(job_id)
    typer.echo(job_id)


@app.command("resume")
def resume(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    api.resume_job(job_id)
    typer.echo(job_id)


@app.command("cancel")
def cancel(job_id: str, settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    api.cancel_job(job_id)
    typer.echo(job_id)


@app.command("preload")
def preload(
    baseline_model_id: str,
    baseline_model_path: Path,
    loader_target: str | None = typer.Option(None, "--loader-target"),
    pin: bool = typer.Option(False, "--pin"),
    settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config"),
) -> None:
    api = _build_api(settings)
    api.preload_model(baseline_model_id, str(baseline_model_path), loader_target=loader_target, pin=pin)
    typer.echo(baseline_model_id)


@app.command("cache-stats")
def cache_stats(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    typer.echo(json.dumps(api.cache_stats(), indent=2, sort_keys=True))


@app.command("report")
def report(settings: str | None = typer.Option(None, "--settings", help="Path to scheduler YAML config")) -> None:
    api = _build_api(settings)
    typer.echo(json.dumps(api.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
