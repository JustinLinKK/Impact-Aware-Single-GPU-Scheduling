"""Public command and query DTOs for the scheduler client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import CommandType, JobRun, JobSpec


@dataclass(slots=True)
class SubmitJobRequest:
    spec: JobSpec
    run: JobRun | None = None


@dataclass(slots=True)
class JobCommandRequest:
    job_id: str
    command_type: CommandType


@dataclass(slots=True)
class PreloadRequest:
    model_id: str
    model_path: str | Path
    loader_target: str | None = None
    pin: bool = False


@dataclass(slots=True)
class JobQuery:
    job_id: str


@dataclass(slots=True)
class ReportQuery:
    include_profiles: bool = False
    metadata: dict[str, Any] | None = None

