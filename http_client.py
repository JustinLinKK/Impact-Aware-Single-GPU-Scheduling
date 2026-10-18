"""Lightweight Python SDK for a scheduler on the same host."""

from __future__ import annotations

from http.client import HTTPConnection
import json
import math
import time
from typing import Any
from urllib.parse import quote, urlsplit

from .domain import JobRun, JobSpec, TrainingJob
from .dto import SubmitJobRequest


class SchedulerHTTPClient:
    """Reuse one HTTP connection per agent; use a separate client per thread."""

    def __init__(self, base_url: str = "http://127.0.0.1:8766", *, token: str, timeout: float = 35.0):
        url = urlsplit(base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost"}
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.username is not None
        ):
            raise ValueError("base_url must be a local HTTP endpoint")
        if not token.strip():
            raise ValueError("a scheduler API token is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._token = token.strip()
        self._timeout = timeout
        self._connection = HTTPConnection(url.hostname, url.port or 8766, timeout=timeout)
        self._last_request_at = time.monotonic()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SchedulerHTTPClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        if time.monotonic() - self._last_request_at >= 30.0:
            self.close()
        try:
            self._connection.request(
                method,
                path,
                body=body,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            response = self._connection.getresponse()
            data = response.read()
            self._last_request_at = time.monotonic()
        except Exception:
            # Never replay submissions after an ambiguous transport failure.
            self.close()
            raise
        result = json.loads(data) if data else None
        if response.status == 404:
            raise KeyError(path)
        if response.status >= 400:
            raise RuntimeError(f"scheduler HTTP {response.status}: {result.get('error', 'request failed')}")
        return result

    def submit(self, request: SubmitJobRequest | JobSpec | TrainingJob) -> TrainingJob:
        if isinstance(request, SubmitJobRequest):
            request = TrainingJob.from_parts(request.spec, request.run or JobRun())
        elif isinstance(request, JobSpec):
            request = TrainingJob.from_parts(request, JobRun())
        return self.submit_from_payload(request.to_dict())

    def submit_from_payload(self, payload: dict[str, Any]) -> TrainingJob:
        return TrainingJob.from_dict(self._request("POST", "/jobs", payload))

    def submit_many(self, requests: list[SubmitJobRequest | JobSpec | TrainingJob]) -> list[TrainingJob]:
        return [self.submit(request) for request in requests]

    def inspect(self, job_id: str) -> TrainingJob | None:
        try:
            result = self._request("GET", f"/jobs/{quote(job_id, safe='')}")
        except KeyError:
            return None
        return TrainingJob.from_dict(result)

    def list_jobs(self) -> list[TrainingJob]:
        return [TrainingJob.from_dict(item) for item in self._request("GET", "/jobs")]

    def pause(self, job_id: str) -> int:
        return self._request("POST", f"/jobs/{quote(job_id, safe='')}/commands", {"command": "PAUSE"})["command_id"]

    def resume(self, job_id: str) -> int:
        return self._request("POST", f"/jobs/{quote(job_id, safe='')}/commands", {"command": "RESUME"})["command_id"]

    def cancel(self, job_id: str) -> int:
        return self._request("POST", f"/jobs/{quote(job_id, safe='')}/commands", {"command": "CANCEL"})["command_id"]

    def wait_for_completion(self, job_id: str, *, timeout: float = 30.0) -> TrainingJob | None:
        """Receive a completion event; bounded long polls avoid frequent status RPCs."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            wait = min(remaining, 30.0, self._timeout / 2.0)
            result = self._request("GET", f"/jobs/{quote(job_id, safe='')}/completion?timeout={wait}")
            if result is not None:
                return TrainingJob.from_dict(result["job"])
            if time.monotonic() >= deadline:
                return None
