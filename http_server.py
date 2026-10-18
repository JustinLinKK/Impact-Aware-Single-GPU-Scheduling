"""Authenticated loopback HTTP access to one persistent scheduler service."""

from __future__ import annotations

from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from threading import Lock
from urllib.parse import parse_qs, unquote, urlsplit

from .client import SchedulerClient
from .domain import CommandType
from .dto import JobCommandRequest
from .scheduler.service import SchedulerService


class SchedulerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: SchedulerService, *, token: str, port: int = 8766):
        if not token.strip():
            raise ValueError("a scheduler API token is required")
        self.service = service
        self.token = token.strip()
        self.client = SchedulerClient(service.settings)
        self.client.store = service.store
        self.submission_lock = Lock()
        super().__init__(("127.0.0.1", port), _SchedulerHandler)


class _SchedulerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: SchedulerHTTPServer
    timeout = 40.0

    def log_message(self, format, *args) -> None:
        # Request URLs can contain job identifiers; scheduler events record actions.
        return

    def _reply(self, status: int, payload=None) -> None:
        data = json.dumps(payload).encode("utf-8") if payload is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        expected = f"Bearer {self.server.token}".encode("utf-8")
        supplied = self.headers.get("Authorization", "").encode("utf-8")
        if not compare_digest(supplied, expected):
            self.close_connection = True
            self._reply(401, {"error": "unauthorized"})
            return
        try:
            payload = None
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 1024 * 1024:
                raise ValueError("request body must be at most 1 MiB")
            if self.command == "POST":
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("request body must be application/json")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
            elif length:
                raise ValueError("GET requests must not contain a body")
            url = urlsplit(self.path)
            parts = [unquote(part) for part in url.path.strip("/").split("/")]
            client = self.server.client
            if parts == ["jobs"]:
                if self.command == "GET":
                    self._reply(200, [job.to_dict() for job in client.list_jobs()])
                else:
                    with self.server.submission_lock:
                        job = client._normalize_job_payload(payload)
                        if client.inspect(job.job_id) is not None:
                            self._reply(409, {"error": "job_id already exists"})
                            return
                        job = client.submit(job)
                    self._reply(201, job.to_dict())
                return
            if len(parts) not in {2, 3} or parts[0] != "jobs":
                self._reply(404, {"error": "unknown endpoint"})
                return
            job = client.inspect(parts[1])
            if job is None:
                self._reply(404, {"error": "unknown job"})
            elif len(parts) == 2 and self.command == "GET":
                self._reply(200, job.to_dict())
            elif len(parts) == 3 and parts[2] == "commands" and self.command == "POST":
                command = CommandType(payload["command"])
                if command not in {CommandType.PAUSE, CommandType.RESUME, CommandType.CANCEL}:
                    raise ValueError("unsupported job command")
                command_id = client.command(JobCommandRequest(job_id=job.job_id, command_type=command))
                self._reply(202, {"command_id": command_id})
            elif len(parts) == 3 and parts[2] == "completion" and self.command == "GET":
                timeout = float(parse_qs(url.query).get("timeout", ["30"])[0])
                if not math.isfinite(timeout) or not 0 <= timeout <= 30:
                    raise ValueError("timeout must be between 0 and 30 seconds")
                completed = self.server.service.wait_for_completion(job.job_id, timeout=timeout)
                self._reply(200 if completed is not None else 204, {
                    "event": f"job_{completed.status.value.lower()}",
                    "job": completed.to_dict(),
                } if completed is not None else None)
            else:
                self._reply(404, {"error": "unknown endpoint"})
        except (ValueError, TypeError, KeyError) as exc:
            self.close_connection = True
            self._reply(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            self.server.service.logger.exception("Scheduler HTTP request failed")
            self._reply(503, {"error": "scheduler request failed"})
