"""Simple local IPC server for baseline model cache access."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle
import socket
import socketserver
import struct
import threading

from .baseline_cache import BaselineModelCache
from ..settings import SchedulerSettings


def _recv_message(reader) -> Any:
    header = reader.read(4)
    if not header:
        return None
    size = struct.unpack("!I", header)[0]
    payload = reader.read(size)
    return pickle.loads(payload)


def _send_message(writer, message: Any) -> None:
    payload = pickle.dumps(message)
    writer.write(struct.pack("!I", len(payload)))
    writer.write(payload)
    writer.flush()


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class CacheRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            request = _recv_message(self.rfile)
            if request is None:
                return
            response = self.server.cache_server.handle_request(request)  # type: ignore[attr-defined]
            _send_message(self.wfile, response)
            if request.get("action") == "close":
                return


class CacheServer:
    """Expose the baseline cache over a local socket for worker subprocesses."""

    def __init__(self, settings: SchedulerSettings, cache: BaselineModelCache):
        self.settings = settings
        self.cache = cache
        self._thread: threading.Thread | None = None
        self._server: socketserver.BaseServer | None = None

    def _build_server(self) -> socketserver.BaseServer:
        address = self.settings.cache_address()
        if isinstance(address, str):
            socket_path = Path(address)
            if socket_path.exists():
                socket_path.unlink()
            server = _ThreadingUnixServer(address, CacheRequestHandler)
        else:
            server = _ThreadingTCPServer(address, CacheRequestHandler)
        server.cache_server = self  # type: ignore[attr-defined]
        return server

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = self._build_server()
        self._thread = threading.Thread(target=self._server.serve_forever, name="cache-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        address = self.settings.cache_address()
        if isinstance(address, str) and Path(address).exists():
            Path(address).unlink()
        self._server = None
        self._thread = None

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        try:
            if action == "ping":
                return {"ok": True, "result": "pong"}
            if action == "preload":
                ok = self.cache.preload(
                    request["model_id"],
                    request["baseline_model_path"],
                    loader_target=request.get("loader_target"),
                    pin=bool(request.get("pin", False)),
                    metadata=request.get("metadata"),
                )
                return {"ok": True, "result": ok}
            if action == "get":
                payload = self.cache.get(
                    request["model_id"],
                    request["baseline_model_path"],
                    loader_target=request.get("loader_target"),
                    metadata=request.get("metadata"),
                )
                return {"ok": True, "result": payload}
            if action == "evict":
                return {"ok": True, "result": self.cache.evict(request["model_id"])}
            if action == "pin":
                return {"ok": True, "result": self.cache.pin(request["model_id"])}
            if action == "unpin":
                return {"ok": True, "result": self.cache.unpin(request["model_id"])}
            if action == "stats":
                return {"ok": True, "result": self.cache.stats().to_dict(), "entries": self.cache.snapshot_entries()}
            if action == "close":
                return {"ok": True, "result": True}
        except Exception as exc:  # pragma: no cover - exercised indirectly in worker paths
            return {"ok": False, "error": repr(exc)}
        return {"ok": False, "error": f"Unsupported action: {action}"}


class CacheClient:
    """Small client for talking to the local cache server."""

    def __init__(self, settings: SchedulerSettings):
        self.settings = settings

    def _connect(self):
        address = self.settings.cache_address()
        if isinstance(address, str):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(address)
            return sock
        return socket.create_connection(address, timeout=5.0)

    def request(self, action: str, **payload: Any) -> Any:
        with self._connect() as sock:
            reader = sock.makefile("rb")
            writer = sock.makefile("wb")
            _send_message(writer, {"action": action, **payload})
            response = _recv_message(reader)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        if action == "stats":
            return response
        return response["result"]

    def ping(self) -> bool:
        return self.request("ping") == "pong"

    def preload(self, model_id: str, baseline_model_path: str, *, loader_target: str | None = None, pin: bool = False, metadata: dict[str, Any] | None = None) -> bool:
        return bool(
            self.request(
                "preload",
                model_id=model_id,
                baseline_model_path=baseline_model_path,
                loader_target=loader_target,
                pin=pin,
                metadata=metadata or {},
            )
        )

    def get(self, model_id: str, baseline_model_path: str, *, loader_target: str | None = None, metadata: dict[str, Any] | None = None) -> bytes:
        return bytes(
            self.request(
                "get",
                model_id=model_id,
                baseline_model_path=baseline_model_path,
                loader_target=loader_target,
                metadata=metadata or {},
            )
        )

    def evict(self, model_id: str) -> bool:
        return bool(self.request("evict", model_id=model_id))

    def pin(self, model_id: str) -> bool:
        return bool(self.request("pin", model_id=model_id))

    def unpin(self, model_id: str) -> bool:
        return bool(self.request("unpin", model_id=model_id))

    def stats(self) -> dict[str, Any]:
        return self.request("stats")
