"""Bounded persistent client for NVIDIA's single search_cuda_docs MCP tool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time

from .models import CudaDocsSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RemoteCallResult:
    outcome: str
    result: Any = None
    latency_ms: float | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "success" and self.result is not None


class CudaDocsMCPClient:
    """Own one SDK session on a private event loop and fail open on errors."""

    def __init__(
        self,
        settings: CudaDocsSettings | Any,
        *,
        search_callable: Callable[[str, float], Any] | None = None,
        tool_schema_hash: str = "unknown",
    ):
        self.settings = CudaDocsSettings.from_any(settings)
        self._search_callable = search_callable
        self._schema_hash = str(tool_schema_hash or "unknown")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._schema_ready = threading.Event()
        self._state_lock = threading.Lock()
        self._session: Any = None
        self._query_argument = "query"
        self._auth_unavailable = False
        self._closed = False
        self._async_connect_lock: asyncio.Lock | None = None
        self._async_session_ready: asyncio.Event | None = None
        self._owner_stop: asyncio.Event | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._owner_error: BaseException | None = None
        self._callable_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="cuda-docs-fake-transport"
            )
            if search_callable is not None
            else None
        )

    @property
    def tool_schema_hash(self) -> str:
        return self._schema_hash

    @property
    def auth_unavailable(self) -> bool:
        return self._auth_unavailable

    def preconnect(self) -> None:
        """Start connection/schema discovery without delaying the caller."""

        if self._search_callable is not None or self._closed:
            self._schema_ready.set()
            return
        if not self._has_preestablished_auth():
            self._auth_unavailable = True
            self._schema_ready.set()
            return
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._connect(), loop)
        future.add_done_callback(self._consume_preconnect_result)

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        """Wait for schema discovery from a background worker, never an agent."""

        return self._schema_ready.wait(timeout=max(0.0, float(timeout_seconds)))

    def _consume_preconnect_result(
        self, future: concurrent.futures.Future[Any]
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            text = str(exc).lower()
            if any(
                token in text
                for token in ("401", "403", "unauthorized", "authentication")
            ):
                self._auth_unavailable = True
            LOGGER.info(
                "CUDA docs MCP preconnect failed open: %s", exc.__class__.__name__
            )
        finally:
            self._schema_ready.set()

    def search(
        self, query: str, *, timeout_seconds: float | None = None
    ) -> RemoteCallResult:
        if self._closed:
            return RemoteCallResult("unavailable", reason="client_closed")
        if self._search_callable is None and not self._has_preestablished_auth():
            self._auth_unavailable = True
        if self._auth_unavailable:
            return RemoteCallResult(
                "auth_unavailable", reason="authentication_unavailable"
            )
        timeout = min(
            float(timeout_seconds or self.settings.hard_timeout_seconds),
            self.settings.hard_timeout_seconds,
        )
        started = time.monotonic()
        if self._search_callable is not None:
            try:
                assert self._callable_executor is not None
                call = self._callable_executor.submit(
                    self._search_callable, str(query), timeout
                )
                result = call.result(timeout=timeout)
                return RemoteCallResult(
                    "success",
                    result=result,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                )
            except (TimeoutError, concurrent.futures.TimeoutError):
                call.cancel()
                return RemoteCallResult(
                    "timeout",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    reason="hosted_call_timeout",
                )
            except Exception as exc:
                return self._failure(exc, started)

        future = asyncio.run_coroutine_threadsafe(
            self._search_async(str(query), timeout), self._ensure_loop()
        )
        try:
            result = future.result(timeout=timeout)
            return RemoteCallResult(
                "success",
                result=result,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        except concurrent.futures.TimeoutError:
            future.cancel()
            return RemoteCallResult(
                "timeout",
                latency_ms=(time.monotonic() - started) * 1000.0,
                reason="hosted_call_timeout",
            )
        except Exception as exc:
            return self._failure(exc, started)

    def _failure(self, exc: Exception, started: float) -> RemoteCallResult:
        text = str(exc).lower()
        if any(
            token in text for token in ("401", "403", "unauthorized", "authentication")
        ):
            self._auth_unavailable = True
            outcome = "auth_unavailable"
        elif "429" in text:
            outcome = "rate_limited"
        elif any(token in text for token in ("timeout", "timed out")):
            outcome = "timeout"
        else:
            outcome = "error"
        LOGGER.info("CUDA docs hosted call failed open: outcome=%s", outcome)
        return RemoteCallResult(
            outcome,
            latency_ms=(time.monotonic() - started) * 1000.0,
            reason=exc.__class__.__name__,
        )

    def _has_preestablished_auth(self) -> bool:
        """Check only for token presence; never serialize or log its value."""

        return bool(os.getenv(self.settings.auth_token_env, "").strip())

    async def _search_async(self, query: str, timeout: float) -> Any:
        await self._connect()
        if self._session is None:
            raise RuntimeError("CUDA docs MCP session unavailable")
        return await asyncio.wait_for(
            self._session.call_tool(
                "search_cuda_docs",
                {self._query_argument: query},
                read_timeout_seconds=timedelta(seconds=timeout),
            ),
            timeout=timeout,
        )

    async def _connect(self) -> None:
        if self._session is not None or self._auth_unavailable or self._closed:
            self._schema_ready.set()
            return
        if self._async_connect_lock is None:
            self._async_connect_lock = asyncio.Lock()
        async with self._async_connect_lock:
            if self._session is not None:
                return
            if self._owner_task is None or self._owner_task.done():
                assert self._async_session_ready is not None
                self._async_session_ready.clear()
                self._owner_stop = asyncio.Event()
                self._owner_error = None
                self._owner_task = asyncio.create_task(
                    self._session_owner(), name="cuda-docs-session-owner"
                )
                self._owner_task.add_done_callback(self._consume_owner_result)
        assert self._async_session_ready is not None
        await asyncio.wait_for(
            self._async_session_ready.wait(),
            timeout=self.settings.hard_timeout_seconds,
        )
        if self._session is None:
            if self._owner_error is not None:
                raise RuntimeError(str(self._owner_error)) from self._owner_error
            raise RuntimeError("CUDA docs MCP session unavailable")

    async def _session_owner(self) -> None:
        """Enter and exit SDK contexts in one task, as AnyIO requires."""

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx

        try:
            token = os.getenv(self.settings.auth_token_env, "").strip()
            if not token:
                self._auth_unavailable = True
                raise PermissionError("authentication unavailable")
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True,
                timeout=httpx.Timeout(self.settings.hard_timeout_seconds),
            ) as http_client:
                async with streamable_http_client(
                    self.settings.endpoint,
                    http_client=http_client,
                    terminate_on_close=True,
                ) as (read_stream, write_stream, _session_id):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(
                            seconds=self.settings.hard_timeout_seconds
                        ),
                    ) as session:
                        await session.initialize()
                        await self._discover_tool(session)
                        with self._state_lock:
                            self._session = session
                        assert self._async_session_ready is not None
                        self._async_session_ready.set()
                        self._schema_ready.set()
                        assert self._owner_stop is not None
                        await self._owner_stop.wait()
        except BaseException as exc:
            self._owner_error = exc
            text = str(exc).lower()
            if any(
                token in text
                for token in ("401", "403", "unauthorized", "authentication")
            ):
                self._auth_unavailable = True
            raise
        finally:
            with self._state_lock:
                self._session = None
            if self._async_session_ready is not None:
                self._async_session_ready.set()
            self._schema_ready.set()

    async def _discover_tool(self, session: Any) -> None:
        tools_result = await session.list_tools()
        tools = list(getattr(tools_result, "tools", []) or [])
        tool = next(
            (item for item in tools if getattr(item, "name", "") == "search_cuda_docs"),
            None,
        )
        if tool is None:
            raise RuntimeError("NVIDIA MCP session does not expose search_cuda_docs")
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {}
        )
        encoded = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        self._schema_hash = hashlib.sha256(encoded).hexdigest()
        properties = (
            dict(schema.get("properties") or {}) if isinstance(schema, dict) else {}
        )
        required = (
            list(schema.get("required") or []) if isinstance(schema, dict) else []
        )
        if "query" in properties:
            self._query_argument = "query"
        elif required:
            self._query_argument = str(required[0])
        elif properties:
            self._query_argument = str(next(iter(properties)))

    @staticmethod
    def _consume_owner_result(task: asyncio.Task[Any]) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._state_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._loop = loop
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(loop,),
                    name="cuda-docs-mcp",
                    daemon=True,
                )
                self._thread.start()
        self._loop_ready.wait(timeout=1.0)
        assert self._loop is not None
        return self._loop

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        self._async_connect_lock = asyncio.Lock()
        self._async_session_ready = asyncio.Event()
        self._loop_ready.set()
        loop.run_forever()

    def close(self) -> None:
        self._closed = True
        if self._callable_executor is not None:
            self._callable_executor.shutdown(wait=False, cancel_futures=True)
        loop = self._loop
        if loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._disconnect(), loop)
            future.result(timeout=1.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)

    async def _disconnect(self) -> None:
        if self._owner_stop is not None:
            self._owner_stop.set()
        task = self._owner_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=0.8)
            except (asyncio.CancelledError, TimeoutError, Exception):
                task.cancel()
