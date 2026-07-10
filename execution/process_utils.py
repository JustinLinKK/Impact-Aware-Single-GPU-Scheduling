"""Process-group helpers for scheduler-managed worker trees."""

from __future__ import annotations

import os
import signal
import subprocess


def start_new_session_kwargs() -> dict[str, bool]:
    """Return Popen kwargs that isolate child trees on POSIX."""
    return {"start_new_session": True} if os.name == "posix" else {}


def terminate_process_tree(process: subprocess.Popen, *, timeout: float = 2.0) -> None:
    """Terminate a process and its descendants when it owns a POSIX process group."""
    if process.poll() is not None:
        return

    pgid: int | None = None
    if os.name == "posix":
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None
        if pgid == os.getpgrp():
            pgid = None

    _signal_process(process, pgid=pgid, sig=signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal_process(process, pgid=pgid, sig=signal.SIGKILL)
    process.wait()


def _signal_process(process: subprocess.Popen, *, pgid: int | None, sig: int) -> None:
    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        return
