"""Process-group helpers for scheduler-managed worker trees."""

from __future__ import annotations

import os
import signal
import subprocess


def start_new_session_kwargs() -> dict[str, bool]:
    """Return Popen kwargs that isolate child trees on POSIX."""
    return {"start_new_session": True} if os.name == "posix" else {}


def signal_process_tree(process: subprocess.Popen, sig: int) -> None:
    """Signal a process and its descendants when it owns a POSIX process group."""
    if process.poll() is not None:
        return
    _signal_process(process, pgid=_isolated_process_group_id(process), sig=sig)


def terminate_process_tree(process: subprocess.Popen, *, timeout: float = 2.0) -> None:
    """Terminate a process and its descendants when it owns a POSIX process group."""
    if process.poll() is not None:
        return

    _signal_process(process, pgid=_isolated_process_group_id(process), sig=signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal_process(process, pgid=_isolated_process_group_id(process), sig=signal.SIGKILL)
    process.wait()


def _isolated_process_group_id(process: subprocess.Popen) -> int | None:
    pgid: int | None = None
    if os.name == "posix":
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None
        if pgid == os.getpgrp():
            pgid = None
    return pgid


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
