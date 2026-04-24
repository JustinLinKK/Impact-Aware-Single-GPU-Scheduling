"""Scheduler runner for raw MLEvolve-generated Python scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import signal
import subprocess
import sys
import time

import humanize

from ..execution.runner_protocol import RunnerContext


def load_raw_file(path: str) -> bytes:
    """Cache loader for scheduler-managed raw script jobs."""
    return Path(path).read_bytes()


def _parse_exception(stderr_text: str, working_dir: Path, script_path: Path) -> tuple[str, dict[str, Any], list[tuple[str, int, str, str]]]:
    exc_type = "RuntimeError"
    exc_info: dict[str, Any] = {}
    exc_stack: list[tuple[str, int, str, str]] = []

    exc_patterns = [
        ("KeyboardInterrupt", "KeyboardInterrupt"),
        ("TimeoutError", "TimeoutError"),
        ("CUDA", "RuntimeError"),
        ("cuda", "RuntimeError"),
        ("ValueError", "ValueError"),
        ("TypeError", "TypeError"),
        ("AttributeError", "AttributeError"),
        ("KeyError", "KeyError"),
        ("IndexError", "IndexError"),
        ("FileNotFoundError", "FileNotFoundError"),
        ("ImportError", "ImportError"),
        ("AssertionError", "AssertionError"),
        ("NameError", "NameError"),
        ("RuntimeError", "RuntimeError"),
    ]
    for pattern, exc_name in exc_patterns:
        if pattern in stderr_text:
            exc_type = exc_name
            break

    stderr_lines = stderr_text.splitlines()
    for line in stderr_lines:
        if 'File "' not in line or "line" not in line:
            continue
        try:
            file_start = line.find('File "') + 6
            file_end = line.find('"', file_start)
            filename = line[file_start:file_end]
            line_start = line.find("line ") + 5
            line_end = line.find(",", line_start)
            if line_end == -1:
                line_end = len(line)
            line_num_str = line[line_start:line_end].strip()
            if not line_num_str.isdigit():
                continue
            func_name = ""
            if "in " in line:
                func_start = line.find("in ") + 3
                func_name = line[func_start:].strip()
            filename_short = filename.replace(str(script_path), script_path.name)
            filename_short = os.path.basename(filename_short.replace(str(working_dir), ""))
            exc_stack.append((filename_short, int(line_num_str), func_name, ""))
        except Exception:
            continue

    for line in reversed(stderr_lines):
        line = line.strip()
        if line and not line.startswith("File") and not line.startswith("Traceback") and ":" in line:
            exc_info["message"] = line.split(":", 1)[1].strip()
            break

    return exc_type, exc_info, exc_stack


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def run_mlevolve_script_job(context: RunnerContext) -> dict[str, Any]:
    """Run one generated MLEvolve script and persist an ExecutionResult payload."""
    kwargs = context.job.config.runner_kwargs
    script_path = Path(kwargs["script_path"]).resolve()
    working_dir = Path(kwargs["working_dir"]).resolve()
    result_path = Path(kwargs["result_path"]).resolve()
    timeout = int(kwargs["timeout"])
    python_executable = context.job.config.python_executable or sys.executable

    start_time = time.time()
    proc = subprocess.Popen(
        [python_executable, str(script_path)],
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    exc_type: str | None = None
    exc_info: dict[str, Any] = {}
    exc_stack: list[tuple[str, int, str, str]] = []
    stdout = ""
    stderr = ""

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        exec_time = time.time() - start_time
        if proc.returncode != 0:
            exc_type, exc_info, exc_stack = _parse_exception(stderr, working_dir, script_path)
    except subprocess.TimeoutExpired:
        try:
            proc.send_signal(signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        exec_time = time.time() - start_time
        exc_type = "TimeoutError"

    output: list[str] = []
    if stdout:
        output.extend(stdout.splitlines(keepends=True))
    if stderr:
        output.extend(stderr.splitlines(keepends=True))
    if not output:
        output = [""]
    if output and output[-1] and not output[-1].endswith("\n"):
        output.append("\n")

    if exc_type == "TimeoutError":
        output.append(f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(timeout)}")
    else:
        output.append(f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(timeout)}).")

    result = {
        "term_out": output,
        "exec_time": exec_time,
        "exc_type": exc_type,
        "exc_info": exc_info,
        "exc_stack": exc_stack,
    }
    _write_json_atomic(result_path, result)
    return {
        "reason": "mlevolve script executed",
        "execution_result_path": str(result_path),
        "candidate_returncode": proc.returncode,
        "candidate_exc_type": exc_type,
    }
