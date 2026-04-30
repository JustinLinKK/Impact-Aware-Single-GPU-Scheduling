"""Scheduler runner for raw MLEvolve-generated Python scripts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
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
from ..scheduler.telemetry import GpuTelemetrySample, NvidiaSmiTelemetrySampler
from ..schemas import BatchProbeTrialResult

_BATCH_SIZE_NAMES = {
    "batch_size",
    "train_batch_size",
    "eval_batch_size",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
}
_BATCH_OVERRIDE_VAR = "_MLEVOLVE_BATCH_SIZE_OVERRIDE"
_EPOCH_COUNT_NAMES = {
    "epochs",
    "num_epochs",
    "n_epochs",
    "max_epochs",
    "train_epochs",
}
_EPOCH_OVERRIDE_VAR = "_MLEVOLVE_PROBE_MAX_EPOCHS"
_PROBE_MODE_VAR = "_MLEVOLVE_PROBE_MODE"


@dataclass(slots=True)
class InstrumentedScript:
    path: Path
    had_batch_rewrite: bool


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


def _override_batch_expr(original_value: ast.expr) -> ast.expr:
    override_name = ast.Name(id=_BATCH_OVERRIDE_VAR, ctx=ast.Load())
    return ast.IfExp(
        test=ast.Compare(left=override_name, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
        body=ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[override_name], keywords=[]),
        orelse=original_value,
    )


def _override_epoch_expr(original_value: ast.expr) -> ast.expr:
    override_name = ast.Name(id=_EPOCH_OVERRIDE_VAR, ctx=ast.Load())
    probe_mode = ast.Name(id=_PROBE_MODE_VAR, ctx=ast.Load())
    return ast.IfExp(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                probe_mode,
                ast.Compare(left=override_name, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
            ],
        ),
        body=ast.Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=[
                ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[override_name], keywords=[]),
                original_value,
            ],
            keywords=[],
        ),
        orelse=original_value,
    )


class _BatchOverrideTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.modified = False

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name in _BATCH_SIZE_NAMES:
                node.value = _override_batch_expr(node.value)
                self.modified = True
            elif target_name in _EPOCH_COUNT_NAMES:
                node.value = _override_epoch_expr(node.value)
                self.modified = True
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            target_name = node.target.id
            if target_name in _BATCH_SIZE_NAMES:
                node.value = _override_batch_expr(node.value)
                self.modified = True
            elif target_name in _EPOCH_COUNT_NAMES:
                node.value = _override_epoch_expr(node.value)
                self.modified = True
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        modified = False
        for keyword in node.keywords:
            if keyword.arg in _BATCH_SIZE_NAMES and keyword.value is not None:
                keyword.value = _override_batch_expr(keyword.value)
                modified = True
        if modified:
            self.modified = True
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node = self.generic_visit(node)
        if (
            isinstance(node.target, ast.Name)
            and node.target.id in {"epoch", "ep", "epoch_idx"}
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and node.iter.args
        ):
            node.iter.args[0] = _override_epoch_expr(node.iter.args[0])
            self.modified = True
        return node


def _materialize_instrumented_script(script_path: Path, working_dir: Path) -> InstrumentedScript:
    source = script_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(script_path))
    except SyntaxError:
        return InstrumentedScript(path=script_path, had_batch_rewrite=False)

    transformer = _BatchOverrideTransformer()
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    if not transformer.modified:
        return InstrumentedScript(path=script_path, had_batch_rewrite=False)

    helper_module = ast.parse(
        "import os\n"
        f"{_BATCH_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_BATCH_SIZE_OVERRIDE')\n"
        f"{_PROBE_MODE_VAR} = os.environ.get('MLEVOLVE_PROBE_MODE') == '1'\n"
        f"{_EPOCH_OVERRIDE_VAR} = int(os.environ['MLEVOLVE_PROBE_MAX_EPOCHS']) if os.environ.get('MLEVOLVE_PROBE_MAX_EPOCHS') else None\n"
        "_MLEVOLVE_PROBE_MAX_TRAIN_BATCHES = int(os.environ['MLEVOLVE_PROBE_MAX_TRAIN_BATCHES']) if os.environ.get('MLEVOLVE_PROBE_MAX_TRAIN_BATCHES') else None\n"
        "def _mlevolve_apply_probe_limits():\n"
        "    if not _MLEVOLVE_PROBE_MODE or _MLEVOLVE_PROBE_MAX_TRAIN_BATCHES is None:\n"
        "        return\n"
        "    try:\n"
        "        from torch.utils.data import DataLoader\n"
        "    except Exception:\n"
        "        return\n"
        "    _original_iter = DataLoader.__iter__\n"
        "    def _limited_iter(self):\n"
        "        iterator = _original_iter(self)\n"
        "        for _idx, item in enumerate(iterator):\n"
        "            if _idx >= _MLEVOLVE_PROBE_MAX_TRAIN_BATCHES:\n"
        "                break\n"
        "            yield item\n"
        "    DataLoader.__iter__ = _limited_iter\n"
        "_mlevolve_apply_probe_limits()\n",
        filename=str(script_path),
    )
    module.body = helper_module.body + module.body
    ast.fix_missing_locations(module)

    instrumented_dir = working_dir / "working" / "instrumented_scripts"
    instrumented_dir.mkdir(parents=True, exist_ok=True)
    instrumented_path = instrumented_dir / f"{script_path.stem}_instrumented.py"
    instrumented_path.write_text(ast.unparse(module), encoding="utf-8")
    return InstrumentedScript(path=instrumented_path, had_batch_rewrite=True)


def _base_script_env(
    batch_size_override: int | None = None,
    *,
    probe_mode: bool = False,
    probe_max_epochs: int | None = None,
    probe_max_train_batches: int | None = None,
) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if batch_size_override is not None:
        env["MLEVOLVE_BATCH_SIZE_OVERRIDE"] = str(int(batch_size_override))
    if probe_mode:
        env["MLEVOLVE_PROBE_MODE"] = "1"
    if probe_max_epochs is not None:
        env["MLEVOLVE_PROBE_MAX_EPOCHS"] = str(max(1, int(probe_max_epochs)))
    if probe_max_train_batches is not None:
        env["MLEVOLVE_PROBE_MAX_TRAIN_BATCHES"] = str(max(1, int(probe_max_train_batches)))
    return env


def _resolved_batch_size(context: RunnerContext) -> int | None:
    raw_value = context.job.metadata.get("resolved_batch_size")
    try:
        return int(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_batch_size_failure(stderr_text: str) -> str | None:
    lowered = stderr_text.lower()
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return "cuda out of memory"
    return None


def _run_probe_subprocess(
    *,
    python_executable: str,
    script_path: Path,
    working_dir: Path,
    batch_size: int,
    timeout_seconds: int,
    poll_interval_seconds: float,
    device_index: int,
    probe_max_epochs: int,
    probe_max_train_batches: int,
) -> tuple[bool, list[GpuTelemetrySample], str, str]:
    stdout_path = working_dir / "working" / f"probe_stdout_bs_{batch_size}.log"
    stderr_path = working_dir / "working" / f"probe_stderr_bs_{batch_size}.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = NvidiaSmiTelemetrySampler(device_index)

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            [python_executable, str(script_path)],
            cwd=str(working_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            env=_base_script_env(
                batch_size_override=batch_size,
                probe_mode=True,
                probe_max_epochs=probe_max_epochs,
                probe_max_train_batches=probe_max_train_batches,
            ),
        )

        samples: list[GpuTelemetrySample] = []
        deadline = time.time() + max(1, timeout_seconds)
        while time.time() < deadline:
            sample = sampler.sample()
            if sample is not None:
                samples.append(sample)
            if proc.poll() is not None:
                break
            time.sleep(max(0.1, poll_interval_seconds))

        fits = proc.poll() == 0
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            fits = True

    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    return fits, samples, stdout_text, stderr_text


def probe_mlevolve_script_job(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> BatchProbeTrialResult:
    """Probe a candidate batch size for a raw MLEvolve script."""
    kwargs = context.job.config.runner_kwargs
    script_path = Path(kwargs["script_path"]).resolve()
    working_dir = Path(kwargs["working_dir"]).resolve()
    python_executable = context.job.config.python_executable or sys.executable
    instrumented = _materialize_instrumented_script(script_path, working_dir)

    if not instrumented.had_batch_rewrite:
        return BatchProbeTrialResult(
            fits=True,
            peak_vram_mb=None,
            memory_total_mb=None,
            avg_step_time_ms=None,
            message="no recognizable batch-size knob found; probe skipped with original script",
        )

    timeout_seconds = int(kwargs.get("probe_timeout_seconds", max(20, warmup_steps + measure_steps)))
    poll_interval_seconds = float(kwargs.get("probe_poll_interval_seconds", 0.5))
    probe_max_epochs = max(1, int(kwargs.get("probe_max_epochs", 1)))
    probe_max_train_batches = max(1, int(kwargs.get("probe_max_train_batches", 3)))
    started_at = time.time()
    fits, samples, _stdout_text, stderr_text = _run_probe_subprocess(
        python_executable=python_executable,
        script_path=instrumented.path,
        working_dir=working_dir,
        batch_size=int(batch_size),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        device_index=context.settings.gpu_scheduler.device_index,
        probe_max_epochs=probe_max_epochs,
        probe_max_train_batches=probe_max_train_batches,
    )
    failure_reason = _parse_batch_size_failure(stderr_text)
    if failure_reason is not None:
        fits = False

    peak_vram_mb = max((sample.memory_used_mb for sample in samples), default=None)
    memory_total_mb = max((sample.memory_total_mb for sample in samples), default=None)
    elapsed_ms = (time.time() - started_at) * 1000.0
    return BatchProbeTrialResult(
        fits=bool(fits),
        peak_vram_mb=peak_vram_mb,
        memory_total_mb=memory_total_mb,
        avg_step_time_ms=elapsed_ms / max(1, len(samples)) if samples else None,
        message=failure_reason or ("probe window completed" if fits else stderr_text.strip()[:400]),
    )


def run_mlevolve_script_job(context: RunnerContext) -> dict[str, Any]:
    """Run one generated MLEvolve script and persist an ExecutionResult payload."""
    kwargs = context.job.config.runner_kwargs
    script_path = Path(kwargs["script_path"]).resolve()
    working_dir = Path(kwargs["working_dir"]).resolve()
    result_path = Path(kwargs["result_path"]).resolve()
    timeout = int(kwargs["timeout"])
    python_executable = context.job.config.python_executable or sys.executable

    instrumented = _materialize_instrumented_script(script_path, working_dir)
    executable_script = instrumented.path
    batch_size_override = _resolved_batch_size(context) if instrumented.had_batch_rewrite else None

    start_time = time.time()
    proc = subprocess.Popen(
        [python_executable, str(executable_script)],
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_base_script_env(batch_size_override=batch_size_override),
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
            exc_type, exc_info, exc_stack = _parse_exception(stderr, working_dir, executable_script)
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
        "batch_size_override": batch_size_override,
    }
