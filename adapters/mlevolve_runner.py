"""Scheduler runner for raw MLEvolve-generated Python scripts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

import humanize

from ..execution.runner_protocol import RunnerContext
from ..scheduler.telemetry import GpuTelemetrySample, NvidiaSmiTelemetrySampler
from ..domain import BatchProbeTrialResult, BatchResolution, JobStatus, ProgressSnapshot, utc_now
from ..runtime_environment import repair_generated_training_code
from utils.candidate_timing import materialize_phase_instrumented_file, parse_phase_timing_log

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
    syntax_error: str | None = None
    precision_repair_count: int = 0


@dataclass(slots=True)
class ProbeSubprocessResult:
    fits: bool
    samples: list[GpuTelemetrySample]
    stdout_text: str
    stderr_text: str
    returncode: int | None
    timed_out: bool = False


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
    repair_result = repair_generated_training_code(source, stage="scheduler_materialize")
    source = str(repair_result.get("code") or source)
    precision_repair_count = int(repair_result.get("replacement_count", 0) or 0)
    try:
        module = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        return InstrumentedScript(
            path=script_path,
            had_batch_rewrite=False,
            syntax_error=str(exc),
            precision_repair_count=precision_repair_count,
        )

    transformer = _BatchOverrideTransformer()
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    instrumented_dir = working_dir / "working" / "instrumented_scripts"
    instrumented_dir.mkdir(parents=True, exist_ok=True)

    if not transformer.modified:
        if precision_repair_count <= 0:
            return InstrumentedScript(path=script_path, had_batch_rewrite=False)
        guarded_path = instrumented_dir / f"{script_path.stem}_precision_guarded.py"
        guarded_path.write_text(source, encoding="utf-8")
        return InstrumentedScript(
            path=guarded_path,
            had_batch_rewrite=False,
            precision_repair_count=precision_repair_count,
        )

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

    instrumented_path = instrumented_dir / f"{script_path.stem}_instrumented.py"
    instrumented_path.write_text(ast.unparse(module), encoding="utf-8")
    return InstrumentedScript(
        path=instrumented_path,
        had_batch_rewrite=True,
        precision_repair_count=precision_repair_count,
    )


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
    if raw_value is None:
        return None
    return BatchResolution.resolved_batch_size(context.job)


def _short_excerpt(text: str, *, limit: int = 1000) -> str | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


_METRIC_PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_\-./]*)\s*[:=]\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_METRIC_KEY_HINTS = (
    "loss",
    "acc",
    "accuracy",
    "auc",
    "f1",
    "score",
    "precision",
    "recall",
    "iou",
    "map",
    "error",
    "rmse",
    "mae",
    "mse",
    "lr",
    "learning_rate",
)


def _metric_key_allowed(key: str) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    return any(token in normalized for token in _METRIC_KEY_HINTS)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_metric_line(line: str) -> tuple[dict[str, float], int | None, int | None] | None:
    text = str(line or "").strip()
    if not text:
        return None

    if "MLEVOLVE_METRIC" in text:
        payload_text = text.split("MLEVOLVE_METRIC", 1)[1].strip()
        payload_text = payload_text[1:].strip() if payload_text.startswith(":") else payload_text
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            metrics: dict[str, float] = {}
            epoch = None
            global_step = None
            for key, value in payload.items():
                normalized = str(key).strip().lower()
                numeric = _as_float(value)
                if numeric is None:
                    continue
                if normalized in {"epoch", "epochs"}:
                    epoch = int(numeric)
                elif normalized in {"step", "global_step", "iteration", "iter"}:
                    global_step = int(numeric)
                elif _metric_key_allowed(normalized):
                    metrics[str(key)] = numeric
            return (metrics, epoch, global_step) if metrics else None

    metrics = {}
    epoch = None
    global_step = None
    for match in _METRIC_PAIR_RE.finditer(text):
        key = match.group("key")
        numeric = _as_float(match.group("value"))
        if numeric is None:
            continue
        normalized = key.strip().lower().replace("-", "_")
        if normalized in {"epoch", "epochs"}:
            epoch = int(numeric)
        elif normalized in {"step", "global_step", "iteration", "iter"}:
            global_step = int(numeric)
        elif _metric_key_allowed(normalized):
            metrics[key] = numeric
    return (metrics, epoch, global_step) if metrics else None


def _enqueue_stream_lines(stream, stream_name: str, output_queue: Queue[tuple[str, str]]) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            output_queue.put((stream_name, line))
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _record_raw_metric_line(
    context: RunnerContext,
    line: str,
    counters: dict[str, int],
) -> None:
    parsed = _parse_metric_line(line)
    if parsed is None:
        return
    metrics, epoch, global_step = parsed
    counters["sample_count"] = counters.get("sample_count", 0) + 1
    if global_step is None:
        global_step = max(counters.get("last_global_step", 0) + 1, counters["sample_count"])
    if epoch is None:
        epoch = counters.get("last_epoch", 0)
    counters["last_global_step"] = int(global_step)
    counters["last_epoch"] = int(epoch)
    created_at = utc_now()
    heartbeat = ProgressSnapshot(
        job_id=context.job.job_id,
        epoch=int(epoch),
        global_step=int(global_step),
        phase="train",
        metrics=metrics,
        last_safe_point="metric_log",
    )
    heartbeat.heartbeat_at = created_at
    context.control_hook.control_plane.write_heartbeat(heartbeat)
    if hasattr(context.store, "record_job_metric_sample"):
        context.store.record_job_metric_sample(
            job_id=context.job.job_id,
            created_at=created_at,
            epoch=int(epoch),
            global_step=int(global_step),
            metrics=metrics,
        )
    context.store.update_job(context.job.job_id, last_heartbeat_at=created_at)
    if getattr(context.event_logger, "log_store", None) is not None:
        context.event_logger.log_store.record_job_metric_sample(
            job_id=context.job.job_id,
            created_at=created_at,
            epoch=int(epoch),
            global_step=int(global_step),
            avg_step_time_ms=None,
            estimated_total_runtime_seconds=None,
            remaining_runtime_seconds=None,
            metrics=metrics,
        )


def _classify_probe_failure(
    *,
    stdout_text: str,
    stderr_text: str,
    returncode: int | None,
    timed_out: bool,
) -> tuple[str | None, str | None]:
    combined = f"{stdout_text}\n{stderr_text}".lower()
    if timed_out:
        return "timeout", "probe subprocess timed out"
    if returncode == 0:
        return None, None
    if "cuda out of memory" in combined or "out of memory" in combined or "cublas_status_alloc_failed" in combined:
        return "oom", "cuda out of memory"
    if "syntaxerror" in combined:
        return "syntax_error", _short_excerpt(stderr_text, limit=400) or "syntax error"
    if "modulenotfounderror" in combined or "importerror" in combined:
        return "import_error", _short_excerpt(stderr_text, limit=400) or "import error"
    invalid_model_markers = (
        "not a valid model identifier",
        "not a local folder",
        "does not appear to have a file named",
        "unknown model",
        "invalid model",
        "model not found",
        "no pretrained weights exist",
    )
    if any(marker in combined for marker in invalid_model_markers):
        return "invalid_model", _short_excerpt(stderr_text, limit=400) or "invalid model"
    dtype_markers = (
        "unsupported scalartype",
        "unsupported dtype",
        "expected scalar type",
        "bfloat16",
        "bf16",
        "float8",
        "fp8",
        "mxfp8",
        "nvfp4",
        "mxfp4",
        "fp4",
        "dtype mismatch",
        "mat1 and mat2 must have the same dtype",
    )
    if any(marker in combined for marker in dtype_markers):
        return "dtype_error", _short_excerpt(stderr_text, limit=400) or "dtype error"
    if "traceback" in combined:
        return "script_exception", _short_excerpt(stderr_text, limit=400) or "script exception"
    if returncode not in (0, None):
        return "unknown", _short_excerpt(stderr_text or stdout_text, limit=400) or f"probe subprocess failed with code {returncode}"
    return None, None


def _parse_batch_size_failure(stderr_text: str, stdout_text: str = "", returncode: int | None = None, timed_out: bool = False) -> tuple[str | None, str | None]:
    failure_kind, message = _classify_probe_failure(
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        returncode=returncode,
        timed_out=timed_out,
    )
    if failure_kind == "oom":
        return failure_kind, message or "cuda out of memory"
    return failure_kind, message


def classify_mlevolve_probe_failure(
    *,
    stdout_text: str = "",
    stderr_text: str = "",
    returncode: int | None = None,
    timed_out: bool = False,
) -> tuple[str | None, str | None]:
    """Public helper used by tests and probe wrappers to classify raw script failures."""
    return _classify_probe_failure(
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        returncode=returncode,
        timed_out=timed_out,
    )


def _message_indicates_oom(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return True
    return False


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
) -> ProbeSubprocessResult:
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

        timed_out = False
        fits = proc.poll() == 0
        if proc.poll() is None:
            timed_out = True
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            fits = False

    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    return ProbeSubprocessResult(
        fits=fits,
        samples=samples,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        returncode=proc.returncode,
        timed_out=timed_out,
    )


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

    if instrumented.syntax_error:
        return BatchProbeTrialResult(
            fits=False,
            peak_vram_mb=None,
            memory_total_mb=None,
            avg_step_time_ms=None,
            message=instrumented.syntax_error,
            failure_kind="syntax_error",
            returncode=None,
            stdout_excerpt=None,
            stderr_excerpt=instrumented.syntax_error,
        )

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
    probe_result = _run_probe_subprocess(
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
    failure_kind, failure_reason = _parse_batch_size_failure(
        probe_result.stderr_text,
        stdout_text=probe_result.stdout_text,
        returncode=probe_result.returncode,
        timed_out=probe_result.timed_out,
    )
    fits = bool(probe_result.fits)
    if failure_kind is not None or failure_reason is not None:
        fits = False

    peak_vram_mb = max((sample.memory_used_mb for sample in probe_result.samples), default=None)
    memory_total_mb = max((sample.memory_total_mb for sample in probe_result.samples), default=None)
    elapsed_ms = (time.time() - started_at) * 1000.0
    return BatchProbeTrialResult(
        fits=bool(fits),
        peak_vram_mb=peak_vram_mb,
        memory_total_mb=memory_total_mb,
        avg_step_time_ms=elapsed_ms / max(1, len(probe_result.samples)) if probe_result.samples else None,
        message=failure_reason or ("probe window completed" if fits else _short_excerpt(probe_result.stderr_text, limit=400)),
        failure_kind=failure_kind,
        returncode=probe_result.returncode,
        stdout_excerpt=_short_excerpt(probe_result.stdout_text),
        stderr_excerpt=_short_excerpt(probe_result.stderr_text),
    )


def run_mlevolve_script_job(context: RunnerContext) -> dict[str, Any]:
    """Run one generated MLEvolve script and persist an ExecutionResult payload."""
    kwargs = context.job.config.runner_kwargs
    script_path = Path(kwargs["script_path"]).resolve()
    working_dir = Path(kwargs["working_dir"]).resolve()
    result_path = Path(kwargs["result_path"]).resolve()
    raw_timeout = kwargs.get("timeout")
    timeout = None
    if raw_timeout is not None:
        try:
            parsed_timeout = float(raw_timeout)
            if parsed_timeout > 0:
                timeout = parsed_timeout
        except (TypeError, ValueError):
            timeout = None
    python_executable = context.job.config.python_executable or sys.executable

    instrumented = _materialize_instrumented_script(script_path, working_dir)
    phase_log_path = working_dir / "working" / "phase_timings" / f"phase_{context.job.job_id}.jsonl"
    phase_output_path = working_dir / "working" / "instrumented_scripts" / f"{instrumented.path.stem}_phase.py"
    executable_script, phase_metadata = materialize_phase_instrumented_file(
        instrumented.path,
        phase_output_path,
        phase_log_path,
    )
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
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    output_queue: Queue[tuple[str, str]] = Queue()
    stdout_thread = threading.Thread(target=_enqueue_stream_lines, args=(proc.stdout, "stdout", output_queue), daemon=True)
    stderr_thread = threading.Thread(target=_enqueue_stream_lines, args=(proc.stderr, "stderr", output_queue), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    counters = {"sample_count": 0, "last_global_step": 0, "last_epoch": 0}
    early_stop_reason: str | None = None
    timed_out = False
    deadline = start_time + timeout if timeout is not None else None

    def drain_output() -> None:
        while True:
            try:
                stream_name, line = output_queue.get_nowait()
            except Empty:
                break
            if stream_name == "stderr":
                stderr_lines.append(line)
            else:
                stdout_lines.append(line)
            _record_raw_metric_line(context, line, counters)

    while True:
        drain_output()
        if proc.poll() is not None:
            break

        command = context.control_hook.control_plane.read_command(context.job.job_id)
        if command.action == "early_stop":
            early_stop_reason = command.reason or "scheduler early stop requested"
            context.control_hook.control_plane.clear_command(context.job.job_id)
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            break

        if deadline is not None and time.time() >= deadline:
            timed_out = True
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            break

        time.sleep(0.1)

    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    drain_output()
    exec_time = time.time() - start_time
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    if early_stop_reason is not None:
        exc_info["scheduler_early_stop_reason"] = early_stop_reason
    elif timed_out:
        exc_type = "TimeoutError"
    elif proc.returncode != 0:
        exc_type, exc_info, exc_stack = _parse_exception(stderr, working_dir, executable_script)

    output: list[str] = []
    if stdout_lines:
        output.extend(stdout_lines)
    if stderr_lines:
        output.extend(stderr_lines)
    if not output:
        output = [""]
    if output and output[-1] and not output[-1].endswith("\n"):
        output.append("\n")

    if early_stop_reason is not None:
        output.append(f"Scheduler early stop: {early_stop_reason}\n")
        output.append(f"Execution time: {humanize.naturaldelta(exec_time)} seconds (stopped early by scheduler).\n")
    elif exc_type == "TimeoutError" and timeout is not None:
        output.append(f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(timeout or 0)}")
    elif exc_type == "TimeoutError":
        output.append(f"Execution time: TimeoutError raised after {humanize.naturaldelta(exec_time)} seconds (no time limit).")
    else:
        limit_text = "no time limit" if timeout is None else f"time limit is {humanize.naturaldelta(timeout)}"
        output.append(f"Execution time: {humanize.naturaldelta(exec_time)} seconds ({limit_text}).")

    phase_timings = parse_phase_timing_log(phase_log_path, exec_time=exec_time, instrumentation=phase_metadata)
    instrumentation = dict(phase_metadata or {})
    if instrumented.precision_repair_count:
        instrumentation["precision_numpy_export_repair"] = {
            "replacement_count": instrumented.precision_repair_count,
            "script_path": str(instrumented.path),
        }
    if early_stop_reason is not None:
        samples = context.store.list_job_metric_samples(context.job.job_id) if hasattr(context.store, "list_job_metric_samples") else []
        artifact_payload: dict[str, Any] = {}
        try:
            from ..scheduler.training_plot import render_training_process

            artifact_payload = render_training_process(
                samples,
                context.settings.job_runtime_dir(context.job.job_id),
                decision=None,
            )
        except Exception as exc:
            artifact_payload = {"plot_error": str(exc)}
        completed_at = utc_now()
        context.store.update_job(
            context.job.job_id,
            status=JobStatus.EARLY_STOPPED,
            reason=early_stop_reason,
            hold=True,
            metadata_updates={
                "scheduler_early_stop_pending": False,
                "scheduler_early_stop_completed_at": completed_at,
                "scheduler_early_stop_runtime_seconds": exec_time,
                "scheduler_early_stop_plot_path": artifact_payload.get("plot_path"),
                "scheduler_early_stop_summary_path": artifact_payload.get("summary_path"),
            },
        )
        context.event_logger.emit(
            "job_early_stopped",
            job_id=context.job.job_id,
            payload={
                "reason": early_stop_reason,
                "runtime_seconds": exec_time,
                "plot_path": artifact_payload.get("plot_path"),
                "summary_path": artifact_payload.get("summary_path"),
            },
        )
        instrumentation["scheduler_early_stop"] = {
            "reason": early_stop_reason,
            "runtime_seconds": exec_time,
            "plot_path": artifact_payload.get("plot_path"),
            "summary_path": artifact_payload.get("summary_path"),
            "sample_count": len(samples),
        }
    result = {
        "term_out": output,
        "exec_time": exec_time,
        "exc_type": exc_type,
        "exc_info": exc_info,
        "exc_stack": exc_stack,
        "phase_timings": phase_timings,
        "instrumentation": instrumentation,
    }
    _write_json_atomic(result_path, result)
    if early_stop_reason is None and proc.returncode == 0 and context.job.packing.signature:
        backend_name = str(context.job.metadata.get("placement_backend") or "exclusive")
        context.upsert_runtime_profile(
            backend_name=backend_name,
            strategy=context.job.runtime_probe.strategy,
            startup_seconds=None,
            epoch_1_seconds=None,
            steps_per_epoch=None,
            avg_step_time_ms=exec_time * 1000.0,
            estimated_total_runtime_seconds=exec_time,
            confidence=1.0,
            source="successful_execution",
            observations=1,
            metadata={
                "success": True,
                "candidate_returncode": proc.returncode,
                "scheduler_session_id": context.job.metadata.get("scheduler_session_id"),
                "resolved_batch_size": BatchResolution.resolved_batch_size(context.job),
                "batch_size_override": batch_size_override,
                "phase_timing_available": bool(phase_timings),
            },
        )
    return {
        "reason": "mlevolve script early-stopped" if early_stop_reason is not None else "mlevolve script executed",
        "execution_result_path": str(result_path),
        "candidate_returncode": proc.returncode,
        "candidate_exc_type": exc_type,
        "batch_size_override": batch_size_override,
        "phase_timings": phase_timings,
        "instrumentation": instrumentation,
    }


def run_mlevolve_model_family_probe_job(context: RunnerContext) -> dict[str, Any]:
    """Complete after batch-probe preflight has written the model-family profile."""
    return {
        "kind": "mlevolve_model_family_probe",
        "model_family": context.job.metadata.get("model_family"),
        "profile_key": context.job.batch_probe.profile_key,
        "resolved_batch_size": context.job.metadata.get("resolved_batch_size"),
    }
