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
from ..domain import BatchProbeTrialResult, BatchResolution, FailureDiagnostic, JobStatus, ProgressSnapshot, utc_now
from ..runtime_environment import repair_generated_training_code
from engine.script_introspection import TrainingBatchContract, analyze_training_batch_contract
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
_TRAIN_BATCH_OVERRIDE_VAR = "_MLEVOLVE_PROBE_MAX_TRAIN_BATCHES"
_PROBE_EVENT_PATH_VAR = "_MLEVOLVE_PROBE_EVENT_PATH"
_PROBE_OPTIMIZER_STEPS_VAR = "_MLEVOLVE_PROBE_OPTIMIZER_STEPS"
_GRADIENT_ACCUMULATION_NAMES = {
    "gradient_accumulation_steps",
    "grad_accum_steps",
    "accumulation_steps",
    "GRADIENT_ACCUMULATION_STEPS",
    "GRAD_ACCUM_STEPS",
    "ACCUMULATION_STEPS",
}


@dataclass(slots=True)
class InstrumentedScript:
    path: Path
    had_batch_rewrite: bool
    syntax_error: str | None = None
    precision_repair_count: int = 0
    batch_contract: TrainingBatchContract | None = None


@dataclass(slots=True)
class ProbeSubprocessResult:
    fits: bool
    samples: list[GpuTelemetrySample]
    stdout_text: str
    stderr_text: str
    returncode: int | None
    timed_out: bool = False
    timeout_phase: str | None = None
    completion_event: dict[str, Any] | None = None


def load_raw_file(path: str) -> bytes:
    """Cache loader for scheduler-managed raw script jobs."""
    return Path(path).read_bytes()


def _parse_exception(stderr_text: str, working_dir: Path, script_path: Path) -> tuple[str, dict[str, Any], list[tuple[str, int, str, str]]]:
    terminal_type, terminal_message = _terminal_exception(stderr_text)
    exc_type = terminal_type or "RuntimeError"
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
    if terminal_type is None:
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

    if terminal_message:
        exc_info["message"] = terminal_message

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
    return ast.IfExp(
        test=ast.Compare(left=override_name, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
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
    def __init__(self, contract: TrainingBatchContract) -> None:
        self.modified = False
        self.batch_modified = False
        self._train_sites = {(site.lineno, site.col_offset): site.argument for site in contract.train_sites}

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name in _EPOCH_COUNT_NAMES:
                node.value = _override_epoch_expr(node.value)
                self.modified = True
            elif target_name in _GRADIENT_ACCUMULATION_NAMES:
                node.value = ast.IfExp(
                    test=ast.Name(id=_PROBE_MODE_VAR, ctx=ast.Load()),
                    body=ast.Constant(value=1),
                    orelse=node.value,
                )
                self.modified = True
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            target_name = node.target.id
            if target_name in _EPOCH_COUNT_NAMES:
                node.value = _override_epoch_expr(node.value)
                self.modified = True
            elif target_name in _GRADIENT_ACCUMULATION_NAMES:
                node.value = ast.IfExp(
                    test=ast.Name(id=_PROBE_MODE_VAR, ctx=ast.Load()),
                    body=ast.Constant(value=1),
                    orelse=node.value,
                )
                self.modified = True
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        argument = self._train_sites.get((int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))))
        if argument is None:
            return node
        if argument.startswith("keyword:"):
            keyword_name = argument.split(":", 1)[1]
            for keyword in node.keywords:
                if keyword.arg == keyword_name:
                    keyword.value = _override_batch_expr(keyword.value)
                    self.modified = True
                    self.batch_modified = True
                    break
        elif argument.startswith("positional:"):
            try:
                position = int(argument.split(":", 1)[1])
            except ValueError:
                return node
            if len(node.args) > position:
                node.args[position] = _override_batch_expr(node.args[position])
                self.modified = True
                self.batch_modified = True
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


def _prepend_after_module_preamble(module: ast.Module, statements: list[ast.stmt]) -> None:
    """Keep the module docstring and future imports in their required leading positions."""
    insertion_index = 0
    if module.body and isinstance(module.body[0], ast.Expr):
        value = module.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion_index = 1
    while insertion_index < len(module.body):
        node = module.body[insertion_index]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        insertion_index += 1
    module.body[insertion_index:insertion_index] = statements


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

    batch_contract = analyze_training_batch_contract(source)
    transformer = _BatchOverrideTransformer(batch_contract)
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    instrumented_dir = working_dir / "working" / "instrumented_scripts"
    instrumented_dir.mkdir(parents=True, exist_ok=True)

    if not transformer.modified:
        if precision_repair_count <= 0:
            return InstrumentedScript(path=script_path, had_batch_rewrite=False, batch_contract=batch_contract)
        guarded_path = instrumented_dir / f"{script_path.stem}_precision_guarded.py"
        guarded_path.write_text(source, encoding="utf-8")
        return InstrumentedScript(
            path=guarded_path,
            had_batch_rewrite=False,
            precision_repair_count=precision_repair_count,
            batch_contract=batch_contract,
        )

    helper_source = (
        "import json as _mlevolve_json\n"
        "import os\n"
        "import time as _mlevolve_time\n"
        f"{_BATCH_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_BATCH_SIZE_OVERRIDE')\n"
        f"{_PROBE_MODE_VAR} = os.environ.get('MLEVOLVE_PROBE_MODE') == '1'\n"
        f"{_EPOCH_OVERRIDE_VAR} = int(os.environ['MLEVOLVE_PROBE_MAX_EPOCHS']) if os.environ.get('MLEVOLVE_PROBE_MAX_EPOCHS') else None\n"
        f"{_PROBE_EVENT_PATH_VAR} = os.environ.get('MLEVOLVE_PROBE_EVENT_PATH')\n"
        f"{_PROBE_OPTIMIZER_STEPS_VAR} = max(1, int(os.environ.get('MLEVOLVE_PROBE_OPTIMIZER_STEPS', '1')))\n"
        "_mlevolve_probe_step_started = None\n"
        "_mlevolve_probe_steps_completed = 0\n"
        "def _mlevolve_probe_emit(event_type, **payload):\n"
        f"    if not {_PROBE_EVENT_PATH_VAR}:\n"
        "        return\n"
        "    record = {'event': event_type, 'monotonic': _mlevolve_time.monotonic(), **payload}\n"
        f"    with open({_PROBE_EVENT_PATH_VAR}, 'a', encoding='utf-8') as _probe_handle:\n"
        "        _probe_handle.write(_mlevolve_json.dumps(record, sort_keys=True) + '\\n')\n"
        "        _probe_handle.flush()\n"
        "def _mlevolve_install_optimizer_probe():\n"
        f"    if not {_PROBE_MODE_VAR}:\n"
        "        return\n"
        "    import torch as _mlevolve_torch\n"
        "    try:\n"
        "        if _mlevolve_torch.cuda.is_available():\n"
        "            _mlevolve_torch.cuda.reset_peak_memory_stats()\n"
        "    except Exception:\n"
        "        pass\n"
        "    _mlevolve_probe_emit('probe_started')\n"
        "    def _probe_pre_hook(_optimizer, _args, _kwargs):\n"
        "        global _mlevolve_probe_step_started\n"
        "        _mlevolve_probe_step_started = _mlevolve_time.monotonic()\n"
        "        _mlevolve_probe_emit('optimizer_step_started')\n"
        "    def _probe_post_hook(_optimizer, _args, _kwargs):\n"
        "        global _mlevolve_probe_steps_completed\n"
        "        if _mlevolve_torch.cuda.is_available():\n"
        "            _mlevolve_torch.cuda.synchronize()\n"
        "            allocated = int(_mlevolve_torch.cuda.max_memory_allocated())\n"
        "            reserved = int(_mlevolve_torch.cuda.max_memory_reserved())\n"
        "            total = int(_mlevolve_torch.cuda.get_device_properties(_mlevolve_torch.cuda.current_device()).total_memory)\n"
        "        else:\n"
        "            allocated = reserved = total = 0\n"
        "        elapsed_ms = None if _mlevolve_probe_step_started is None else (_mlevolve_time.monotonic() - _mlevolve_probe_step_started) * 1000.0\n"
        "        _mlevolve_probe_steps_completed += 1\n"
        "        _mlevolve_probe_emit('optimizer_step_completed', step=_mlevolve_probe_steps_completed, peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, memory_total_bytes=total, optimizer_step_time_ms=elapsed_ms)\n"
        f"        if _mlevolve_probe_steps_completed >= {_PROBE_OPTIMIZER_STEPS_VAR}:\n"
        "            raise SystemExit(0)\n"
        "    _optimizer_module = getattr(_mlevolve_torch.optim, 'optimizer', None)\n"
        "    _register_pre = getattr(_optimizer_module, 'register_optimizer_step_pre_hook', None)\n"
        "    _register_post = getattr(_optimizer_module, 'register_optimizer_step_post_hook', None)\n"
        "    if callable(_register_pre) and callable(_register_post):\n"
        "        _register_pre(_probe_pre_hook)\n"
        "        _register_post(_probe_post_hook)\n"
        "        return\n"
        "    _original_init = _mlevolve_torch.optim.Optimizer.__init__\n"
        "    def _probe_optimizer_init(self, *args, **kwargs):\n"
        "        _original_init(self, *args, **kwargs)\n"
        "        self.register_step_pre_hook(_probe_pre_hook)\n"
        "        self.register_step_post_hook(_probe_post_hook)\n"
        "    _mlevolve_torch.optim.Optimizer.__init__ = _probe_optimizer_init\n"
        "_mlevolve_install_optimizer_probe()\n"
    )
    helper_module = ast.parse(helper_source, filename=str(script_path))
    _prepend_after_module_preamble(module, helper_module.body)
    ast.fix_missing_locations(module)

    instrumented_path = instrumented_dir / f"{script_path.stem}_instrumented.py"
    instrumented_path.write_text(ast.unparse(module), encoding="utf-8")
    return InstrumentedScript(
        path=instrumented_path,
        had_batch_rewrite=transformer.batch_modified,
        precision_repair_count=precision_repair_count,
        batch_contract=batch_contract,
    )


def _base_script_env(
    batch_size_override: int | None = None,
    *,
    probe_mode: bool = False,
    probe_max_epochs: int | None = None,
    probe_max_train_batches: int | None = None,
    probe_event_path: Path | None = None,
    probe_optimizer_steps: int | None = None,
) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for key in (
        "MLEVOLVE_BATCH_SIZE_OVERRIDE",
        "MLEVOLVE_PROBE_MODE",
        "MLEVOLVE_PROBE_MAX_EPOCHS",
        "MLEVOLVE_PROBE_MAX_TRAIN_BATCHES",
        "MLEVOLVE_PROBE_EVENT_PATH",
        "MLEVOLVE_PROBE_OPTIMIZER_STEPS",
    ):
        env.pop(key, None)
    if batch_size_override is not None:
        env["MLEVOLVE_BATCH_SIZE_OVERRIDE"] = str(int(batch_size_override))
    if probe_mode:
        env["MLEVOLVE_PROBE_MODE"] = "1"
    if probe_max_epochs is not None:
        env["MLEVOLVE_PROBE_MAX_EPOCHS"] = str(max(1, int(probe_max_epochs)))
    if probe_max_train_batches is not None:
        env["MLEVOLVE_PROBE_MAX_TRAIN_BATCHES"] = str(max(1, int(probe_max_train_batches)))
    if probe_event_path is not None:
        env["MLEVOLVE_PROBE_EVENT_PATH"] = str(probe_event_path)
    if probe_optimizer_steps is not None:
        env["MLEVOLVE_PROBE_OPTIMIZER_STEPS"] = str(max(1, int(probe_optimizer_steps)))
    return env


def _resolved_batch_size(context: RunnerContext) -> int | None:
    raw_value = context.job.metadata.get("resolved_batch_size")
    if raw_value is None:
        return None
    return BatchResolution.resolved_batch_size(context.job)


def _max_epochs_override(context: RunnerContext) -> int | None:
    for value in (
        context.job.config.runner_kwargs.get("max_epochs"),
        context.job.config.max_epochs,
        context.job.max_epochs,
    ):
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _short_excerpt(text: str, *, limit: int = 1000) -> str | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


def _head_tail(text: str, *, limit: int = 1000) -> tuple[str | None, str | None]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None, None
    if len(cleaned) <= limit:
        return cleaned, cleaned
    return cleaned[:limit], cleaned[-limit:]


_TERMINAL_EXCEPTION_RE = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):\s*(?P<message>.*)$"
)


def _terminal_exception(stderr_text: str) -> tuple[str | None, str | None]:
    for raw_line in reversed(str(stderr_text or "").splitlines()):
        match = _TERMINAL_EXCEPTION_RE.match(raw_line.strip())
        if match:
            return match.group("type").rsplit(".", 1)[-1], match.group("message").strip()
    return None, None


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
    exception_type, exception_message = _terminal_exception(stderr_text)
    traceback_tail = "\n".join(str(stderr_text or "").splitlines()[-80:])
    terminal = f"{exception_type or ''}: {exception_message or ''}\n{traceback_tail}".lower()
    terminal_summary = f"{exception_type or ''}: {exception_message or ''}".lower()
    if timed_out:
        return "timeout", "probe subprocess timed out"
    if returncode == 0:
        return None, None
    if "cuda out of memory" in terminal or "out of memory" in terminal or "cublas_status_alloc_failed" in terminal:
        return "oom", "cuda out of memory"
    message = exception_message or _short_excerpt(traceback_tail, limit=400)
    if exception_type == "SyntaxError":
        return "syntax_error", message or "syntax error"
    if exception_type in {"ModuleNotFoundError", "ImportError"}:
        return "import_error", message or "import error"
    invalid_model_markers = (
        "not a valid model identifier",
        "not a local folder",
        "does not appear to have a file named",
        "unknown model",
        "invalid model",
        "model not found",
        "no pretrained weights exist",
    )
    if any(marker in terminal_summary for marker in invalid_model_markers):
        return "invalid_model", message or "invalid model"
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
    if any(marker in terminal_summary for marker in dtype_markers):
        return "dtype_error", message or "dtype error"
    if exception_type or "traceback" in terminal:
        return "script_exception", message or "script exception"
    if returncode not in (0, None):
        return "unknown", message or _short_excerpt(stderr_text or stdout_text, limit=400) or f"probe subprocess failed with code {returncode}"
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


def _stdout_indicates_training_progress(stdout_text: str) -> bool:
    lowered = stdout_text.lower()
    markers = (
        "final validation score",
        "training finished",
        "train loss",
        "val loss",
        "val auc",
        "epoch ",
    )
    return any(marker in lowered for marker in markers)


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
    probe_optimizer_steps: int = 1,
    step_timeout_seconds: int | None = None,
) -> ProbeSubprocessResult:
    stdout_path = working_dir / "working" / f"probe_stdout_bs_{batch_size}.log"
    stderr_path = working_dir / "working" / f"probe_stderr_bs_{batch_size}.log"
    event_path = working_dir / "working" / f"probe_events_bs_{batch_size}.jsonl"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.unlink(missing_ok=True)
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
                probe_event_path=event_path,
                probe_optimizer_steps=probe_optimizer_steps,
            ),
        )

        samples: list[GpuTelemetrySample] = []
        deadline = time.time() + max(1, timeout_seconds)
        timeout_phase = "startup"
        optimizer_step_seen = False
        while time.time() < deadline:
            sample = sampler.sample()
            if sample is not None:
                samples.append(sample)
            if proc.poll() is not None:
                break
            if not optimizer_step_seen and event_path.exists():
                try:
                    optimizer_step_seen = any(
                        json.loads(line).get("event") == "optimizer_step_started"
                        for line in event_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                except (OSError, json.JSONDecodeError):
                    optimizer_step_seen = False
                if optimizer_step_seen:
                    timeout_phase = "optimizer_step"
                    deadline = time.time() + max(1, int(step_timeout_seconds or timeout_seconds))
            time.sleep(max(0.1, poll_interval_seconds))

        timed_out = False
        if proc.poll() is None:
            timed_out = True
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    completion_event = None
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "optimizer_step_completed":
                completion_event = event
    fits = proc.returncode == 0 and completion_event is not None and not timed_out
    return ProbeSubprocessResult(
        fits=fits,
        samples=samples,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        returncode=proc.returncode,
        timed_out=timed_out,
        timeout_phase=timeout_phase if timed_out else None,
        completion_event=completion_event,
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

    contract = instrumented.batch_contract or TrainingBatchContract(unsupported_reason="batch contract unavailable")
    if not instrumented.had_batch_rewrite or not contract.supported:
        return BatchProbeTrialResult(
            fits=False,
            peak_vram_mb=None,
            memory_total_mb=None,
            avg_step_time_ms=None,
            message=contract.unsupported_reason or "no safe training batch-size knob found",
            failure_kind="probe_unsupported",
            diagnostic=FailureDiagnostic(
                kind="probe_unsupported",
                phase="instrumentation",
                exception_message=contract.unsupported_reason or "no safe training batch-size knob found",
                batch_size=int(batch_size),
            ),
        )

    timeout_seconds = int(kwargs.get("probe_startup_timeout_seconds", kwargs.get("probe_timeout_seconds", max(20, warmup_steps + measure_steps))))
    step_timeout_seconds = int(kwargs.get("probe_step_timeout_seconds", 30))
    poll_interval_seconds = float(kwargs.get("probe_poll_interval_seconds", 0.5))
    probe_max_epochs = max(1, int(kwargs.get("probe_max_epochs", 2)))
    probe_max_train_batches = max(1, int(kwargs.get("probe_max_train_batches", 3)))
    probe_optimizer_steps = max(1, int(kwargs.get("probe_optimizer_steps", 1)))
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
        probe_optimizer_steps=probe_optimizer_steps,
        step_timeout_seconds=step_timeout_seconds,
    )
    failure_kind, failure_reason = _parse_batch_size_failure(
        probe_result.stderr_text,
        stdout_text=probe_result.stdout_text,
        returncode=probe_result.returncode,
        timed_out=probe_result.timed_out,
    )

    completion = probe_result.completion_event or {}
    allocator_peak_mb = int(completion.get("peak_allocated_bytes", 0) / (1024 * 1024)) or None
    sampled_peak_mb = max((sample.memory_used_mb for sample in probe_result.samples), default=None)
    peak_vram_mb = max(value for value in (allocator_peak_mb, sampled_peak_mb) if value is not None) if any(
        value is not None for value in (allocator_peak_mb, sampled_peak_mb)
    ) else None
    event_total_mb = int(completion.get("memory_total_bytes", 0) / (1024 * 1024)) or None
    memory_total_mb = event_total_mb or max((sample.memory_total_mb for sample in probe_result.samples), default=None)
    stdout_head, stdout_tail = _head_tail(probe_result.stdout_text)
    stderr_head, stderr_tail = _head_tail(probe_result.stderr_text)
    exception_type, exception_message = _terminal_exception(probe_result.stderr_text)
    completed_without_hook = (
        probe_result.returncode == 0
        and probe_result.completion_event is None
        and not probe_result.timed_out
        and failure_kind is None
        and peak_vram_mb is not None
        and _stdout_indicates_training_progress(probe_result.stdout_text)
    )
    fits = bool(probe_result.fits or completed_without_hook)
    if probe_result.returncode == 0 and probe_result.completion_event is None and not probe_result.timed_out and not completed_without_hook:
        failure_kind = "probe_incomplete"
        failure_reason = "probe exited without completing an optimizer step"
    if failure_kind is not None or failure_reason is not None:
        fits = False
    phase = probe_result.timeout_phase or ("optimizer_step" if completion else "subprocess")
    diagnostic = None
    if not fits:
        diagnostic = FailureDiagnostic(
            kind=failure_kind or "unknown",
            phase=phase,
            exception_type=exception_type,
            exception_message=exception_message or failure_reason,
            batch_size=int(batch_size),
            returncode=probe_result.returncode,
            timed_out=probe_result.timed_out,
            stdout_head=stdout_head,
            stdout_tail=stdout_tail,
            stderr_head=stderr_head,
            stderr_tail=stderr_tail,
        )
    return BatchProbeTrialResult(
        fits=bool(fits),
        peak_vram_mb=peak_vram_mb,
        memory_total_mb=memory_total_mb,
        avg_step_time_ms=_as_float(completion.get("optimizer_step_time_ms")),
        message=failure_reason
        or (
            "optimizer step completed"
            if probe_result.fits
            else "probe process completed with training output; using sampled VRAM telemetry"
            if completed_without_hook
            else _short_excerpt(probe_result.stderr_text, limit=400)
        ),
        failure_kind=failure_kind,
        returncode=probe_result.returncode,
        stdout_excerpt=stdout_tail,
        stderr_excerpt=stderr_tail,
        diagnostic=diagnostic,
        probe_completed=completion.get("event") == "optimizer_step_completed",
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
    max_epochs_override = _max_epochs_override(context)

    start_time = time.time()
    proc = subprocess.Popen(
        [python_executable, str(executable_script)],
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_base_script_env(batch_size_override=batch_size_override, probe_max_epochs=max_epochs_override),
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
    if max_epochs_override is not None:
        instrumentation["max_epochs_override"] = max_epochs_override
    failure_diagnostic = None
    if exc_type is not None:
        stdout_head, stdout_tail = _head_tail(stdout)
        stderr_head, stderr_tail = _head_tail(stderr)
        failure_diagnostic = FailureDiagnostic(
            kind="execution_timeout" if exc_type == "TimeoutError" else "candidate_exception",
            phase="execution",
            exception_type=exc_type,
            exception_message=str(exc_info.get("message") or ("execution timed out" if exc_type == "TimeoutError" else "candidate failed")),
            returncode=proc.returncode,
            timed_out=exc_type == "TimeoutError",
            stdout_head=stdout_head,
            stdout_tail=stdout_tail,
            stderr_head=stderr_head,
            stderr_tail=stderr_tail,
        )
        instrumentation["failure_diagnostic"] = failure_diagnostic.to_dict()
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
        "failure_diagnostic": failure_diagnostic.to_dict() if failure_diagnostic else None,
        "success": bool(early_stop_reason is None and exc_type is None and proc.returncode == 0),
        "outcome": (
            "execution_timeout"
            if exc_type == "TimeoutError"
            else "candidate_exception"
            if exc_type is not None
            else "early_stopped"
            if early_stop_reason is not None
            else "success"
        ),
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
                "max_epochs_override": max_epochs_override,
                "phase_timing_available": bool(phase_timings),
            },
        )
    return {
        "reason": "mlevolve script early-stopped" if early_stop_reason is not None else "mlevolve script executed",
        "execution_result_path": str(result_path),
        "candidate_returncode": proc.returncode,
        "candidate_exc_type": exc_type,
        "success": result["success"],
        "outcome": result["outcome"],
        "batch_size_override": batch_size_override,
        "phase_timings": phase_timings,
        "instrumentation": instrumentation,
    }


def run_mlevolve_model_family_probe_job(context: RunnerContext) -> dict[str, Any]:
    """Complete after batch-probe preflight has written the branch profile."""
    return {
        "kind": "mlevolve_model_family_probe",
        "branch_name": context.job.metadata.get("branch_name") or context.job.metadata.get("model_family"),
        "branch_profile_key": context.job.batch_probe.profile_namespace,
        "model_family": context.job.metadata.get("model_family"),
        "profile_key": context.job.metadata.get("batch_probe_key"),
        "resolved_batch_size": context.job.metadata.get("resolved_batch_size"),
    }
