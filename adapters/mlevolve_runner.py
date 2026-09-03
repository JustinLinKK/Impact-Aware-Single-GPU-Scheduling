"""Scheduler runner for raw MLEvolve-generated Python scripts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

import humanize

from engine.script_introspection import (
    GRADIENT_ACCUMULATION_PARAM_NAMES,
    LEARNING_RATE_PARAM_NAMES,
    SCHEDULER_TOTAL_STEPS_PARAM_NAMES,
    WARMUP_STEPS_PARAM_NAMES,
    analyze_training_batch_contract,
)

from ..execution.runner_protocol import RunnerContext
from ..scheduler.telemetry import GpuTelemetrySample, NvidiaSmiTelemetrySampler
from ..domain import (
    BatchProbeTrialResult,
    BatchResolution,
    BatchSizeObservation,
    ProgressSnapshot,
    build_batch_probe_shape_signature,
    build_batch_size_observation_key,
    utc_now,
)
from ..profiling.runtime_probe import estimate_total_runtime_from_epoch_1

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
_GRADIENT_ACCUMULATION_OVERRIDE_VAR = "_MLEVOLVE_GRADIENT_ACCUMULATION_OVERRIDE"
_LEARNING_RATE_OVERRIDE_VAR = "_MLEVOLVE_LEARNING_RATE_OVERRIDE"
_WARMUP_STEPS_OVERRIDE_VAR = "_MLEVOLVE_WARMUP_STEPS_OVERRIDE"
_SCHEDULER_TOTAL_STEPS_OVERRIDE_VAR = "_MLEVOLVE_SCHEDULER_TOTAL_STEPS_OVERRIDE"


def _request_process_stop(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        proc.terminate()


@dataclass(slots=True)
class InstrumentedScript:
    path: Path
    had_batch_rewrite: bool


def load_raw_file(path: str) -> bytes:
    """Cache loader for scheduler-managed raw script jobs."""
    return Path(path).read_bytes()


def _parse_exception(
    stderr_text: str, working_dir: Path, script_path: Path
) -> tuple[str, dict[str, Any], list[tuple[str, int, str, str]]]:
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
            filename_short = os.path.basename(
                filename_short.replace(str(working_dir), "")
            )
            exc_stack.append((filename_short, int(line_num_str), func_name, ""))
        except Exception:
            continue

    for line in reversed(stderr_lines):
        line = line.strip()
        if (
            line
            and not line.startswith("File")
            and not line.startswith("Traceback")
            and ":" in line
        ):
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
        test=ast.Compare(
            left=override_name,
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        body=ast.Call(
            func=ast.Name(id="int", ctx=ast.Load()), args=[override_name], keywords=[]
        ),
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
                ast.Compare(
                    left=override_name,
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
            ],
        ),
        body=ast.Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="int", ctx=ast.Load()),
                    args=[override_name],
                    keywords=[],
                ),
                original_value,
            ],
            keywords=[],
        ),
        orelse=original_value,
    )


def _override_training_expr(
    original_value: ast.expr, override_var: str, cast_name: str
) -> ast.expr:
    override_name = ast.Name(id=override_var, ctx=ast.Load())
    return ast.IfExp(
        test=ast.Compare(
            left=override_name,
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        body=ast.Call(
            func=ast.Name(id=cast_name, ctx=ast.Load()),
            args=[override_name],
            keywords=[],
        ),
        orelse=original_value,
    )


class _BatchOverrideTransformer(ast.NodeTransformer):
    """Apply probe overrides only where static analysis proves training use."""

    def __init__(self, training_sites: dict[tuple[int, int], str]) -> None:
        self.training_sites = training_sites
        self.modified = False
        self.had_batch_rewrite = False

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name in _EPOCH_COUNT_NAMES:
                node.value = _override_epoch_expr(node.value)
                self.modified = True
            elif target_name in GRADIENT_ACCUMULATION_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _GRADIENT_ACCUMULATION_OVERRIDE_VAR, "int"
                )
                self.modified = True
            elif target_name in LEARNING_RATE_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _LEARNING_RATE_OVERRIDE_VAR, "float"
                )
                self.modified = True
            elif target_name in WARMUP_STEPS_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _WARMUP_STEPS_OVERRIDE_VAR, "int"
                )
                self.modified = True
            elif target_name in SCHEDULER_TOTAL_STEPS_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _SCHEDULER_TOTAL_STEPS_OVERRIDE_VAR, "int"
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
            elif target_name in GRADIENT_ACCUMULATION_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _GRADIENT_ACCUMULATION_OVERRIDE_VAR, "int"
                )
                self.modified = True
            elif target_name in LEARNING_RATE_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _LEARNING_RATE_OVERRIDE_VAR, "float"
                )
                self.modified = True
            elif target_name in WARMUP_STEPS_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _WARMUP_STEPS_OVERRIDE_VAR, "int"
                )
                self.modified = True
            elif target_name in SCHEDULER_TOTAL_STEPS_PARAM_NAMES:
                node.value = _override_training_expr(
                    node.value, _SCHEDULER_TOTAL_STEPS_OVERRIDE_VAR, "int"
                )
                self.modified = True
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        argument = self.training_sites.get((node.lineno, node.col_offset))
        if argument is None:
            return node
        kind, _, name_or_index = argument.partition(":")
        if kind == "keyword":
            for keyword in node.keywords:
                if keyword.arg == name_or_index and keyword.value is not None:
                    keyword.value = _override_batch_expr(keyword.value)
                    self.modified = True
                    self.had_batch_rewrite = True
                    break
        elif kind == "positional" and name_or_index.isdigit():
            index = int(name_or_index)
            if index < len(node.args):
                node.args[index] = _override_batch_expr(node.args[index])
                self.modified = True
                self.had_batch_rewrite = True
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


def _materialize_instrumented_script(
    script_path: Path, working_dir: Path
) -> InstrumentedScript:
    source = script_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(script_path))
    except SyntaxError:
        return InstrumentedScript(path=script_path, had_batch_rewrite=False)

    contract = analyze_training_batch_contract(source)
    training_sites = {
        (site.lineno, site.col_offset): site.argument for site in contract.train_sites
    }
    transformer = _BatchOverrideTransformer(training_sites)
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    if not transformer.modified:
        return InstrumentedScript(path=script_path, had_batch_rewrite=False)

    helper_module = ast.parse(
        "import os\n"
        f"{_BATCH_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_BATCH_SIZE_OVERRIDE')\n"
        f"{_PROBE_MODE_VAR} = os.environ.get('MLEVOLVE_PROBE_MODE') == '1'\n"
        f"{_EPOCH_OVERRIDE_VAR} = int(os.environ['MLEVOLVE_PROBE_MAX_EPOCHS']) if os.environ.get('MLEVOLVE_PROBE_MAX_EPOCHS') else None\n"
        f"{_GRADIENT_ACCUMULATION_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_GRADIENT_ACCUMULATION_OVERRIDE')\n"
        f"{_LEARNING_RATE_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_LEARNING_RATE_OVERRIDE')\n"
        f"{_WARMUP_STEPS_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_WARMUP_STEPS_OVERRIDE')\n"
        f"{_SCHEDULER_TOTAL_STEPS_OVERRIDE_VAR} = os.environ.get('MLEVOLVE_SCHEDULER_TOTAL_STEPS_OVERRIDE')\n"
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
    return InstrumentedScript(
        path=instrumented_path,
        had_batch_rewrite=transformer.had_batch_rewrite,
    )


def _base_script_env(
    batch_size_override: int | None = None,
    *,
    gradient_accumulation_override: int | None = None,
    learning_rate_override: float | None = None,
    warmup_steps_override: int | None = None,
    scheduler_total_steps_override: int | None = None,
    probe_mode: bool = False,
    probe_max_epochs: int | None = None,
    probe_max_train_batches: int | None = None,
) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if batch_size_override is not None:
        env["MLEVOLVE_BATCH_SIZE_OVERRIDE"] = str(int(batch_size_override))
    if gradient_accumulation_override is not None:
        env["MLEVOLVE_GRADIENT_ACCUMULATION_OVERRIDE"] = str(
            max(1, int(gradient_accumulation_override))
        )
    if learning_rate_override is not None:
        env["MLEVOLVE_LEARNING_RATE_OVERRIDE"] = str(float(learning_rate_override))
    if warmup_steps_override is not None:
        env["MLEVOLVE_WARMUP_STEPS_OVERRIDE"] = str(max(0, int(warmup_steps_override)))
    if scheduler_total_steps_override is not None:
        env["MLEVOLVE_SCHEDULER_TOTAL_STEPS_OVERRIDE"] = str(
            max(1, int(scheduler_total_steps_override))
        )
    if probe_mode:
        env["MLEVOLVE_PROBE_MODE"] = "1"
    if probe_max_epochs is not None:
        env["MLEVOLVE_PROBE_MAX_EPOCHS"] = str(max(1, int(probe_max_epochs)))
    if probe_max_train_batches is not None:
        env["MLEVOLVE_PROBE_MAX_TRAIN_BATCHES"] = str(
            max(1, int(probe_max_train_batches))
        )
    return env


def _resolved_batch_size(context: RunnerContext) -> int | None:
    raw_value = context.job.metadata.get("resolved_batch_size")
    if raw_value is None:
        return None
    return BatchResolution.resolved_batch_size(context.job)


def _parse_batch_size_failure(stderr_text: str) -> str | None:
    lowered = stderr_text.lower()
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return "cuda out of memory"
    return None


def _metric_direction(metric_name: str | None, explicit: Any = None) -> bool | None:
    if explicit is not None:
        return bool(explicit)
    lowered = str(metric_name or "").lower()
    if any(token in lowered for token in ("loss", "error", "rmse", "mae", "mse")):
        return False
    if any(
        token in lowered
        for token in ("score", "accuracy", "auc", "f1", "precision", "recall")
    ):
        return True
    return None


def _parse_training_quality(
    stdout: str, *, planned_epochs: int | None, metric_maximize: Any = None
) -> dict[str, Any]:
    """Parse structured epoch markers first, then common validation log formats."""
    curve: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if "MLEVOLVE_EPOCH_METRIC" in line:
            payload_text = line.split("MLEVOLVE_EPOCH_METRIC", 1)[1].lstrip(" :")
            try:
                payload = json.loads(payload_text)
                epoch = int(payload["epoch"])
                metric = float(payload["metric"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            curve.append(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "metric_name": str(
                        payload.get("metric_name") or "validation_metric"
                    ),
                }
            )
            continue
        epoch_match = re.search(
            r"\bepoch\s*[:#]?\s*(\d+)(?:\s*/\s*\d+)?", line, re.IGNORECASE
        )
        metric_match = re.search(
            r"\b((?:val|valid|validation)[_\s-]*(?:score|metric|loss|accuracy|auc|f1|rmse|mae|mse))\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            line,
            re.IGNORECASE,
        )
        if epoch_match and metric_match:
            curve.append(
                {
                    "epoch": int(epoch_match.group(1)),
                    "metric": float(metric_match.group(2)),
                    "metric_name": metric_match.group(1).strip().replace(" ", "_"),
                }
            )

    final_match = None
    for final_match in re.finditer(
        r"Final Validation Score\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        stdout,
        re.IGNORECASE,
    ):
        pass
    explicit_best_epoch = re.search(
        r"\bbest[_\s-]*epoch\s*[:=]\s*(\d+)", stdout, re.IGNORECASE
    )
    completed_epochs = max(
        (int(point["epoch"]) for point in curve),
        default=0,
    )
    metric_name = curve[-1].get("metric_name") if curve else None
    direction = _metric_direction(metric_name, metric_maximize)
    best_metric = None
    best_epoch = int(explicit_best_epoch.group(1)) if explicit_best_epoch else None
    if curve:
        selector = min if direction is False else max
        best_point = selector(curve, key=lambda point: float(point["metric"]))
        best_metric = float(best_point["metric"])
        best_epoch = best_epoch or int(best_point["epoch"])
    if final_match is not None:
        best_metric = float(final_match.group(1))
        metric_name = "Final Validation Score"
        direction = _metric_direction(metric_name, metric_maximize)
        if best_epoch is None:
            best_epoch = completed_epochs or None
    if completed_epochs == 0 and best_epoch is not None:
        completed_epochs = best_epoch
    if planned_epochs is not None:
        completed_epochs = min(int(planned_epochs), completed_epochs)
    return {
        "planned_epochs": planned_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "metric_maximize": direction,
        "convergence_curve": curve,
    }


def _record_training_quality_observation(
    context: RunnerContext, quality: dict[str, Any], *, exec_time: float
) -> None:
    physical_batch = _resolved_batch_size(context)
    if physical_batch is None:
        return
    job = context.job
    backend_name = str(job.metadata.get("placement_backend") or "exclusive")
    hardware_key = context.store.hardware_key()
    model_key = str(job.batch_probe.model_key or job.baseline_model_id)
    shape_signature = build_batch_probe_shape_signature(job)
    existing = context.store.get_batch_size_observation(
        model_key=model_key,
        shape_signature=shape_signature,
        hardware_key=hardware_key,
        backend_name=backend_name,
        batch_size=physical_batch,
    )
    metadata = dict(existing.metadata if existing else {})
    samples = [
        float(value)
        for value in metadata.get("quality_samples") or []
        if isinstance(value, (int, float))
    ]
    current_metric = quality.get("best_metric")
    seed_samples = [
        dict(item)
        for item in metadata.get("quality_samples_by_seed") or []
        if isinstance(item, dict) and item.get("metric") is not None
    ]
    if current_metric is not None:
        samples.append(float(current_metric))
        seed_samples.append(
            {
                "seed": job.metadata.get("random_seed"),
                "metric": float(current_metric),
                "job_id": job.job_id,
            }
        )
    seed_metrics: dict[int, list[float]] = {}
    unlabelled_metrics: list[float] = []
    for item in seed_samples:
        try:
            metric = float(item["metric"])
        except (KeyError, TypeError, ValueError):
            continue
        if item.get("seed") is None:
            unlabelled_metrics.append(metric)
            continue
        try:
            seed = int(item["seed"])
        except (TypeError, ValueError):
            unlabelled_metrics.append(metric)
            continue
        seed_metrics.setdefault(seed, []).append(metric)
    per_seed_metrics = [statistics.fmean(values) for values in seed_metrics.values()]
    variance_values = (
        per_seed_metrics
        if len(per_seed_metrics) > 1
        else (unlabelled_metrics if not per_seed_metrics else [])
    )
    metadata.update(
        {
            "quality_samples": samples[-32:],
            "quality_samples_by_seed": seed_samples[-32:],
            "seconds_per_epoch": (
                float(exec_time) / max(1, int(quality.get("completed_epochs") or 1))
            ),
            "training_parameter_resolution": dict(
                job.metadata.get("training_parameter_resolution") or {}
            ),
            "early_stopping_required": bool(
                (job.metadata.get("training_quality_contract") or {}).get(
                    "early_stopping_required"
                )
            ),
            "has_validation_early_stopping": bool(
                job.metadata.get("has_validation_early_stopping")
            ),
        }
    )
    maximize = quality.get("metric_maximize")
    best_metric = quality.get("best_metric")
    best_epoch = quality.get("best_epoch")
    if existing and existing.best_metric is not None:
        if best_metric is None:
            best_metric, best_epoch = existing.best_metric, existing.best_epoch
        elif maximize is True and existing.best_metric > best_metric:
            best_metric, best_epoch = existing.best_metric, existing.best_epoch
        elif maximize is False and existing.best_metric < best_metric:
            best_metric, best_epoch = existing.best_metric, existing.best_epoch

    context.store.upsert_batch_size_observation(
        BatchSizeObservation(
            observation_key=build_batch_size_observation_key(
                model_key,
                shape_signature,
                hardware_key,
                backend_name,
                physical_batch,
            ),
            model_key=model_key,
            shape_signature=shape_signature,
            hardware_key=hardware_key,
            backend_name=backend_name,
            batch_param_name=BatchResolution.param_name(job),
            batch_size=physical_batch,
            effective_batch_size=job.metadata.get("resolved_effective_batch_size"),
            peak_vram_mb=existing.peak_vram_mb if existing else None,
            avg_vram_mb=existing.avg_vram_mb if existing else None,
            memory_total_mb=existing.memory_total_mb if existing else None,
            avg_step_time_ms=existing.avg_step_time_ms if existing else None,
            avg_gpu_utilization=existing.avg_gpu_utilization if existing else None,
            avg_memory_utilization=(
                existing.avg_memory_utilization if existing else None
            ),
            best_metric=best_metric,
            metric_name=quality.get("metric_name")
            or (existing.metric_name if existing else None),
            metric_maximize=maximize
            if maximize is not None
            else (existing.metric_maximize if existing else None),
            best_epoch=best_epoch,
            planned_epochs=quality.get("planned_epochs"),
            completed_epochs=quality.get("completed_epochs"),
            convergence_curve=list(quality.get("convergence_curve") or []),
            seed_variance=(
                statistics.pvariance(variance_values)
                if len(variance_values) > 1
                else None
            ),
            observations=(existing.observations + 1) if existing else 1,
            last_job_id=job.job_id,
            metadata=metadata,
        )
    )


def _record_live_epoch_metric(
    context: RunnerContext, line: str, *, started_at: float
) -> None:
    """Persist a duration prediction as soon as a generated script ends an epoch."""
    if "MLEVOLVE_EPOCH_METRIC" not in line:
        return
    payload_text = line.split("MLEVOLVE_EPOCH_METRIC", 1)[1].lstrip(" :")
    try:
        payload = json.loads(payload_text)
        epoch = max(1, int(payload["epoch"]))
        metric = float(payload["metric"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return

    elapsed = max(0.001, time.time() - started_at)
    total_epochs = (
        context.job.max_epochs
        or context.job.config.max_epochs
        or context.job.metadata.get("proposed_epochs")
        or context.job.metadata.get("planned_epochs")
    )
    try:
        total_epochs = max(epoch, int(total_epochs)) if total_epochs is not None else epoch
    except (TypeError, ValueError):
        total_epochs = epoch
    backend_name = str(context.job.metadata.get("placement_backend") or "exclusive")
    existing = context.get_runtime_profile(backend_name=backend_name)
    estimate = (
        float(existing.estimated_total_runtime_seconds)
        if existing is not None and existing.estimated_total_runtime_seconds is not None
        else None
    )
    if estimate is None and context.job.runtime_probe.enabled:
        estimate = estimate_total_runtime_from_epoch_1(
            startup_seconds=0.0,
            epoch_1_seconds=elapsed / float(epoch),
            total_epochs=int(total_epochs),
        )
        context.upsert_runtime_profile(
            backend_name=backend_name,
            strategy="epoch_1",
            startup_seconds=0.0,
            epoch_1_seconds=elapsed / float(epoch),
            steps_per_epoch=None,
            avg_step_time_ms=None,
            estimated_total_runtime_seconds=estimate,
            confidence=0.75,
            source="mlevolve_stdout_epoch_marker",
            observations=1,
            metadata={"epoch": epoch, "metric_name": payload.get("metric_name")},
        )

    remaining = max(0.0, float(estimate) - elapsed) if estimate is not None else None
    heartbeat_at = utc_now()
    context.store.update_job(
        context.job.job_id,
        last_heartbeat_at=heartbeat_at,
        metadata_updates={
            "last_completed_epoch": epoch,
            "runtime_estimated_total_runtime_seconds": estimate,
            "runtime_remaining_runtime_seconds": remaining,
            "runtime_profile_strategy": "epoch_1" if estimate is not None else None,
            "runtime_profile_confidence": 0.75 if estimate is not None else None,
        },
    )
    context.control_hook.control_plane.write_heartbeat(
        ProgressSnapshot(
            job_id=context.job.job_id,
            epoch=epoch,
            global_step=0,
            phase="train",
            metrics={str(payload.get("metric_name") or "validation_metric"): metric},
            last_safe_point="epoch",
            message="MLEVOLVE_EPOCH_METRIC",
            estimated_total_runtime_seconds=estimate,
            remaining_runtime_seconds=remaining,
            heartbeat_at=heartbeat_at,
        )
    )


def _stream_script_process(
    proc: subprocess.Popen[str],
    context: RunnerContext,
    *,
    started_at: float,
    timeout: int | None,
) -> tuple[str, str, bool]:
    """Stream stdout/stderr so an epoch-one profile exists before completion."""
    lines: queue.Queue[tuple[str, str]] = queue.Queue()

    def pump(name: str, stream: Any) -> None:
        if stream is None:
            return
        try:
            for text in iter(stream.readline, ""):
                lines.put((name, text))
        finally:
            stream.close()

    readers = [
        threading.Thread(target=pump, args=(name, stream), daemon=True)
        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    for reader in readers:
        reader.start()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    timed_out = False
    deadline = started_at + float(timeout) if timeout is not None else None
    termination_deadline: float | None = None
    while proc.poll() is None or any(reader.is_alive() for reader in readers):
        remaining = (deadline - time.time()) if deadline is not None else None
        if (
            remaining is not None
            and remaining <= 0
            and proc.poll() is None
            and termination_deadline is None
        ):
            timed_out = True
            _request_process_stop(proc)
            deadline = time.time() + 2.0
            termination_deadline = deadline
            remaining = deadline - time.time()
        elif remaining is not None and remaining <= 0 and proc.poll() is None:
            proc.kill()
            deadline = time.time() + 0.1
            remaining = deadline - time.time()
        try:
            name, text = lines.get(
                timeout=max(
                    0.01,
                    min(0.1, remaining if remaining is not None and remaining > 0 else 0.1),
                )
            )
        except queue.Empty:
            continue
        if name == "stdout":
            stdout_lines.append(text)
            _record_live_epoch_metric(context, text, started_at=started_at)
        else:
            stderr_lines.append(text)

    for reader in readers:
        reader.join(timeout=0.2)
    while not lines.empty():
        name, text = lines.get_nowait()
        if name == "stdout":
            stdout_lines.append(text)
            _record_live_epoch_metric(context, text, started_at=started_at)
        else:
            stderr_lines.append(text)
    try:
        proc.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    return "".join(stdout_lines), "".join(stderr_lines), timed_out


def _calibrate_runtime_profile(
    context: RunnerContext, *, exec_time: float, training_summary: dict[str, Any]
) -> None:
    """Replace the first-epoch extrapolation with the observed wall-clock runtime."""
    if not context.job.runtime_probe.enabled:
        return
    backend_name = str(context.job.metadata.get("placement_backend") or "exclusive")
    existing = context.get_runtime_profile(backend_name=backend_name)
    context.upsert_runtime_profile(
        backend_name=backend_name,
        strategy="epoch_1",
        startup_seconds=(existing.startup_seconds if existing is not None else None),
        epoch_1_seconds=(existing.epoch_1_seconds if existing is not None else None),
        steps_per_epoch=(existing.steps_per_epoch if existing is not None else None),
        avg_step_time_ms=(existing.avg_step_time_ms if existing is not None else None),
        estimated_total_runtime_seconds=max(0.0, float(exec_time)),
        confidence=0.95,
        source="mlevolve_completed_wall_clock",
        observations=(int(existing.observations) + 1) if existing is not None else 1,
        metadata={
            "completed_epochs": training_summary.get("completed_epochs"),
            "best_epoch": training_summary.get("best_epoch"),
        },
    )
    context.store.update_job(
        context.job.job_id,
        metadata_updates={
            "runtime_estimated_total_runtime_seconds": max(0.0, float(exec_time)),
            "runtime_remaining_runtime_seconds": 0.0,
            "runtime_profile_strategy": "completed_wall_clock",
            "runtime_profile_confidence": 0.95,
        },
    )


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

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
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
                _request_process_stop(proc)
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
            fits = True

    stdout_text = (
        stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    )
    stderr_text = (
        stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    )
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

    timeout_seconds = int(
        kwargs.get("probe_timeout_seconds", max(20, warmup_steps + measure_steps))
    )
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
    avg_vram_mb = (
        sum(sample.memory_used_mb for sample in samples) / len(samples)
        if samples
        else None
    )
    memory_total_mb = max((sample.memory_total_mb for sample in samples), default=None)
    elapsed_ms = (time.time() - started_at) * 1000.0
    return BatchProbeTrialResult(
        fits=bool(fits),
        peak_vram_mb=peak_vram_mb,
        avg_vram_mb=avg_vram_mb,
        memory_total_mb=memory_total_mb,
        avg_step_time_ms=elapsed_ms / max(1, len(samples)) if samples else None,
        message=failure_reason
        or ("probe window completed" if fits else stderr_text.strip()[:400]),
    )


def run_mlevolve_script_job(context: RunnerContext) -> dict[str, Any]:
    """Run one generated MLEvolve script and persist an ExecutionResult payload."""
    kwargs = context.job.config.runner_kwargs
    script_path = Path(kwargs["script_path"]).resolve()
    working_dir = Path(kwargs["working_dir"]).resolve()
    result_path = Path(kwargs["result_path"]).resolve()
    configured_timeout = kwargs.get("timeout")
    timeout = int(configured_timeout) if configured_timeout is not None else None
    python_executable = context.job.config.python_executable or sys.executable

    instrumented = _materialize_instrumented_script(script_path, working_dir)
    executable_script = instrumented.path
    batch_size_override = (
        _resolved_batch_size(context) if instrumented.had_batch_rewrite else None
    )
    parameter_resolution = dict(
        context.job.metadata.get("training_parameter_resolution") or {}
    )

    start_time = time.time()
    proc = subprocess.Popen(
        [python_executable, "-u", str(executable_script)],
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_base_script_env(
            batch_size_override=batch_size_override,
            gradient_accumulation_override=parameter_resolution.get(
                "gradient_accumulation_steps"
            ),
            learning_rate_override=parameter_resolution.get("learning_rate"),
            warmup_steps_override=parameter_resolution.get("warmup_steps"),
            scheduler_total_steps_override=parameter_resolution.get(
                "scheduler_total_steps"
            ),
        ),
    )

    exc_type: str | None = None
    exc_info: dict[str, Any] = {}
    exc_stack: list[tuple[str, int, str, str]] = []
    stdout = ""
    stderr = ""

    stdout, stderr, timed_out = _stream_script_process(
        proc, context, started_at=start_time, timeout=timeout
    )
    exec_time = time.time() - start_time
    if timed_out:
        exc_type = "TimeoutError"
    elif proc.returncode != 0:
        exc_type, exc_info, exc_stack = _parse_exception(
            stderr, working_dir, executable_script
        )

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
        output.append(
            f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(timeout)}"
        )
    elif timeout is None:
        output.append(
            f"Execution time: {humanize.naturaldelta(exec_time)} seconds (no execution time limit configured)."
        )
    else:
        output.append(
            f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(timeout)})."
        )

    planned_epochs = context.job.max_epochs or context.job.config.max_epochs
    training_summary = _parse_training_quality(
        stdout,
        planned_epochs=int(planned_epochs) if planned_epochs is not None else None,
        metric_maximize=context.job.metadata.get("metric_maximize"),
    )
    try:
        if exc_type is None:
            _calibrate_runtime_profile(
                context, exec_time=exec_time, training_summary=training_summary
            )
        context.store.update_job(
            context.job.job_id,
            metadata_updates={
                "planned_epochs": training_summary.get("planned_epochs"),
                "completed_epochs": training_summary.get("completed_epochs"),
                "last_completed_epoch": training_summary.get("completed_epochs"),
                "best_epoch": training_summary.get("best_epoch"),
                "best_metric": training_summary.get("best_metric"),
                "convergence_curve": training_summary.get("convergence_curve"),
            },
        )
        _record_training_quality_observation(
            context, training_summary, exec_time=exec_time
        )
    except Exception:
        # Quality evidence must never turn a completed training run into a
        # scheduler failure; the execution payload still carries the summary.
        pass

    result = {
        "term_out": output,
        "exec_time": exec_time,
        "exc_type": exc_type,
        "exc_info": exc_info,
        "exc_stack": exc_stack,
        "training_summary": training_summary,
    }
    _write_json_atomic(result_path, result)
    return {
        "reason": "mlevolve script executed",
        "execution_result_path": str(result_path),
        "candidate_returncode": proc.returncode,
        "candidate_exc_type": exc_type,
        "batch_size_override": batch_size_override,
        "training_parameter_resolution": parameter_resolution,
        "training_summary": training_summary,
    }
