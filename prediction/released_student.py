"""Verified 17-tensor paper-release student, used as a CPU ranking signal.

The six outputs already have physical units. NVML busy is kernel-active time,
not SM occupancy or FLOP utilization. Memory outputs never replace branch
profiles or live memory admission. No training labels are read here.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from .job_contract import prediction_metadata
from .released_artifacts import digest, hardware_matches, load_verified_student_files
from .released_outputs import deployment_prediction, VRAM_BUCKET_BYTES


def worker_resident_mib(pid):
    """Linux child RSS, not virtual address space reserved by PyTorch libraries."""
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    except (OSError, ValueError, IndexError):
        return None

INPUT_NAMES = ("x_cont", "op_exact_id", "phase_id", "dtype_id", "node_flags", "edge_index",
               "edge_cont", "edge_role_id", "edge_flags", "node_graph", "u_cont", "quality",
               "precision_id", "optimizer_id", "hardware_cont", "hardware_mask", "raw_steps")
TARGETS = ("train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
           "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib")
CONTRACT = {
    "version": "perfseer-paper-student-v1", "inputs": INPUT_NAMES,
    "targets": TARGETS, "units": ("ms", "ms", "ms", "percent", "MiB", "MiB"),
    "epoch_semantics": "wall step milliseconds multiplied by explicit optimizer steps per epoch",
    "busy_semantics": "NVML kernel-active-time workload proxy; not true SM occupancy",
    "memory_semantics": "advisory predictions only; branch profiles and live VRAM gate admission",
    "vram_postprocessing": {"method": "ceiling", "bucket_bytes": VRAM_BUCKET_BYTES,
                            "minimum_bytes": VRAM_BUCKET_BYTES, "transport_unit": "MiB",
                            "scope": "predict_options; predict_tensors retains raw model outputs"},
    "time_evaluation": {"metric": "pairwise ordering accuracy",
                        "material_measured_ratio_min": 1.10,
                        "close_material_measured_ratio_max_exclusive": 1.25,
                        "prediction_tie_credit": 0.5, "requires_measured_durations": True},
}



def steps_for_job(job, batch_size):
    """Return optimizer steps, recalculated when physical batch size changes."""
    if batch_size <= 0:
        raise ValueError("physical batch_size must be positive")
    metadata = prediction_metadata(job)
    explicit = metadata.get("steps_per_epoch_by_batch", {}).get(str(batch_size))
    if explicit is None:
        explicit = metadata.get("steps_per_epoch_by_batch", {}).get(batch_size)
    if explicit is not None:
        value = float(explicit)
    elif "steps_per_epoch_by_batch" in metadata:
        raise ValueError(f"steps_per_epoch_by_batch has no entry for physical batch {batch_size}")
    elif metadata.get("steps_per_epoch") is not None:
        from ..domain import BatchResolution
        if batch_size != BatchResolution.resolved_batch_size(job):
            raise ValueError("steps_per_epoch_by_batch is required for another physical batch when explicit steps_per_epoch is set")
        value = float(metadata["steps_per_epoch"])
    elif metadata.get("dataset_size") is not None:
        size = int(metadata["dataset_size"])
        training = (metadata.get("workload_contract") or {}).get("training") or {}
        accumulation = int(training.get("gradient_accumulation_steps", metadata.get("gradient_accumulation_steps", 1)))
        if size <= 0 or accumulation <= 0:
            raise ValueError("dataset_size and gradient_accumulation_steps must be positive")
        microsteps = size // batch_size if metadata.get("drop_last", False) else math.ceil(size / batch_size)
        value = math.ceil(microsteps / accumulation)
    else:
        from ..domain import BatchResolution
        if batch_size != BatchResolution.resolved_batch_size(job):
            raise ValueError("steps_per_epoch_by_batch or dataset_size is required for another physical batch")
        value = float(metadata.get("steps_per_epoch", job.metadata.get("runtime_steps_per_epoch", 0)))
    if not math.isfinite(value) or value <= 0 or not float(value).is_integer():
        raise ValueError("positive integer optimizer steps_per_epoch is required; epoch time is never inferred from one step")
    return int(value)


class ReleasedStudentRuntime:
    requires_branch_memory = True

    def __init__(self, release_path, settings, hardware):
        import torch
        self.settings = settings
        self.hardware = hardware
        self.release_root = Path(release_path).expanduser().resolve()
        self.model_root, selected_id, trained, self.artifact_hash = load_verified_student_files(
            self.release_root, settings.student_model_path, hardware.gpu_name)
        self.artifact_path = self.model_root / "student.ts"
        self.normalization_path = self.model_root / "normalization.pt"
        self.runtime_src = self.release_root / "portable/runtime/src"
        if not self.runtime_src.is_dir() or not (self.runtime_src.parent / "aliases").is_dir():
            raise ValueError("portable graph feature runtime is missing")
        extra = {"student_config.json": ""}
        torch.set_num_threads(settings.cpu_threads)
        self.model = torch.jit.load(str(self.artifact_path), map_location="cpu", _extra_files=extra).eval()
        self.metadata = json.loads(extra["student_config.json"])
        version = self.metadata.get("version")
        if version not in {"perfseer_a1001_student_v1", "release_v1_student"}:
            raise ValueError("unsupported released student metadata version")
        if self.metadata.get("hardware") != trained:
            raise ValueError("student metadata hardware differs from release identity")
        self.normalization_hash = digest(self.normalization_path)
        provenance = self.metadata.get("provenance", {})
        for name, path in (("normalization", self.normalization_path),
                           ("architecture", self.model_root / "architecture.pt")):
            expected = provenance.get(name + "_sha256")
            if expected is not None and digest(path) != expected:
                raise ValueError(f"student {name} provenance hash mismatch")
        names = tuple(item.name for item in self.model.forward.schema.arguments)[1:]
        if names != INPUT_NAMES or tuple(self.metadata.get("inputs", INPUT_NAMES)) != INPUT_NAMES:
            raise ValueError("unsupported student tensor interface")
        self.model_id = "release/" + selected_id
        self.last_capture_audits = {}
        self.last_conversion_error = None
        self.cache = OrderedDict()
        self.capture_audit_cache = {}
        self.failures = OrderedDict()
        self.contract = {**CONTRACT, "artifact": str(self.artifact_path), "sha256": self.artifact_hash,
                         "normalization_sha256": self.normalization_hash,
                         "trained_hardware": trained, "out_of_distribution": False,
                         "quality_scope": f"{trained} training distribution; hardware match alone does not establish workload accuracy"}

    def predict_tensors(self, tensors):
        import torch
        if len(tensors) != len(INPUT_NAMES) or any(t.device.type != "cpu" for t in tensors):
            raise ValueError("student requires exactly 17 CPU tensors")
        with torch.inference_mode():
            outputs = self.model(*tensors)
        pred, upper = outputs["prediction"], outputs["memory_upper"]
        if pred.ndim != 2 or pred.shape[1] != 6 or upper.shape != pred[:, 4:6].shape:
            raise ValueError("invalid student output shape")
        if not torch.isfinite(pred).all() or not torch.isfinite(upper).all():
            raise ValueError("nonfinite student output")
        if not ((pred[:, [0, 1, 2, 4, 5]] > 0).all() and ((pred[:, 3] >= 0) & (pred[:, 3] <= 100)).all()
                and (upper >= pred[:, 4:6]).all()):
            raise ValueError("student outputs violate physical ranges")
        if not torch.allclose(pred[:, 2], pred[:, 0] * tensors[-1].float(), rtol=1e-5, atol=1e-5):
            raise ValueError("student epoch output violates wall-step-times-steps contract")
        return {"prediction": pred, "memory_upper": upper}

    def predict_options(self, job, batch_sizes):
        from .ml_predictor import released_source_request, precision_for_job
        from .source_contract import fingerprint_source
        from .source_worker import convert_released_source_options_worker
        import torch
        metadata = prediction_metadata(job)
        self.last_conversion_error = None
        values, missing = {}, []
        for batch in dict.fromkeys(int(n) for n in batch_sizes):
            steps = steps_for_job(job, batch)
            graph = (metadata.get("training_graph_paths") or {}).get(str(batch))
            if graph is None and metadata.get("graph_batch_size") == batch:
                graph = metadata.get("training_graph_path")
            if graph:
                request = {"graph_path": str(Path(graph).expanduser().resolve()),
                           "precision": precision_for_job(job)}
                fingerprint = digest(request["graph_path"])
                request["cacheable"] = True
            else:
                request = released_source_request(job, batch)
                fingerprint = fingerprint_source(request["source_path"], metadata)
                request["cacheable"] = fingerprint["complete"]
            request.update({"raw_steps": steps, "batch_size": batch, "metadata": metadata,
                            "hardware": {"name": self.hardware.gpu_name,
                                         "memory_bytes": int((self.hardware.total_vram_mb or 0) * 1024 * 1024) or None,
                                         "compute_capability": self.hardware.compute_capability}})
            key = sha256(json.dumps({"request": request, "source": fingerprint,
                                     "artifact": self.artifact_hash, "normalization": self.normalization_hash,
                                     "python": sys.executable, "torch": torch.__version__},
                                    sort_keys=True, default=str).encode()).hexdigest()
            if key in self.failures:
                self.failures.move_to_end(key)
                error, expires = self.failures[key]
                if time.monotonic() < expires:
                    raise ValueError(error)
                del self.failures[key]
            if request["cacheable"] and key in self.cache:
                values[batch] = self.cache[key]
                self.last_capture_audits[batch] = self.capture_audit_cache.get(key, {})
                self.cache.move_to_end(key)
            else:
                missing.append((batch, key, request))
        if missing:
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=convert_released_source_options_worker,
                args=(sender, str(self.runtime_src), str(self.normalization_path),
                      [r for _, _, r in missing], self.settings.cpu_threads), daemon=True)
            process.start()
            sender.close()
            try:
                deadline = time.monotonic() + self.settings.conversion_timeout_seconds * len(missing)
                while not receiver.poll(0.05):
                    rss = worker_resident_mib(process.pid)
                    if rss is not None and rss > self.settings.conversion_max_rss_mb:
                        error = (f"source_capture_cpu_memory: worker RSS {rss:.0f} MiB exceeds "
                                 f"conversion_max_rss_mb={self.settings.conversion_max_rss_mb}")
                        self._remember_failure(missing, error)
                        raise ValueError(error)
                    if time.monotonic() >= deadline:
                        self._remember_failure(missing, "released student source conversion timed out")
                        raise ValueError("released student source conversion timed out")
                    if not process.is_alive():
                        error = f"source worker exited before a result (exit code {process.exitcode})"
                        self._remember_failure(missing, error)
                        raise ValueError(error)
                try:
                    response = receiver.recv()
                except (EOFError, OSError) as exc:
                    error = f"source worker exited before a result (exit code {process.exitcode}); inspect CPU memory budget and source imports"
                    self._remember_failure(missing, error)
                    raise ValueError(error) from exc
            finally:
                receiver.close()
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            if not response.get("ok"):
                self.last_conversion_error = response.get("error_detail")
                self._remember_failure(missing, response.get("error", "released student conversion failed"))
                raise ValueError(response.get("error", "released student conversion failed"))
            options = response.get("tensors_by_option", [])
            if len(options) != len(missing):
                raise ValueError("incomplete released student feature set")
            for index, ((batch, key, request), arrays) in enumerate(zip(missing, options, strict=True)):
                audits = response.get("audits", [])
                audit = dict(audits[index]) if len(audits) == len(missing) else {}
                outputs = self.predict_tensors(tuple(torch.from_numpy(a) for a in arrays))
                if len(outputs["prediction"]) != 1:
                    raise ValueError("one graph required per scheduling option")
                raw = outputs["prediction"][0].tolist()
                value = deployment_prediction(raw)
                audit["raw_prediction"] = raw
                audit["vram_postprocessing"] = dict(CONTRACT["vram_postprocessing"])
                self.last_capture_audits[batch] = audit
                values[batch] = value
                if request["cacheable"]:
                    self.cache[key] = value
                    self.capture_audit_cache[key] = self.last_capture_audits.get(batch, {})
                while len(self.cache) > self.settings.cache_size:
                    evicted, _ = self.cache.popitem(last=False)
                    self.capture_audit_cache.pop(evicted, None)
        return values

    def _remember_failure(self, missing, error):
        # Unchanged unsupported sources must not block every scheduler tick on
        # another subprocess timeout. Source/options/artifact hashes key retries.
        for _, key, request in missing:
            self.failures[key] = (str(error), time.monotonic() + 60.0)
        while len(self.failures) > self.settings.cache_size:
            self.failures.popitem(last=False)
