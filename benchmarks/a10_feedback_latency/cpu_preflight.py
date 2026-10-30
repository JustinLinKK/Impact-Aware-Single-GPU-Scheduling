"""CPU-only published A10 source capture; never executes real-image training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run CPU proof with CUDA_VISIBLE_DEVICES='' before importing torch")
    import psutil
    import torch
    import timm
    import torchvision
    from ...client import SchedulerClient
    from ...config import PredictionSettings
    from ...hardware import HardwareProfile, build_hardware_key
    from ...prediction.ml_predictor import MLVramPredictor
    from .run import STUDENT_SHA256, accept_job, atomic_json, build_job, file_hash, make_settings
    if torch.cuda.is_initialized() or torch.cuda.is_available(): raise RuntimeError("CPU-only proof required")
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    hardware_fields = dict(os_name="linux", gpu_name="NVIDIA A10", total_vram_mb=22600,
        compute_capability="8.6", cuda_runtime=torch.version.cuda, torch_version=torch.__version__,
        driver_version=None)
    hardware = HardwareProfile(hardware_key=build_hardware_key(**hardware_fields), **hardware_fields)
    predictor_settings = PredictionSettings(mode="ml_predictor", student_release_path=str(args.release.resolve()),
        student_model_path=str(args.release.resolve() / "checkpoints/S_A10"),
        conversion_timeout_seconds=180, conversion_max_rss_mb=16384, cpu_threads=1)
    predictor = MLVramPredictor(predictor_settings, hardware)
    if not predictor.available or predictor.artifact_hash != STUDENT_SHA256:
        raise ValueError("published S_A10 unavailable or wrong artifact:" + str(predictor.unavailable_reason))
    settings = make_settings({}, output, {"total_mib": hardware.total_vram_mb})
    client = SchedulerClient(settings)
    client.store.backend._hardware_profile = hardware
    state = {"complete": False, "scope": "CPU synthetic graph capture; no real-data fit or GPU allocation",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "timm": timm.__version__},
        "explicit_target_hardware": hardware.to_dict(), "physical_local_gpu_inspected": False,
        "conversion_max_rss_mib": 16384, "conversion_timeout_seconds": 180,
        "manifest_sha256": file_hash(manifest_path), "student_sha256": predictor.artifact_hash,
        "driver_sha256": file_hash(__file__), "cuda_initialized_before": torch.cuda.is_initialized(),
        "cases": []}
    atomic_json(output / "cpu-preflight.json", state)
    unique = {}
    for spec in manifest["jobs"]: unique.setdefault((spec["model_family"], spec["batch"]), spec)
    for (family, batch), spec in unique.items():
        case = {"name": spec["name"], "family": family, "batch": batch, "complete": False,
                "source_sha256": spec["source_sha256"], "expected_runtime_signature": spec["expected_runtime_signature"]}
        stop = threading.Event()
        peaks = {}
        def monitor():
            parent = psutil.Process()
            while not stop.wait(.1):
                for child in parent.children(recursive=True):
                    try:
                        rss = child.memory_info().rss / 1024**2
                        peaks[str(child.pid)] = max(peaks.get(str(child.pid), 0.), rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        watcher = threading.Thread(target=monitor, daemon=True)
        started = time.monotonic()
        watcher.start()
        print(json.dumps({"event": "cpu_capture_started", "family": family, "batch": batch}), flush=True)
        try:
            job, work = build_job(spec, manifest_path=manifest_path, output=output,
                manifest=manifest, memory_budget_mib=hardware.total_vram_mb * .9)
            job = accept_job(client, job, time.time())
            profile = predictor.predict_profile_options(job, [batch])[batch]
            if predictor.last_sources.get(job.job_id) != "ml_predictor": raise ValueError("predictor fallback")
            if profile["hardware"] != "A10": raise ValueError("wrong target hardware")
            case.update(complete=True, profile=profile,
                        capture_audit=predictor._released.last_capture_audits.get(batch, {}))
            atomic_json(work / "cpu-prepared-profile.json", case)
        except Exception:
            case.update(error=traceback.format_exc(), predictor_errors=dict(predictor.last_errors))
        finally:
            stop.set(); watcher.join(timeout=2)
            case.update(elapsed_seconds=time.monotonic() - started, child_peak_rss_mib=peaks,
                        maximum_child_rss_mib=max(peaks.values(), default=0.),
                        cuda_initialized_after=torch.cuda.is_initialized())
            state["cases"].append(case)
            atomic_json(output / "cpu-preflight.json", state)
            print(json.dumps({"event": "cpu_capture_finished", "family": family,
                "complete": case["complete"], "elapsed_seconds": case["elapsed_seconds"],
                "maximum_child_rss_mib": case["maximum_child_rss_mib"],
                "error": case.get("error")}), flush=True)
    state.update(complete=len(state["cases"]) == 3 and all(c["complete"] for c in state["cases"]),
                 cuda_initialized_after=torch.cuda.is_initialized())
    atomic_json(output / "cpu-preflight.json", state)
    if not state["complete"] or state["cuda_initialized_after"]: raise SystemExit(1)


if __name__ == "__main__": main()
