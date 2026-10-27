#!/usr/bin/env python3
"""Run actual A10G student predictions on source-bound, generated MLEvolve jobs.

This does not execute candidates' training loops or measure GPU accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.config import PredictionSettings
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction.ml_predictor import MLVramPredictor
from localml_scheduler.prediction.job_contract import prediction_metadata
from localml_scheduler.prediction.released_artifacts import digest
from localml_scheduler.prediction.released_outputs import MIB_BYTES, relative_time_costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.sources.glob("candidate_*.py"))
    if not 2 <= len(sources) <= 3:
        raise ValueError("the integration test requires 2–3 generated candidate sources")
    settings = PredictionSettings(mode="ml_predictor", student_release_path=args.release_root,
        student_model_path=args.model_dir, conversion_timeout_seconds=90, conversion_max_rss_mb=4096)
    target = HardwareProfile("declared-deployment", "linux", "NVIDIA A10G", 23028,
                             "8.6", "declared target, not local measurement", torch.__version__)
    predictor = MLVramPredictor(settings, target)
    assert predictor.available, predictor.unavailable_reason
    assert predictor.model_id == "release/S_A10G"
    rows = []
    for source in sources:
        job = build_mlevolve_job(workflow_id="petfinder-code-agent-integration",
            baseline_model_id=source.stem, baseline_model_path=str(source.resolve()),
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            runner_kwargs={"batch_size": 4}, max_epochs=3)
        metadata = prediction_metadata(job)
        assert metadata["steps_per_epoch_by_batch"]["4"] == 8
        start = time.perf_counter()
        values = predictor._released_options(job, [4])[4]
        elapsed = time.perf_counter() - start
        audit = predictor._released.last_capture_audits[4]
        assert audit["numeric_parity"] and audit["backward_node_count"] > 0
        assert audit["mode"] == "candidate_adapter"
        assert audit["materialized_input_numel"] == 4 * (3 * 256 * 256 + 12 + 1)
        assert not torch.cuda.is_initialized()
        # Public scheduler-facing APIs must consume the very same cached result.
        assert predictor.predict_busy_percent_options(job, [4])[4] == values[3]
        assert predictor.predict_seconds_per_epoch_options(job, [4])[4] == values[2] / 1000
        assert predictor.predict_avg_vram_options(job, [4])[4] == values[4]
        assert predictor.last_sources[job.job_id] == "ml_predictor"
        assert values[:4] == audit["raw_prediction"][:4]
        rows.append({"candidate": source.stem, "source": source.name,
            "source_sha256": digest(source), "sidecar_sha256": digest(source.with_suffix(".py.perfseer.json")),
            "physical_batch": 4, "optimizer_steps_per_epoch": 8, "max_epochs": 3,
            "predicted_wall_step_ms": values[0], "predicted_gpu_step_ms": values[1],
            "predicted_epoch_ms": values[2], "predicted_busy_percent": values[3],
            "predicted_avg_vram_mib": values[4], "predicted_peak_vram_mib": values[5],
            "predicted_avg_vram_decimal_mb": values[4] * MIB_BYTES / 1e6,
            "predicted_peak_vram_decimal_mb": values[5] * MIB_BYTES / 1e6,
            "predicted_training_seconds": values[2] * 3 / 1000,
            "cpu_capture_and_prediction_seconds": elapsed, "capture_audit": audit})
    timing_comparison = {}
    for metric in ("wall_step", "gpu_step", "training"):
        field = "predicted_training_seconds" if metric == "training" else f"predicted_{metric}_ms"
        durations = {row["candidate"]: row[field] for row in rows}
        ratios = relative_time_costs(durations)
        timing_comparison[metric] = {"reference_candidate": min(durations, key=durations.get),
                                     "ratio_to_fastest": ratios}
        for row in rows:
            row[f"predicted_{metric}_relative_cost"] = ratios[row["candidate"]]
    result = {"schema": "mlevolve-petfinder-a10g-integration-v2",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "student": predictor.model_id, "student_sha256": predictor.artifact_hash,
        "torch": torch.__version__, "target_gpu": "NVIDIA A10G", "cuda_initialized": False,
        "training_launched": False, "measured_accuracy": None,
        "vram_postprocessing": predictor._released.contract["vram_postprocessing"],
        "time_evaluation": {**predictor._released.contract["time_evaluation"],
                            "status": "not_measured", "material_ordering_accuracy": None,
                            "close_material_ordering_accuracy": None},
        "relative_time_comparison": timing_comparison,
        "scope": "Actual generated source, native backward replay, CPU A10G TorchScript inference and scheduler APIs; not a GPU performance benchmark",
        "busy_order_low_to_high": [row["candidate"] for row in sorted(rows, key=lambda row: row["predicted_busy_percent"])],
        "candidates": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "candidates": len(rows), "student": predictor.model_id,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
