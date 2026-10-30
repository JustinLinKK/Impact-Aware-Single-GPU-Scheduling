"""CPU-only manifest preparation before any pilot outcome is observed."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from localml_scheduler.benchmarks.a10_feedback_latency.workloads.core import (
    HERE, build_model, file_hash, spec_digest, expected_signature, prediction_metadata,
)


def prepare(data_root, archive_root, output=None):
    output = Path(output or HERE / "manifest.json")
    frame = pd.read_csv(data_root / "train_labels.csv", dtype={"id": str})
    training, heldout = train_test_split(frame.id.tolist(), test_size=.2, random_state=42,
        stratify=frame.label.to_numpy())
    labels = dict(zip(frame.id, frame.label))
    validation, _ = train_test_split(heldout, train_size=256, random_state=42,
        stratify=[labels[x] for x in heldout])
    split = {"schema": "a10-pilot-fixed-split-v1", "seed": 42, "training_ids": training,
             "heldout_ids": heldout, "validation_ids": validation}
    (HERE / "split.json").write_text(json.dumps(split, sort_keys=True, indent=2) + "\n")
    data = {"schema": "histopathologic-cancer-data-v1", "total_rows": len(frame),
        "training_rows": len(training), "heldout_rows": len(heldout), "validation_rows": len(validation),
        "validation_class_counts": [sum(labels[x] == y for x in validation) for y in (0, 1)],
        "labels_sha256": file_hash(data_root / "train_labels.csv"),
        "images_manifest_sha256": file_hash(data_root / "cancer-data-manifest.json"),
        "split_sha256": file_hash(HERE / "split.json"),
        "subset_provenance_sha256": file_hash(data_root / "subset-provenance.json"),
        "source_dataset": "Kaggle histopathologic-cancer-detection real 40960-image stratified subset"}
    definitions = [(17,"efficientnet_b4",380,16,"parent",0), (20,"efficientnet_b4",380,16,"child",0),
        (14,"convnext_small",224,16,"parent",1), (16,"convnext_small",224,16,"child",1),
        (22,"custom_cnn",96,64,"parent",2), (18,"convnext_small",224,16,"child",2),
        (8,"efficientnet_b0",224,16,"calibration",None), (8,"efficientnet_b0",224,32,"calibration",None),
        (12,"convnext_tiny",224,8,"calibration",None), (12,"convnext_tiny",224,16,"calibration",None)]
    counts = {}
    for _, family, size, *_ in definitions:
        if family not in counts:
            model = build_model({"model_family": family, "image_size": size, "seed": 42,
                                 "channels_last": family != "custom_cnn"})
            counts[family] = (sum(p.numel() for p in model.parameters()),
                             sum(p.numel() for p in model.parameters() if p.requires_grad))
            del model
    archive = HERE / "archived_sources"
    archive.mkdir(exist_ok=True)
    origins = {}
    for source_id in {x[0] for x in definitions}:
        paths = [p for p in archive_root.glob(f"runfile_{source_id}_*.py") if not p.name.endswith("_instrumented.py")]
        if len(paths) != 1:
            raise ValueError(f"ambiguous archived source {source_id}: {paths}")
        original = paths[0]
        copied = archive / original.name
        shutil.copyfile(original, copied)
        origins[source_id] = {"path": str(copied.relative_to(HERE)), "sha256": file_hash(copied),
            "original_archive_path": str(original),
            "provenance": "archived agent source; prior archive probe/repair changes retained; not pristine original"}
    package = {name: file_hash(HERE / name) for name in ("core.py", "candidate.py", "__init__.py", "split.json")}
    jobs, calibration = [], []
    for source_id, family, size, batch, role, chain in definitions:
        spec = {"name": f"source{source_id}_{role}_b{batch}", "source_id": source_id,
            "source": "candidate.py", "source_sha256": file_hash(HERE / "candidate.py"),
            "model_family": family, "image_size": size, "batch": batch, "epochs": 2,
            "steps_per_epoch": 32, "seed": 42, "role": role, "chain": chain,
            "metric_name": "validation_auc", "maximize": True, "validation_size": 256,
            "parameter_count": counts[family][0], "trainable_parameter_count": counts[family][1],
            "data_identity": data, "package_pins": package,
            "env": {}, "metadata": {"trace_derivation": "TRACE-DERIVED", "archived_source": origins[source_id],
                "constructed_dependency": role != "calibration", "selection_metric_internal": "negative_validation_auc",
                "returned_metric": "validation_auc", "returned_metric_maximize": True,
                "deviations": ["deterministic random initialization instead of pretrained/downloaded weights",
                    "IEEE FP32 instead of original mixed precision", "shared_training.v1 synchronous map-style loop",
                    "fixed AdamW LR 0.0001 instead of original cosine schedule", "bounded 2 epochs x 32 successful updates",
                    "pilot batch differs from original", "fixed 256-image single-view validation; no test/submission or TTA",
                    "shared stratified 80/20 seed42 split; custom source22 original 90/10 split changed"],
                "preprocessing": {"native_size": 96, "model_input_size": size,
                    "normalization": "0.5/0.5" if family == "custom_cnn" else "ImageNet",
                    "augmentation": "H/V flip, rotation90 custom/20 others, color jitter(.1,.1,.1,.05)"}}}
        spec["metadata"]["perfseer_model"] = prediction_metadata(spec)
        spec["spec_sha256"] = spec_digest(spec)
        spec["expected_runtime_signature"] = expected_signature(spec)
        (calibration if role == "calibration" else jobs).append(spec)
    manifest = {"schema": "a10-feedback-latency-workloads-v1", "deadline_utc": "2026-10-30T03:10:00Z",
        "job_timeout_seconds": 480, "episode_timeout_seconds": 720,
        "jobs": jobs, "calibration": calibration,
        "data_identity": data, "budget_policy": "64 updates provisional; calibration-only shortening requires a NEW frozen manifest before evaluation",
        "dependency_semantics": "CONSTRUCTED three 2-job chains; no claimed historical causal edges",
        "additional_pins": [{"path": name, "sha256": value} for name, value in package.items()] +
             [{"path": value["path"], "sha256": value["sha256"]} for value in origins.values()],
        "evaluation_model_groups": ["efficientnet_b4", "convnext_small", "custom_cnn"],
        "calibration_model_groups": ["efficientnet_b0", "convnext_tiny"],
        "feedback_contract": "positive finite AUC maximize; negative AUC internal minimization; real-image BCE separate; both validation classes required",
        "phase_proxy": "calibration timings are proxies, not direct eval measurements; all arms share identical frozen calibration"}
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": file_hash(output), "class_counts": data["validation_class_counts"], "parameter_counts": counts}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.archive_root)
