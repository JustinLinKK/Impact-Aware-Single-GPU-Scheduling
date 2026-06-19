"""Test Filter1 & Filter2 with real graph DB data."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "feature_filter",
        str(Path(__file__).resolve().parents[1] / "hardware_knowledge" / "feature_filter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_filter_module()

    which = input("Which filter? (1 / 2): ").strip()
    gpu_name = input("Enter GPU name (e.g. rtx 4090): ").strip() or "rtx 4090"
    stage = input("Pipeline stage (model_structure / datatype / training_parameters / all): ").strip()
    if not stage or stage == "all":
        stage = None

    if which == "2":
        result = mod.query_hardware_features(gpu_name, stage)
        label = f'query_hardware_features("{gpu_name}", "{stage or "None"}")'
    else:
        result = mod.query_hardware_node(gpu_name, stage)
        label = f'query_hardware_node("{gpu_name}", "{stage or "None"}")'

    print(f"\n{'=' * 50}")
    print(label)
    print(f"{'=' * 50}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
