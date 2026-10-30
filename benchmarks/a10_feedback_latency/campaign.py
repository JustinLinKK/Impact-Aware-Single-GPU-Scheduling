"""Serial, balanced 16-cohort controller. No calibration fitting or cloud calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from ...execution.process_utils import terminate_process_tree
from .plot import comparison_plot
from .run import ARMS, CAUSAL_ARMS, atomic_json, check_deadline, file_hash, source_snapshot, with_hard_deadline


def episode_order(arms=ARMS):
    return [{"repeat": repeat + 1, "mode": mode, "arm": arm}
            for repeat in range(2)
            for mode in (("individual", "group") if repeat == 0 else ("group", "individual"))
            for arm in (arms if repeat == 0 else tuple(reversed(arms)))]


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-max-seconds", type=int, default=600)
    parser.add_argument("--campaign-max-seconds", type=int, default=7200)
    parser.add_argument("--deadline-utc", default="2026-10-30T03:10:00Z")
    parser.add_argument("--causal-iteration-one", action="store_true",
                        help="Five common-calibration preparation-ablation arms, 20 cohorts")
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError("campaign terminated; cleaning up the owned episode child")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    manifest_path, root = args.manifest.resolve(), args.output.resolve()
    manifest = with_hard_deadline(json.loads(manifest_path.read_text()), args.deadline_utc)
    check_deadline(manifest)
    before = source_snapshot(manifest, manifest_path)
    root.mkdir(parents=True, exist_ok=False)
    frozen = {"order": episode_order(CAUSAL_ARMS if args.causal_iteration_one else ARMS), "source_pins": before,
        "causal_iteration_one": args.causal_iteration_one,
        "settings_sha256": file_hash(args.settings), "calibration_sha256": file_hash(args.calibration),
        "episode_max_seconds": args.episode_max_seconds, "campaign_max_seconds": args.campaign_max_seconds,
        "effective_deadline_utc": manifest["deadline_utc"],
        "seed_policy": "same immutable per-job seeds and budgets in every arm and paired repeat",
        "cache_policy": "fresh scheduler/predictor process and runtime per episode; shared OS/data caches warmed by charged dedicated calibration",
        "requestor": "colocated Python controller; no network result-return latency claim"}
    atomic_json(root / "campaign-protocol-before.json", frozen)
    started, results = time.monotonic(), []
    for index, episode in enumerate(frozen["order"]):
        check_deadline(manifest, time.monotonic() - started, args.campaign_max_seconds)
        if source_snapshot(manifest, manifest_path) != before:
            raise ValueError("source changed after campaign freeze")
        if (file_hash(args.settings) != frozen["settings_sha256"]
                or file_hash(args.calibration) != frozen["calibration_sha256"]):
            raise ValueError("settings/calibration changed after campaign freeze")
        name = f"{index:02d}-r{episode['repeat']}-{episode['mode']}-{episode['arm']}"
        output = root / name
        command = [sys.executable, "-m", "localml_scheduler.benchmarks.a10_feedback_latency.run", "run",
            "--manifest", str(manifest_path), "--settings", str(args.settings.resolve()),
            "--calibration", str(args.calibration.resolve()), "--output", str(output),
            "--arm", episode["arm"], "--mode", episode["mode"],
            "--deadline-utc", args.deadline_utc,
            "--max-seconds", str(args.episode_max_seconds)]
        with (root / (name + ".log")).open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                atomic_json(root / "active-child.json", {"pid": child.pid, "command": command, **episode})
                while child.poll() is None:
                    check_deadline(manifest, time.monotonic() - started, args.campaign_max_seconds)
                    time.sleep(.5)
            except BaseException as exc:
                atomic_json(root / "campaign-aborted.json", {"complete": False, "episode": episode,
                    "output": name, "elapsed_seconds": time.monotonic() - started,
                    "error": repr(exc), "prior_completed_episodes": len(results)})
                raise
            finally:
                terminate_process_tree(child, timeout=15.)
        path = output / "result.json"
        row = json.loads(path.read_text()) if path.exists() else {"complete": False, "error": "child returned no result"}
        results.append({**episode, "output": name, "exit_code": child.returncode, "result": row})
        atomic_json(root / "campaign-results.json", results)
        comparison_plot(root)
        if child.returncode or not row.get("complete") or row.get("predictor_comparison_eligible") is not True:
            raise RuntimeError("incomplete required episode; campaign stopped without completed-only ranking")
    after = source_snapshot(manifest, manifest_path)
    if after != before: raise ValueError("source changed during final episode")
    atomic_json(root / "campaign-complete.json", {"complete": True, "episodes": len(results),
        "elapsed_seconds": time.monotonic() - started, "source_pins_after": after})


if __name__ == "__main__": main()
