"""Render failed attempts and the final local GPU integration in one PNG."""
from pathlib import Path
from datetime import datetime
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts = sorted(p for p in args.record_root.glob("gpu-attempt-*") if p.is_dir() and (p / "timeline.json").exists())
    roots = [(p, False) for p in attempts] + [(p, True) for p in attempts if (p / "recovery-complete.json").exists()]
    fig, axes = plt.subplots(4, len(roots), figsize=(5 * len(roots), 11), squeeze=False,
                             gridspec_kw={"height_ratios": [1.1, 1.3, 1, 1]})
    colors = {"RUNNING": "#238b65", "PAUSED": "#f3ab38", "PAUSING": "#f3ab38", "READY": "#c4cbd4",
              "PENDING": "#c4cbd4", "COMPLETED": "#1976d2", "FAILED": "#c44545", "CANCELLED": "#c44545"}
    for col, (root, recovery) in enumerate(roots):
        timeline = json.loads((root / "timeline.json").read_text())
        result = json.loads((root / "result.json").read_text()) if (root / "result.json").exists() else None
        predictions = json.loads((root / "predictions.json").read_text())
        names = {jid: f"Draft {i:02d}" for i, (jid, _) in enumerate(sorted(predictions["profiles"].items(), key=lambda item: item[1]["busy_indicator"]))}
        if recovery:
            receipt = json.loads((root / "recovery-complete.json").read_text())
            jid = receipt["job_id"]
            names = {jid: names[jid]}
            events = [json.loads(line) for line in (root / "runtime/logs/events.jsonl").read_text().splitlines()]
            points = [(0, receipt["resumed_step"])] + [
                (datetime.fromisoformat(e["created_at"]).timestamp() - receipt["started_unix"], e["payload"]["global_step"])
                for e in events if e["event_type"] == "checkpoint_saved" and e["job_id"] == jid
                and datetime.fromisoformat(e["created_at"]).timestamp() >= receipt["started_unix"]]
            points.append((receipt["finished_unix"] - receipt["started_unix"], receipt["final_step"]))
            timeline = [{"seconds": seconds, "jobs": {jid: {"status": "RUNNING", "steps": steps}}} for seconds, steps in points]
            result = {"jobs": [{"job_id": jid, "history": receipt["history"]}]}
        for row_index, (jid, name) in enumerate(names.items()):
            for first, second in zip(timeline, timeline[1:]):
                job = first["jobs"].get(jid)
                if job:
                    axes[0, col].broken_barh([(first["seconds"], second["seconds"] - first["seconds"])],
                        (row_index - .32, .64), facecolors=colors.get(job["status"], "#c4cbd4"))
            available = [sample for sample in timeline if jid in sample["jobs"]]
            axes[1, col].plot([s["seconds"] for s in available], [s["jobs"][jid]["steps"] for s in available], label=name)
            if result:
                job_result = next(j for j in result["jobs"] if j["job_id"] == jid)
                axes[2, col].plot([h["epoch"] for h in job_result["history"]], [h["rmse"] for h in job_result["history"]], marker="o", label=name)
        axes[0, col].set_yticks(range(len(names)), list(names.values()))
        axes[0, col].set_title(root.name + (" — checkpoint recovery" if recovery else " — completed" if result else " — diagnosed failure"))
        axes[0, col].set_xlabel("Elapsed seconds")
        axes[1, col].set(xlabel="Elapsed seconds", ylabel="Completed optimizer updates")
        axes[1, col].legend(fontsize=8)
        axes[2, col].set(xlabel="Validation epoch", ylabel="RMSE (32-image smoke subset)")
        if not result:
            axes[2, col].text(.5, .5, "No completed validation", transform=axes[2, col].transAxes, ha="center")
        else:
            axes[2, col].legend(fontsize=8)
        if recovery:
            axes[3, col].text(.5, .5, "Recovery reused the allocator guard.\nNo device-memory time series recorded.\nProgress above is sampled at checkpoints.", transform=axes[3, col].transAxes, ha="center")
            axes[3, col].axis("off")
        else:
            axes[3, col].plot([s["seconds"] for s in timeline], [s["gpu"]["used_mib"] / 1024 for s in timeline], color="#5b4ba1")
            axes[3, col].axhline(timeline[0]["gpu"]["total_mib"] * .9 / 1024, color="#c44545", linestyle="--", label="90% device budget")
            axes[3, col].set(xlabel="Elapsed seconds", ylabel="Whole-device VRAM (GiB)")
            axes[3, col].legend(fontsize=8)
        for row in range(1, 4):
            axes[row, col].grid(alpha=.2)
    fig.suptitle("PetFinder shared-wrapper verification | A10G student on CPU, training on RTX 5090\n"
                 "Existing A10 scratch training remained active; these are control tests, not accuracy or speedup benchmarks", fontsize=14)
    fig.legend(handles=[Patch(color=colors[k], label=k.title()) for k in ("RUNNING", "PAUSED", "READY", "FAILED")], loc="lower center", ncol=4)
    fig.tight_layout(rect=[0, .035, 1, .93])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
