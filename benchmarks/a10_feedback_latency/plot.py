"""Standalone plots from saved real-run telemetry, with no GPU interaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def episode_plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    result = json.loads((output / "result.json").read_text())
    telemetry_path = output / "timeline.jsonl"
    if not telemetry_path.exists(): return None
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
    if not rows: return None
    names = sorted({name for row in rows for name in row["jobs"]})
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    palette = plt.get_cmap("tab10")
    preparation = output / "preparation-timeline.json"
    prep = json.loads(preparation.read_text()) if preparation.exists() else {}
    origin = result.get("episode_started_monotonic", prep.get("episode_started_monotonic"))
    for index, name in enumerate(names):
        color = palette(index % 10)
        seen = [(row["elapsed_seconds"], row["jobs"][name]) for row in rows if name in row["jobs"]]
        for (a, state), (b, _) in zip(seen, seen[1:]):
            if state["status"] in {"RUNNING", "PAUSING"}:
                axes[0].broken_barh([(a, b - a)], (index - .35, .7), facecolors=[color],
                                   alpha=.5 if state["status"] == "PAUSING" else .9)
        times, steps = [x[0] for x in seen], [x[1]["global_step"] for x in seen]
        axes[1].step(times, steps, where="post", color=color, label=name)
        rates = [(steps[i] - steps[i - 1]) / (times[i] - times[i - 1]) for i in range(1, len(times))]
        axes[2].step(times[1:], rates, where="post", color=color)
        memory = [(t, state["reserved_mib"] if state["status"] in {"RUNNING", "PAUSING"} else float("nan"))
                  for t, state in seen if state.get("reserved_mib") is not None]
        if memory: axes[3].plot([x[0] for x in memory], [x[1] for x in memory], color=color, alpha=.6)
        allocated = [(t, state["allocated_mib"] if state["status"] in {"RUNNING", "PAUSING"} else float("nan"))
                     for t, state in seen if state.get("allocated_mib") is not None]
        if allocated:
            axes[3].plot([x[0] for x in allocated], [x[1] for x in allocated],
                         color=color, alpha=.8, linestyle=":")
        job_file = output / "jobs" / name / "job-final.json"
        if origin is not None and job_file.exists():
            job_id = json.loads(job_file.read_text())["job_id"]
            for event in prep.get("events", []):
                if event.get("job_id") == job_id and event.get("phase") == "end":
                    a, b = event["started_monotonic_seconds"] - origin, event["ended_monotonic_seconds"] - origin
                    axes[0].broken_barh([(a, b-a)], (index+.24, .18), facecolors="none",
                                       edgecolors="black", hatch="////", linewidth=.5)
            receipt = result.get("receipts", {}).get(job_id)
            if receipt:
                axes[0].scatter([receipt["feedback_ready_monotonic"] - origin], [index],
                                marker="|", s=90, c="black", zorder=5)
    axes[0].set_yticks(range(len(names)), names, fontsize=8)
    axes[0].set_ylabel("Worker state")
    axes[1].set_ylabel("Successful updates")
    axes[2].set_ylabel("Sampled updates/s")
    axes[3].plot([r["elapsed_seconds"] for r in rows], [r["gpu"]["used_mib"] for r in rows],
                 color="black", linewidth=2, label="Device used")
    axes[3].set_ylabel("VRAM MiB\nsolid=reserved; dotted=allocated")
    axes[3].set_xlabel("Seconds from first request acceptance")
    axes[3].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{result.get('arm', 'calibration')} / {result.get('mode', '')}: worker states; hatched=preparation, | = receipt")
    for axis in axes: axis.grid(alpha=.2)
    fig.tight_layout()
    target = output / "cohort-measurements.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def comparison_plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    records = []
    for path in sorted(root.glob("*/result.json")):
        row = json.loads(path.read_text())
        repeat = next((part for part in path.parent.name.split("-")
                       if len(part) > 1 and part[0] == "r" and part[1:].isdigit()), "repeat unknown")
        if "arm" in row: records.append({"path": str(path.relative_to(root)), "observed_repeat": repeat, **row})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    protocol = root / "campaign-protocol-before.json"
    arms = tuple(dict.fromkeys(r["arm"] for r in json.loads(protocol.read_text())["order"])) if protocol.exists() else tuple(dict.fromkeys(r["arm"] for r in records))
    for axis, mode in zip(axes, ("individual", "group")):
        for index, arm in enumerate(arms):
            matched = [r for r in records if r.get("complete") and r.get("predictor_comparison_eligible")
                       and r.get("arm") == arm and r.get("mode") == mode]
            if matched:
                values = [r["episode_duration_seconds"] for r in matched]
                axis.scatter([index] * len(values), values, label=arm)
                for record, value in zip(matched, values):
                    axis.annotate(record["observed_repeat"], (index, value), xytext=(5, 3),
                                  textcoords="offset points", fontsize=8)
            axis.text(index, .01, f"eligible {len(matched)}/2", ha="center", fontsize=8,
                      transform=axis.get_xaxis_transform())
        axis.set_xticks(range(len(arms)), [a.replace("_", "\n") for a in arms], fontsize=8)
        axis.set_ylabel("Whole episode result-ready seconds")
        axis.set_title(mode + ": individual observed eligible repeats; no unmatched means")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Descriptive A10 pilot; incomplete/ineligible episodes remain in comparison.json")
    fig.tight_layout()
    fig.savefig(root / "combined-comparison.png", dpi=150)
    plt.close(fig)
    (root / "comparison.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    cohorts = sorted(root.glob("*/cohort-measurements.png"))
    if cohorts:
        import math
        columns = min(4, len(cohorts))
        rows = math.ceil(len(cohorts) / columns)
        contact, panels = plt.subplots(rows, columns, figsize=(8 * columns, 8 * rows),
                                       squeeze=False)
        for panel, path in zip(panels.flat, cohorts):
            panel.imshow(plt.imread(path))
            panel.set_title(path.parent.name, fontsize=12)
        for panel in panels.flat: panel.axis("off")
        contact.suptitle("Observed cohort Gantt + update/rate/VRAM metrics; includes partial episode plots", fontsize=16)
        contact.tight_layout()
        contact.savefig(root / "combined-cohort-gantt-metrics.png", dpi=200)
        plt.close(contact)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    if args.comparison: comparison_plot(args.output)
    else: episode_plot(args.output)


if __name__ == "__main__": main()
