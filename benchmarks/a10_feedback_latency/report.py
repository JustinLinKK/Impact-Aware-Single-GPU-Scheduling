"""CPU-only, coverage-preserving descriptive export for the A10 pilot.

No completed-only ranking: missing/failed required jobs make an aggregate or
paired comparison unavailable. Clean calibration windows and end-to-end update
rates have separate meanings and are never relabeled as full solo slope labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics

ARMS = ("reference_v4", "calibrated_v4", "calibrated_revised", "fixed_three")
MODES = ("individual", "group")
OOM = re.compile(r"out[ _-]of[ _-]memory|cuda.*memory allocation|cudaerror.*memoryallocation", re.I)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def number(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def summary(values):
    """Linear-interpolation p95, explicitly descriptive for the small sample."""
    if not values or any(not number(v) for v in values):
        return None
    values = sorted(float(v) for v in values)
    x = .95 * (len(values) - 1)
    lo, hi = math.floor(x), math.ceil(x)
    return {"n": len(values), "mean": statistics.mean(values),
            "p95": values[lo] + (x - lo) * (values[hi] - values[lo]), "max": values[-1]}


def valid_receipt(row):
    result = row.get("returned_result") or {}
    return (row.get("worker_exit_code") == 0 and number(row.get("jct_seconds"))
        and type(row.get("expected_updates")) is int and row["expected_updates"] > 0
        and result.get("global_step") == row["expected_updates"]
        and result.get("epochs") == row.get("expected_epochs")
        and bool(row.get("expected_runtime_signature"))
        and result.get("signature") == row.get("expected_runtime_signature")
        and row.get("auc_maximize") is True and number(row.get("selected_auc"))
        and row["selected_auc"] <= 1 and number(row.get("selected_validation_bce")))


def event_rows(output):
    seen, rows = set(), []
    paths = list(output.glob("jobs/*/events.json")) + list(output.glob("runtime/**/events.jsonl"))
    if (output / "failure-events.json").exists():
        paths.append(output / "failure-events.json")
    for path in paths:
        if path.suffix == ".jsonl":
            items = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        else:
            items = read_json(path, [])
        for item in items:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                rows.append(item)
                seen.add(key)
    return rows


def cohort_record(root, spec, results, expected_jobs):
    key = (spec["repeat"], spec["mode"], spec["arm"])
    saved = results.get(key, {})
    name = saved.get("output") or spec.get("output")
    if name is None:
        matches = sorted(root.glob(f"*-r{key[0]}-{key[1]}-{key[2]}"))
        name = matches[0].name if matches else f"missing-r{key[0]}-{key[1]}-{key[2]}"
    output = root / name
    result = read_json(output / "result.json", saved.get("result", {})) or {}
    receipts = dict(result.get("receipts") or {})
    for path in output.glob("jobs/*/receipt.json"):
        row = read_json(path)
        receipts.setdefault(row["job_id"], row)
    verified = {job: row for job, row in receipts.items() if valid_receipt(row)}
    events = event_rows(output) if output.exists() else []
    job_states = [read_json(p) for p in output.glob("jobs/*/job-final.json")]
    known_ids = {r.get("job_id") for r in job_states}
    job_states += [r for r in read_json(output / "failure-jobs.json", []) if r.get("job_id") not in known_ids]
    failed = {r.get("job_id", str(i)) for i, r in enumerate(job_states)
              if str(r.get("status", "")).upper() in {"FAILED", "CANCELLED"}}
    memory_ids = {e.get("job_id") for e in events
                  if OOM.search(json.dumps(e)) and e.get("job_id")}
    memory_ids.update(r.get("job_id", str(i)) for i, r in enumerate(job_states) if OOM.search(json.dumps(r)))
    error = result.get("error") or ""
    unattributed_oom = bool(OOM.search(error)) and not memory_ids
    requests = result.get("request_latencies_seconds") or {}
    n_requests = expected_jobs if spec["mode"] == "individual" else 2
    all_requests = len(requests) == n_requests and all(number(v) for v in requests.values())
    complete = (result.get("complete") is True and saved.get("exit_code", 0) == 0
                and len(verified) == expected_jobs and not failed)
    eligible = (complete and result.get("predictor_comparison_eligible") is True
                and all_requests and number(result.get("episode_duration_seconds")))
    chains = result.get("chain_root_to_child_seconds") or {}
    profiles = dict(result.get("prepared_numeric_profiles") or {})
    for path in output.glob("jobs/*/predicted-profiles.json"):
        final = read_json(path.parent / "job-final.json", {})
        profiles.setdefault(final.get("job_id", path.parent.name), read_json(path, {}))
    progress_rates = []
    first, last = result.get("first_progress_observations", {}), result.get("last_progress_observations", {})
    for job, a in first.items():
        b = last.get(job, {})
        elapsed = b.get("observed_seconds", 0) - a.get("observed_seconds", 0)
        steps = b.get("step", 0) - a.get("step", 0)
        if elapsed > 0 and steps > 0:
            progress_rates.append({"job": job, "successful_updates": steps, "observation_seconds": elapsed,
                                   "sampled_updates_per_second": steps / elapsed})
    return {**spec, "output": name, "complete": complete, "comparison_eligible": eligible,
        "status": "complete" if complete else ("incomplete" if output.exists() else "not_attempted"),
        "expected_jobs": expected_jobs, "verified_receipts": len(verified), "unreturned_jobs": max(0, expected_jobs-len(verified)),
        "excess_receipts": max(0, len(verified)-expected_jobs),
        "invalid_receipts": len(receipts)-len(verified), "failed_or_cancelled_job_ids": sorted(failed),
        "memory_failure_job_ids": sorted(memory_ids), "unattributed_memory_failure": unattributed_oom,
        "error": error or None, "expected_requests": n_requests,
        "job_status_counts": dict(Counter(str(r.get("status", "unknown")) for r in job_states)),
        "incomplete_observation_seconds": result.get("elapsed_seconds", read_json(output / "status.json", {}).get("elapsed_seconds")) if not complete else None,
        "returned_requests": sum(number(v) for v in requests.values()),
        "job_jct_seconds": summary([r["jct_seconds"] for r in verified.values()]) if complete else None,
        "request_latency_seconds": summary(list(requests.values())) if complete and all_requests else None,
        "workflow_duration_seconds": result.get("episode_duration_seconds") if complete else None,
        "chain_latency_seconds": summary(list(chains.values())) if complete and len(chains) == expected_jobs//2 else None,
        "peak_device_used_mib": result.get("peak_device_used_mib"),
        "predictor_errors": result.get("prediction_errors", {}),
        "prediction_sources": result.get("prediction_sources", {}),
        "prepared_numeric_profiles": profiles,
        "numeric_profile_note": "saved actual preparation outputs only; no new inference and no co-located timing-error labels",
        "observed_successful_job_jct_seconds": summary([r["jct_seconds"] for r in verified.values()]),
        "observed_only_note": "partial-success values are coverage diagnostics, never a completed-arm comparison",
        "sampled_progress": progress_rates,
        "progress_note": "co-located sampled safe-point progress, includes stalls/pauses; not solo labels or complete sustained training rate",
        "phase_model_role": result.get("phase_model_role", "calibrated_cross_workload_phase_proxy_v1"),
        "source_pins_match": (read_json(output / "source-pins-before.json") == read_json(output / "source-pins-after.json"))
            if (output / "source-pins-after.json").exists() else None}


def calibration_evidence(rows, calibration=None):
    calibration = calibration or {}
    evidence = []
    for row in rows:
        profile, windows = row.get("predicted_profile", {}), row.get("windows", [])
        clean = (row.get("role") == "calibration" and row.get("complete") is True and bool(windows)
            and all(w.get("complete") is True and w.get("kind") == "calibration"
                    and w.get("excludes_nonupdate_phases") is True and number(w.get("steps"), True)
                    and number(w.get("elapsed_seconds"), True) for w in windows)
            and number(row.get("isolated_seconds_per_update"), True)
            and profile.get("hardware") == "A10" and number(profile.get("step_cost"), True)
            and bool(row.get("runtime_completion_identity"))
            and valid_receipt(row.get("receipt", {}))
            and row["runtime_completion_identity"] == row["receipt"].get("expected_runtime_signature"))
        item = {"name": row.get("name"), "clean_isolated_window_evidence": clean,
                "role": "dedicated calibration; fit-set diagnostics, not held-out evaluation accuracy",
                "source_identity": row.get("source_identity"),
                "runtime_completion_identity": row.get("runtime_completion_identity"),
                "predicted_profile": profile}
        if clean:
            observed, predicted = row["isolated_seconds_per_update"], profile["step_cost"] / 1000.
            item.update({"measured_window_updates_per_second": 1 / observed,
                "predicted_updates_per_second": 1 / predicted,
                "window_rate_relative_error": abs(observed / predicted - 1),
                "window_duration_relative_error": abs(predicted / observed - 1),
                "window_updates_total": sum(w["steps"] for w in windows),
                "window_seconds_total": sum(w["elapsed_seconds"] for w in windows),
                "pooled_window_updates_per_second": sum(w["steps"] for w in windows)/sum(w["elapsed_seconds"] for w in windows),
                "window_note": "short isolated phase-clean safe-point training windows; includes training input/control costs; not full-run sustained slope"})
        multiplier = calibration.get("wall_multiplier")
        if clean and number(multiplier, True):
            corrected = predicted * multiplier
            item["calibrated_window_updates_per_second"] = 1 / corrected
            item["calibrated_window_rate_relative_error"] = abs(observed / corrected - 1)
            item["calibrated_fit_set_note"] = "correction fit on these same dedicated calibration rows; no held-out improvement claim"
        receipt = row.get("receipt", {})
        if valid_receipt(receipt) and number(receipt.get("jct_seconds"), True):
            item["whole_job_update_equivalent_per_second"] = receipt["expected_updates"] / receipt["jct_seconds"]
            item["whole_job_note"] = "updates divided by accepted-to-receipt JCT; startup, prediction, validation, pause/resume, checkpoint and delivery included"
        item["full_sustained_solo_updates_per_second"] = None
        item["full_sustained_note"] = "no complete phase-excluded full-job training interval supplied; not inferred by subtracting estimated phases"
        item["memory_comparison_note"] = "observed reserved memory is admission evidence, not an allocated-VRAM target label"
        item["observed_peak_reserved_mib"] = row.get("peak_reserved_mib")
        evidence.append(item)
    return evidence


def paired_comparisons(cohorts):
    index = {(r["repeat"], r["mode"], r["arm"]): r for r in cohorts}
    result = []
    metrics = {"average_job_jct_seconds": lambda r: (r["job_jct_seconds"] or {}).get("mean"),
        "average_request_latency_seconds": lambda r: (r["request_latency_seconds"] or {}).get("mean"),
        "p95_job_jct_seconds": lambda r: (r["job_jct_seconds"] or {}).get("p95"),
        "max_job_jct_seconds": lambda r: (r["job_jct_seconds"] or {}).get("max"),
        "workflow_duration_seconds": lambda r: r["workflow_duration_seconds"],
        "average_chain_latency_seconds": lambda r: (r["chain_latency_seconds"] or {}).get("mean")}
    for mode in MODES:
        for arm in ARMS[1:]:
            pairs = []
            for repeat in (1, 2):
                base, candidate = index.get((repeat, mode, ARMS[0])), index.get((repeat, mode, arm))
                valid = bool(base and candidate and base["comparison_eligible"] and candidate["comparison_eligible"])
                deltas = {name: fn(candidate)-fn(base) if valid and number(fn(base)) and number(fn(candidate)) else None
                          for name, fn in metrics.items()}
                pairs.append({"repeat": repeat, "eligible": valid, "candidate_minus_reference": deltas})
            result.append({"mode": mode, "arm": arm, "reference": ARMS[0], "pairs": pairs,
                "paired_mean_delta": {name: statistics.mean(p["candidate_minus_reference"][name] for p in pairs)
                    if all(p["candidate_minus_reference"][name] is not None for p in pairs) else None for name in metrics},
                "interpretation": "negative favors candidate; exactly two paired repeats are descriptive, no independent significance or winner claim"})
    return result


def build_report(campaign, calibration_rows=None, manifest=None):
    root = Path(campaign)
    protocol = read_json(root / "campaign-protocol-before.json", {})
    expected = protocol.get("order") or [{"repeat": r, "mode": m, "arm": a}
                                       for r in (1, 2) for m in MODES for a in ARMS]
    rows = read_json(root / "campaign-results.json", [])
    results = {(r["repeat"], r["mode"], r["arm"]): r for r in rows}
    manifest_value = read_json(manifest, {}) if manifest else {}
    expected_jobs = len(manifest_value.get("jobs", [])) or 6
    cohorts = [cohort_record(root, spec, results, expected_jobs) for spec in expected]
    calibration = read_json(calibration_rows, []) if calibration_rows else []
    if isinstance(calibration, dict):
        calibration = calibration.get("rows", [])
    grouped = []
    for mode in MODES:
        for arm in ARMS:
            members = [r for r in cohorts if r["mode"] == mode and r["arm"] == arm]
            complete = len(members) == 2 and all(r["comparison_eligible"] for r in members)
            grouped.append({"mode": mode, "arm": arm, "planned_cohorts": 2,
                "eligible_cohorts": sum(r["comparison_eligible"] for r in members),
                "expected_jobs": 2*expected_jobs, "verified_receipts": sum(r["verified_receipts"] for r in members),
                "mean_of_repeat_average_job_jct_seconds": statistics.mean(r["job_jct_seconds"]["mean"] for r in members) if complete else None,
                "mean_of_repeat_average_request_latency_seconds": statistics.mean(r["request_latency_seconds"]["mean"] for r in members)
                    if complete and all(r["request_latency_seconds"] for r in members) else None,
                "mean_workflow_duration_seconds": statistics.mean(r["workflow_duration_seconds"] for r in members) if complete else None,
                "comparison_available": complete})
    inputs = [root / "campaign-protocol-before.json", root / "campaign-results.json"]
    inputs += [Path(p) for p in (calibration_rows, manifest) if p]
    inputs += list(root.glob("*/result.json"))
    if calibration_rows:
        inputs.append(Path(calibration_rows).parent / "calibration.json")
    return {"schema": "a10-feedback-descriptive-report-v1", "comparison_scope": "A10 student S_A10; not A100 parent P",
        "planned_cohorts": len(expected), "expected_jobs": len(expected)*expected_jobs,
        "deadline_utc": manifest_value.get("deadline_utc"),
        "verified_receipts": sum(r["verified_receipts"] for r in cohorts),
        "complete_cohorts": sum(r["complete"] for r in cohorts),
        "comparison_eligible_cohorts": sum(r["comparison_eligible"] for r in cohorts),
        "failed_or_cancelled_jobs": sum(len(r["failed_or_cancelled_job_ids"]) for r in cohorts),
        "memory_failure_jobs": sum(len(r["memory_failure_job_ids"]) for r in cohorts),
        "unattributed_memory_failure_cohorts": sum(r["unattributed_memory_failure"] for r in cohorts),
        "unreturned_jobs": sum(r["unreturned_jobs"] for r in cohorts),
        "campaign_complete": read_json(root / "campaign-complete.json", {}).get("complete") is True,
        "cohorts": cohorts, "arm_mode_summary": grouped, "paired_comparisons": paired_comparisons(cohorts),
        "calibration_evidence": calibration_evidence(calibration, read_json(Path(calibration_rows).parent / "calibration.json", {}) if calibration_rows else {}),
        "notes": ["All planned jobs/cohorts stay in coverage denominators; missing/failed required results disable aggregate comparisons.",
            "Individual primary request metric is average JCT; group primary metric is average all-required bracket latency. Job JCT also reported in both modes.",
            "The requestor is colocated; feedback readiness includes actual worker exit and local result parse/hash, not network RTT.",
            "Cross-workload phase costs are calibrated proxies; parameter scaling is an estimate, not a measured bound for evaluation jobs.",
            "Two paired repeats provide descriptive evidence only; no fabricated winner or global promotion.",
            "Offline A100 timing-tree slope ARE32.12% versus P36.25% did not satisfy all ordering/tail guards and does not establish A10 predictor improvement.",
            "Calibration windows are fit-set train-step proxies. Co-located progress is never a solo target; full sustained solo timing is unavailable without clean complete interval evidence."],
        "input_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()}}


def markdown(report):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.3f}"
    lines = ["# A10 feedback-latency pilot", "", report["comparison_scope"], "",
        f"Coverage: {report['verified_receipts']}/{report['expected_jobs']} planned jobs returned verified receipts; "
        f"{report['complete_cohorts']}/{report['planned_cohorts']} cohorts complete. Failed and unattempted jobs remain in the denominator.", "",
        "| Mode | Arm | Eligible repeats | Receipts | Mean job JCT s | Mean request/bracket s | Mean workflow s |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in report["arm_mode_summary"]:
        lines.append(f"| {r['mode']} | {r['arm']} | {r['eligible_cohorts']}/2 | {r['verified_receipts']}/{r['expected_jobs']} | "
            f"{fmt(r['mean_of_repeat_average_job_jct_seconds'])} | {fmt(r['mean_of_repeat_average_request_latency_seconds'])} | {fmt(r['mean_workflow_duration_seconds'])} |")
    lines += ["", f"Failures: {report['failed_or_cancelled_jobs']} failed/cancelled jobs; {report['memory_failure_jobs']} jobs with explicit memory-failure evidence; {report['unreturned_jobs']} planned jobs lack verified feedback (includes unattempted jobs).", "", "Paired changes below are candidate minus reference_v4 in seconds; negative favors the candidate. Both repeats must be eligible.", "",
        "| Mode | Arm | Δ average JCT | Δ request/bracket | Δ p95 JCT | Δ max JCT | Δ workflow | Δ chain |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["paired_comparisons"]:
        d = r["paired_mean_delta"]
        lines.append(f"| {r['mode']} | {r['arm']} | " + " | ".join(fmt(d[k]) for k in (
            "average_job_jct_seconds", "average_request_latency_seconds", "p95_job_jct_seconds", "max_job_jct_seconds",
            "workflow_duration_seconds", "average_chain_latency_seconds")) + " |")
    lines += ["", "Cohort tails, chain latencies, all failure identities/errors, predictor errors and per-repeat paired changes are retained in the JSON export.", "",
        "| Calibration workload | Clean short window | Predicted updates/s | Measured window updates/s | Window rate error | Full sustained solo rate |",
        "|---|---|---:|---:|---:|---|"]
    for r in report["calibration_evidence"]:
        lines.append(f"| {r['name']} | {r['clean_isolated_window_evidence']} | {fmt(r.get('predicted_updates_per_second'))} | "
            f"{fmt(r.get('measured_window_updates_per_second'))} | {fmt(r.get('window_rate_relative_error'))} | unavailable |")
    lines += [""] + [note + "\n" for note in report["notes"]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--calibration-rows", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, help="directory, default campaign directory")
    args = parser.parse_args(argv)
    report = build_report(args.campaign, args.calibration_rows, args.manifest)
    output = args.output or args.campaign
    output.mkdir(parents=True, exist_ok=True)
    (output / "feedback-report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    (output / "feedback-report.md").write_text(markdown(report)+"\n")


if __name__ == "__main__":
    main()
