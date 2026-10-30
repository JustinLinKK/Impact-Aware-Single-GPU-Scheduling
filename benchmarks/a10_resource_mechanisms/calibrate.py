"""Dedicated solo calibration with retained process, update and memory evidence.

The resource class names are hypotheses. Only observed successful-update
windows fit the measured-template table; no neural prediction is invoked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import traceback
import uuid

from ...client import SchedulerClient
from ...domain import JobStatus, utc_now
from ...execution.control import ControlPlane
from ...execution.executor import SubprocessExecutor
from ...execution.process_utils import terminate_process_tree
from ..a10_feedback_latency import run as lifecycle
from ..a10_feedback_latency.receipts import receive
from .manifest import RESOURCE_CLASSES, make_spec, resource_config
from .profiles import fit_templates, positive
from . import workloads

GPU_LOCK_PATH = "/tmp/localml-a10-feedback-pilot.lock"


@contextmanager
def controller_guard():
    """Serialize controllers and convert termination into owned-worker cleanup."""
    import fcntl
    with open(GPU_LOCK_PATH, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        previous = {number: signal.getsignal(number) for number in signals}
        def interrupted(number, frame):
            raise InterruptedError("calibration controller interrupted; cleaning up owned workers")
        try:
            for number in signals:
                signal.signal(number, interrupted)
            yield
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            fcntl.flock(lock, fcntl.LOCK_UN)


def calibration_specs(*, epochs, steps_per_epoch, replicas, source_path,
                      seed_start=1001, pool_seed_start=10001):
    """Seal caller-budgeted replicas of all four hypotheses, with disjoint seeds."""
    if type(replicas) is not int or replicas < 1:
        raise ValueError("positive calibration replica count required")
    rows = []
    for resource_class in RESOURCE_CLASSES:
        for replica in range(replicas):
            index = len(rows)
            rows.append(make_spec(resource_config(resource_class),
                name=f"{resource_class}-calibration-{replica}", role="calibration", chain=None,
                epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed_start + index,
                train_pool_seed=pool_seed_start + 2 * index,
                validation_pool_seed=pool_seed_start + 2 * index + 1, source_path=source_path))
    return rows


def process_snapshot(proc_root=Path("/proc")):
    """Record visible PID ancestry and all kernel-reported namespace aliases."""
    records = []
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            lines = (directory / "status").read_text().splitlines()
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                      for line in lines if ":" in line}
            pid, parent = int(fields["Pid"]), int(fields["PPid"])
            aliases = [int(v) for v in fields.get("NSpid", str(pid)).split()]
            # PID reuse can be inspected from the retained kernel start tick.
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            records.append({"pid": pid, "ppid": parent, "nspid": aliases,
                            "start_ticks": int(stat[19]), "name": fields.get("Name", "")})
        except (FileNotFoundError, ProcessLookupError):
            continue  # A vanished task cannot still be an owned descendant.
        except (PermissionError, KeyError, ValueError, IndexError) as exc:
            raise ValueError("cannot verify visible process ancestry: " + str(directory)) from exc
    return records


def compute_apps(gpu_uuid):
    result = subprocess.run(["nvidia-smi", "--id=" + gpu_uuid,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    apps = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or parts[0] != gpu_uuid or not parts[1].isdigit():
            raise ValueError("unrecognized compute-app process evidence")
        apps.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "process_name": parts[2],
                     "used_gpu_memory_mib": None if parts[3] in {"N/A", "[N/A]"} else positive(float(parts[3]), zero=True)})
    return apps


def verify_isolation(sample, *, gpu_uuid, root_pid=None):
    """Derive ownership from retained ancestry/NSpid; unknown GPU PIDs fail."""
    if sample["gpu"]["gpu_uuid"] != gpu_uuid:
        raise ValueError("GPU UUID changed during calibration")
    records = sample["process_snapshot"]
    by_pid = {row["pid"]: row for row in records}
    if len(by_pid) != len(records):
        raise ValueError("duplicate process identity")
    owned = set()
    if root_pid is not None and root_pid in by_pid:
        owned.add(root_pid)
        while True:
            extra = {pid for pid, row in by_pid.items() if row["ppid"] in owned} - owned
            if not extra:
                break
            owned.update(extra)
    aliases = {alias for pid in owned for alias in [pid, *by_pid[pid]["nspid"]]}
    for app in sample["compute_apps"]:
        if app["gpu_uuid"] != gpu_uuid or app["pid"] not in aliases:
            raise ValueError("unknown GPU compute process; solo isolation unverified")
        # A host PID must resolve uniquely to an owned visible process.
        matches = [pid for pid, row in by_pid.items() if app["pid"] in [pid, *row["nspid"]]]
        if len(matches) != 1 or matches[0] not in owned:
            raise ValueError("ambiguous namespace process ownership")
    return {"owned_pids": sorted(owned), "owned_pid_aliases": sorted(aliases),
            "observed_compute_pids": sorted(app["pid"] for app in sample["compute_apps"])}


def collect_sample(gpu_uuid, *, root_pid=None, launch=None, latest=None, hardware_key=None,
                   evidence_path=None):
    started = time.time()
    metadata = latest.metadata if latest is not None else {}
    sample = {"sample_started_unix": started,
        "root_pid": root_pid, "launch": launch, "hardware_key": hardware_key,
        "worker": {key: metadata.get(key) for key in (
            "runtime_global_step", "runtime_completion_identity", "trial_allocated_mb",
            "trial_peak_allocated_mb", "trial_reserved_mb", "trial_peak_reserved_mb")},
        "worker_observed_unix": metadata.get("trial_memory_sampled_at")}
    try:
        sample["gpu"] = lifecycle.gpu_sample()
        sample["compute_apps"] = compute_apps(gpu_uuid)
        sample["process_snapshot"] = process_snapshot()
        sample["isolation_audit"] = verify_isolation(sample, gpu_uuid=gpu_uuid, root_pid=root_pid)
        if sample["gpu"]["used_mib"] > sample["gpu"]["total_mib"] * .94:
            raise RuntimeError("calibration live GPU memory emergency cutoff")
        return sample
    except Exception as exc:
        sample["evidence_error"] = str(exc)
        raise
    finally:
        sample["sample_finished_unix"] = time.time()
        if evidence_path is not None:
            with Path(evidence_path).open("a") as stream:
                stream.write(json.dumps(sample, allow_nan=False) + "\n")


def audit_pause_checkpoint(path, spec):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    state = value["state"]
    total = spec["epochs"] * spec["steps_per_epoch"]
    step = state["global_step"]
    if (state.get("signature") != spec["expected_runtime_signature"] or type(step) is not int
            or not 0 < step < total or state.get("contract") != shared_contract()):
        raise ValueError("pause checkpoint lacks matching partial-update state")
    counters = [int(row["step"]) for row in state["optimizer_state"]["state"].values()]
    if not counters or any(counter != step for counter in counters):
        raise ValueError("pause checkpoint AdamW counters do not match")
    return {"global_step": step, "signature": state["signature"],
            "sha256": lifecycle.file_hash(path), "bytes": Path(path).stat().st_size}


def shared_contract():
    from ...adapters.shared_training import CONTRACT
    return CONTRACT


def _phase_costs(phase_history, signature, hardware_key):
    costs = {name: [] for name in ("validation", "checkpoint", "finalization")}
    for row in phase_history:
        if row.get("identity") != signature or row.get("hardware_key") != hardware_key:
            raise ValueError("phase evidence source/hardware identity mismatch")
        if row["phase"] in costs:
            costs[row["phase"]].append(positive(row["seconds"], zero=True))
    if any(not values for values in costs.values()):
        raise ValueError("validation/checkpoint/finalization phase evidence missing")
    return {name: max(values) for name, values in costs.items()}


def extract_row(spec, *, hardware_key, gpu_uuid, samples, windows, launches,
                events, phase_history, receipt, final_audit, pause, resources):
    """Fail closed while turning retained raw evidence into fit_templates input."""
    signature = workloads.expected_signature(spec)
    if signature != spec["expected_runtime_signature"] or spec["template_key"] != workloads.template_key(spec):
        raise ValueError("frozen source/spec/signature mismatch")
    total = spec["epochs"] * spec["steps_per_epoch"]
    ballast = workloads.adapter_context(spec)["ballast_bytes"]
    if (resources.get("signature") != signature or resources.get("spec_sha256") != spec["spec_sha256"]
            or resources.get("allocated_ballast_bytes") != ballast
            or resources.get("declared_ballast_bytes") != ballast
            or resources.get("trainable_parameter_bytes") != spec["parameter_count"] * 4
            or resources.get("ballast_initializations_this_launch") != int(ballast > 0)
            or resources.get("ballast_checkpoint_persistent") is not False
            or resources.get("precision") != "fp32_ieee"
            or not str(resources.get("device", "")).startswith("cuda")):
        raise ValueError("actual CUDA ballast/trainable memory report mismatch")
    if (spec.get("role") != "calibration" or len(launches) != 2 or len(windows) < 2
            or receipt["worker_exit_code"] != 0 or receipt["runtime_completion_identity"] != signature
            or final_audit["all_adamw_steps"] != total or final_audit["global_step"] != total):
        raise ValueError("completed updates and two real launches/windows required")
    if (pause["checkpoint"]["signature"] != signature or pause["exit_code"] != 0
            or pause["status"] != "PAUSED" or pause["checkpoint"]["global_step"] != launches[1]["resume_step"]):
        raise ValueError("charged pause/resume checkpoint evidence mismatch")
    pause_seconds = positive(pause["finished_unix"] - pause["requested_unix"])
    if not launches[0]["launched_unix"] <= pause["requested_unix"] <= pause["finished_unix"] <= launches[1]["launched_unix"]:
        raise ValueError("pause/resume ordering invalid")
    if not pause["job_finished_unix"] <= launches[1]["exited_unix"] <= receipt["feedback_ready_unix"]:
        raise ValueError("final process-exit and receipt ordering invalid")
    ready = [event for event in events if event["event_type"] == "trial_runner_ready"]
    if len(ready) != 2:
        raise ValueError("two actual runner-ready events required")
    phase_costs = _phase_costs(phase_history, signature, hardware_key)
    phase_costs.update(startup=positive(lifecycle.timestamp(ready[0]["created_at"]) - launches[0]["launched_unix"]),
        resume=positive(lifecycle.timestamp(ready[1]["created_at"]) - launches[1]["launched_unix"]),
        pause=pause_seconds, delivery=positive(receipt["feedback_ready_unix"] - pause["job_finished_unix"], zero=True))
    for index, event in enumerate(ready):
        if event.get("payload", {}).get("global_step") != launches[index]["resume_step"]:
            raise ValueError("runner readiness does not match launch checkpoint position")
    if len([p for p in phase_history if p["phase"] == "validation"]) != spec["epochs"]:
        raise ValueError("validation phase count does not match frozen epochs")
    if not samples:
        raise ValueError("raw GPU process and allocator samples required")
    device_total = positive(samples[0]["gpu"]["total_mib"])
    for sample in samples:
        index = sample["launch"]
        if (type(index) is not int or index not in (0, 1)
                or sample["root_pid"] != launches[index]["pid"]
                or sample.get("hardware_key") != hardware_key):
            raise ValueError("sample launch/source hardware identity mismatch")
        root = next((record for record in sample["process_snapshot"] if record["pid"] == sample["root_pid"]), None)
        if root is not None and root["start_ticks"] != launches[index]["pid_start_ticks"]:
            raise ValueError("owned worker PID was reused")
        if not launches[index]["launched_unix"] <= sample["sample_started_unix"] <= sample["sample_finished_unix"] <= launches[index]["exited_unix"]:
            raise ValueError("sample lies outside owning launch observation")
        verify_isolation(sample, gpu_uuid=gpu_uuid, root_pid=sample["root_pid"])
        if sample["gpu"]["name"] != "NVIDIA A10":
            raise ValueError("calibration hardware changed")
        if (sample["gpu"]["total_mib"] != device_total
                or positive(sample["gpu"]["used_mib"], zero=True) > .94 * device_total):
            raise ValueError("device memory identity or live guard invalid")
        worker_signature = sample["worker"].get("runtime_completion_identity")
        if worker_signature is not None and worker_signature != signature:
            raise ValueError("worker telemetry identity mismatch")
    selected, allocated, busy, used_launches = [], [], [], set()
    previous_end, previous_finish = -1, -math.inf
    for window in sorted(windows, key=lambda row: row["start_step"]):
        launch_index = window["launch"]
        if type(launch_index) is not int or launch_index not in (0, 1):
            raise ValueError("window launch identity invalid")
        launch = launches[launch_index]
        start, end = window["start_step"], window["end_step"]
        begin, finish = positive(window["started_at"]), lifecycle.timestamp(window["finished_at"])
        if (window.get("complete") is not True or window.get("excludes_nonupdate_phases") is not True
                or type(start) is not int or type(end) is not int or start < launch["resume_step"] + 2
                or start < previous_end or end > total or end - start != window["steps"]
                or end - start < 8 or begin < previous_finish or finish <= begin
                or begin < launch["launched_unix"] or finish > launch["exited_unix"]
                or type(window.get("nonupdate_phase_counter")) is not int
                or window.get("completed_nonupdate_phase_counter") != window["nonupdate_phase_counter"]
                or window.get("completion_identity") != signature):
            raise ValueError("invalid, overlapping or insufficiently warmed update window")
        if launch_index == 0 and (end > pause["checkpoint"]["global_step"] or finish > pause["requested_unix"]):
            raise ValueError("pause checkpoint predates the first measured updates")
        positive(window["elapsed_seconds"])
        raw_window = {key: value for key, value in window.items() if key != "launch"}
        if not any(event["event_type"] == "trial_step_window" and event["payload"] == raw_window for event in events):
            raise ValueError("window lacks matching persisted worker event")
        for phase in phase_history:
            phase_end = lifecycle.timestamp(phase["finished_at"])
            if phase_end > begin and phase_end - phase["seconds"] < finish:
                raise ValueError("non-update phase overlaps training window")
        within = [sample for sample in samples if sample["launch"] == launch_index
            and begin <= sample["sample_started_unix"] <= sample["sample_finished_unix"] <= finish
            and sample["worker_observed_unix"] is not None
            and begin <= sample["worker_observed_unix"] <= finish
            and start <= (sample["worker"].get("runtime_global_step") or -1) <= end
            and sample["compute_apps"]]
        if not within:
            raise ValueError("no owned GPU/memory samples bounded to update window")
        for sample in within:
            allocated.append(positive(sample["worker"]["trial_allocated_mb"]))
            value = positive(sample["gpu"]["utilization_percent"], zero=True)
            if value > 100:
                raise ValueError("invalid GPU-busy sample")
            busy.append(value)
        selected.extend(samples.index(sample) for sample in within)
        used_launches.add(launch_index)
        previous_end, previous_finish = end, finish
    if used_launches != {0, 1}:
        raise ValueError("fresh and resumed clean update windows required")
    def peak(field):
        values = [sample["worker"].get(field) for sample in samples if sample["worker"].get(field) is not None]
        if not values:
            raise ValueError("missing allocator peak evidence: " + field)
        return positive(max(positive(value, zero=True) for value in values))
    peak_allocated, peak_reserved = peak("trial_peak_allocated_mb"), peak("trial_peak_reserved_mb")
    if peak_allocated > peak_reserved:
        raise ValueError("allocator allocated peak exceeds reserved peak")
    if peak_allocated * workloads.MIB < ballast:
        raise ValueError("allocator peak does not cover declared CUDA ballast")
    return {"name": spec["name"], "role": "calibration", "complete": True,
        "hardware_key": hardware_key, "gpu_uuid": gpu_uuid, "job_signature": signature,
        "runtime_completion_identity": signature, "template_key": workloads.template_key(spec),
        "successful_updates": total, "windows": deepcopy(windows),
        "solo_process_isolation_verified": True,
        "isolation_derivation": "every raw compute PID resolved to the sampled owned process tree or NSpid alias",
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        "peak_device_used_mib": positive(max(positive(s["gpu"]["used_mib"], zero=True) for s in samples)),
        "steady_allocated_mib": allocated, "steady_gpu_busy_percent": busy,
        "phase_seconds": phase_costs,
        # Bind observation time to retained evidence so later replay is exact.
        "observed_at": datetime.fromtimestamp(receipt["feedback_ready_unix"], timezone.utc).isoformat(),
        "steady_sample_indices": selected, "raw_evidence_sha256": workloads.digest({
            "samples": samples, "windows": windows, "launches": launches, "events": events,
            "phase_history": phase_history, "pause": pause, "receipt": receipt,
            "final_audit": final_audit, "resources": resources}),
        "resource_accounting": deepcopy(resources),
        "source_identity": {"spec_sha256": spec["spec_sha256"], "source_sha256": spec["source_sha256"],
                            "source": workloads.source_metadata()}}


def _check_sources(snapshot, manifest, manifest_path):
    after = lifecycle.source_snapshot(manifest, manifest_path)
    if after != snapshot:
        raise ValueError("source/manifest snapshot changed during calibration")
    return after


def calibrate(manifest, manifest_path, raw_settings, output, max_seconds):
    """Run dedicated workers only; retain diagnostics even when evidence fails."""
    positive(max_seconds)
    output, manifest_path = Path(output), Path(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("calibration output must be fresh and empty")
    started = time.monotonic()
    rows, audits, active = [], {}, None
    try:
        snapshot = lifecycle.source_snapshot(manifest, manifest_path)
        lifecycle.atomic_json(output / "source-snapshot.json", snapshot)
        specs = manifest.get("calibration") or []
        if not specs or len({s["name"] for s in specs}) != len(specs):
            raise ValueError("unique dedicated calibration specifications required")
        eval_seeds = {s[key] for s in manifest.get("jobs", [])
                      for key in ("seed", "train_pool_seed", "validation_pool_seed")}
        calibration_seeds = []
        for spec in specs:
            if (spec.get("role") != "calibration" or spec.get("workload_kind") != "synthetic_resource_control"
                    or spec["expected_runtime_signature"] != workloads.expected_signature(spec)
                    or spec["template_key"] != workloads.template_key(spec)
                    or spec["steps_per_epoch"] < 32 or spec["epochs"] * spec["steps_per_epoch"] < 64):
                raise ValueError("frozen resource calibration spec or measurement budget invalid")
            calibration_seeds.extend(spec[key] for key in ("seed", "train_pool_seed", "validation_pool_seed"))
        if len(set(calibration_seeds)) != len(calibration_seeds) or set(calibration_seeds) & eval_seeds:
            raise ValueError("dedicated calibration initialization/pool seeds must be disjoint")
        lifecycle.check_deadline(manifest, time.monotonic() - started, max_seconds)
        hardware = lifecycle.gpu_sample()
        if hardware["used_mib"] > 512:
            raise RuntimeError("dedicated calibration requires initially idle GPU")
        initial = collect_sample(hardware["gpu_uuid"], evidence_path=output / "idle-samples.jsonl")
        lifecycle.atomic_json(output / "initial-idle-evidence.json", initial)
        if initial["compute_apps"]:
            raise ValueError("GPU has compute applications before calibration")
        settings = lifecycle.make_settings(raw_settings, output, hardware)
        client, executor, controls = SchedulerClient(settings), SubprocessExecutor(settings), ControlPlane(settings)
        hardware_key = client.store.hardware_key()
        for spec in specs:
            job_started = time.monotonic()
            job_timeout = positive(manifest["job_timeout_seconds"])
            job, work = lifecycle.build_job(spec, manifest_path=manifest_path, output=output,
                manifest=manifest, memory_budget_mib=hardware["total_mib"] * .9)
            def command():
                return {"action": "measure", "owner": "dedicated-resource-calibration", "kind": "reference",
                    "token": uuid.uuid4().hex, "window_min_seconds": positive(manifest.get("calibration_window_seconds", .5)),
                    "window_max_steps": 1024,
                    "expires_at": time.time() + min(job_timeout, max_seconds)}
            job.metadata["trial_step_command"] = command()
            accepted_mono, accepted_unix = time.monotonic(), time.time()
            job = lifecycle.accept_job(client, job, accepted_unix)
            frozen = lifecycle.submission(spec, job, work, accepted_mono, accepted_unix)
            samples, windows, launches, pause = [], [], [], {}
            def launch(latest, resume_step):
                lifecycle.check_deadline(manifest, time.monotonic() - started, max_seconds)
                if time.monotonic() - job_started >= job_timeout:
                    raise TimeoutError("predeclared dedicated calibration job timeout before launch")
                entry = {"launched_unix": time.time(), "resume_step": resume_step}
                handle = executor.start(latest)
                try:
                    entry["pid"] = handle.process.pid
                    entry["pid_start_ticks"] = next(record["start_ticks"] for record in process_snapshot()
                                                      if record["pid"] == handle.process.pid)
                except Exception:
                    terminate_process_tree(handle.process)
                    raise
                launches.append(entry)
                return handle
            client.store.set_job_status(job.job_id, JobStatus.READY, reason="dedicated resource calibration", hold=False)
            active = launch(client.store.get_job(job.job_id), 0)
            try:
                while True:
                    lifecycle.check_deadline(manifest, time.monotonic() - started, max_seconds)
                    if time.monotonic() - job_started >= job_timeout:
                        raise TimeoutError("predeclared dedicated calibration job timeout")
                    latest = client.store.get_job(job.job_id)
                    sample = collect_sample(hardware["gpu_uuid"], root_pid=active.process.pid,
                        launch=len(launches) - 1, latest=latest, hardware_key=hardware_key,
                        evidence_path=work / "samples.jsonl")
                    samples.append(sample)
                    if sample["gpu"]["total_mib"] != hardware["total_mib"]:
                        raise ValueError("GPU device memory identity changed")
                    window = latest.metadata.get("trial_step_window")
                    if window and window.get("complete") and not any(w["token"] == window["token"] for w in windows):
                        windows.append({**deepcopy(window), "launch": len(launches) - 1})
                        if len(launches) == 1:
                            pause["requested_unix"] = time.time()
                            controls.request_pause(job.job_id, reason="charged dedicated calibration pause/resume", hold=True)
                        else:
                            client.store.update_job(job.job_id, metadata_updates={"trial_step_command": {}})
                    code = active.process.poll()
                    if code is not None:
                        launches[-1].setdefault("exited_unix", time.time())
                    if code is not None and latest.status == JobStatus.PAUSED:
                        if code != 0 or len(launches) != 1 or not latest.latest_checkpoint_path:
                            raise ValueError("unexpected dedicated calibration pause exit")
                        pause.update(finished_unix=time.time(), exit_code=code, status="PAUSED",
                                     checkpoint=audit_pause_checkpoint(latest.latest_checkpoint_path, spec))
                        lifecycle.atomic_json(work / "pause-checkpoint-audit.json", pause)
                        # Retain the pause artifact even if the normal checkpoint retention later prunes it.
                        import shutil
                        shutil.copyfile(latest.latest_checkpoint_path, work / "pause-checkpoint.pt")
                        controls.clear_command(job.job_id)
                        latest = client.store.update_job(job.job_id, status=JobStatus.READY, hold=False,
                            resume_from_checkpoint=latest.latest_checkpoint_path,
                            metadata_updates={"trial_step_window": None, "trial_step_command": command()})
                        active = launch(latest, pause["checkpoint"]["global_step"])
                        continue
                    receipt = receive(frozen, latest, code)
                    if receipt is not None:
                        break
                    time.sleep(.01)
            finally:
                terminate_process_tree(active.process)
                active = None
                lifecycle.atomic_json(work / "launches.json", launches)
                lifecycle.atomic_json(work / "windows.json", windows)
                lifecycle.atomic_json(work / "pause-evidence.json", pause)
            pause["job_finished_unix"] = lifecycle.timestamp(latest.finished_at)
            events = lifecycle.events_for(settings.events_jsonl_path, job.job_id)
            phases = latest.metadata.get("runtime_phase_timing_history") or []
            audits[spec["name"]] = lifecycle.audit_checkpoint(spec, work, latest)
            evidence = {"samples": samples, "windows": windows, "launches": launches, "events": events,
                "phase_history": phases, "pause": pause, "receipt": receipt, "final_audit": audits[spec["name"]],
                "resources": json.loads((work / "working/shared_training/workload_resources.json").read_text())}
            lifecycle.atomic_json(work / "raw-calibration-evidence.json", evidence)
            lifecycle.atomic_json(work / "receipt.json", receipt)
            lifecycle.atomic_json(work / "job-final.json", latest.to_dict())
            _check_sources(snapshot, manifest, manifest_path)
            row = extract_row(spec, hardware_key=hardware_key, gpu_uuid=hardware["gpu_uuid"], **evidence)
            rows.append(row)
            lifecycle.atomic_json(output / "calibration-rows.json", rows)
            lifecycle.atomic_json(output / "validity-audits.json", audits)
            # The preceding worker has exited; require a clean idle process list before the next replica.
            idle = collect_sample(hardware["gpu_uuid"], evidence_path=output / "idle-samples.jsonl")
            lifecycle.atomic_json(work / "post-exit-idle-evidence.json", idle)
            if idle["compute_apps"]:
                raise ValueError("compute process remains after dedicated worker exit")
        table = fit_templates(rows, hardware_key=hardware_key, gpu_uuid=hardware["gpu_uuid"], source_identity=snapshot)
        lifecycle.atomic_json(output / "source-snapshot-after.json", _check_sources(snapshot, manifest, manifest_path))
        lifecycle.atomic_json(output / "calibration.json", table)
        result = {"complete": True, "kind": "dedicated_resource_calibration", "calibration": table,
            "elapsed_seconds": time.monotonic() - started, "hardware": hardware,
            "calibration_rows": rows, "source_and_budget_audits": audits,
            "class_note": "resource class names are hypotheses; only retained measurements populate templates"}
        lifecycle.atomic_json(output / "result.json", result)
        return result
    except BaseException as exc:
        if active is not None:
            terminate_process_tree(active.process)
        lifecycle.atomic_json(output / "result.json", {"complete": False, "kind": "dedicated_resource_calibration",
            "error": str(exc), "exception": type(exc).__name__, "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started, "calibration_rows": rows})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    positive(args.max_seconds)
    with controller_guard():
        args.output.mkdir(parents=True, exist_ok=False)
        calibrate(json.loads(args.manifest.read_text()), args.manifest, json.loads(args.settings.read_text()),
                  args.output, args.max_seconds)


if __name__ == "__main__":
    main()
