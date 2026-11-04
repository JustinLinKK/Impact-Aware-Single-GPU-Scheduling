"""Opt-in V5.1.1 FIFO episode; no provider management or calibration training.

Run only with an explicitly sealed manifest/calibration and bounded deadline.
Real graph capture and profile preparation are charged after job acceptance.
Optional service_factory/gpu_reader are CPU test seams, never an evaluation proof.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import json,math,re,signal,statistics,time,traceback
from ...scheduler.hardware_concurrency import MAX_START
from ...client import SchedulerClient
from ...domain import JobStatus
from ...execution.executor import SubprocessExecutor
from ...execution.process_utils import terminate_process_tree
from ..a10_feedback_latency.run import (atomic_json,check_deadline,with_hard_deadline,
    accept_job,submission,parent_due,fixed_fifo_fits,require_predictor_exposure,
    audit_checkpoint,events_for,prune_owned_weights,policy_activation,make_settings,
    gpu_sample as default_gpu_sample,source_snapshot as baseline_source_snapshot,STUDENT_SHA256)
from ..a10_v5_ablation.job_builder import build_job
from ..a10_feedback_latency.calibration import CalibratedPredictor,RecordingPredictor,phase_model,trial_reserve
from ..a10_feedback_latency.receipts import receive,request_latency
from ..a10_feedback_latency.async_preparation import AsyncV4Service

SOURCE_COMMIT_NOTE='Caller-declared base commit; complete Python SHA map identifies current bytes. No clean Git claim.'

ARMS=('v5_1_1',)

def arm_flags(arm, *, development_shadow=False):
    if arm!='v5_1_1' or development_shadow:raise ValueError('explicit V5.1 FIFO arm required')
    return dict(continuation=False,deferral=False,bracket=False)

def make_service(settings,*,store,arm,flags):
    return AsyncV4Service(settings,store=store) # plumbing only; background scheduler never starts

def validate_manifest(manifest,*,arm,mode):
    arm_flags(arm)
    if 'development' in manifest and type(manifest['development']) is not bool:
        raise ValueError('explicit development boolean required')
    if 'development_fixed_cap' in manifest:
        cap = manifest['development_fixed_cap']
        if (manifest.get('development') is not True or arm != 'fixed_three'
                or type(cap) is not int or not 1 <= cap <= 4):
            raise ValueError('development fixed cap1..4 requires development fixed_three')
    if 'development_shadow' in manifest:
        if (manifest.get('development') is not True or arm != 'forecast_only'
                or manifest['development_shadow'] is not True):
            raise ValueError('development shadow requires development forecast_only')
    if mode not in ('individual','group'):raise ValueError('explicit request mode required')
    specs=manifest['jobs'];names=[s['name']for s in specs]
    chains=sorted({s['chain']for s in specs})
    if (len(chains)not in (3,6)or chains!=list(range(len(chains)))or len(specs)!=2*len(chains)
            or len(set(names))!=len(names)or any(s['role']not in ('parent','child')for s in specs)
            or any(sum(s['chain']==c and s['role']==role for s in specs)!=1 for c in chains for role in ('parent','child'))
            or any(s['epochs']!=2 or type(s['steps_per_epoch'])is not int or s['steps_per_epoch']<=0 for s in specs)):
        raise ValueError('exactly3or6 sealed parent-child chains and positive two-epoch budgets required')
    delays=manifest.get('preparation_delay_seconds_by_name',{})
    if (not isinstance(delays,dict)or not set(delays)<=set(names)
            or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<0 for v in delays.values())):
        raise ValueError('explicit finite nonnegative preparation delays required')
    source=manifest.get('profile_source','ml_predictor')
    if source not in ('dedicated_solo_template_proxy','ml_predictor'):raise ValueError('explicit supported common profile condition required')
    return chains

def source_snapshot(manifest,path,source_commit):
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('explicit source base commit pin required')
    value=baseline_source_snapshot(manifest,path)
    value['scheduler_reference_commit']=source_commit
    value['commit_identity_note']=SOURCE_COMMIT_NOTE
    return value

def v5_activation(events):
    selected = [e for e in events if e.get('event_type', '').startswith(('v5_', 'bracket_bootstrap_'))]
    measured = Counter()
    outcomes = Counter()
    for event in selected:
        if event.get('event_type') != 'bracket_bootstrap_decided': continue
        payload = event.get('payload') or {}
        decision = str(payload.get('decision', 'unknown'))
        outcomes[decision] += 1
        if (decision in {'accepted', 'rejected'} and payload.get('measured_full_group') is True
                and payload.get('error') is None and isinstance(payload.get('halves'), list)
                and len(payload['halves']) == 2):
            measured[decision] += 1
    return {'events': selected, 'counts': dict(Counter(e['event_type'] for e in selected)),
        'bootstrap_outcomes': dict(outcomes), 'bootstrap_measured_decisions': dict(measured),
        'bootstrap_actionable': sum(measured.values())}


def _run_episode(manifest, manifest_path, raw_settings, output, calibration, *, arm, mode, max_seconds, source_commit, service_factory=None, gpu_reader=None, controller_factory=None):
    chains_declared = validate_manifest(manifest, arm=arm, mode=mode)
    gpu_sample = gpu_reader or default_gpu_sample
    setup_started = time.monotonic()
    hardware = gpu_sample()
    if hardware["used_mib"] > 512: raise RuntimeError("episode requires an otherwise idle A10")
    settings = make_settings(raw_settings, output, hardware)
    client = SchedulerClient(settings)
    resource_control = manifest.get("profile_source") == "dedicated_solo_template_proxy"
    fixed_three = True # manual no-preemption FIFO executor lifecycle, not fixed-three policy
    if controller_factory is None:
        from ...scheduler.adaptive_fifo_v511 import AdaptiveFIFO511Controller
        from ...scheduler.fifo_experience import ExperienceStore
        trace = manifest['v511_trace']
        if not trace['history_namespace'] or Path(trace['history_path']).name != trace['history_namespace']:
            raise ValueError('explicit owned history namespace required')
        history = ExperienceStore(trace['history_namespace'], path=trace['history_path'])
        controller_factory = lambda mode: AdaptiveFIFO511Controller(mode, store=history,
            episode=trace['episode'], gpu_name=hardware['name'])
    controller = controller_factory(mode=mode)
    revised, causal, asynchronous = False, True, True
    flags = arm_flags(arm, development_shadow=manifest.get('development_shadow', False))
    fixed_cap = manifest.get('development_fixed_cap', 3)
    service = (service_factory or make_service)(settings, store=client.store, arm=arm, flags=flags)
    supervisor_exit_observations, harness_exit_observations = [], {}
    if causal:
        original_record_exit = service.supervisor._record_process_exit
        def record_exit(job_id, returncode):
            observed = time.monotonic()
            original_record_exit(job_id, returncode)
            supervisor_exit_observations.append({"job_id": job_id, "returncode": returncode,
                "observed_monotonic": observed,
                "source": "supervisor_record_process_exit_entry_after_process_poll"})
        service.supervisor._record_process_exit = record_exit
    released_for_fingerprint = service.planner.estimator.ml_predictor._released
    source_tag, phase_factory = "ml_predictor", phase_model
    if resource_control:
        from ..a10_resource_mechanisms.profiles import MeasuredTemplatePredictor, SOURCE, phase_model as template_phases
        source_tag, phase_factory = SOURCE, template_phases
        service.planner.estimator.ml_predictor = MeasuredTemplatePredictor(calibration,
            client.store.hardware_profile(), gpu_uuid=hardware["gpu_uuid"])
    else:
        predictor = service.planner.estimator.ml_predictor
        if not predictor or not predictor.available: raise ValueError("published A10 predictor unavailable")
        if predictor.artifact_hash != STUDENT_SHA256: raise ValueError("published S_A10 student hash mismatch")
        if causal or arm in {"calibrated_v4", "calibrated_revised"}:
            service.planner.estimator.ml_predictor = CalibratedPredictor(predictor, calibration, client.store.hardware_key())
        elif predictor.artifact_hash != calibration["predictor_artifact_sha256"]:
            raise ValueError("reference artifact differs from dedicated calibration")
    jobs, works, specs = {}, {}, {}
    for spec in manifest["jobs"]:
        job, work = build_job(spec, manifest_path=manifest_path, output=output, manifest=manifest,
                              memory_budget_mib=hardware["total_mib"] * .9)
        job.metadata["feedback_phase_model"] = phase_factory(calibration, job,
            hardware_key=client.store.hardware_key(), batch_size=spec["batch"])
        if resource_control:
            entry = calibration["entries"][job.metadata["resource_template_key"]]
            c = entry["phase_seconds"]
            reserve = c["startup"] + c["pause"] + c["resume"] + 20 / entry["updates_per_second"]
        else:
            reserve = trial_reserve(calibration)
        if resource_control:
            job.metadata["v5_startup_envelope"] = {
                "source": "dedicated_same_host_peak_device_plus_headroom",
                "peak_mib": entry["admission_reservation_mib"],
                "hardware_key": client.store.hardware_key(), "artifact_sha256": calibration["artifact_sha256"],
                "job_signature": job.packing.signature or job.job_id, "batch_size": spec["batch"]}
        # Neural predictions are not measured startup envelopes; leave unknown.
        job.metadata["feedback_trial_reserve_seconds"] = reserve
        jobs[spec["name"]], works[spec["name"]], specs[spec["name"]] = job, work, spec
    names_by_id = {job.job_id: name for name, job in jobs.items()}
    # Template controls already use cheap common prepared lookups. Process
    # isolation is needed only for costly neural source capture/inference.
    preparation_workers = 2 if mode == 'group' and not resource_control else 1
    preparation_class = None
    if preparation_workers == 2:
        from ...scheduler.parallel_preparation import IsolatedPredictionBackend, ParallelPreparedPredictor
        service.planner.estimator.ml_predictor = IsolatedPredictionBackend(
            service.planner.estimator.ml_predictor, kind=source_tag, settings_dict=settings.to_dict(),
            calibration=calibration, hardware=client.store.hardware_profile(),
            gpu_uuid=hardware['gpu_uuid'], workers=2)
        from .preparation import GraphParallelPreparedPredictor
        preparation_class = GraphParallelPreparedPredictor
    from ...prediction.fingerprint_preparation import FingerprintPredictor
    from dataclasses import asdict
    graph_preparation = FingerprintPredictor(service.planner.estimator.ml_predictor,
        released_for_fingerprint, gpu_uuid=hardware['gpu_uuid'],
        runtime_identity=asdict(client.store.hardware_profile()),
        backend=settings.gpu_scheduler.packing_backend, profile_source=source_tag,
        source_closure=source_snapshot(manifest, manifest_path, source_commit)['scheduler_python_sources'], optional_async=True)
    service.planner.estimator.ml_predictor = graph_preparation
    recording = RecordingPredictor(service.planner.estimator.ml_predictor,
        lambda job_id, profiles: atomic_json(works[names_by_id[job_id]] / "predicted-profiles.json", profiles))
    service.planner.estimator.ml_predictor = recording
    preparation = None
    if causal:
        from ..a10_feedback_latency.async_preparation import PreparedPredictor
        prepare_type = preparation_class or PreparedPredictor
        preparation = prepare_type(recording, asynchronous=asynchronous,
            prepare_on_read=not asynchronous, hardware="A10", profile_source=source_tag,
            **({'workers': 2} if preparation_workers == 2 else {}))
        service.planner.estimator.ml_predictor = preparation
        if asynchronous and not fixed_three:
            service.predictor_preparation = preparation
    parents = {s["chain"]: s["name"] for s in manifest["jobs"] if s["role"] == "parent"}
    children = {s["chain"]: s["name"] for s in manifest["jobs"] if s["role"] == "child"}
    if set(parents) != set(chains_declared) or set(children) != set(chains_declared): raise ValueError("sealed chain membership changed")
    start, start_unix = time.monotonic(), time.time()
    submitted, receipts, request_specs, handles, exit_codes = {}, {}, {}, {}, {}
    arrivals, timeline, audits, first_progress, last_progress = [], [], {}, {}, {}
    executor = SubprocessExecutor(settings)
    prediction_pool = None  # All arms share charged asynchronous preparation.
    prediction_futures, fixed_profiles = {}, {}
    if not fixed_three: service.start(background=True)
    next_sample, group_child_release = 0., None
    observation_sample, observation_sample_at = hardware, 0.
    telemetry_seconds = []
    delayed_preparation = {}
    release_observations = {}
    process_exit_observed = {}
    dispatch_events=[];head_block_seconds=Counter();occupancy_seconds=Counter();observe_wall=[];observe_cpu=[];last_alpha_state=None
    last_accounted=start;previous_block=None;previous_occupancy=0;head_name=None;head_block=None
    try:
        while len(receipts) != len(jobs):
            now_accounted=time.monotonic();delta=max(0.,now_accounted-last_accounted)
            occupancy_seconds[str(previous_occupancy)]+=delta
            if previous_block:head_block_seconds[previous_block]+=delta
            last_accounted=now_accounted
            elapsed = now_accounted - start
            if not fixed_three:
                failure = getattr(service, 'v5_failure', None)
                scheduler_thread = getattr(service, '_thread', None)
                if failure or (scheduler_thread is not None and not scheduler_thread.is_alive()):
                    raise RuntimeError('scheduler failed before complete feedback: ' + str(failure or 'thread exited'))
            check_deadline(manifest, elapsed, max_seconds)
            require_predictor_exposure(recording, jobs, specs, required_source=source_tag)
            if preparation is not None and preparation.failures:
                raise RuntimeError("prediction preparation failed: " + str(preparation.failures))
            if mode == "group" and group_child_release is None and all(jobs[n].job_id in receipts for n in parents.values()):
                group_child_release = max(receipts[jobs[n].job_id]["feedback_ready_monotonic"] for n in parents.values()) + 2.
            for name, spec in specs.items():
                if name in submitted: continue
                if spec["role"] == "parent": due = parent_due(start, spec["chain"], mode)
                elif mode == "group": due = group_child_release
                else:
                    parent = jobs[parents[spec["chain"]]].job_id
                    due = receipts[parent]["feedback_ready_monotonic"] + 2. if parent in receipts else None
                if due is None or time.monotonic() < due: continue
                accepted_mono, accepted_unix = time.monotonic(), time.time()
                job = jobs[name]
                required = ([job.job_id] if mode == "individual" else
                    sorted(jobs[n].job_id for n in (parents if spec["role"] == "parent" else children).values()))
                request_id = name if mode == "individual" else spec["role"] + "-bracket"
                if request_id not in request_specs:
                    request_specs[request_id] = {"mode": mode, "required_ids": required,
                        "submitted_monotonic": accepted_mono, "submitted_unix": accepted_unix}
                request = request_specs[request_id]
                job.metadata["feedback_request"] = {"request_id": request_id,
                    "workflow_id": "a10-v5-ablation", "completion_policy": mode,
                    "required_job_ids": required, "membership_finalized": True,
                    "submitted_at": datetime.fromtimestamp(request["submitted_unix"], timezone.utc).isoformat()}
                job = accept_job(client, job, accepted_unix)
                submitted[name] = submission(spec, job, works[name], accepted_mono, accepted_unix)
                if preparation is not None and asynchronous:
                    delay = manifest.get("preparation_delay_seconds_by_name", {}).get(name, 0.)
                    delayed_preparation[name] = accepted_mono + delay
                if prediction_pool is not None:
                    prediction_futures[name] = prediction_pool.submit(service.planner.estimator.predicted_profile,
                                                                      job, spec["batch"])
                arrivals.append({"name": name, "due_seconds": due - start,
                                 "accepted_seconds": accepted_mono - start})
            for name, due_preparation in list(delayed_preparation.items()):
                if time.monotonic() >= due_preparation:
                    preparation.enqueue(jobs[name], specs[name]["batch"])
                    del delayed_preparation[name]
            current = {n: client.store.get_job(jobs[n].job_id) for n in submitted}
            if fixed_three:
                for name, handle in list(handles.items()):
                    code = handle.process.poll()
                    if code is not None:
                        exit_codes[name] = code
                        process_exit_observed.setdefault(name,time.monotonic())
                        del handles[name]
                pending = [n for n in submitted if n not in handles and n not in exit_codes]
                current = {n: client.store.get_job(jobs[n].job_id) for n in submitted}
                ready_profiles = {}
                for name in submitted:
                    if preparation is not None and preparation.ready(jobs[name], specs[name]['batch']):
                        ready_profiles[jobs[name].job_id] = graph_preparation.augment(jobs[name], specs[name]['batch'],
                            service.planner.estimator.predicted_profile(jobs[name], specs[name]['batch']))
                for name, job in current.items():
                    job.metadata['v51_accepted_at']=submitted[name].accepted_monotonic
                    job.metadata['v51_hardware_key']=client.store.hardware_key()
                    job.metadata['v51_backend']=settings.gpu_scheduler.packing_backend
                    job.metadata['v51_phase_model_ttl_seconds']=settings.gpu_scheduler.feedback_latency.phase_model_ttl_seconds
                    request_id=job.metadata['feedback_request']['request_id']
                    job.metadata['v51_request_accepted_at']=request_specs[request_id]['submitted_monotonic']
                    if name in process_exit_observed:
                        job.metadata['v51_phase']='delivering'
                        job.metadata['v51_resource_released_at']=process_exit_observed[name]
                # Shared episode telemetry is sampled once per second. Each
                # dispatch still performs its independent fresh memory check.
                if time.monotonic() - observation_sample_at >= 1.:
                    telemetry_started = time.perf_counter()
                    observation_sample = gpu_sample()
                    observation_sample_at = time.monotonic()
                    telemetry_seconds.append(time.perf_counter() - telemetry_started)
                for j in current.values():
                    if j.job_id in ready_profiles:
                        j.metadata['trial_reserved_mb'] = max(j.metadata.get('trial_reserved_mb') or 0.,
                            ready_profiles[j.job_id]['v511_reservation_mib'])
                observe_started=time.perf_counter();observe_cpu_started=time.thread_time()
                controller.observe(now=time.monotonic(),queued_jobs=[current[n] for n in submitted if n not in handles],
                    running_jobs=[current[n] for n in handles],profiles=ready_profiles,
                    gpu_sample=observation_sample,completed_job_ids={key:row['feedback_ready_monotonic'] for key,row in receipts.items()})
                observe_wall.append(time.perf_counter()-observe_started);observe_cpu.append(time.thread_time()-observe_cpu_started)
                # Charge the whole sampled observer, including durable history,
                # when assessing whether a later forecast gain clears overhead.
                if controller._last_observation >= observation_sample_at:
                    controller._last_overhead_seconds = max(controller._last_overhead_seconds, observe_wall[-1])
                fixed_cap=controller.admission_limit
                if type(fixed_cap)is not int or not 1<=fixed_cap<=MAX_START:raise ValueError('finite positive actual-cohort admission limit required')
                head_name=pending[0] if pending else None
                head_block='alpha' if pending and len(handles)>=fixed_cap else None
                for name in pending[:max(0, fixed_cap - len(handles))]:
                    latest=client.store.get_job(jobs[name].job_id)
                    if latest.hold:
                        head_block='held';break
                    if preparation is not None:
                        if not preparation.ready(jobs[name], specs[name]["batch"]):
                            head_block="prep";break
                        fixed_profiles[name] = service.planner.estimator.predicted_profile(jobs[name], specs[name]["batch"])
                    else:
                        future = prediction_futures[name]
                        if not future.done(): break  # Strict FIFO head remains pending during charged conversion.
                        fixed_profiles[name] = future.result()
                    sample = gpu_sample()
                    running = {n: client.store.get_job(jobs[n].job_id) for n in handles}
                    admission_profiles = {n: {**p, 'peak_vram_mib': p['v511_reservation_mib']}
                                          for n, p in fixed_profiles.items()}
                    if not fixed_fifo_fits(admission_profiles[name], running, admission_profiles,
                            device_used_mib=sample["used_mib"], budget_mib=hardware["total_mib"] * .9):
                        if not handles: raise ValueError("fixed FIFO head does not fit A10 budget even alone")
                        head_block="memory";break
                    latest = current[name]
                    client.store.set_job_status(latest.job_id, JobStatus.READY, reason="V5.1 strict FIFO adaptive alpha limit" + str(fixed_cap))
                    launch_started=time.monotonic()
                    handles[name] = executor.start(client.store.get_job(latest.job_id))
                    controller.on_launch(latest.job_id,launch_started)
                    dispatch_events.append({'job_id':latest.job_id,'name':name,'launched_monotonic':launch_started,'alpha':fixed_cap,'resident_count':len(handles),
                        'memory_gate': {'device_used_mib': sample['used_mib'], 'budget_mib': hardware['total_mib'] * .9,
                            'candidate_reservation_mib': admission_profiles[name]['peak_vram_mib'],
                            'resident_reservations_mib': {n: max(admission_profiles[n]['peak_vram_mib'],
                                float(j.metadata.get('trial_reserved_mb') or 0.)) for n,j in running.items()}}})
                    if hasattr(controller,'export_state'):
                        exported=controller.export_state()
                        if exported!=last_alpha_state:atomic_json(output/'alpha-state.json',exported);last_alpha_state=exported
                remaining=[n for n in pending if n not in handles]
                head_name=remaining[0] if remaining else None
                if head_name and current[head_name].hold:head_block='held'
                elif head_name and len(handles)>=fixed_cap:head_block='alpha'
                previous_block=head_block if head_name else None;previous_occupancy=len(handles)
                if hasattr(controller,'export_state'):
                    exported=controller.export_state()
                    if exported!=last_alpha_state:atomic_json(output/'alpha-state.json',exported);last_alpha_state=exported
            for name, latest in current.items():
                step = latest.metadata.get("runtime_global_step", 0)
                if step > 0:
                    first_progress.setdefault(name, {"observed_seconds": elapsed, "step": step})
                    if name not in last_progress or step > last_progress[name]["step"]:
                        last_progress[name] = {"observed_seconds": elapsed, "step": step}
                if latest.job_id in receipts: continue
                code = exit_codes.get(name) if fixed_three else service.supervisor.observed_exit_code(latest.job_id)
                if code is not None and latest.status == JobStatus.COMPLETED:
                    release_observations.setdefault(latest.job_id, {
                        "observed_monotonic": time.monotonic(), "returncode": code,
                        "source": "worker_exit_observed_resource_release_upper_bound",
                        "physical_cuda_release_exact_time_known": False})
                    harness_exit_observations.setdefault(latest.job_id, {
                        "observed_monotonic": time.monotonic(), "returncode": code,
                        "source": "harness_observation_before_receipt_parse"})
                receipt = receive(submitted[name], latest, code)
                if receipt is not None:
                    receipts[latest.job_id] = receipt
                    atomic_json(works[name] / "receipt.json", receipt)
                    client.store.update_job(latest.job_id, metadata_updates={
                        "feedback_ready_at": datetime.fromtimestamp(time.time(), timezone.utc).isoformat(),
                        "feedback_ready_monotonic": receipt["feedback_ready_monotonic"],
                        "v5_resource_release_observation": release_observations.get(latest.job_id)})
            if elapsed >= next_sample:
                sample = observation_sample
                if sample["used_mib"] > sample["total_mib"] * .94:
                    raise RuntimeError("GPU live memory exceeds predeclared emergency cutoff")
                timeline.append({"elapsed_seconds": elapsed, "gpu": sample,"target_alpha":controller.admission_limit,"actual_resident_count":len(handles),"fifo_head":head_name,"head_block_reason":head_block if head_name else None,
                    "jobs": {n: {"status": j.status.value, "global_step": j.metadata.get("runtime_global_step", 0),
                        "allocated_mib": j.metadata.get("trial_allocated_mb"),
                        "peak_allocated_mib": j.metadata.get("trial_peak_allocated_mb"),
                        "reserved_mib": j.metadata.get("trial_reserved_mb"),
                        "peak_reserved_mib": j.metadata.get("trial_peak_reserved_mb")} for n, j in current.items()}})
                with (output / "timeline.jsonl").open("a") as stream: stream.write(json.dumps(timeline[-1]) + "\n")
                atomic_json(output / "status.json", {"elapsed_seconds": elapsed, "submitted": len(submitted), "receipts": len(receipts)})
                next_sample = elapsed + 1.
            time.sleep(.02)
        require_predictor_exposure(recording, jobs, specs, require_all=True, required_source=source_tag)
    finally:
        if not fixed_three: service.stop()
        for handle in handles.values(): terminate_process_tree(handle.process)
        if prediction_pool is not None: prediction_pool.shutdown(wait=False, cancel_futures=True)
        atomic_json(output / "alpha-decisions.json", {"events":controller.events,"ordered_launch_events":dispatch_events,"observe_wall_seconds_total":sum(observe_wall),"observe_wall_seconds_max":max(observe_wall,default=0.),"observe_thread_cpu_seconds_total":sum(observe_cpu),"observe_thread_cpu_seconds_max":max(observe_cpu,default=0.),"head_block_seconds":dict(head_block_seconds),"occupancy_seconds_by_resident_count":dict(occupancy_seconds),"final_admission_limit":controller.admission_limit,"accepted_monotonic_by_job":{jobs[n].job_id:row.accepted_monotonic for n,row in submitted.items()},"resource_release_observed_monotonic_by_job":{jobs[n].job_id:t for n,t in process_exit_observed.items()},"clock_note":"resource release is upper bounded by actual process poll observation, never inferred from optimizer completion"})
        if causal:
            atomic_json(output / "exit-observations.json", {
                "episode_started_monotonic": start,
                "supervisor_events_including_pause_exits": supervisor_exit_observations,
                "completed_job_harness_observations": harness_exit_observations,
                "note": "observations bound process exit from above; worker finished_at precedes OS exit"})
        if preparation is not None:
            stopped = preparation.stop(timeout=3.)
            atomic_json(output / "preparation-timeline.json", {
                "episode_started_monotonic": start, "asynchronous": asynchronous,
                "worker_stopped": stopped, "events": preparation.events,
                "failures": preparation.failures,
                "isolated_process_cleanup": getattr(getattr(preparation, "_backend", None), "cleanup_report", None),
                "timeout_contract": getattr(getattr(preparation, "_backend", None), "timeout_contract", None)})
        features_stopped = graph_preparation.stop(timeout=3.)
        atomic_json(output / 'optional-feature-preparation.json', {
            'worker_stopped': features_stopped, 'events': graph_preparation.capture_events,
            'failures': graph_preparation.failures,
            'ready_fingerprints': {j + ':' + str(b): fp for (j, b), fp in graph_preparation._results.items()},
            'contract': 'numeric profiles gate FIFO; optional features never gate launch; no cross-seed cache substitution'})
        if not features_stopped:
            raise RuntimeError('optional graph worker cleanup incomplete')
        atomic_json(output / "predictor-exposure.json", {"failures": recording.failures,
            "prediction_sources": dict(recording.last_sources), "prediction_errors": dict(recording.last_errors),
            "prepared_numeric_profiles": recording.records})
        if preparation is not None and not stopped:
            raise RuntimeError("bounded preparation stop incomplete; owned process-tree cleanup required")
    end = time.monotonic()
    result_ready = max(row["feedback_ready_monotonic"] for row in receipts.values())
    for name, spec in specs.items():
        latest = client.store.get_job(jobs[name].job_id)
        audits[name] = audit_checkpoint(spec, works[name], latest)
        atomic_json(works[name] / "job-final.json", latest.to_dict())
        atomic_json(works[name] / "events.json", events_for(settings.events_jsonl_path, latest.job_id))
    atomic_json(output / "validity-audits.json", audits)
    latencies = {name: request_latency(row["mode"], row["required_ids"], receipts,
                                     row["submitted_monotonic"]) for name, row in request_specs.items()}
    chains = {str(chain): receipts[jobs[children[chain]].job_id]["feedback_ready_monotonic"]
              - submitted[parents[chain]].accepted_monotonic for chain in chains_declared}
    result = {"complete": True, "arm": arm, "mode": mode, "requestor": "colocated_python_controller",
        "common_calibrated_profile_access": causal,
        "profile_source": source_tag,
        "evaluation_scope": "scheduler mechanism control; no neural accuracy claim" if resource_control else "released student pipeline",
        "harness_exit_observations": harness_exit_observations,
        "preparation_treatment": ("acceptance_enqueued_serial_async" if asynchronous else
            "first_demand_sync") if causal else "legacy_pilot",
        "episode_started_monotonic": start,
        "transport": "local filesystem JSON read+parse+sha256 after actual successful worker exit",
        "episode_started_unix": start_unix, "episode_duration_seconds": result_ready - start,
        "pre_request_setup_seconds": start - setup_started,
        "scheduler_shutdown_seconds": end - result_ready,
        "request_latencies_seconds": latencies, "mean_request_latency_seconds": statistics.mean(latencies.values()),
        "chain_root_to_child_seconds": chains, "arrivals": arrivals,
        "first_progress_observations": first_progress, "last_progress_observations": last_progress,
        "calibration_sha256": calibration["artifact_sha256"] if resource_control else calibration["calibration_sha256"], "receipts": receipts,
        "source_and_budget_audits": audits,
        "phase_model_role": "dedicated_same_configuration_template_proxy" if resource_control else "calibrated_cross_workload_phase_proxy_v1",
        "group_arrival_note": "group parents at start; individual parents at2*chain seconds; children after usable feedback+2s",
        "chains": len(chains_declared), "v5_flags": flags, "preparation_workers": preparation_workers,
        "preparation_shutdown_note": "Bounded preparation shutdown verified; isolated workers additionally retain birth-checked process cleanup proof. Serial capture retains its declared timeout and owned episode tree bound.",
        "declared_source_commit": source_commit, "release_observations": release_observations,
        "fixed_reference_memory_profiles": fixed_profiles,
        "prepared_numeric_profiles": recording.records,
        "hardware": hardware, "peak_device_used_mib": max(x["gpu"]["used_mib"] for x in timeline)}
    result["prediction_sources"] = dict(service.planner.estimator.ml_predictor.last_sources)
    result["prediction_errors"] = dict(service.planner.estimator.ml_predictor.last_errors)
    saved_events = client.store.list_events()
    result["policy_activation"] = policy_activation(saved_events)
    activation = v5_activation(saved_events)
    result["v5_events"] = activation["events"]
    result["v5_activation_counts"] = activation["counts"]
    result["bootstrap_outcomes"] = activation["bootstrap_outcomes"]
    result["bootstrap_measured_decisions"] = activation["bootstrap_measured_decisions"]
    result["v5_activation_required_for_causal_performance_claim"] = True
    result["admission_policy_comparison_eligible"] = (fixed_three
        or result["policy_activation"]["actionable_forecasts"] > 0
        or activation["bootstrap_actionable"] > 0
        or (revised and result["policy_activation"]["amortization_skips"] > 0))
    result["policy_activation_note"] = "V5.1.1 alpha decisions distinguish evidence-supported changes, productive exploration, warm initial priors and holds; zero changes does not invalidate receipt timings"
    result["development_only"] = manifest.get("development", False)
    result["heldout_comparison_eligible"] = not result["development_only"]
    result["fixed_concurrency_cap"] = None
    result["adaptive_fifo_admission_limit_final"] = controller.admission_limit
    if hasattr(controller, 'close_episode'):
        controller.close_episode()
        atomic_json(output / 'alpha-state.json', controller.export_state())
        result['history_parent_sha256'] = controller.store.parent
        result['history_namespace'] = controller.store.namespace
    result['fingerprint_capture_events'] = graph_preparation.capture_events
    result['telemetry_seconds'] = telemetry_seconds
    result["alpha_events"] = controller.events
    result["ordered_launch_events"] = dispatch_events
    result["head_block_seconds"] = dict(head_block_seconds)
    result["occupancy_seconds_by_resident_count"] = dict(occupancy_seconds)
    result["preparation_treatment"] = "acceptance_enqueued_parallel_isolated_async" if preparation_workers==2 else "acceptance_enqueued_serial_async"
    result["whole_system_comparison_note"] = "V5.1.1 numeric memory preparation gates FIFO; optional low-priority CPU graph capture runs independently and may abstain; neural group profile inference retains two isolated consumers. Fixed-three retains original serial preparation. Whole-system screening, not isolated controller overhead"
    result["development_shadow"] = manifest.get("development_shadow", False)
    if result["development_only"]:
        result["admission_policy_comparison_eligible"] = False
        result["evaluation_scope"] += "; development diagnostic only, not held-out comparison"
    result["predictor_comparison_eligible"] = all(
        result["prediction_sources"].get(job.job_id) == source_tag
        and specs[name]["batch"] in recording.records.get(job.job_id, {})
        for name, job in jobs.items())
    if not result["predictor_comparison_eligible"]:
        raise RuntimeError("incomplete predictor exposure; episode is diagnostic, not a comparison")
    if manifest.get("prune_completed_weights", True): prune_owned_weights(output)
    return result


def run_episode(manifest, manifest_path, raw_settings, output, calibration, *, arm, mode,
                max_seconds, source_commit, service_factory=None, gpu_reader=None, controller_factory=None):
    validate_manifest(manifest,arm=arm,mode=mode)
    if isinstance(max_seconds,bool)or not math.isfinite(max_seconds)or max_seconds<=0:
        raise ValueError('finite positive episode bound required')
    check_deadline(manifest)
    output=Path(output);manifest_path=Path(manifest_path)
    before=source_snapshot(manifest,manifest_path,source_commit)
    atomic_json(output/'source-pins-before.json',before)
    try:
        result=_run_episode(manifest,manifest_path,raw_settings,output,calibration,arm=arm,mode=mode,
            max_seconds=max_seconds,source_commit=source_commit,service_factory=service_factory,gpu_reader=gpu_reader,controller_factory=controller_factory)
    finally:
        after=source_snapshot(manifest,manifest_path,source_commit)
        atomic_json(output/'source-pins-after.json',after)
        if before!=after:raise ValueError('source/manifest changed during V5 episode')
    result['source_files_unchanged']=True
    result['source_snapshot']=before
    atomic_json(output/'result.json',result)
    from ..a10_feedback_latency.plot import episode_plot
    try:episode_plot(output)
    except Exception:atomic_json(output/'plot-error.json',{'error':traceback.format_exc()})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=('run','check'),default='run')
    for name in ('manifest','settings','output','source-commit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--calibration');parser.add_argument('--arm',choices=ARMS,default='v4')
    parser.add_argument('--mode',choices=('individual','group'),default='individual')
    parser.add_argument('--max-seconds',type=float,default=600.);parser.add_argument('--deadline-utc')
    args=parser.parse_args(argv);path=Path(args.manifest).resolve();output=Path(args.output).resolve()
    manifest=json.loads(path.read_text());raw_settings=json.loads(Path(args.settings).read_text())
    if args.deadline_utc:manifest=with_hard_deadline(manifest,args.deadline_utc)
    validate_manifest(manifest,arm=args.arm,mode=args.mode);check_deadline(manifest)
    if output.exists():raise FileExistsError('V5 episode output already exists; no overwrite/resume')
    if args.command=='run'and not args.calibration:raise ValueError('same explicit calibration required for every arm')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'invocation.json',{**vars(args),'manifest':str(path),'output':str(output)})
    if args.command=='check':
        snapshot=source_snapshot(manifest,path,args.source_commit)
        atomic_json(output/'source-pins-before.json',snapshot);atomic_json(output/'source-pins-after.json',snapshot)
        atomic_json(output/'result.json',{'complete':True,'scope':'CPU source check only; no evaluation'})
        return
    import fcntl
    signals=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP);previous={s:signal.getsignal(s)for s in signals}
    def interrupted(number,frame):raise InterruptedError('V5 episode interrupted; owned cleanup required')
    with open('/tmp/localml-a10-feedback-pilot.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for number in signals:signal.signal(number,interrupted)
            result=run_episode(manifest,path,raw_settings,output,json.loads(Path(args.calibration).read_text()),
                arm=args.arm,mode=args.mode,max_seconds=args.max_seconds,source_commit=args.source_commit)
        except BaseException:
            atomic_json(output/'result.json',{'complete':False,'arm':args.arm,'mode':args.mode,'error':traceback.format_exc(),
                'planned_jobs':len(manifest['jobs']),'note':'Failed/unfinished jobs remain in the denominator.'})
            raise
        finally:
            for number,handler in previous.items():signal.signal(number,handler)
            fcntl.flock(lock,fcntl.LOCK_UN)
    print(json.dumps({'complete':result['complete'],'arm':args.arm,'mode':args.mode,'jobs':len(result['receipts']),
        'v5_flags':result['v5_flags'],'episode_duration_seconds':result['episode_duration_seconds']},sort_keys=True))

if __name__=='__main__':main()
