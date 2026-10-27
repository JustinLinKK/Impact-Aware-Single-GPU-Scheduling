"""Slow-start/slow-input integration preflight, not a performance benchmark."""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import torch
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings,PredictionSettings
from localml_scheduler.domain import CheckpointPolicy,JobStatus,ResourceRequirements
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction.ml_predictor import MLVramPredictor

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].split(', ')
    return {'name':raw[0],'total_mib':float(raw[1]),'used_mib':float(raw[2])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--release-root',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cancer-jobs',type=Path);p.add_argument('--cancer-dataset',type=Path)
    p.add_argument('--max-incumbent-slowdown-fraction',type=float,choices=[.10,.50])
    p.add_argument('--scheduler-commit',required=True);p.add_argument('--local-rtx-test',action='store_true');a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    os.environ.update(MLEVOLVE_INPUT_DIR=str(a.dataset.resolve()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    initial=gpu();assert initial['used_mib']+8000<.9*initial['total_mib'],'insufficient live GPU headroom for bounded test'
    if a.local_rtx_test:assert initial['name']=='NVIDIA GeForce RTX 5090'
    else:assert initial['name']=='NVIDIA A10' and initial['used_mib']<250
    prediction=PredictionSettings(mode='ml_predictor',student_release_path=a.release_root,student_model_path=a.model_dir,conversion_timeout_seconds=150,conversion_max_rss_mb=8192)
    settings=SchedulerSettings(runtime_root=root/'runtime',scheduler_poll_interval_seconds=.05,prediction=prediction,
        graph_db={'enabled':False},hardware_knowledge_graph={'enabled':False},hardware_feature_db={'enabled':False},
        early_stopping={'enabled':False},gpu_scheduler={'packing_backend':'cuda_process','batch_probe_enabled':False,
            'colocation':({'cold_start_max_incumbent_slowdown_fraction':a.max_incumbent_slowdown_fraction}
                          if a.max_incumbent_slowdown_fraction is not None else {}),
            'memory':{'gpu_vram_gib':initial['total_mib']/1024,'predicted_budget_fraction':.9}})
    settings.ensure_runtime_layout();save(settings.runtime_root/'scheduler_settings.json',settings.to_dict())
    client=SchedulerClient(settings);service=client.create_service()
    # Observe the production control boundary rather than infer add-only
    # behavior from job status (a yielded worker still has RUNNING status).
    admission_commands=[];admission_pauses=[]
    original_command=service._cold_command
    def checked_command(trial,job,action,kind,**kwargs):
        is_candidate=job.job_id==trial.candidate_job_id
        admission_commands.append({'trial_id':trial.trial_id,'job_id':job.job_id,
            'candidate':is_candidate,'action':action,'kind':kind,
            'step':job.metadata.get('runtime_global_step')})
        assert is_candidate or action!='yield','admission yielded an incumbent'
        return original_command(trial,job,action,kind,**kwargs)
    service._cold_command=checked_command
    original_pause=service.supervisor.request_pause
    def checked_pause(job_id,*args,**kwargs):
        trial=service._colocation_trial
        if trial is not None:
            admission_pauses.append({'trial_id':trial.trial_id,'job_id':job_id,
                'candidate_job_id':trial.candidate_job_id})
            assert job_id==trial.candidate_job_id,'admission paused an incumbent'
        return original_pause(job_id,*args,**kwargs)
    service.supervisor.request_pause=checked_pause
    if a.local_rtx_test:
        # Test-only target injection. Telemetry and allocator retain real RTX
        # capacity; production GPU identity checks are never modified.
        declared=HardwareProfile('integration-declared-a10','linux','NVIDIA A10',23028,'8.6','test only',torch.__version__)
        service.planner.estimator.ml_predictor=MLVramPredictor(prediction,declared)
    predictor=service.planner.estimator.ml_predictor
    assert predictor.available,predictor.unavailable_reason
    assert predictor.model_id=='release/S_A10'
    runtime=predictor._released
    model_devices=sorted({str(t.device) for t in list(runtime.model.parameters())+list(runtime.model.buffers())})
    assert model_devices==['cpu'],'released student weights must reside on CPU'
    cpu_calls=[]
    original_predict=runtime.predict_tensors
    def checked_cpu_predict(tensors):
        assert all(t.device.type=='cpu' for t in tensors)
        outputs=original_predict(tensors)
        assert all(t.device.type=='cpu' for t in outputs.values())
        cpu_calls.append({'input_devices':sorted({str(t.device) for t in tensors}),
                          'output_devices':sorted({str(t.device) for t in outputs.values()})})
        return outputs
    runtime.predict_tensors=checked_cpu_predict
    original=json.loads((a.jobs/'manifest.json').read_text());jobs=[];profiles={};source_hashes={}
    for name in ['pet_00','pet_01','pet_06']:
        spec=next(j for j in original['jobs'] if j['name']==name);source_path=a.jobs/spec['source']
        assert hashlib.sha256(source_path.read_bytes()).hexdigest()==spec['source_sha256']
        work=root/name;work.mkdir();source=work/source_path.name
        code=source_path.read_text();code=re.sub(r'^EPOCHS = \d+$','EPOCHS = 2',code,flags=re.M)
        code=re.sub(r'^MAX_STEPS_PER_EPOCH = \d+$','MAX_STEPS_PER_EPOCH = 128',code,flags=re.M)
        code=code.replace('import math\n','import math\nimport time\n')
        code=code.replace('def prepare_training():\n','def prepare_training():\n    time.sleep(4.0)  # preflight-only slow startup\n')
        code=code.replace('    def __getitem__(self, index):\n','    def __getitem__(self, index):\n        time.sleep(0.025)  # preflight-only slow input pipeline\n')
        source.write_text(code);doc=json.loads(source_path.with_suffix('.py.perfseer.json').read_text())
        doc['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();source_hashes[name]=doc['source_sha256']
        doc['perfseer_model'].update(epochs=2,steps_per_epoch_by_batch={str(spec['batch']):128})
        save(source.with_suffix('.py.perfseer.json'),doc)
        job=build_mlevolve_job(workflow_id='admission-slow-start-preflight',baseline_model_id=name,baseline_model_path=str(source),
            runner_target='localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job',
            runner_kwargs={'script_path':str(source),'working_dir':str(work),'result_path':str(work/'execution.json'),'batch_size':spec['batch'],'timeout':600},
            loader_target='localml_scheduler.adapters.mlevolve_runner:load_raw_file',max_epochs=2,
            resource_requirements=ResourceRequirements(requires_gpu=True),checkpoint_policy=CheckpointPolicy(save_every_n_steps=64,save_every_epoch=True,keep_last_n=3),
            packing_eligible=True,packing_family=name,packing_backend_allowlist=['cuda_process'],
            metadata={'cooperative_trial':True,'skip_active_scheduler_probes':True,'placement_backend':'cuda_process','trial_memory_limit_mb':initial['total_mib']*.9})
        profiles[name]=predictor.predict_profile_options(job,[spec['batch']])[spec['batch']]
        jobs.append(job)
    # Hardware discovery may initialize CUDA in the coordinator. Verify the
    # actual inference tensors and model devices instead of that global flag.
    cancer_checks=[]
    if a.cancer_jobs:
        assert a.cancer_dataset,'cancer data path is required with cancer jobs'
        os.environ['MLEVOLVE_CANCER_DIR']=str(a.cancer_dataset.resolve())
        cancer_manifest=json.loads((a.cancer_jobs/'manifest.json').read_text())
        for spec in cancer_manifest['jobs']:
            if spec.get('dataset')!='cancer':
                continue
            source_path=a.cancer_jobs/spec['source']
            assert hashlib.sha256(source_path.read_bytes()).hexdigest()==spec['source_sha256']
            work=root/spec['name'];work.mkdir();source=work/source_path.name
            code=re.sub(r'^EPOCHS = \d+$','EPOCHS = 1',source_path.read_text(),flags=re.M)
            code=re.sub(r'^MAX_STEPS_PER_EPOCH = \d+$','MAX_STEPS_PER_EPOCH = 16',code,flags=re.M)
            source.write_text(code);doc=json.loads(source_path.with_suffix('.py.perfseer.json').read_text())
            doc['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            doc['perfseer_model'].update(epochs=1,steps_per_epoch_by_batch={str(spec['batch']):16})
            save(source.with_suffix('.py.perfseer.json'),doc)
            check_job=build_mlevolve_job(workflow_id='cancer-preflight',baseline_model_id=spec['name'],baseline_model_path=str(source),
                runner_target='localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job',
                runner_kwargs={'script_path':str(source),'working_dir':str(work),'result_path':str(work/'execution.json'),'batch_size':spec['batch'],'timeout':240},
                loader_target='localml_scheduler.adapters.mlevolve_runner:load_raw_file',max_epochs=1,
                resource_requirements=ResourceRequirements(requires_gpu=True),
                metadata={'cooperative_trial':True,'skip_active_scheduler_probes':True,'placement_backend':'cuda_process'})
            profile=predictor.predict_profile_options(check_job,[spec['batch']])[spec['batch']]
            assert profile,'cancer source must convert through actual S_A10'
            begun=time.monotonic()
            with (work/'training.log').open('w') as log:
                subprocess.run([sys.executable,str(source)],cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
            state=torch.load(work/'working/shared_training/training_state.pt',map_location='cpu',weights_only=False)
            assert state['global_step']==16 and all(int(v['step'])==16 for v in state['optimizer_state']['state'].values())
            training_result=json.loads((work/'working/shared_training/result.json').read_text())
            assert training_result['device'].startswith('cuda'),'cancer preflight must actually train on CUDA'
            cancer_checks.append({'name':spec['name'],'updates':16,'source_sha256':doc['source_sha256'],
                                  'frozen_source_sha256':spec['source_sha256'],'predicted_profile':profile,
                                  'device':training_result['device'],'elapsed_seconds':time.monotonic()-begun})
        assert len(cancer_checks)==3,'the mixed preflight requires all three cancer models'
    assert len(cpu_calls)==3+len(cancer_checks),'each new source must execute verified CPU inference'
    save(root/'predictions.json',{'profiles':profiles,'student_sha256':predictor.artifact_hash,'local_target_injection':a.local_rtx_test,'source_sha256':source_hashes,
                                 'student_model_devices':model_devices,'cpu_inference_calls':cpu_calls,'cancer_checks':cancer_checks})
    jobs.sort(key=lambda j:service.planner.estimator.workload_rank(j,prefer_high=True))
    anchor=jobs[0];client.submit(anchor);service.start(background=True)
    started=time.monotonic();paused_state=None;pause_sent=False;resume_sent=False;joined=False;error=None;samples=[];exit_receipts={}
    try:
        while time.monotonic()-started<600:
            current={j.job_id:j for j in client.store.list_jobs()};first=current.get(anchor.job_id)
            if first.status==JobStatus.RUNNING and first.metadata.get('runtime_global_step',0)>=24 and not pause_sent:
                client.pause(first.job_id);pause_sent=True
            if first.status==JobStatus.PAUSED and pause_sent and not resume_sent:
                paused_state=torch.load(first.latest_checkpoint_path,map_location='cpu',weights_only=False)['state']
                assert 24<=paused_state['global_step']<256
                client.resume(first.job_id);resume_sent=True
            if first.status==JobStatus.RUNNING and resume_sent and first.metadata.get('runtime_global_step',0)>paused_state['global_step'] and not joined:
                for job in jobs[1:]:client.submit(job)
                joined=True
            if any(j.status in {JobStatus.FAILED,JobStatus.CANCELLED} for j in current.values()):raise RuntimeError('candidate failed; inspect preserved execution logs')
            samples.append({'seconds':time.monotonic()-started,'jobs':{j.baseline_model_id:{'status':j.status.value,'step':j.metadata.get('runtime_global_step',0)} for j in current.values()}})
            if joined and len(current)==3 and all(j.status==JobStatus.COMPLETED
                    and service.supervisor.has_reaped_process(j.job_id)
                    and not service.supervisor.has_live_process(j.job_id)
                    and service.supervisor.observed_exit_code(j.job_id)==0 for j in current.values()):
                exit_receipts={j.job_id:service.supervisor.observed_exit_code(j.job_id) for j in current.values()}
                break
            time.sleep(.1)
        else:raise TimeoutError('slow-start preflight exceeded ten-minute cap')
    except BaseException:error=traceback.format_exc()
    finally:
        if error:
            for j in client.store.list_jobs():
                if not j.status.is_terminal:client.cancel(j.job_id)
            until=time.monotonic()+15
            while service.supervisor.active_job_ids() and time.monotonic()<until:time.sleep(.1)
            service.supervisor.shutdown()
        service.stop()
        events=client.store.list_events();save(root/'events.json',events);save(root/'timeline.json',samples)
        save(root/'admission-controls.json',{'commands':admission_commands,'pauses':admission_pauses})
    if error:
        save(root/'verification.json',{'complete':False,'error':error,'scheduler_commit':a.scheduler_commit});raise RuntimeError(error)
    completed=[]
    for job in client.store.list_jobs():
        state=torch.load(root/job.baseline_model_id/'working/shared_training/training_state.pt',map_location='cpu',weights_only=False)
        assert state['global_step']==256 and all(int(v['step'])==256 for v in state['optimizer_state']['state'].values())
        assert exit_receipts.get(job.job_id)==0,'successful worker exit was not observed'
        completed.append({'name':job.baseline_model_id,'updates':state['global_step'],'status':job.status.value,
                          'worker_exit_observed':True,'worker_exit_code':exit_receipts[job.job_id]})
    trials=[e for e in events if e['event_type']=='colocation_trial_started']
    windows=[e['payload'] for e in events if e['event_type']=='trial_step_window' and e['payload'].get('complete')]
    assert trials and windows,'test did not exercise real overlap measurements'
    assert any(w['elapsed_seconds']>1 for w in windows),'slow-input regression condition was not reached'
    # Three candidates and at most two distinct retirements per candidate:
    # the retry bound includes the reviewed retirement recovery mechanism.
    assert len(trials)<=27,'admission retries exceeded the finite cohort bound'
    assert not any(e['event_type']=='cold_start_phase' and e['payload']['phase'] not in
        ['prepare','pre_add','overlap_warmup','measure_overlap'] for e in events),'unexpected non-add-only profiling phase'
    incumbent_steps={}
    for command in admission_commands:
        if not command['candidate'] and isinstance(command['step'],int):
            incumbent_steps.setdefault((command['trial_id'],command['job_id']),[]).append(command['step'])
    assert any(max(values)>min(values) for values in incumbent_steps.values()),'no incumbent progress observed during admission'
    decisions=[e['payload'] for e in events if e['event_type']=='cold_start_trial_decided']
    scored=[d for d in decisions if d.get('admission_policy')=='add_only_bounded_interference_v1'
        and d.get('objective')=='bounded_incumbent_interference' and d.get('incumbent_step_time_ratios')
        and d.get('common_measurement_interval',{}).get('finished_at',0)>
            d.get('common_measurement_interval',{}).get('started_at',0)]
    assert scored and any(w.get('kind')=='measure_overlap' for w in windows),'new admission scorer did not execute on real overlapping windows'
    threshold=settings.gpu_scheduler.colocation.cold_start_max_incumbent_slowdown_fraction
    assert all(d.get('max_incumbent_slowdown_fraction')==threshold for d in scored)
    incumbent_yields=sum(not c['candidate'] and c['action']=='yield' for c in admission_commands)
    incumbent_pauses=sum(p['job_id']!=p['candidate_job_id'] for p in admission_pauses)
    assert incumbent_yields==incumbent_pauses==0,'admission interrupted an incumbent'
    result={'complete':True,'scheduler_commit':a.scheduler_commit,'scope':'integration/liveness; not performance or predictor accuracy',
        'student':'S_A10','student_sha256':predictor.artifact_hash,'actual_gpu':initial,'test_only_target_override':a.local_rtx_test,
        'cancer_checks':cancer_checks,'cpu_inference_calls':cpu_calls,
        'admission_policy':'add_only_bounded_interference_v1','incumbent_yields':incumbent_yields,
        'max_incumbent_slowdown_fraction':threshold,'scored_overlap_decisions':len(scored),
        'incumbent_admission_pauses':incumbent_pauses,'candidate_rollback_count':len(admission_pauses),
        'jobs':completed,'manual_pause_step':paused_state['global_step'],'resumed':resume_sent,
        'trial_count':len(trials),'completed_measurement_windows':len(windows),'longest_window_seconds':max(w['elapsed_seconds'] for w in windows),
        'pause_count':sum(e['event_type']=='job_paused' for e in events),'exclusive_profile_phases':0,'elapsed_seconds':time.monotonic()-started,
        'max_observed_running_jobs':max(sum(j['status']=='RUNNING' for j in s['jobs'].values()) for s in samples)}
    save(root/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
