"""Injected CPU workers test real receipt/cache/causal boundaries; never GPU evidence."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import json,time
import pytest
import torch
from localml_scheduler.domain import JobStatus
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.benchmarks.a10_v5_ablation import episode as e
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import make_spec,resource_config
from localml_scheduler.tests.test_a10_resource_cpu_episode import calibration_fixture
H=HardwareProfile(hardware_key='CPU-V5-TEST',os_name='fixture',gpu_name='NVIDIA A10',total_vram_mb=24000,compute_capability='8.6',cuda_runtime=None,torch_version=torch.__version__)
GPU=dict(name='NVIDIA A10',gpu_uuid='CPU-TEST-NO-GPU',total_mib=24000.,used_mib=0.,utilization_percent=0.,memory_controller_busy_percent=0.,power_watts=0.)

def manifest(chains=3):
    jobs=[]
    for role in ('parent','child'):
        for chain in range(chains):
            row=make_spec(resource_config('tiny',width=8,input_features=8,batch=2,pool_size=16,validation_size=128),name=f'{role}-{chain}',role=role,chain=chain%3,epochs=2,steps_per_epoch=3,seed=100+len(jobs),train_pool_seed=1000+2*len(jobs),validation_pool_seed=1001+2*len(jobs),source_path=workloads.HERE/'candidate.py')
            row['chain']=chain;row['spec_sha256']=workloads.spec_digest(row);row['expected_runtime_signature']=workloads.expected_signature(row);jobs.append(row)
    return dict(jobs=jobs,calibration=[],profile_source='dedicated_solo_template_proxy',job_timeout_seconds=30,prune_completed_weights=False,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=3)).isoformat(),fixture_only='Injected CPU artifacts; not GPU evaluation')
class Estimator:
    ml_predictor=None
    def predicted_profile(self,job,batch):return self.ml_predictor.predict_profile_options(job,[batch])[batch]
class Worker:
    def __init__(self,store,job):self.store,self.job,self.started,self.written,self.pid=store,job,time.monotonic(),False,99999999
    def poll(self):
        if not self.written and time.monotonic()-self.started>=.01:
            job=self.store.get_job(self.job.job_id);assert 'feedback_ready_at'not in job.metadata
            spec=job.metadata['resource_workload_spec'];steps=2*spec['steps_per_epoch'];work=Path(job.config.runner_kwargs['working_dir']);directory=work/'working/shared_training';directory.mkdir(parents=True,exist_ok=True)
            history=[dict(epoch=i,validation_auc=.7+i*.1,negative_validation_auc=-(.7+i*.1),validation_bce=.5,validation_negative_count=spec['validation_size']//2,validation_positive_count=spec['validation_size']//2)for i in range(2)]
            (directory/'result.json').write_text(json.dumps(dict(global_step=steps,epochs=2,signature=spec['expected_runtime_signature'],metric_name='negative_validation_auc',history=history,selected_metric=-.8,fixture_only=True)))
            (work/'execution.json').write_text(json.dumps({'exc_type':None,'fixture_only':True}))
            torch.save(dict(global_step=steps,completed_epochs=2,signature=spec['expected_runtime_signature'],optimizer_state={'state':{0:{'step':torch.tensor(steps)}}}),directory/'training_state.pt')
            self.store.update_job(job.job_id,status=JobStatus.COMPLETED,metadata_updates={'runtime_global_step':steps,'runtime_total_optimizer_updates':steps,'runtime_finalization_complete':True,'runtime_completion_identity':spec['expected_runtime_signature']});self.written=True
        if time.monotonic()-self.started<.04:
            assert 'feedback_ready_at'not in self.store.get_job(self.job.job_id).metadata
            return None
        return 0
class StubService:
    def __init__(self,store,arm,flags):
        self.store,self.arm,self.flags=store,arm,flags;self.planner=SimpleNamespace(estimator=Estimator());self.workers={};self.exits={};self.predictor_preparation=None
        self.supervisor=SimpleNamespace(_record_process_exit=lambda job_id,code:self.exits.setdefault(job_id,code),observed_exit_code=self.observed)
    def start(self,background=True):return self
    def stop(self):pass
    def observed(self,job_id):
        for job in self.store.runnable_jobs():
            spec=job.metadata['resource_workload_spec']
            if self.predictor_preparation.ready(job,spec['batch'])and job.job_id not in self.workers:
                self.store.update_job(job.job_id,status=JobStatus.RUNNING);self.workers[job.job_id]=Worker(self.store,job)
        if job_id in self.workers:
            code=self.workers[job_id].poll()
            if code is not None:self.supervisor._record_process_exit(job_id,code)
        return self.exits.get(job_id)
@pytest.fixture
def harness(monkeypatch):
    from localml_scheduler.benchmarks.a10_feedback_latency import plot
    initial_cuda_initialized=torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU harness must not initialize CUDA'))
    monkeypatch.setattr('localml_scheduler.storage.sqlite_store.detect_hardware_profile',lambda **_:H)
    monkeypatch.setattr(e,'source_snapshot',lambda manifest,path,source_commit:{'fixture_only':True,'scheduler_reference_commit':source_commit})
    monkeypatch.setattr(plot,'episode_plot',lambda output:None);services=[]
    def factory(settings,*,store,arm,flags):
        service=StubService(store,arm,flags);services.append(service);return service
    def executor(settings):
        from localml_scheduler.storage.state_store import StateStore
        store=StateStore(settings)
        def start(job):
            store.update_job(job.job_id,status=JobStatus.RUNNING);return SimpleNamespace(process=Worker(store,job))
        return SimpleNamespace(start=start)
    monkeypatch.setattr(e,'SubprocessExecutor',executor);monkeypatch.setattr(e,'terminate_process_tree',lambda process:None)
    return factory,services,initial_cuda_initialized

def execute(tmp_path,harness,chains=3,arm='v4',mode='group',delay=False):
    m=manifest(chains);table=calibration_fixture(m['jobs'][0]['template_key'],H)
    if delay:m['preparation_delay_seconds_by_name']={'parent-0':.1}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m));output=tmp_path/'episode';output.mkdir()
    result=e.run_episode(m,path,{},output,table,arm=arm,mode=mode,max_seconds=35,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(GPU))
    return result,output,harness[1][-1]
@pytest.mark.parametrize('arm',e.ARMS)
def test_six_arm_flags_select_only_intended_mechanism(arm):
    assert e.arm_flags(arm)=={'continuation':arm in ('forecast_only','combined'),'deferral':arm in ('deferral_only','combined'),'bracket':arm in ('bracket_only','combined')}
@pytest.mark.parametrize('arm',e.ARMS)
def test_cpu_episode_has_real_receipt_gates_async_profiles_and_group_units(tmp_path,harness,monkeypatch,arm):
    from localml_scheduler.scheduler import parallel_preparation
    monkeypatch.setattr(parallel_preparation,'IsolatedPredictionBackend',
        lambda *a,**kw:pytest.fail('measured table lookup must not spawn an isolated capture pool'))
    result,out,service=execute(tmp_path,harness,arm=arm,delay=True)
    assert result['complete']and result['predictor_comparison_eligible']and len(result['receipts'])==6
    assert set(result['request_latencies_seconds'])=={'parent-bracket','child-bracket'}
    assert result['v5_flags']==e.arm_flags(arm)==service.flags and result['preparation_workers']==1
    parent_ids={r['job_id']for r in result['receipts'].values()if '/parent-'in r['result_path']};release=max(result['receipts'][pid]['feedback_ready_monotonic']for pid in parent_ids)+2
    assert all(r['accepted_seconds']+result['episode_started_monotonic']>=release for r in result['arrivals']if r['name'].startswith('child-'))
    prep=json.loads((out/'preparation-timeline.json').read_text());accepted={r['job_id']:r['accepted_monotonic']for r in result['receipts'].values()}
    assert all(event['monotonic_seconds']>=accepted[event['job_id']]for event in prep['events'])
    first=next(r for r in result['receipts'].values()if '/parent-0/'in r['result_path']);enqueue=next(x for x in prep['events']if x['job_id']==first['job_id']and x['phase']=='enqueue');assert enqueue['monotonic_seconds']>=first['accepted_monotonic']+.1
    for r in result['receipts'].values():
        assert r['expected_updates']==6 and r['worker_exit_code']==0
        job=json.loads((Path(r['execution_path']).parent/'job-final.json').read_text());assert job['metadata']['feedback_ready_monotonic']==r['feedback_ready_monotonic']and job['metadata']['feedback_ready_at'];assert job['metadata']['v5_startup_envelope']['peak_mib']>0
        assert result['release_observations'][r['job_id']]['observed_monotonic']<=r['feedback_ready_monotonic']
    assert result['source_files_unchanged']and torch.cuda.is_initialized()==harness[2]
def test_six_individual_chains_have_twelve_receipts_and_actual_parent_offsets(tmp_path,harness):
    result,out,_=execute(tmp_path,harness,chains=6,mode='individual');assert len(result['receipts'])==len(result['request_latencies_seconds'])==12 and result['chains']==6
    arrivals={r['name']:r for r in result['arrivals']};assert [arrivals[f'parent-{chain}']['due_seconds']for chain in range(6)]==[0,2,4,6,8,10]
    by_name={Path(r['execution_path']).parent.name:r for r in result['receipts'].values()}
    for chain in range(6):assert arrivals[f'child-{chain}']['accepted_seconds']+result['episode_started_monotonic']>=by_name[f'parent-{chain}']['feedback_ready_monotonic']+2
@pytest.mark.parametrize('bad',['chains','epochs','duplicate','delay'])
def test_bad_seal_fails_before_gpu_probe(tmp_path,harness,bad):
    m=manifest()
    if bad=='chains':m['jobs']=m['jobs'][:4]
    elif bad=='epochs':m['jobs'][0]['epochs']=1
    elif bad=='duplicate':m['jobs'][1]['name']=m['jobs'][0]['name']
    else:m['preparation_delay_seconds_by_name']={'future-child':1}
    with pytest.raises(ValueError):e.run_episode(m,tmp_path/'m',{},tmp_path,{},arm='v4',mode='group',max_seconds=30,source_commit='a'*40,gpu_reader=lambda:pytest.fail('invalid seal must not probe GPU'))


@pytest.mark.parametrize('arm',['bracket_only','combined'])
def test_neural_group_parallel_adapter_has_two_consumers_and_finite_stop_without_real_processes(tmp_path,harness,monkeypatch,arm):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from localml_scheduler.domain import TrainingJob
    from localml_scheduler.scheduler import parallel_preparation as parallel
    original=parallel.IsolatedPredictionBackend;observed=[]
    class InlineBackend(original):
        def __init__(self,prototype,**kwargs):
            self.fixture_active=0;self.fixture_peak=0;self.fixture_lock=Lock();outer=self
            class TestExecutor(ThreadPoolExecutor):
                def shutdown(self,**kwargs):
                    outer.fixture_stopped=True
                    return super().shutdown(**kwargs)
            def predict(spec,raw,batches):
                with outer.fixture_lock:outer.fixture_active+=1;outer.fixture_peak=max(outer.fixture_peak,outer.fixture_active)
                try:
                    time.sleep(.05);job=TrainingJob.from_dict(raw,historical_read=True)
                    outputs=prototype.predict_profile_options(job,batches)
                    return outputs,prototype.last_sources.get(job.job_id),prototype.last_errors.get(job.job_id)
                finally:
                    with outer.fixture_lock:outer.fixture_active-=1
            super().__init__(prototype,**kwargs,_executor=TestExecutor(max_workers=2),_predict=predict)
            observed.append(self)
    monkeypatch.setattr(parallel,'IsolatedPredictionBackend',InlineBackend)
    result,out,_=execute_neural(tmp_path,harness,arm=arm)
    assert result['preparation_workers']==2 and result['complete'] and result['profile_source']=='ml_predictor'
    assert observed[0]._spec[0]=='ml_predictor'
    for receipt in result['receipts'].values():
        job=json.loads((Path(receipt['execution_path']).parent/'job-final.json').read_text())
        assert 'v5_startup_envelope'not in job['metadata']
    assert observed[0].fixture_peak==2 and observed[0].fixture_stopped
    prep=json.loads((out/'preparation-timeline.json').read_text());assert prep['worker_stopped']is True and not prep['failures']


class NeuralFixture:
    available=True;artifact_hash=e.STUDENT_SHA256;model_id='SYNTHETIC-published-shape-only';hardware=H
    def __init__(self):self.last_sources={};self.last_errors={}
    def predict_profile_options(self,job,batches):
        self.last_sources[job.job_id]='ml_predictor'
        return {b:dict(kind='predicted_workload',artifact_sha256=self.artifact_hash,hardware='A10',job_signature=job.packing.signature or job.job_id,batch_size=b,step_cost=10.,epoch_cost=30.,avg_vram_mib=1.,peak_vram_mib=2.,busy_indicator=1.)for b in batches}


def execute_neural(tmp_path,harness,arm='v4'):
    m=manifest();m['profile_source']='ml_predictor';path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
    table=dict(predictor_artifact_sha256=e.STUDENT_SHA256,calibration_sha256='c'*64,hardware_key=H.hardware_key,wall_multiplier=1.,peak_add_mib=0.,costs=dict.fromkeys(['startup','pause','resume','validation','checkpoint','finalization','delivery'],.01),isolated_seconds_per_update_max=.01,calibration_parameter_count_max=100000,observed_at=datetime.now(timezone.utc).isoformat(),measurement_job_signatures=['fixture'])
    def factory(settings,**kwargs):
        service=harness[0](settings,**kwargs);service.planner.estimator.ml_predictor=NeuralFixture();return service
    result=e.run_episode(m,path,{},out,table,arm=arm,mode='group',max_seconds=30,source_commit='a'*40,service_factory=factory,gpu_reader=lambda:dict(GPU))
    return result,out,harness[1][-1]


def test_neural_profile_route_keeps_original_artifact_gate_and_unknown_startup_envelope(tmp_path,harness):
    result,out,_=execute_neural(tmp_path,harness)
    assert result['profile_source']=='ml_predictor'and result['predictor_comparison_eligible']
    for r in result['receipts'].values():
        job=json.loads((Path(r['execution_path']).parent/'job-final.json').read_text());assert 'v5_startup_envelope'not in job['metadata']
        assert job['metadata']['feedback_phase_model']['calibration_sha256']=='c'*64


def test_wrong_neural_weight_hash_rejected_before_any_worker(tmp_path,harness):
    m=manifest();m['profile_source']='ml_predictor';path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
    def factory(settings,**kwargs):
        service=harness[0](settings,**kwargs);service.planner.estimator.ml_predictor=NeuralFixture();service.planner.estimator.ml_predictor.artifact_hash='0'*64;return service
    with pytest.raises(ValueError,match='student hash mismatch'):e.run_episode(m,path,{},out,{},arm='v4',mode='group',max_seconds=30,source_commit='a'*40,service_factory=factory,gpu_reader=lambda:dict(GPU))


def test_source_changed_during_episode_fails_closed(tmp_path,harness,monkeypatch):
    values=iter([{'pin':'before'},{'pin':'changed'}]);monkeypatch.setattr(e,'source_snapshot',lambda *a,**kw:next(values))
    with pytest.raises(ValueError,match='source/manifest changed'):execute(tmp_path,harness)


@pytest.mark.parametrize('arm,extra',[
    ('fixed_three',{'development_fixed_cap':2}),('v4',{'development':True,'development_fixed_cap':2}),
    ('fixed_three',{'development':True,'development_fixed_cap':0}),('fixed_three',{'development':True,'development_fixed_cap':True}),
    ('forecast_only',{'development_shadow':True}),('combined',{'development':True,'development_shadow':True})])
def test_development_controls_cannot_leak_into_heldout_or_wrong_arm(arm,extra):
    m=manifest();m.update(extra)
    with pytest.raises(ValueError):e.validate_manifest(m,arm=arm,mode='group')


def test_shadow_configuration_is_diagnostic_and_never_enables_forecast_dispatch():
    m=manifest();m.update(development=True,development_shadow=True)
    assert len(e.validate_manifest(m,arm='forecast_only',mode='group'))==3
    flags=e.arm_flags('forecast_only',development_shadow=True)
    assert flags['shadow']is True and flags['continuation']is False


def test_development_fixed_cap_changes_actual_slots_without_weakening_memory_gate(tmp_path,harness,monkeypatch):
    m=manifest();m.update(development=True,development_fixed_cap=1)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir();table=calibration_fixture(m['jobs'][0]['template_key'],H)
    original=e.fixed_fifo_fits;running_counts=[]
    def checked(profile,running,profiles,**kwargs):
        running_counts.append(len(running));return original(profile,running,profiles,**kwargs)
    monkeypatch.setattr(e,'fixed_fifo_fits',checked)
    result=e.run_episode(m,path,{},out,table,arm='fixed_three',mode='group',max_seconds=30,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(GPU))
    assert result['fixed_concurrency_cap']==1 and result['development_only']and not result['heldout_comparison_eligible']
    assert not result['admission_policy_comparison_eligible']and running_counts and max(running_counts)==0


def test_receipt_artifact_budget_mismatch_prevents_feedback_or_child_acceptance(tmp_path,harness,monkeypatch):
    original=Worker.poll
    def bad_result(self):
        code=original(self)
        if self.written:
            job=self.store.get_job(self.job.job_id);work=Path(job.config.runner_kwargs['working_dir']);path=work/'working/shared_training/result.json';payload=json.loads(path.read_text());payload['global_step']+=1;path.write_text(json.dumps(payload))
        return code
    monkeypatch.setattr(Worker,'poll',bad_result)
    with pytest.raises(ValueError,match='artifact identity or budget mismatch'):execute(tmp_path,harness)
    jobs=harness[1][-1].store.list_jobs()
    assert len(jobs)==3 and all(job.baseline_model_id.startswith('parent-')for job in jobs)
    assert all('feedback_ready_at'not in job.metadata for job in jobs)


def test_preparation_stop_failure_does_not_silently_complete_episode(tmp_path,harness,monkeypatch):
    from localml_scheduler.benchmarks.a10_feedback_latency.async_preparation import PreparedPredictor
    original=PreparedPredictor.stop
    def failed_stop(self,timeout=3.):original(self,timeout);return False
    monkeypatch.setattr(PreparedPredictor,'stop',failed_stop)
    with pytest.raises(RuntimeError,match='bounded preparation stop incomplete'):execute(tmp_path,harness)


def test_bootstrap_activation_distinguishes_measured_decisions_and_inconclusive():
    measured={'decision':'accepted','measured_full_group':True,'halves':[{},{}],'error':None}
    events=[{'event_type':'bracket_bootstrap_decided','payload':measured},
        {'event_type':'bracket_bootstrap_decided','payload':{**measured,'decision':'inconclusive'}},
        {'event_type':'bracket_bootstrap_decided','payload':{**measured,'decision':'rejected','error':'invalid'}},
        {'event_type':'bracket_bootstrap_started','payload':{}},
        {'event_type':'v5_reference_started','payload':{}},
        {'event_type':'ordinary','payload':{}}]
    result=e.v5_activation(events)
    assert result['bootstrap_actionable']==1
    assert result['bootstrap_measured_decisions']=={'accepted':1}
    assert result['bootstrap_outcomes']=={'accepted':1,'inconclusive':1,'rejected':1}
    assert len(result['events'])==5
