"""Real SQLite/receipt lifecycle with injected CPU artifacts; not policy/GPU evidence."""
import json
from pathlib import Path
import pytest
from localml_scheduler.benchmarks.a10_v5_1_ablation import fifo_episode as f
from localml_scheduler.tests import test_a10_v5_episode as original

class ControllerFixture:
 def __init__(self,*,mode):self.admission_limit=3;self.events=[];self.launches=[];self.observations=[]
 def observe(self,**row):
  self.observations.append(row)
  # Synthetic limit decrease tests drain without any worker pause/eviction.
  if len(self.launches)>=3:self.admission_limit=1
 def on_launch(self,job_id,now):self.launches.append(job_id);self.events.append({'job_id':job_id,'launched_monotonic':now})

@pytest.fixture
def harness(monkeypatch):
 monkeypatch.setattr(original,'e',f)
 return original.harness.__wrapped__(monkeypatch)

def test_adaptive_fifo_retains_causal_receipts_initial_three_and_drain(tmp_path,harness,monkeypatch):
 controllers=[]
 original_poll=original.Worker.poll
 def retained_context(worker):
  code=original_poll(worker)
  return None if __import__('time').monotonic()-worker.started<.25 else code
 monkeypatch.setattr(original.Worker,'poll',retained_context)
 def factory(**kw):c=ControllerFixture(**kw);controllers.append(c);return c
 m=original.manifest();m['preparation_delay_seconds_by_name']={'parent-0':.15}
 table=original.calibration_fixture(m['jobs'][0]['template_key'],original.H)
 path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
 result=f.run_episode(m,path,{},out,table,arm='v5_1',mode='group',max_seconds=35,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(original.GPU),controller_factory=factory)
 c=controllers[0]
 assert result['complete']and len(result['receipts'])==6 and result['preparation_workers']==1
 assert result['fixed_concurrency_cap']is None and result['adaptive_fifo_admission_limit_final']==1
 assert len(c.launches)==6 and len(set(c.launches))==6
 ids={Path(row['result_path']).parents[2].name:job_id for job_id,row in result['receipts'].items()}
 assert c.launches==[ids[f'{role}-{i}']for role in ('parent','child')for i in range(3)]
 assert c.events[0]['launched_monotonic']>=result['receipts'][ids['parent-0']]['accepted_monotonic']+.15
 # Original prepared-profile stricthead gate delays ALL first dispatches until head ready.
 observations=[r for r in c.observations if r['queued_jobs']]
 allids=set(result['receipts'])
 assert all({j.job_id for j in r['queued_jobs']+r['running_jobs']}<=allids for r in observations)
 assert max(len(r['running_jobs'])for r in c.observations)>=3 # decrease1 drains existing3
 assert all(row['worker_exit_code']==0 and row['expected_updates']==6 for row in result['receipts'].values())
 timeline=json.loads((out/'preparation-timeline.json').read_text());accepted={key:row['accepted_monotonic']for key,row in result['receipts'].items()}
 assert all(event['monotonic_seconds']>=accepted[event['job_id']]for event in timeline['events'])
 assert json.loads((out/'alpha-decisions.json').read_text())['events']==c.events
 assert len(result['request_latencies_seconds'])==2 and result['source_files_unchanged']


def test_controller_cannot_override_memory_head_gate(tmp_path,harness,monkeypatch):
 monkeypatch.setattr(f,'fixed_fifo_fits',lambda *a,**k:False)
 m=original.manifest();table=original.calibration_fixture(m['jobs'][0]['template_key'],original.H)
 path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
 with pytest.raises(ValueError,match='does not fit'):
  f.run_episode(m,path,{},out,table,arm='v5_1',mode='group',max_seconds=35,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(original.GPU),controller_factory=ControllerFixture)
 assert json.loads((out/'alpha-decisions.json').read_text())['events']==[]


def test_default_controller_real_builder_has_known_queued_budget_and_successful_fifo(tmp_path,harness):
 from localml_scheduler.scheduler.adaptive_fifo import AdaptiveFIFOController
 captured=[]
 class RealController(AdaptiveFIFOController):
  def observe(self,**row):
   for job in row['queued_jobs']:
    if job.job_id not in self._launched:
     remaining=self._remaining(job,queued=True)
     assert remaining is not None, 'real build_job queued remaining budget must be known'
   captured.append(row);return super().observe(**row)
 m=original.manifest();m['development']=True;m['condition']={'name':'short-mixed'};table=original.calibration_fixture(m['jobs'][0]['template_key'],original.H)
 path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
 result=f.run_episode(m,path,{},out,table,arm='v5_1',mode='group',max_seconds=35,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(original.GPU),controller_factory=RealController)
 assert result['complete']and len(result['receipts'])==6 and captured
 state=json.loads((out/'alpha-state.json').read_text());assert len(state['launched'])==6
 assert all(job.metadata['v51_hardware_key']==original.H.hardware_key for row in captured for job in row['queued_jobs'])


 # Dedicated development auditor accepts validity without forced alpha decisions.
 from localml_scheduler.benchmarks.a10_v5_1_ablation import development as d
 audit=d.audit_episode(out,{'arm':'v5_1','mode':'group','kind':'fifo_alpha'},m,json.loads((out/'source-pins-before.json').read_text()),table)
 assert audit['valid'],audit['errors']
 assert result['development_only']and not result['heldout_comparison_eligible']


def test_explicit_held_head_never_unheld_or_bypassed(tmp_path,harness,monkeypatch):
 accepted=[];real_accept=f.accept_job;real_deadline=f.check_deadline
 def accept(client,job,when):
  result=real_accept(client,job,when);accepted.append(result)
  if len(accepted)==1:client.store.update_job(result.job_id,hold=True)
  return result
 def deadline(manifest,elapsed=None,max_seconds=None):
  if elapsed is not None and elapsed>.3:raise TimeoutError('bounded held fixture')
  return real_deadline(manifest,elapsed,max_seconds)
 monkeypatch.setattr(f,'accept_job',accept);monkeypatch.setattr(f,'check_deadline',deadline)
 m=original.manifest();table=original.calibration_fixture(m['jobs'][0]['template_key'],original.H)
 path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
 with pytest.raises(TimeoutError,match='held fixture'):
  f.run_episode(m,path,{},out,table,arm='v5_1',mode='group',max_seconds=35,source_commit='a'*40,service_factory=harness[0],gpu_reader=lambda:dict(original.GPU),controller_factory=ControllerFixture)
 proof=json.loads((out/'alpha-decisions.json').read_text())
 assert proof['ordered_launch_events']==[] and proof['head_block_seconds']['held']>0
 assert harness[1][-1].store.get_job(accepted[0].job_id).hold is True


@pytest.mark.parametrize('mode',['individual','group'])
def test_neural_development_branch_uses_exact_calibration_and_rejects_matching_bad_logs(tmp_path,harness,monkeypatch,mode):
 from datetime import datetime,timezone
 from concurrent.futures import ThreadPoolExecutor
 from localml_scheduler.scheduler import parallel_preparation as parallel
 from localml_scheduler.domain import TrainingJob
 from localml_scheduler.benchmarks.a10_v5_1_ablation import development as d
 original_backend=parallel.IsolatedPredictionBackend
 class InlineBackend(original_backend):
  def __init__(self,prototype,**kwargs):
   def predict(spec,raw,batches):
    job=TrainingJob.from_dict(raw,historical_read=True);values=prototype.predict_profile_options(job,batches)
    return values,prototype.last_sources.get(job.job_id),prototype.last_errors.get(job.job_id)
   super().__init__(prototype,**kwargs,_executor=ThreadPoolExecutor(max_workers=2),_predict=predict)
 monkeypatch.setattr(parallel,'IsolatedPredictionBackend',InlineBackend)
 m=original.manifest();m.update(profile_source='ml_predictor',development=True,condition={'name':'published-pipeline'})
 path=tmp_path/'manifest.json';path.write_text(json.dumps(m));out=tmp_path/'episode';out.mkdir()
 table=dict(predictor_artifact_sha256=f.STUDENT_SHA256,calibration_sha256='c'*64,hardware_key=original.H.hardware_key,wall_multiplier=1.,peak_add_mib=0.,costs=dict.fromkeys(['startup','pause','resume','validation','checkpoint','finalization','delivery'],.01),isolated_seconds_per_update_max=.01,calibration_parameter_count_max=100000,observed_at=datetime.now(timezone.utc).isoformat(),measurement_job_signatures=['fixture'])
 def factory(settings,**kw):
  service=harness[0](settings,**kw);service.planner.estimator.ml_predictor=original.NeuralFixture();return service
 result=f.run_episode(m,path,{},out,table,arm='v5_1',mode=mode,max_seconds=35,source_commit='a'*40,service_factory=factory,gpu_reader=lambda:dict(original.GPU))
 record={'arm':'v5_1','mode':mode,'kind':'fifo_alpha'};snapshot=json.loads((out/'source-pins-before.json').read_text())
 audit=d.audit_episode(out,record,m,snapshot,table);assert audit['valid'],audit['errors']
 assert result['preparation_workers']==(2 if mode=='group'else 1)
 assert json.loads((out/'preparation-timeline.json').read_text())['worker_stopped']is True
 assert all(not any(k=='v5_startup_envelope' for k in json.loads((Path(row['execution_path']).parent/'job-final.json').read_text())['metadata'])for row in result['receipts'].values())
 # Both duplicate logs are changed identically: semantic alpha validation must reject.
 alpha=json.loads((out/'alpha-decisions.json').read_text());alpha['ordered_launch_events'][0]['resident_count']=99;result['ordered_launch_events']=alpha['ordered_launch_events']
 (out/'alpha-decisions.json').write_text(json.dumps(alpha));(out/'result.json').write_text(json.dumps(result))
 assert not d.audit_episode(out,record,m,snapshot,table)['valid']
 alpha['ordered_launch_events'][0]['resident_count']=1;(out/'alpha-decisions.json').write_text(json.dumps(alpha))
 result['ordered_launch_events']=alpha['ordered_launch_events'];jid=next(iter(result['prepared_numeric_profiles']));batch=next(iter(result['prepared_numeric_profiles'][jid]));profile=result['prepared_numeric_profiles'][jid][batch];expected_artifact=profile['artifact_sha256'];profile['artifact_sha256']='0'*64
 (out/'result.json').write_text(json.dumps(result));assert not d.audit_episode(out,record,m,snapshot,table)['valid']
 profile['artifact_sha256']=expected_artifact;profile['step_cost']=float('nan');(out/'result.json').write_text(json.dumps(result))
 numeric=d.audit_episode(out,record,m,snapshot,table);assert not numeric['valid']and 'nonfinite' in str(numeric['errors'])
