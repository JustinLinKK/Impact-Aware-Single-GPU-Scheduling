"""CPU-only contract tests; no scheduler or GPU execution."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import pytest
from localml_scheduler.benchmarks.a10_v5_1_ablation import protocol,episode
from localml_scheduler.benchmarks.a10_v5_ablation import protocol as old

def candidate():
 p=Path(protocol.__file__).with_name('blueprint.json');bp=json.loads(p.read_text())
 return protocol.candidate_plan(bp,blueprint_sha256=protocol.file_hash(p),source_commit='a'*40,host_id='host-fixture',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=20)).isoformat(),job_timeout_seconds=1800,episode_max_seconds=3600,campaign_max_seconds=70200,calibration_max_seconds=3600)

def test_three_arm_full_withinhost_blocks_preserve_counterbalance_and_denominators():
 p=candidate();rows=protocol.episode_order(p)
 assert p['status']=='candidate_pending_explicit_execution_freeze'
 assert len(rows)==60 and sum(r['expected_jobs']for r in rows)==432
 groups={}
 for r in rows:groups.setdefault(r['block'],[]).append(r['arm'])
 assert len(groups)==20 and all(set(v)==set(protocol.ARMS)and len(v)==3 for v in groups.values())
 assert rows[0]['arm']=='fixed_three' and rows[30]['arm']=='v5_1'
 assert set(r['host_id']for r in rows)=={'host-fixture'}

@pytest.mark.parametrize('arms',[['fixed_three','combined'],['fixed_three','combined','v4'],list(old.ARMS)])
def test_arm_dropping_or_old_ablation_not_silently_accepted(arms):
 p=candidate();p['arms']=arms;p['blueprint']['arms']=arms
 with pytest.raises(ValueError,match='three exact arms'):protocol.validate_plan(p)

def test_recipe_calibration_seed_and_global_budget_specs_unchanged_from_v5():
 p=candidate();v5=deepcopy(p);v5['schema']=old.SCHEMA_PLAN;v5['arms']=list(old.ARMS);v5['blueprint']['arms']=list(old.ARMS)
 a=protocol.calibration_manifest(p);b=old.calibration_manifest(v5)
 assert a['calibration']==b['calibration'] and len(a['calibration'])==8
 assert p['configurations']==v5['configurations'] and p['conditions']==v5['conditions']
 assert p['seed_base']==v5['seed_base']
 assert all(s['epochs']==2 for s in a['calibration'])

@pytest.mark.parametrize('arm',['fixed_three','combined'])
def test_baseline_route_delegates_identical_objects_and_arguments(monkeypatch,tmp_path,arm):
 objects=[{},tmp_path/'manifest.json',{},tmp_path,{}];seen={};factory=object();gpu=object()
 def original(*args,**kwargs):seen.update(args=args,kwargs=kwargs);return {'baseline':True}
 monkeypatch.setattr(episode.baseline,'run_episode',original)
 assert episode.run_episode(*objects,arm=arm,mode='group',max_seconds=3600,source_commit='a'*40,service_factory=factory,gpu_reader=gpu)=={'baseline':True}
 assert all(a is b for a,b in zip(seen['args'],objects))
 assert seen['kwargs']==dict(arm=arm,mode='group',max_seconds=3600,source_commit='a'*40,service_factory=factory,gpu_reader=gpu)
 with pytest.raises(ValueError,match='cannot change a baseline'):episode.run_episode(*objects,arm=arm,mode='group',max_seconds=3600,source_commit='a'*40,fifo_episode=lambda:None)

def test_v51_route_calls_only_explicit_fifo_lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(episode.baseline,'run_episode',lambda *a,**k:pytest.fail('no fallback policy'))
    seen={}
    def fifo(*a,**k):seen.update(k);return {'arm':'v5_1'}
    assert episode.run_episode({},tmp_path/'manifest',{},tmp_path,{},arm='v5_1',mode='group',max_seconds=3600,source_commit='a'*40,fifo_episode=fifo)=={'arm':'v5_1'}
    assert seen['arm']=='v5_1' and seen['controller_factory']is None


def test_six_development_manifests_have_disjoint_seeds_identical_recipes():
 from localml_scheduler.benchmarks.a10_v5_1_ablation import development as d
 p=candidate();rows=d.planned_records()
 assert len(rows)==6 and sum(r['expected_jobs']for r in rows)==48
 assert d.coverage(rows)['planned_episodes']==6 and d.coverage(rows)['planned_jobs']==48
 assert {r['arm']for r in rows}=={'v5_1'}
 assert {(r['condition'],r['mode'])for r in rows}=={(c,m)for c in ('short-mixed','backlog-six','published-pipeline')for m in ('individual','group')}
 # Seed re-sealing is exercised with actual blueprint workload definitions.
 source=protocol.workloads.HERE/'candidate.py';spec=protocol._spec(p,p['configurations']['light'],name='parent-0-light',role='parent',chain=0,steps_per_epoch=1500,index=0,calibration=False)
 manifest={'profile_source':protocol.SOURCE,'condition':{'name':'short-mixed'},'jobs':[spec]}
 # Real manifest requires all causal chains; use direct complete synthetic CPU fixture.
 from localml_scheduler.tests.test_a10_v5_episode import manifest as cpu_manifest
 base=cpu_manifest();base['condition']={'name':'short-mixed'}
 shifted=d.development_manifest(base,rows[0])
 assert [s['seed']for s in shifted['jobs']]==[s['seed']+18000000 for s in base['jobs']]
 assert [s['template_key']for s in shifted['jobs']]==[s['template_key']for s in base['jobs']]
 assert [(s['epochs'],s['steps_per_epoch'])for s in shifted['jobs']]==[(s['epochs'],s['steps_per_epoch'])for s in base['jobs']]


def test_primary_three_pair_contrasts_keep_request_units_and_full_denominator():
 from localml_scheduler.benchmarks.a10_v5_1_ablation import campaign as c
 p=candidate();records=c.planned_records(protocol.episode_order(p))
 for r in records:
  count=r['expected_jobs']if r['mode']=='individual'else 2
  r.update(status='completed',exit_code=0,usable_receipts=r['expected_jobs'],audit={'request_latencies_seconds':{str(i):float(i+1)for i in range(count)},'job_jct_seconds':[1.]*r['expected_jobs'],'episode_duration_seconds':10.})
 contrasts=c.paired_contrasts(records,p)
 assert len(contrasts)==30 and all(x['planned_pairs']==2 and x['valid_pairs']==2 for x in contrasts)
 assert {(x['arm'],x['reference'])for x in contrasts}=={('combined','fixed_three'),('v5_1','fixed_three'),('v5_1','combined')}
 assert c.coverage(records,protocol.episode_order(p))['usable_receipts']==432
 with pytest.raises(ValueError,match='coverage'):c.coverage(records[:-1],protocol.episode_order(p))


def test_failed_cleanup_receipts_are_observed_not_usable_primary_results():
 from localml_scheduler.benchmarks.a10_v5_1_ablation import campaign as c
 p=candidate();order=protocol.episode_order(p);records=c.planned_records(order)
 records[0].update(status='failed',exit_code=1,usable_receipts=records[0]['expected_jobs'])
 coverage=c.coverage(records,order)
 assert coverage['usable_receipts']==0 and coverage['failed_observed_receipts']==6 and coverage['observed_audited_receipts']==6


def test_actual_development_commands_route_neural_table_only_to_published_smokes(tmp_path):
 from localml_scheduler.benchmarks.a10_v5_1_ablation import development as d
 measured=tmp_path/'measured.json';neural=tmp_path/'neural.json'
 bundle={'settings':tmp_path/'settings.json','source_commit':'a'*40,'plan':{'episode_max_seconds':3600,'deadline_utc':'2030-01-01T00:00:00Z'},'inputs':{c:{'calibration':neural if c=='published-pipeline' else measured}for c in ('short-mixed','backlog-six','published-pipeline')}}
 records=d.planned_records();seen=[]
 for record in records:
  command=d.episode_command(bundle,record,{'path':tmp_path/(record['output']+'.json')},tmp_path)
  selected=command[command.index('--calibration')+1];seen.append(selected)
  assert selected==str(neural if record['condition']=='published-pipeline'else measured)
  assert command[command.index('--arm')+1]=='v5_1'and command[command.index('--mode')+1]==record['mode']
 assert seen.count(str(neural))==2 and seen.count(str(measured))==4
