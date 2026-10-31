"""CPU stubs only: no GPU worker, provider, policy tuning or held-out launch."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

import pytest

from localml_scheduler.benchmarks.a10_v5_ablation import development as d,protocol
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import resource_config
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE


def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def snapshot(manifest,path,source_commit):
    return {'scheduler_python_sources':{'fixture.py':'a'*64},'manifest_sha256':d.file_hash(path),'source_commit':source_commit}


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    manifests=tmp_path/'sealed';manifests.mkdir();settings=tmp_path/'settings.json';cal=tmp_path/'table.json'
    write(settings,{});write(cal,{'fixture':'not a measured table'})
    monkeypatch.setattr(d,'source_snapshot',snapshot)
    selected={};snapshots={}
    for condition,chains in [('short-mixed',3),('backlog-six',6)]:
        jobs=[]
        for role in ('parent','child'):
            for chain in range(chains):
                i=len(jobs)
                jobs.append(protocol.make_spec(resource_config('tiny'),name=f'{role}-{chain}-tiny',role=role,
                    chain=chain,epochs=2,steps_per_epoch=32,seed=1000+i,train_pool_seed=2000+2*i,
                    validation_pool_seed=2001+2*i,source_path=workloads.HERE/'candidate.py'))
        manifest={'jobs':jobs,'profile_source':SOURCE,'condition':{'name':condition},'job_timeout_seconds':600,
            'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()}
        path=manifests/(condition+'.json');write(path,manifest)
        selected[condition]={'path':path,'manifest':manifest};snapshots[condition]=snapshot(manifest,path,'b'*40)
    return {'plan':{'episode_max_seconds':600,'deadline_utc':selected['short-mixed']['manifest']['deadline_utc']},
        'settings':settings,'calibration':cal,'source_commit':'b'*40,'inputs':selected,'snapshots':snapshots,
        'table':{'fixture':'not a measured table'},'hashes':{str(settings):d.file_hash(settings),str(cal):d.file_hash(cal)}}


def fake_run(monkeypatch,*,exit_code=0,stuck=False,on_launch=None):
    launches=[];cleaned=[]
    class Child:
        def __init__(self,command,**kwargs):
            assert command[:3]==[d.sys.executable,'-m','localml_scheduler.benchmarks.a10_v5_ablation.episode']
            assert kwargs['start_new_session']
            self.command=command;self.pid=9000+len(launches);self.returncode=None;launches.append(self)
            if on_launch:on_launch(self)
        def poll(self):
            if not stuck:self.returncode=exit_code
            return self.returncode
    def cleanup(child,timeout):
        cleaned.append(child)
        if child.returncode is None:child.returncode=-15
    def audit(output,record,manifest,*a):
        n=record['expected_jobs']if record['mode']=='individual'else 2
        return {'valid':True,'usable_receipts':record['expected_jobs'],'errors':[],
            'request_latencies_seconds':{str(i):float(record['cap']or 1)for i in range(n)}}
    monkeypatch.setattr(d.subprocess,'Popen',Child);monkeypatch.setattr(d,'terminate_process_tree',cleanup)
    monkeypatch.setattr(d,'audit_episode',audit);monkeypatch.setattr(d,'shadow_samples',lambda output:[])
    return launches,cleaned


def test_exact_fixed_twelve_episodes_and_complete_denominators():
    records=d.planned_records();cov=d.coverage(records)
    assert len(records)==12 and cov['planned_jobs']==120
    for cap in (1,2,3,4):assert {r['mode']for r in records if r['cap']==cap}=={'individual','group'}
    assert len([r for r in records if r['kind']=='shadow'])==2
    assert len([r for r in records if r['kind']=='smoke'])==2
    with pytest.raises(ValueError):d.coverage(records[:-1])


def test_causal_seeds_disjoint_identity_rebuilt_but_budget_and_templates_fixed(bundle):
    for record in d.planned_records():
        original=bundle['inputs'][record['condition']]['manifest'];before=deepcopy(original)
        manifest=d.development_manifest(original,record)
        assert manifest['development']is True
        for source,spec in zip(original['jobs'],manifest['jobs']):
            for key in ('seed','train_pool_seed','validation_pool_seed'):assert spec[key]==source[key]+9000000
            assert spec['steps_per_epoch']==source['steps_per_epoch']and spec['epochs']==2
            assert spec['chain']==source['chain']and spec['role']==source['role']
            assert spec['template_key']==source['template_key']
            assert spec['spec_sha256']==workloads.spec_digest(spec)!=source['spec_sha256']
            assert spec['expected_runtime_signature']==workloads.expected_signature(spec)!=source['expected_runtime_signature']
        assert original==before
        if record['kind']=='shadow':
            assert manifest['development_shadow']is True
            assert d.arm_flags(record['arm'],development_shadow=True)['continuation']is False
        if record['kind']=='fixed_cap':assert manifest['development_fixed_cap']==record['cap']


def test_full_controller_cpu_stub_never_changes_sealed_inputs_or_launches_heldout(bundle,tmp_path,monkeypatch):
    originals={str(x['path']):x['path'].read_bytes()for x in bundle['inputs'].values()}
    launches,cleaned=fake_run(monkeypatch)
    root=tmp_path/'development';records=d.execute(bundle,root,1000.)
    assert launches==cleaned and len(launches)==12
    assert d.load(root/'coverage.json')['usable_receipts']==120
    assert all(Path(c.command[c.command.index('--manifest')+1]).is_relative_to(root)for c in launches)
    assert all(d.load(Path(c.command[c.command.index('--manifest')+1]))['development']is True for c in launches)
    assert all(Path(p).read_bytes()==raw for p,raw in originals.items())
    assert d.load(root/'fixed-cap-summary.json')['best_caps_by_mode']=={'individual':[1],'group':[1]}
    assert d.load(root/'development-complete.json')['no_automatic_retuning']
    with pytest.raises(FileExistsError):d.execute(bundle,root,1000.)


def test_failure_stops_all_following_episodes_without_survivor_best_cap(bundle,tmp_path,monkeypatch):
    launches,cleaned=fake_run(monkeypatch,exit_code=1);root=tmp_path/'failed'
    with pytest.raises(RuntimeError):d.execute(bundle,root,1000.)
    cov=d.load(root/'coverage.json')
    assert len(launches)==1 and launches==cleaned
    assert cov['planned_episodes']==12 and cov['failed_episodes']==1 and cov['unattempted_episodes']==11
    assert cov['planned_jobs']==120 and cov['unattempted_jobs']==108
    assert d.load(root/'fixed-cap-summary.json')['best_caps_by_mode']=={'individual':None,'group':None}
    assert not (root/'development-complete.json').exists()


def test_finite_external_bound_terminates_only_owned_episode(bundle,tmp_path,monkeypatch):
    launches,cleaned=fake_run(monkeypatch,stuck=True);now=[0.]
    monkeypatch.setattr(d.time,'monotonic',lambda:now[0]);monkeypatch.setattr(d.time,'sleep',lambda t:now.__setitem__(0,now[0]+601))
    with pytest.raises(TimeoutError):d.execute(bundle,tmp_path/'timeout',1000.)
    assert len(launches)==1 and launches==cleaned


def test_shadow_export_matches_observed_initial_action_only_and_labels_future_arrivals(tmp_path):
    p={'trial_id':'trial','context':{'snapshot_at':100.},'halves':[{
        'admit':{'job_completions':{'a':110.}},'defer':{'job_completions':{'a':120.}}}]*2}
    result={'v5_events':[{'event_type':'v5_continuation_forecast','payload':p},
        {'event_type':'cold_start_trial_decided','payload':{'trial_id':'trial','decision':'accepted'}}],
        'receipts':{'a':{'feedback_ready_unix':112.,'accepted_unix':90.},
                    'later':{'feedback_ready_unix':130.,'accepted_unix':105.}}}
    write(tmp_path/'result.json',result);rows=d.shadow_samples(tmp_path)
    assert len(rows)==4
    for row in rows:
        assert row['future_cohort_changed']and row['later_arrival_ids']==['later']
        assert row['unchosen_counterfactual_observed']is False and row['exact_future_continuation_verified']is False
        if row['alternative']=='admit':assert row['executed_initial_action_error_seconds']==2.
        else:assert row['observed_receipt_completion_unix']is None and row['executed_initial_action_error_seconds']is None


def test_unknown_shadow_actual_decision_has_no_error_or_counterfactual_claim(tmp_path):
    write(tmp_path/'result.json',{'v5_events':[{'event_type':'v5_continuation_forecast','payload':{
        'trial_id':'unknown','halves':[{'admit':{'job_completions':{'a':110.}}}]}}],
        'receipts':{'a':{'feedback_ready_unix':112.}}})
    rows=d.shadow_samples(tmp_path)
    assert rows[0]['observed_initial_action']is None
    assert rows[0]['executed_initial_action_error_seconds']is None


def test_development_manifest_rejected_by_heldout_auditor_without_weakening_it(bundle,tmp_path):
    from localml_scheduler.benchmarks.a10_v5_ablation.campaign import audit_episode
    record=d.planned_records()[0];manifest=d.development_manifest(bundle['inputs']['backlog-six']['manifest'],record)
    result=audit_episode(tmp_path,record,manifest,{},bundle['table'])
    assert not result['valid'] and 'development' in str(result['errors'])


def test_development_audit_rejects_missing_physical_shadow_flag(bundle,tmp_path):
    record=next(r for r in d.planned_records()if r['kind']=='shadow')
    manifest=d.development_manifest(bundle['inputs']['short-mixed']['manifest'],record)
    write(tmp_path/'result.json',{'complete':False,'receipts':{}});write(tmp_path/'validity-audits.json',{})
    assert not d.audit_episode(tmp_path,record,manifest,{},bundle['table'])['valid']


def test_real_cpu_receipt_causal_and_checkpoint_artifacts_pass_development_audit(tmp_path,monkeypatch):
    # Imported harness injects in-process CPU artifacts and fails if CUDA init
    # occurs. The actual receipt, optimizer audit and preparation checks run.
    from localml_scheduler.tests.test_a10_v5_episode import harness,manifest,H,GPU
    from localml_scheduler.tests.test_a10_resource_cpu_episode import calibration_fixture
    from localml_scheduler.benchmarks.a10_v5_ablation import episode
    cpu=harness.__wrapped__(monkeypatch)
    record=next(r for r in d.planned_records()if r['kind']=='shadow'and r['mode']=='group')
    original=manifest();original['condition']={'name':'short-mixed'}
    actual=d.development_manifest(original,record);path=tmp_path/'manifest.json';write(path,actual)
    output=tmp_path/'episode';output.mkdir();table=calibration_fixture(actual['jobs'][0]['template_key'],H)
    result=episode.run_episode(actual,path,{},output,table,arm=record['arm'],mode=record['mode'],
        max_seconds=30,source_commit='a'*40,service_factory=cpu[0],gpu_reader=lambda:dict(GPU))
    pin=d.load(output/'source-pins-before.json')
    audit=d.audit_episode(output,record,actual,pin,table)
    assert audit['valid'],audit['errors']
    assert audit['usable_receipts']==6 and len(audit['request_latencies_seconds'])==2
    assert audit['development_only']and not audit['heldout_comparison_eligible']
    result['development_shadow']=False;write(output/'result.json',result)
    assert not d.audit_episode(output,record,actual,pin,table)['valid']
