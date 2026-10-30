"""Offline synthetic controller tests; no GPU/worker/cloud execution."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import subprocess

import pytest

from localml_scheduler.benchmarks.a10_resource_mechanisms import campaign,study,workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE,SCHEMA


def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


@pytest.fixture
def bundle_files(tmp_path,monkeypatch):
    plan=json.loads((workloads.HERE/'iteration-2-plan.json').read_text())
    plan.update(status='frozen',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat())
    calibration=study.calibration_manifest(plan)
    table={'source':SOURCE,'schema':SCHEMA,'entries':{s['template_key']:{'updates_per_second':10.} for s in calibration['calibration']}}
    table['artifact_sha256']=workloads.digest(table)
    directory=tmp_path/'sealed';directory.mkdir();settings=tmp_path/'settings.json';calibration_path=tmp_path/'calibration.json'
    write(settings,{});write(calibration_path,table);write(directory/'plan.json',plan)
    pins={}
    for name,manifest in study.evaluation_manifests(plan,table).items():
        path=directory/(name+'.json');write(path,manifest);pins[name]=campaign.file_hash(path)
    protocol={'plan':plan,'plan_sha256':campaign.file_hash(directory/'plan.json'),'manifests':pins,
        'calibration_sha256':campaign.file_hash(calibration_path),'order':study.episode_order(plan),'expected_cohorts':64,'expected_jobs':384,'frozen_before_evaluation':True}
    write(directory/'protocol-before.json',protocol)
    # Current engine/source validation is exercised through this snapshot API;
    # synthetic fixtures carry no measured host identity.
    monkeypatch.setattr(campaign,'source_snapshot',lambda manifest,path:{'manifest_sha256':campaign.file_hash(path),'engine':'synthetic-current-pin'})
    return directory,settings,calibration_path


def bundle(bundle_files):return campaign.sealed_bundle(*bundle_files)


def successful_audit(record):
    n=6 if record['mode']=='individual' else 2
    return {'valid':True,'usable_receipts':6,'errors':[],'profiles_by_name':{'shared':'synthetic'},
        'request_latencies_seconds':{str(i):10. for i in range(n)},'job_jct_seconds':[10.]*6,
        'episode_duration_seconds':12.,'policy_activation':{}}


def fake_execution(monkeypatch,*,exit_code=0,stuck=False,on_launch=None):
    launches=[];cleaned=[]
    class Child:
        def __init__(self,command,**kwargs):
            self.command=command;self.pid=9000+len(launches);self.returncode=None
            launches.append(self)
            assert kwargs['start_new_session']is True
            assert command[:4]==[campaign.sys.executable,'-m','localml_scheduler.benchmarks.a10_feedback_latency.run','run']
            if on_launch:on_launch(self)
        def poll(self):
            if not stuck:self.returncode=exit_code
            return self.returncode
    def cleanup(child,timeout):
        cleaned.append(child)
        assert timeout==15.
        if child.returncode is None:child.returncode=-15
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    monkeypatch.setattr(campaign,'terminate_process_tree',cleanup)
    monkeypatch.setattr(campaign,'audit_episode',lambda output,record,*args:successful_audit(record))
    monkeypatch.setattr(campaign,'final_plots',lambda *args:None)
    return launches,cleaned


def test_sealed_order_and_plan_candidate_gate(bundle_files):
    value=bundle(bundle_files);order=value['order']
    assert len(order)==64 and len(campaign.planned_records(order))==64
    cells=[(e['condition'],e['mode'],e['arm']) for e in order]
    assert cells[:32]==list(reversed(cells[32:]))
    directory,_,_=bundle_files;protocol=campaign.load(directory/'protocol-before.json');plan=campaign.load(directory/'plan.json')
    plan['status']='candidate_protocol_pending_iteration1_review';write(directory/'plan.json',plan)
    protocol.update(plan=plan,plan_sha256=campaign.file_hash(directory/'plan.json'));write(directory/'protocol-before.json',protocol)
    with pytest.raises(ValueError,match='must be frozen'):bundle(bundle_files)


@pytest.mark.parametrize('filename',['plan.json','protocol-before.json','complementary-short.json'])
def test_sealed_bundle_hash_tampering_rejected(bundle_files,filename):
    value=bundle(bundle_files);path=bundle_files[0]/filename
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):campaign.verify_bundle(value)


@pytest.mark.parametrize('index',[1,2])
def test_settings_and_table_tampering_rejected(bundle_files,index):
    value=bundle(bundle_files);path=bundle_files[index]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):campaign.verify_bundle(value)


def test_frozen_source_and_engine_change_rejected(bundle_files,monkeypatch):
    value=bundle(bundle_files)
    monkeypatch.setattr(campaign,'source_snapshot',lambda *args:{'engine':'changed'})
    with pytest.raises(ValueError,match='source/engine'):campaign.verify_bundle(value)


def test_serial_64_cell_controller_retains384_denominator_and_no_resume(bundle_files,tmp_path,monkeypatch):
    value=bundle(bundle_files);launches,cleaned=fake_execution(monkeypatch)
    root=tmp_path/'run';records=campaign.execute(value,root)
    assert len(launches)==len(cleaned)==64
    assert [r['status'] for r in records]==['completed']*64
    coverage=campaign.load(root/'coverage.json')
    assert coverage['planned_cohorts']==64 and coverage['planned_jobs']==coverage['usable_receipts']==384
    assert campaign.load(root/'campaign-complete.json')['complete']
    comparisons=campaign.load(root/'comparisons.json');assert len(comparisons)==32
    assert all(r['planned_repeats']==r['valid_repeats']==2 for r in comparisons)
    assert {r['planned_request_units'] for r in comparisons}=={4,12}
    assert all(r['request_latency_distribution']['n']==r['planned_request_units'] for r in comparisons)
    assert all(s['n']==6 for r in comparisons for s in r['job_jct_distributions_by_repeat'].values())
    assert all(r['observed_valid_request_units']==r['planned_request_units'] for r in comparisons)
    assert all('--max-seconds' in c.command and '--deadline-utc' in c.command for c in launches)
    with pytest.raises(FileExistsError):campaign.execute(value,root)
    assert len(launches)==64


def test_failure_preserves_failed_unattempted_denominators_and_stops(bundle_files,tmp_path,monkeypatch):
    launches,cleaned=fake_execution(monkeypatch,exit_code=7);root=tmp_path/'run'
    with pytest.raises(RuntimeError):campaign.execute(bundle(bundle_files),root)
    assert len(launches)==len(cleaned)==1
    coverage=campaign.load(root/'coverage.json')
    assert coverage['planned_jobs']==384 and coverage['failed_cohorts']==1 and coverage['unattempted_cohorts']==63
    assert not (root/'campaign-complete.json').exists() and campaign.load(root/'campaign-aborted.json')['complete']is False
    assert len(campaign.load(root/'campaign-results.json'))==64
    assert all(r['request_latency_mean_seconds']is None for r in campaign.load(root/'comparisons.json'))


def test_ineligible_zero_exit_child_stops_without_rankings(bundle_files,tmp_path,monkeypatch):
    launches,cleaned=fake_execution(monkeypatch)
    monkeypatch.setattr(campaign,'audit_episode',lambda *args:{'valid':False,'usable_receipts':3,'errors':['ineligible']})
    root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='ineligible'):campaign.execute(bundle(bundle_files),root)
    assert len(launches)==1 and len(cleaned)==1
    assert campaign.load(root/'coverage.json')['usable_receipts']==3


@pytest.mark.parametrize('signum',[signal.SIGTERM,signal.SIGINT,signal.SIGHUP])
def test_signals_propagate_and_cleanup_only_owned_child_without_masking(bundle_files,tmp_path,monkeypatch,signum):
    launches,cleaned=fake_execution(monkeypatch,stuck=True)
    before=signal.getsignal(signum)
    def interrupted_sleep(seconds):
        handler=signal.getsignal(signum)
        assert callable(handler)
        handler(signum,None)
    monkeypatch.setattr(campaign.time,'sleep',interrupted_sleep)
    root=tmp_path/'run'
    with pytest.raises(InterruptedError):campaign.execute(bundle(bundle_files),root)
    assert launches==cleaned and len(cleaned)==1
    assert signal.getsignal(signum)==before
    assert campaign.load(root/'active-child.json')['status']=='failed'
    assert campaign.load(root/'coverage.json')['unattempted_cohorts']==63


def test_episode_external_timeout_terminates_owned_child(bundle_files,tmp_path,monkeypatch):
    value=bundle(bundle_files);launches,cleaned=fake_execution(monkeypatch,stuck=True)
    now=[0.]
    monkeypatch.setattr(campaign.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(campaign.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+601.))
    with pytest.raises(TimeoutError,match='episode'):campaign.execute(value,tmp_path/'run')
    assert launches==cleaned and len(cleaned)==1


def test_final_source_hash_recheck_fails_after_last_child(bundle_files,tmp_path,monkeypatch):
    value=bundle(bundle_files)
    launches,cleaned=fake_execution(monkeypatch)
    def audit(output,record,*args):
        if len(launches)==64:bundle_files[1].write_text('changed after final child')
        return successful_audit(record)
    monkeypatch.setattr(campaign,'audit_episode',audit)
    root=tmp_path/'run'
    with pytest.raises(ValueError,match='changed'):campaign.execute(value,root)
    assert len(launches)==len(cleaned)==64 and not (root/'campaign-complete.json').exists()


def test_shared_output_parent_campaign_flock_is_exclusive(bundle_files,tmp_path):
    import fcntl
    with (tmp_path/'.a10-resource-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):campaign.execute(bundle(bundle_files),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def saved_episode(tmp_path,bundle_files,mode='individual',arm='resource_revised'):
    value=bundle(bundle_files);condition=next(iter(value['manifests']));item=value['manifests'][condition]
    manifest=deepcopy(item['manifest'])
    # Explicit different successful-update budgets per job; all remain two epochs.
    for i,spec in enumerate(manifest['jobs']):spec['steps_per_epoch']+=i
    episode={'condition':condition,'mode':mode,'arm':arm,'repeat':1}
    root=tmp_path/'episode';root.mkdir();receipts={};audits={};profiles={};requests={};exits={};arrivals=[];preparation_events=[]
    specs=manifest['jobs']
    for spec in specs:
        name=spec['name'];job_id='id-'+name;role=spec['role'];chain=spec['chain'];updates=2*spec['steps_per_epoch']
        accepted=1000.+(2*chain if mode=='individual' else 0)+(12 if role=='child' else 0)
        wall=1793323000+accepted-1000
        timestamp=datetime.fromtimestamp(wall,timezone.utc).isoformat()
        receipt={'job_id':job_id,'worker_exit_code':0,'expected_updates':updates,'expected_epochs':2,
            'expected_runtime_signature':spec['expected_runtime_signature'],'expected_workload_sha256':campaign.runtime_digest(spec),
            'accepted_monotonic':accepted,'accepted_unix':wall,'feedback_ready_monotonic':accepted+10,'jct_seconds':10.,
            'auc_maximize':True,'selected_auc':.6,'selected_validation_bce':.7,
            'returned_result':{'signature':spec['expected_runtime_signature'],'global_step':updates,'epochs':2,
                'history':[{'validation_negative_count':spec['validation_size']//2,'validation_positive_count':spec['validation_size']//2}]*2}}
        receipts[job_id]=receipt
        audits[name]={'global_step':updates,'all_adamw_steps':updates,'completed_epochs':2,'optimizer_parameter_states':2}
        profile={'profile_source':SOURCE,'calibration':SCHEMA,'artifact_sha256':value['table']['artifact_sha256'],
            'hardware':'A10','batch_size':spec['batch'],'job_signature':'a'*64,'step_cost':100.,'epoch_cost':100.*spec['steps_per_epoch'],
            'avg_vram_mib':100.,'peak_vram_mib':200.}
        profiles[job_id]={str(spec['batch']):profile}
        required=[job_id] if mode=='individual' else sorted('id-'+s['name'] for s in specs if s['role']==role)
        key=name if mode=='individual' else role+'-bracket';requests[key]=10.
        job={'job_id':job_id,'baseline_model_id':name,'submitted_at':timestamp,'packing':{'signature':'a'*64},
            'metadata':{'runtime_completion_identity':spec['expected_runtime_signature'],'runtime_global_step':updates,
                'runtime_total_optimizer_updates':updates,'runtime_finalization_complete':True,
                'feedback_request':{'request_id':key,'required_job_ids':required,'submitted_at':timestamp,
                    'membership_finalized':True,'completion_policy':mode}}}
        job['metadata'].update(resource_workload_spec=deepcopy(spec),resource_template_key=spec['template_key'],pilot_frozen_workload_sha256=campaign.runtime_digest(spec))
        job['config']={'env':{'A10_PILOT_SPEC_JSON':json.dumps(spec)},'runner_kwargs':{'batch_size':spec['batch']}}
        folder=root/'jobs'/name;folder.mkdir(parents=True)
        returned_path=folder/'working/shared_training/result.json';returned_path.parent.mkdir(parents=True)
        execution_path=folder/'execution.json';write(returned_path,receipt['returned_result']);write(execution_path,{'exc_type':None})
        for kind,path in (('result',returned_path),('execution',execution_path)):
            receipt.update({kind+'_path':str(path),kind+'_bytes':path.stat().st_size,kind+'_sha256':campaign.file_hash(path)})
        write(folder/'job-final.json',job);write(folder/'receipt.json',receipt)
        arrivals.append({'name':name,'due_seconds':accepted-1000.,'accepted_seconds':accepted-1000.})
        identity={'job_id':job_id,'job_signature':'a'*64,'batch_size':spec['batch'],'artifact_sha256':value['table']['artifact_sha256'],
            'hardware':'A10','hardware_key':'fixture-only','prediction_contract_sha256':'c'*64}
        preparation_events.extend([{**identity,'phase':'enqueue','monotonic_seconds':accepted+.1},
            {**identity,'phase':'start','monotonic_seconds':accepted+.3},
            {**identity,'phase':'end','monotonic_seconds':accepted+.5,'started_monotonic_seconds':accepted+.2,
                'ended_monotonic_seconds':accepted+.4,'wall_seconds':(accepted+.4)-(accepted+.2),'status':'ready'}])
        exits[job_id]={'returncode':0,'observed_monotonic':accepted+9,'source':'synthetic observation'}
    result={'episode_started_monotonic':1000.,'arrivals':arrivals,'complete':True,'predictor_comparison_eligible':True,'arm':arm,'mode':mode,'profile_source':SOURCE,
        'calibration_sha256':value['table']['artifact_sha256'],'prediction_errors':{},'receipts':receipts,
        'prediction_sources':{job_id:SOURCE for job_id in receipts},'prepared_numeric_profiles':profiles,
        'source_and_budget_audits':audits,'request_latencies_seconds':requests,'harness_exit_observations':exits,
        'episode_duration_seconds':26.,'policy_activation':{},'fixed_reference_memory_profiles':{s['name']:profiles['id-'+s['name']][str(s['batch'])] for s in specs}}
    write(root/'result.json',result);write(root/'validity-audits.json',audits)
    for name in ('source-pins-before.json','source-pins-after.json'):write(root/name,value['snapshots'][condition])
    write(root/'exit-observations.json',{'completed_job_harness_observations':exits})
    write(root/'preparation-timeline.json',{'episode_started_monotonic':1000.,'asynchronous':True,'worker_stopped':True,'failures':{},'events':preparation_events})
    return root,episode,manifest,value['snapshots'][condition],value['table']


@pytest.mark.parametrize('mode',['individual','group'])
def test_saved_receipts_accept_exact_variable_budgets_and_two_epochs(bundle_files,tmp_path,mode):
    args=saved_episode(tmp_path,bundle_files,mode)
    audit=campaign.audit_episode(*args)
    assert audit['valid'],audit['errors']
    assert audit['usable_receipts']==6
    assert len(audit['request_latencies_seconds'])==(6 if mode=='individual' else 2)
    budgets={r['expected_updates'] for r in campaign.load(args[0]/'result.json')['receipts'].values()}
    assert len(budgets)>1 and 64 not in budgets


@pytest.mark.parametrize('mutation',['updates','epochs','optimizer','receipt','exit','profile','source'])
def test_saved_episode_invalid_evidence_fails_closed(bundle_files,tmp_path,mutation):
    args=saved_episode(tmp_path,bundle_files);root=args[0];result=campaign.load(root/'result.json')
    job_id=next(iter(result['receipts']));name=args[2]['jobs'][0]['name']
    if mutation=='updates':result['receipts'][job_id]['returned_result']['global_step']=64
    elif mutation=='epochs':result['receipts'][job_id]['returned_result']['epochs']=1
    elif mutation=='optimizer':result['source_and_budget_audits'][name]['all_adamw_steps']-=1;write(root/'validity-audits.json',result['source_and_budget_audits'])
    elif mutation=='receipt':(root/'jobs'/name/'receipt.json').unlink()
    elif mutation=='exit':result['harness_exit_observations'][job_id]['returncode']=1;write(root/'exit-observations.json',{'completed_job_harness_observations':result['harness_exit_observations']})
    elif mutation=='profile':next(iter(result['prepared_numeric_profiles'][job_id].values()))['artifact_sha256']='x'*64
    elif mutation=='source':write(root/'source-pins-after.json',{'engine':'changed'})
    write(root/'result.json',result)
    # Keep durable receipt equal when testing the deeper returned artifact budget check.
    if mutation in {'updates','epochs'}:write(root/'jobs'/name/'receipt.json',result['receipts'][job_id])
    audit=campaign.audit_episode(*args)
    assert not audit['valid'] and audit['errors']


def test_fixed_three_audits_checked_actual_profile(bundle_files,tmp_path):
    args=saved_episode(tmp_path,bundle_files,arm='resource_fixed_three');audit=campaign.audit_episode(*args)
    assert audit['valid'],audit['errors']
    result=campaign.load(args[0]/'result.json');name=args[2]['jobs'][0]['name']
    result['fixed_reference_memory_profiles'][name]['peak_vram_mib']+=1;write(args[0]/'result.json',result)
    assert not campaign.audit_episode(*args)['valid']


def test_settings_hash_is_checked_before_second_child(bundle_files,tmp_path,monkeypatch):
    value=bundle(bundle_files);launches,cleaned=fake_execution(monkeypatch)
    def audit(output,record,*args):
        bundle_files[1].write_text('changed after first child')
        return successful_audit(record)
    monkeypatch.setattr(campaign,'audit_episode',audit)
    with pytest.raises(ValueError,match='changed'):campaign.execute(value,tmp_path/'run')
    assert len(launches)==len(cleaned)==1
    coverage=campaign.load(tmp_path/'run/coverage.json')
    assert coverage['completed_valid_cohorts']==1 and coverage['unattempted_cohorts']==63


def test_condition_comparisons_never_pool_different_resource_or_duration_conditions(bundle_files):
    value=bundle(bundle_files);records=campaign.planned_records(value['order'])
    by_condition={row['name']:i+1 for i,row in enumerate(value['plan']['conditions'])}
    for record in records:
        record.update(status='completed',usable_receipts=6,audit=successful_audit(record))
        number=by_condition[record['condition']]
        record['audit']['request_latencies_seconds']={k:v*number for k,v in record['audit']['request_latencies_seconds'].items()}
    rows=campaign.comparisons(records,value['plan'])
    assert all(row['request_latency_mean_seconds']==10.*by_condition[row['condition']] for row in rows)
    records[0]['status']='failed'
    row=next(r for r in campaign.comparisons(records,value['plan']) if (r['condition'],r['mode'],r['arm'])==(records[0]['condition'],records[0]['mode'],records[0]['arm']))
    assert row['planned_repeats']==2 and row['valid_repeats']==row['failed_repeats']==1
    assert row['request_latency_mean_seconds']is None
    assert row['request_latency_distribution']is None


def test_latency_tails_use_request_observations_without_member_job_weighting():
    observed=campaign.latency_distribution([10.,30.,40.,80.])
    assert observed['n']==4 and observed['mean_seconds']==40.
    assert observed['p95_seconds']==pytest.approx(74.) and observed['max_seconds']==80.


def test_final_combined_plots_and_gantt_contact_sheet_are_condition_scoped(bundle_files,tmp_path):
    pytest.importorskip('matplotlib');pytest.importorskip('PIL')
    from PIL import Image
    value=bundle(bundle_files);records=campaign.planned_records(value['order']);root=tmp_path/'plots';root.mkdir()
    for record in records:record.update(status='completed',usable_receipts=6,audit=successful_audit(record))
    first=root/records[0]['output'];first.mkdir();Image.new('RGB',(800,600),'white').save(first/'cohort-measurements.png')
    campaign.final_plots(root,records,value['plan'])
    assert (root/'combined-condition-comparisons.png').exists()
    assert (root/'combined-cohort-gantt-metrics.png').exists()
    assert all((root/(c['name']+'-cohort-gantt-metrics.png')).exists() for c in value['plan']['conditions'])


def test_failed_active_child_receipt_write_still_cleans_owned_child(bundle_files,tmp_path,monkeypatch):
    value=bundle(bundle_files);launches,cleaned=fake_execution(monkeypatch,stuck=True)
    original=campaign.atomic_json
    def fail_running_receipt(path,payload):
        if Path(path).name=='active-child.json' and payload.get('status')=='running':raise OSError('injected active child receipt write failure')
        return original(path,payload)
    monkeypatch.setattr(campaign,'atomic_json',fail_running_receipt)
    with pytest.raises(OSError,match='injected'):campaign.execute(value,tmp_path/'run')
    assert len(launches)==1 and launches==cleaned
    assert campaign.load(tmp_path/'run/coverage.json')['failed_cohorts']==1


@pytest.mark.parametrize('mutation',['early_child','wrong_due','early_preparation','artifact_bytes','artifact_hash','full_spec','float_counts'])
def test_causal_artifact_and_full_spec_audit_gates(bundle_files,tmp_path,mutation):
    args=saved_episode(tmp_path,bundle_files);root=args[0];result=campaign.load(root/'result.json')
    if mutation=='early_child':result['arrivals'][-1]['accepted_seconds']-=2.
    elif mutation=='wrong_due':result['arrivals'][-1]['due_seconds']-=1.
    elif mutation=='early_preparation':
        path=root/'preparation-timeline.json';timeline=campaign.load(path);timeline['events'][0]['monotonic_seconds']=999.;write(path,timeline)
    elif mutation in {'artifact_bytes','artifact_hash'}:
        receipt=next(iter(result['receipts'].values()));path=Path(receipt['result_path'])
        if mutation=='artifact_bytes':path.write_text(path.read_text()+' ')
        else:
            raw=path.read_bytes();path.write_bytes(raw.replace(b'"epochs": 2',b'"epochs": 1'))
    elif mutation=='full_spec':
        spec=args[2]['jobs'][0];path=root/'jobs'/spec['name']/'job-final.json';job=campaign.load(path)
        job['metadata']['resource_workload_spec']['steps_per_epoch']+=1;write(path,job)
    elif mutation=='float_counts':
        for receipt in result['receipts'].values():
            for epoch in receipt['returned_result']['history']:
                for key in ('validation_negative_count','validation_positive_count'):epoch[key]=float(epoch[key])
            path=Path(receipt['result_path']);write(path,receipt['returned_result'])
            receipt.update(result_bytes=path.stat().st_size,result_sha256=campaign.file_hash(path))
            job_id=receipt['job_id'];name=job_id.removeprefix('id-');write(root/'jobs'/name/'receipt.json',receipt)
    write(root/'result.json',result)
    audit=campaign.audit_episode(*args)
    assert audit['valid']is (mutation=='float_counts'),audit['errors']


def test_auditor_matches_real_cpu_resource_episode_schema(tmp_path,monkeypatch):
    # Reuse the actual CPU subprocess integration, with all accelerator readings
    # fixture-labeled and CUDA disabled. No A10 measurements or GPU calls occur.
    from localml_scheduler.tests import test_a10_resource_cpu_episode as integration
    from localml_scheduler.benchmarks.a10_feedback_latency import run
    original=run.run_episode;captured={}
    def capture(manifest,manifest_path,settings,output,table,**kwargs):
        result=original(manifest,manifest_path,settings,output,table,**kwargs)
        captured.update(manifest=manifest,manifest_path=manifest_path,output=output,table=table,result=result)
        return result
    monkeypatch.setattr(run,'run_episode',capture)
    integration.test_real_cpu_episode_has_two_causal_brackets_and_exact_audited_receipts(tmp_path,monkeypatch,'resource_incremental')
    snapshot=run.source_snapshot(captured['manifest'],captured['manifest_path'])
    for name in ('source-pins-before.json','source-pins-after.json'):write(captured['output']/name,snapshot)
    write(captured['output']/'result.json',captured['result'])
    audit=campaign.audit_episode(captured['output'],{'mode':'group','arm':'resource_incremental'},captured['manifest'],snapshot,captured['table'])
    assert audit['valid'],audit['errors']
    assert audit['usable_receipts']==6 and len(audit['request_latencies_seconds'])==2


def test_remote_receipt_relocation_preserves_all_original_json_and_hashes(bundle_files,tmp_path):
    args=saved_episode(tmp_path,bundle_files);root=args[0];result=campaign.load(root/'result.json')
    remote='/workspace/owned_resource/runs/iteration-2/00-exact-cohort'
    for receipt in result['receipts'].values():
        for kind in ('result','execution'):
            receipt[kind+'_path']=remote+'/'+str(Path(receipt[kind+'_path']).relative_to(root))
        name=receipt['job_id'].removeprefix('id-');write(root/'jobs'/name/'receipt.json',receipt)
    write(root/'result.json',result)
    before={str(p):campaign.file_hash(p) for p in root.rglob('*') if p.is_file()}
    audit=campaign.audit_episode(*args,remote_output=remote)
    assert audit['valid'],audit['errors']
    assert {str(p):campaign.file_hash(p) for p in root.rglob('*') if p.is_file()}==before
    assert not campaign.audit_episode(*args)['valid']


@pytest.mark.parametrize('malicious_path',[
    '/workspace/owned_resource/runs/iteration-2/00-exact-cohort/../escape/result.json',
    '/workspace/owned_resource/runs/iteration-2/00-exact-cohort-lookalike/jobs/x/result.json',
    '/etc/passwd','jobs/x/result.json',
    '/workspace/owned_resource/runs/iteration-2/00-exact-cohort/jobs/./x/result.json',
    '/workspace/owned_resource/runs/iteration-2/00-exact-cohort/jobs/..\\x/result.json',
])
def test_remote_receipt_traversal_or_prefix_escape_rejected(tmp_path,malicious_path):
    root=tmp_path/'cohort';expected=root/'jobs/x/result.json';expected.parent.mkdir(parents=True);write(expected,{})
    with pytest.raises(ValueError):campaign.receipt_artifact_path(malicious_path,root,expected,'/workspace/owned_resource/runs/iteration-2/00-exact-cohort')


def test_remote_relocation_rejects_symlink_artifact_directory(tmp_path):
    root=tmp_path/'cohort';root.mkdir();target=root/'target';target.mkdir();write(target/'result.json',{})
    (root/'jobs').symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        campaign.receipt_artifact_path('/workspace/exact/jobs/result.json',root,root/'jobs/result.json','/workspace/exact')
