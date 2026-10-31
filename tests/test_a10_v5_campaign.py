"""Inert protocol/controller and saved-proof tests; never launch GPU workers."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal

import pytest

from localml_scheduler.benchmarks.a10_v5_ablation import campaign, protocol
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import resource_config
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE, SCHEMA


def write(path,value): path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def blueprint():
    return dict(arms=list(protocol.ARMS), modes=['individual','group'], repeats=2,
        configurations={'tiny':resource_config('tiny')},
        conditions=[{'name':'three','classes':['tiny']*3,'steps_per_epoch':16},
                    {'name':'six','classes':['tiny']*6,'steps_per_epoch':24}],
        seed_base=7000,calibration_replicas=2,calibration_steps_per_epoch=1024,maximum_updates=65536)


def plan():
    bp=blueprint()
    p=protocol.candidate_plan(bp,blueprint_sha256=workloads.digest(bp),source_commit='a'*40,
        host_id='host-fixture',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
        job_timeout_seconds=600,episode_max_seconds=600,campaign_max_seconds=14400,calibration_max_seconds=1800)
    p['status']='frozen'
    return p


def mock_snapshot(manifest,path,source_commit):
    return {'scheduler_reference_commit':source_commit,'scheduler_python_sources':{'fixture.py':'b'*64},
            'manifest_sha256':campaign.file_hash(path),'external_engine_script_introspection_sha256':'c'*64}


@pytest.fixture
def bundle_files(tmp_path,monkeypatch):
    p=plan();cm=protocol.calibration_manifest(p);runtime=workloads.source_metadata()
    table={'schema':SCHEMA,'source':SOURCE,'hardware_key':'fixture-hardware','gpu_uuid':'GPU-fixture',
        'source_identity':{'calibration_identity_schema':'resource_solo_runtime_v1',
            'source_runtime':runtime,'torch_version':runtime['torch_version'],
            'scheduler_python_sources':{'fixture.py':'b'*64}},
        'entries':{s['template_key']:{'updates_per_second':10.,'mean_allocated_mib':100.,
            'admission_reservation_mib':200.,'phase_seconds':{'startup':1.,'pause':1.,'resume':1.,
            'checkpoint':1.,'validation':1.,'finalization':1.,'delivery':1.}} for s in cm['calibration']}}
    table['artifact_sha256']=workloads.digest(table)
    settings=tmp_path/'settings.json';cal=tmp_path/'calibration.json';pp=tmp_path/'input-plan.json'
    write(settings,{});write(cal,table);write(pp,p)
    from localml_scheduler.benchmarks.a10_v5_ablation import episode
    monkeypatch.setattr(episode,'source_snapshot',mock_snapshot)
    monkeypatch.setattr(campaign,'source_snapshot',mock_snapshot)
    directory=protocol.seal_evaluation_bundle(pp,cal,settings,tmp_path/'sealed')
    return directory,settings,cal


def bundle(files): return campaign.sealed_bundle(*files)


def successful_audit(record):
    count=record['expected_jobs'];n=count if record['mode']=='individual' else 2
    return {'valid':True,'usable_receipts':count,'errors':[], 'profiles_by_name':{'shared':'synthetic'},
        'request_latencies_seconds':{str(i):10. for i in range(n)},'job_jct_seconds':[10.]*count,
        'episode_duration_seconds':12.,'policy_activation':{},'complete':True,'predictor_comparison_eligible':True}


def fake_execution(monkeypatch,*,exit_code=0,stuck=False,on_launch=None):
    launches=[];cleaned=[]
    class Child:
        def __init__(self,command,**kwargs):
            self.command=command;self.pid=9000+len(launches);self.returncode=None
            launches.append(self)
            assert kwargs['start_new_session'] is True
            assert command[:3]==[campaign.sys.executable,'-m','localml_scheduler.benchmarks.a10_v5_ablation.episode']
            assert command[command.index('--source-commit')+1]=='a'*40
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
    return launches,cleaned

def saved_episode(tmp_path,bundle_files,mode='individual',arm='forecast_only',condition='three'):
    value=bundle(bundle_files);item=value['manifests'][condition]
    manifest=deepcopy(item['manifest'])
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
            'metadata':{'feedback_phase_model':{'calibration_sha256':value['table']['artifact_sha256'],
                'hardware_key':value['table']['hardware_key'],'job_signature':'a'*64,'batch_size':spec['batch'],
                'startup_seconds':1.,'resume_seconds':1.,'pause_seconds':1.,'delivery_seconds':1.,'finish_seconds':6.},
                'runtime_completion_identity':spec['expected_runtime_signature'],'runtime_global_step':updates,
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
        'episode_duration_seconds':32.,'policy_activation':{},'fixed_reference_memory_profiles':{s['name']:profiles['id-'+s['name']][str(s['batch'])] for s in specs}}
    write(root/'result.json',result);write(root/'validity-audits.json',audits)
    for name in ('source-pins-before.json','source-pins-after.json'):write(root/name,value['snapshots'][condition])
    write(root/'exit-observations.json',{'completed_job_harness_observations':exits})
    write(root/'preparation-timeline.json',{'episode_started_monotonic':1000.,'asynchronous':True,'worker_stopped':True,'failures':{},'events':preparation_events})
    return root,episode,manifest,value['snapshots'][condition],value['table']



def test_global_fixed_budgets_do_not_depend_on_any_host_rate(bundle_files):
    p=bundle(bundle_files)['plan'];table=campaign.load(bundle_files[2])
    first=protocol.evaluation_manifests(p,table)
    other=deepcopy(table)
    for entry in other['entries'].values():entry['updates_per_second']*=7
    other['gpu_uuid']='GPU-other';other['artifact_sha256']=workloads.digest({k:v for k,v in other.items() if k!='artifact_sha256'})
    second=protocol.evaluation_manifests(p,other)
    assert [[s['steps_per_epoch'] for s in m['jobs']] for m in first.values()]==[[s['steps_per_epoch'] for s in m['jobs']] for m in second.values()]
    assert [len(m['jobs']) for m in first.values()]==[6,12]
    for manifest in first.values():
        for spec in manifest['jobs']:
            assert spec['spec_sha256']==workloads.spec_digest(spec)
            assert spec['expected_runtime_signature']==workloads.expected_signature(spec)
            assert workloads.recipe(spec).epochs==2


def test_chain_extension_preserves_baseline_spec_and_rejects_seven():
    from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import make_spec as baseline
    kwargs=dict(name='p',role='parent',chain=2,epochs=2,steps_per_epoch=16,seed=1,
        train_pool_seed=2,validation_pool_seed=3,source_path=workloads.HERE/'candidate.py')
    assert protocol.make_spec(resource_config('tiny'),**kwargs)==baseline(resource_config('tiny'),**kwargs)
    kwargs['chain']=5
    spec=protocol.make_spec(resource_config('tiny'),**kwargs)
    assert spec['spec_sha256']==workloads.spec_digest(spec)
    kwargs['chain']=6
    with pytest.raises(ValueError,match='0..5'):protocol.make_spec(resource_config('tiny'),**kwargs)


def test_calibration_manifest_stays_dedicated_replicated_runtime_bound():
    p=plan();manifest=protocol.calibration_manifest(p)
    assert not manifest['jobs'] and len(manifest['calibration'])==2
    assert all(s['epochs']==2 and s['steps_per_epoch']==1024 and s['chain'] is None for s in manifest['calibration'])
    from localml_scheduler.benchmarks.a10_resource_mechanisms.calibrate import calibration_source_identity
    identity=calibration_source_identity({'fixture':'rawsnapshot'},manifest,runtime_metadata=manifest['source_runtime'])
    assert identity['calibration_identity_schema']=='resource_solo_runtime_v1'
    assert identity['source_runtime']==manifest['source_runtime']


@pytest.mark.parametrize('field,value', [('arms',['v4']),('modes',['individual']),('calibration_replicas',1),('calibration_steps_per_epoch',2)])
def test_global_blueprint_mutation_cannot_seal(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):protocol.validate_plan(p)


def test_six_arm_matched_blocks_and_mixed_counts(bundle_files):
    b=bundle(bundle_files);order=b['order']
    assert len(order)==48 and sum(e['expected_jobs'] for e in order)==432
    for i in range(0,len(order),6):
        block=order[i:i+6]
        assert len({r['block'] for r in block})==1
        assert {r['arm'] for r in block}==set(protocol.ARMS)
        assert {r['host_id'] for r in block}=={'host-fixture'}
    assert [r['arm'] for r in order[:6]]==list(reversed([r['arm'] for r in order[-6:]]))


@pytest.mark.parametrize('filename',['plan.json','protocol-before.json','three.json'])
def test_sealed_hashes_checked_before_every_child(bundle_files,filename):
    b=bundle(bundle_files);path=bundle_files[0]/filename
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):campaign.verify_bundle(b)


@pytest.mark.parametrize('index',[1,2])
def test_settings_and_calibration_hashes_are_sealed(bundle_files,index):
    b=bundle(bundle_files);bundle_files[index].write_text(bundle_files[index].read_text()+' ')
    with pytest.raises(ValueError,match='changed'):campaign.verify_bundle(b)


def test_source_runtime_or_python_map_mismatch_cannot_launch(bundle_files,monkeypatch):
    monkeypatch.setattr(campaign,'source_snapshot',lambda *a,**kw:{'changed':True})
    with pytest.raises(ValueError,match='source pins'):bundle(bundle_files)


def test_finite_controller_runs_all_arms_and_preserves_full_denominators(bundle_files,tmp_path,monkeypatch):
    launches,cleaned=fake_execution(monkeypatch)
    root=tmp_path/'run';records=campaign.execute(bundle(bundle_files),root)
    assert launches==cleaned and len(records)==48
    assert campaign.load(root/'coverage.json')['usable_receipts']==432
    assert campaign.load(root/'campaign-complete.json')['receipts']==432
    assert len(campaign.load(root/'paired-contrasts.json'))==32
    with pytest.raises(FileExistsError):campaign.execute(bundle(bundle_files),root)


def test_failed_child_failstops_without_dropping_unattempted_jobs(bundle_files,tmp_path,monkeypatch):
    launches,cleaned=fake_execution(monkeypatch,exit_code=1)
    root=tmp_path/'run'
    with pytest.raises(RuntimeError):campaign.execute(bundle(bundle_files),root)
    cov=campaign.load(root/'coverage.json')
    assert launches==cleaned and len(launches)==1
    assert cov['planned_cohorts']==48 and cov['planned_jobs']==432
    assert cov['failed_cohorts']==1 and cov['unattempted_cohorts']==47 and cov['unattempted_jobs']==426
    assert all(r['request_latency_mean_seconds'] is None for r in campaign.load(root/'comparisons.json'))
    assert not (root/'campaign-complete.json').exists()


def test_external_timeout_cleans_only_owned_child(bundle_files,tmp_path,monkeypatch):
    launches,cleaned=fake_execution(monkeypatch,stuck=True);now=[0.]
    monkeypatch.setattr(campaign.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(campaign.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+601.))
    with pytest.raises(TimeoutError,match='episode'):campaign.execute(bundle(bundle_files),tmp_path/'run')
    assert len(launches)==1 and launches==cleaned


@pytest.mark.parametrize('mode',['individual','group'])
@pytest.mark.parametrize('condition,count',[('three',6),('six',12)])
def test_detailed_saved_receipts_audit_generic_counts_and_zero_activation(bundle_files,tmp_path,mode,condition,count):
    a=saved_episode(tmp_path,bundle_files,mode,condition=condition)
    result=campaign.audit_episode(*a)
    assert result['valid'],result['errors']
    assert result['usable_receipts']==count
    assert len(result['request_latencies_seconds'])==(count if mode=='individual' else 2)
    assert result['policy_activation']=={}


@pytest.mark.parametrize('mutation',['updates','epochs','optimizer','receipt','exit','profile','source','early_child','artifact_hash','float_counts','members'])
def test_all_existing_proof_gates_remain_fail_closed(bundle_files,tmp_path,mutation):
    a=saved_episode(tmp_path,bundle_files,mode='group',condition='six');root=a[0];result=campaign.load(root/'result.json')
    job_id=next(iter(result['receipts']));name=a[2]['jobs'][0]['name']
    if mutation=='updates':result['receipts'][job_id]['returned_result']['global_step']-=1
    elif mutation=='epochs':result['receipts'][job_id]['returned_result']['epochs']=1
    elif mutation=='optimizer':result['source_and_budget_audits'][name]['all_adamw_steps']-=1;write(root/'validity-audits.json',result['source_and_budget_audits'])
    elif mutation=='receipt':(root/'jobs'/name/'receipt.json').unlink()
    elif mutation=='exit':result['harness_exit_observations'][job_id]['returncode']=1;write(root/'exit-observations.json',{'completed_job_harness_observations':result['harness_exit_observations']})
    elif mutation=='profile':next(iter(result['prepared_numeric_profiles'][job_id].values()))['step_cost']+=1
    elif mutation=='source':write(root/'source-pins-after.json',{'changed':True})
    elif mutation=='early_child':next(row for row in result['arrivals'] if row['name'].startswith('child'))['due_seconds']=0
    elif mutation=='artifact_hash':result['receipts'][job_id]['result_sha256']='d'*64
    elif mutation=='float_counts':result['receipts'][job_id]['returned_result']['history'][0]['validation_positive_count']=127.5
    elif mutation=='members':
        path=root/'jobs'/name/'job-final.json';job=campaign.load(path);job['metadata']['feedback_request']['required_job_ids'].pop();write(path,job)
    write(root/'result.json',result)
    if mutation in {'updates','epochs','artifact_hash','float_counts'}:write(root/'jobs'/name/'receipt.json',result['receipts'][job_id])
    assert not campaign.audit_episode(*a)['valid']


def test_fixed_three_profiles_and_recovered_receipt_sha_preserve_source_bytes(bundle_files,tmp_path):
    a=saved_episode(tmp_path,bundle_files,arm='fixed_three');assert campaign.audit_episode(*a)['valid']
    result=campaign.load(a[0]/'result.json');name=a[2]['jobs'][0]['name']
    result['fixed_reference_memory_profiles'][name]['peak_vram_mib']+=1;write(a[0]/'result.json',result)
    assert not campaign.audit_episode(*a)['valid']


def test_per_host_condition_repeat_summaries_and_paired_contrasts_do_not_pool(bundle_files):
    b=bundle(bundle_files);records=campaign.planned_records(b['order'])
    for r in records:
        r.update(status='completed',usable_receipts=r['expected_jobs'],exit_code=0,audit=successful_audit(r))
        value=(1 if r['condition']=='three' else 10)*(1 if r['arm']=='v4' else 2)
        r['audit']['request_latencies_seconds']={k:value for k in r['audit']['request_latencies_seconds']}
    rows=campaign.comparisons(records,b['plan'])
    assert all(r['host_id']=='host-fixture' for r in rows)
    assert next(r for r in rows if (r['condition'],r['arm'])==('three','v4'))['request_latency_mean_seconds']==1
    contrasts=campaign.paired_contrasts(records,b['plan'])
    c=next(c for c in contrasts if (c['condition'],c['mode'],c['arm'],c['reference'])==('six','group','combined','v4'))
    assert [p['mean_latency_delta_seconds'] for p in c['pairs']]==[10,10]
    records[0]['status']='failed'
    c=next(c for c in campaign.paired_contrasts(records,b['plan']) if (c['condition'],c['mode'],c['arm'],c['reference'])==('three','individual','combined','v4'))
    assert c['valid_pairs']==1 and 'mean_latency_delta_seconds' not in c['pairs'][0]
    records[0]['host_id']='other-host'
    with pytest.raises(ValueError,match='coverage'):campaign.comparisons(records,b['plan'])


def test_saved_receipt_path_traversal_and_symlinks_still_reject(tmp_path):
    expected=tmp_path/'artifact.json';write(expected,{})
    with pytest.raises(ValueError):campaign.receipt_artifact_path('/remote/cohort/../artifact.json',tmp_path,expected,'/remote/cohort')
    link=tmp_path/'link.json';link.symlink_to(expected)
    with pytest.raises(ValueError):campaign.receipt_artifact_path(str(link),tmp_path,link)


def neural_table(measured):
    table={'schema':'a10_dedicated_calibration_v1','hardware_key':measured['hardware_key'],
        'gpu_uuid':measured['gpu_uuid'],'predictor_artifact_sha256':protocol.STUDENT_SHA256,
        'measured_calibration_artifact_sha256':measured['artifact_sha256'],
        'wall_multiplier':1.,'peak_add_mib':0.,'calibration_parameter_count_max':100000000,
        'costs':dict.fromkeys(('startup','pause','resume','delivery','validation','checkpoint','finalization'),1.),
        'observed_at':'2026-10-30T00:00:00+00:00','measurement_job_signatures':['fixture-solo'],
        'isolated_seconds_per_update_max':.1}
    table['calibration_sha256']=protocol.neural_digest(table)
    return table


def test_configuration_label_budget_mapping_is_global_not_host_fitted(bundle_files):
    p=deepcopy(bundle(bundle_files)['plan']);p['configurations']['compute']=resource_config('compute')
    p['configuration_order']=list(p['configurations'])
    p['conditions'][1]['classes']=['tiny','compute']*3
    p['conditions'][1]['steps_per_epoch']={'tiny':24,'compute':40}
    p['blueprint'].update(configurations=deepcopy(p['configurations']),conditions=deepcopy(p['conditions']))
    p['blueprint_sha256']=workloads.digest(p['blueprint'])
    table=campaign.load(bundle_files[2]);example=next(iter(table['entries'].values()))
    for spec in protocol.calibration_manifest(p)['calibration']:
        table['entries'][spec['template_key']]=deepcopy(example)
    table['artifact_sha256']=workloads.digest({k:v for k,v in table.items() if k!='artifact_sha256'})
    manifests=protocol.evaluation_manifests(p,table)
    assert [s['steps_per_epoch'] for s in manifests['six']['jobs']]==[24,40]*6
    assert all(s['epochs']==2 for s in manifests['six']['jobs'])
    p['conditions'][1]['steps_per_epoch'].pop('compute')
    p['blueprint']['conditions']=deepcopy(p['conditions'])
    with pytest.raises(ValueError,match='exactly cover'):protocol.validate_plan(p)


def test_condition_selects_and_pins_separate_same_host_neural_table(bundle_files,tmp_path,monkeypatch):
    p=deepcopy(bundle(bundle_files)['plan']);p['conditions'][1]['profile_source']='ml_predictor'
    p['blueprint']['conditions']=deepcopy(p['conditions']);p['blueprint_sha256']=workloads.digest(p['blueprint'])
    pp=tmp_path/'neural-plan.json';np=tmp_path/'neural-calibration.json';write(pp,p)
    table=neural_table(campaign.load(bundle_files[2]));write(np,table)
    output=protocol.seal_evaluation_bundle(pp,bundle_files[2],bundle_files[1],tmp_path/'nn-sealed',np)
    b=campaign.sealed_bundle(output,bundle_files[1],bundle_files[2],np)
    assert b['manifests']['three']['calibration']==bundle_files[2]
    assert b['manifests']['six']['calibration']==np
    assert b['manifests']['six']['manifest']['profile_source']=='ml_predictor'
    launches,cleaned=fake_execution(monkeypatch)
    campaign.execute(b,tmp_path/'neural-run')
    assert launches==cleaned
    for child in launches:
        path=Path(child.command[child.command.index('--manifest')+1])
        expected=np if path.name=='six.json' else bundle_files[2]
        assert child.command[child.command.index('--calibration')+1]==str(expected)
    np.write_text(np.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):campaign.verify_bundle(b)


@pytest.mark.parametrize('key,value',[('gpu_uuid','GPU-other'),('predictor_artifact_sha256','b'*64),
                                     ('measured_calibration_artifact_sha256','c'*64)])
def test_neural_correction_cannot_cross_host_release_or_calibration_fold(bundle_files,key,value):
    measured=campaign.load(bundle_files[2]);table=neural_table(measured)
    table[key]=value;table['calibration_sha256']=protocol.neural_digest(table)
    with pytest.raises(ValueError,match='same host'):protocol.validate_neural_table(table,measured)


@pytest.mark.parametrize('arm',['v4','fixed_three','combined'])
def test_actual_neural_profile_schema_audits_same_artifact_and_calibration(bundle_files,tmp_path,arm):
    from hashlib import sha256
    a=list(saved_episode(tmp_path,bundle_files,arm=arm));root=a[0];measured=a[4];table=neural_table(measured)
    a[2]['profile_source']='ml_predictor';a[2]['calibration_sha256']=table['calibration_sha256'];a[4]=table
    result=campaign.load(root/'result.json');result['profile_source']='ml_predictor';result['calibration_sha256']=table['calibration_sha256']
    artifact=sha256((protocol.STUDENT_SHA256+table['calibration_sha256']).encode()).hexdigest()
    for spec in a[2]['jobs']:
        jid='id-'+spec['name'];result['prediction_sources'][jid]='ml_predictor'
        for profile in result['prepared_numeric_profiles'][jid].values():
            profile.pop('profile_source',None);profile['artifact_sha256']=artifact
            profile['calibration']='frozen_dedicated_a10_calibration_v1';profile['calibration_sha256']=table['calibration_sha256']
        result['fixed_reference_memory_profiles'][spec['name']]=deepcopy(result['prepared_numeric_profiles'][jid][str(spec['batch'])])
        path=root/'jobs'/spec['name']/'job-final.json';job=campaign.load(path)
        job['metadata']['feedback_phase_model']['calibration_sha256']=table['calibration_sha256'];write(path,job)
    write(root/'result.json',result)
    timeline=campaign.load(root/'preparation-timeline.json')
    for e in timeline['events']:e['artifact_sha256']=artifact
    write(root/'preparation-timeline.json',timeline)
    audit=campaign.audit_episode(*a)
    assert audit['valid'],audit['errors']
    first=next(iter(result['prepared_numeric_profiles'].values()))
    next(iter(first.values()))['calibration_sha256']='b'*64;write(root/'result.json',result)
    assert not campaign.audit_episode(*a)['valid']


def test_mismatched_future_phase_costs_fail_even_with_correct_rates(bundle_files,tmp_path):
    a=saved_episode(tmp_path,bundle_files)
    path=a[0]/'jobs'/a[2]['jobs'][0]['name']/'job-final.json';job=campaign.load(path)
    job['metadata']['feedback_phase_model']['finish_seconds']=0.;write(path,job)
    assert not campaign.audit_episode(*a)['valid']


def test_candidate_and_reseal_are_never_implicitly_authorized(bundle_files,tmp_path):
    p=deepcopy(bundle(bundle_files)['plan']);p['status']='candidate_pending_explicit_execution_freeze'
    pp=tmp_path/'candidate.json';write(pp,p)
    with pytest.raises(ValueError,match='frozen'):protocol.seal_evaluation_bundle(pp,bundle_files[2],bundle_files[1],tmp_path/'candidate-out')
    assert not (tmp_path/'candidate-out').exists()
    with pytest.raises(FileExistsError):protocol.seal_evaluation_bundle(bundle_files[0]/'plan.json',bundle_files[2],bundle_files[1],bundle_files[0])


def test_revised_operational_bounds_accept_eight_hour_campaign_and_reject_excess():
    p=plan();p.update(job_timeout_seconds=3600,episode_max_seconds=3600,
        calibration_max_seconds=7200,campaign_max_seconds=28800)
    assert protocol.validate_plan(p) is p
    for key in ('job_timeout_seconds','episode_max_seconds','calibration_max_seconds','campaign_max_seconds'):
        bad=deepcopy(p);bad[key]+=1
        with pytest.raises(ValueError,match='bound'):protocol.validate_plan(bad)


@pytest.mark.parametrize('key',['development','development_fixed_cap','development_shadow'])
def test_no_development_controls_even_false_in_heldout_protocol_or_audit(bundle_files,tmp_path,key):
    p=plan();p['conditions'][0][key]=False;p['blueprint']['conditions']=deepcopy(p['conditions'])
    with pytest.raises(ValueError,match='development'):protocol.validate_plan(p)
    a=saved_episode(tmp_path,bundle_files);a[2][key]=False
    audited=campaign.audit_episode(*a)
    assert not audited['valid'] and 'development' in str(audited['errors'])


def test_parent_five_condition_blueprint_has120_matched_cohorts864_jobs_per_host():
    path=Path(protocol.__file__).with_name('blueprint.json');bp=json.loads(path.read_text())
    p=protocol.candidate_plan(bp,blueprint_sha256=campaign.file_hash(path),source_commit='a'*40,
        host_id='fixture-host',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=10)).isoformat(),
        job_timeout_seconds=3600,episode_max_seconds=3600,campaign_max_seconds=28800,calibration_max_seconds=7200)
    order=protocol.episode_order(p)
    assert len(order)==120 and sum(e['expected_jobs'] for e in order)==864
    assert len({e['block'] for e in order})==20
    assert sum(protocol.profile_source(c)=='ml_predictor' for c in p['conditions'])==1
    assert len(protocol.calibration_manifest(p)['calibration'])==8


def test_sorted_json_round_trip_preserves_explicit_calibration_seeds_and_cli_manifest(tmp_path):
    path=Path(protocol.__file__).with_name('blueprint.json');bp=json.loads(path.read_text())
    p=protocol.candidate_plan(bp,blueprint_sha256=campaign.file_hash(path),source_commit='a'*40,
        host_id='fixture-host',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=10)).isoformat(),
        job_timeout_seconds=1800,episode_max_seconds=3600,campaign_max_seconds=28800,calibration_max_seconds=3600)
    p['status']='frozen';before=protocol.calibration_manifest(p)
    plan_path=tmp_path/'plan.json';write(plan_path,p)
    restored=campaign.load(plan_path)
    assert list(restored['configurations'])!=restored['configuration_order']
    assert restored['configuration_order']==p['configuration_order']
    assert protocol.calibration_manifest(restored)==before
    assert protocol.episode_order(restored)==protocol.episode_order(p)
    output=tmp_path/'calibration-manifest.json'
    protocol.main(['calibration-manifest','--plan',str(plan_path),'--output',str(output)])
    assert campaign.load(output)==before
    names=[s['name']for s in before['calibration']]
    assert names==[f'{label}-cal-{replica}'for label in p['configuration_order']for replica in range(2)]
    assert [s['seed']for s in before['calibration']]==list(range(p['seed_base'],p['seed_base']+8))


@pytest.mark.parametrize('bad_order',[None,[],['tiny','tiny'],['unknown'],{},[None]])
def test_seed_order_must_cover_each_configuration_exactly_once(bad_order):
    p=plan();p['configuration_order']=bad_order
    with pytest.raises(ValueError,match='explicit calibration seed order'):
        protocol.validate_plan(p)


def test_full_blueprint_round_trip_seals_and_loads_both_development_and_heldout(tmp_path,monkeypatch):
    """Real protocol/configuration flow; synthetic calibration, no GPU work."""
    from localml_scheduler.benchmarks.a10_v5_ablation import episode,development
    path=Path(protocol.__file__).with_name('blueprint.json');bp=json.loads(path.read_text())
    p=protocol.candidate_plan(bp,blueprint_sha256=campaign.file_hash(path),source_commit='a'*40,
        host_id='fixture-host',deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=10)).isoformat(),
        job_timeout_seconds=1800,episode_max_seconds=3600,campaign_max_seconds=28800,calibration_max_seconds=3600)
    p['status']='frozen';pp=tmp_path/'plan.json';write(pp,p)
    cm=protocol.calibration_manifest(campaign.load(pp));runtime=workloads.source_metadata()
    table={'schema':SCHEMA,'source':SOURCE,'hardware_key':'fixture-hardware','gpu_uuid':'GPU-fixture',
        'source_identity':{'calibration_identity_schema':'resource_solo_runtime_v1',
            'source_runtime':runtime,'torch_version':runtime['torch_version'],
            'scheduler_python_sources':{'fixture.py':'b'*64}},
        'entries':{s['template_key']:{'updates_per_second':10.,'mean_allocated_mib':100.,
            'admission_reservation_mib':200.,'phase_seconds':{'startup':1.,'pause':1.,'resume':1.,
            'checkpoint':1.,'validation':1.,'finalization':1.,'delivery':1.}}for s in cm['calibration']}}
    table['artifact_sha256']=workloads.digest(table)
    settings=tmp_path/'settings.json';cal=tmp_path/'calibration.json';neural=tmp_path/'neural.json'
    write(settings,{});write(cal,table);write(neural,neural_table(table))
    for target in (episode,campaign,development):monkeypatch.setattr(target,'source_snapshot',mock_snapshot)
    sealed=protocol.seal_evaluation_bundle(pp,cal,settings,tmp_path/'sealed',neural)
    heldout=campaign.sealed_bundle(sealed,settings,cal,neural);campaign.verify_bundle(heldout)
    dev=development.sealed_inputs(sealed,settings,cal,'a'*40);development.verify_inputs(dev)
    assert len(heldout['order'])==120 and sum(r['expected_jobs']for r in heldout['order'])==864
    assert sum(r['expected_jobs']for r in development.planned_records())==120
    for record in development.planned_records():
        original=dev['inputs'][record['condition']]['manifest']
        amended=development.development_manifest(original,record)
        assert [s['template_key']for s in original['jobs']]==[s['template_key']for s in amended['jobs']]
        assert [s['steps_per_epoch']for s in original['jobs']]==[s['steps_per_epoch']for s in amended['jobs']]
