"""CPU-only replay fixtures, not measured GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from localml_scheduler.benchmarks.a10_resource_mechanisms import context_isolation as c
from localml_scheduler.tests.test_a10_visibility_qualification import fixture as q_fixture,xml,GPU
from localml_scheduler.tests.test_a10_resource_calibration import evidence
from localml_scheduler.benchmarks.a10_resource_mechanisms import calibrate,workloads


def qualified(gpu=GPU,runtime='fixture'):
    p=q_fixture();p['source_sha256']=c.visibility.sha(c.visibility.SOURCE);p['gpu_uuid']=gpu
    for w in p['workers']:w['source_sha256']=p['source_sha256'];w['gpu_uuid']=gpu;w['runtime_identity']={'torch_version':runtime,'cuda_version':'fixture'}
    for phase in p['phases']:
        for sample in phase['observations']:
            sample['command'][-1]=gpu;sample['raw_xml']=sample['raw_xml'].replace(GPU,gpu)
    return p

def association(*,gpu=GPU,spec=None,root_pid=20,ticks=70,mono=200.,wall=100.,nonce='f'*32):
    spec=spec or {'spec_sha256':'b'*64,'expected_runtime_signature':'signature','torch_version':'fixture'}
    q=qualified(gpu,spec['torch_version']);root={'pid':root_pid,'start_ticks':ticks}
    pipe={'path':'/private/empty','device':1,'inode':2,'uid':1000,'mode':0o700,'empty':True}
    request={'source_sha256':c.visibility.sha(c.SOURCE),'nonce':nonce,'gpu_uuid':gpu,'spec_sha256':spec['spec_sha256'],
        'signature':spec['expected_runtime_signature'],'mps_identity':pipe,'issued_monotonic':mono,'issued_unix':wall}
    ready={'pid':root_pid+1,'ppid':root_pid,'start_ticks':ticks+1,'source_sha256':c.visibility.sha(c.SOURCE),'request':request,
        'gpu_uuid':gpu,'mps_identity':pipe,'checksum':c.checksum(nonce),'sentinel_bytes':8,'synchronized':True,
        'runtime_identity':q['workers'][0]['runtime_identity'],'ready_monotonic':mono+.5}
    records=[{**root,'ppid':1,'nspid':[root_pid]}, {'pid':ready['pid'],'ppid':root_pid,'start_ticks':ticks+1,'nspid':[ready['pid']]}]
    value={'schema':c.SCHEMA,'request':request,'ready':ready,'root':root,'mps_identity':pipe,'qualification_sha256':c.canonical(q),
        'ack':{'nonce':nonce,'pid':ready['pid'],'start_ticks':ready['start_ticks']},'acknowledged_monotonic':mono+.9,
        'eligible_after_monotonic':mono+1.1,
        'finished':{'nonce':nonce,'pid':ready['pid'],'start_ticks':ready['start_ticks'],
                    'source_sha256':ready['source_sha256'],'released_monotonic':mono+1.1},
        'released_monotonic':mono+1.,'compute_pid':5001,'idle_queries':[],'singleton_queries':[],'ownership_verified':True}
    for index in range(3):
        value['idle_queries'].append({'command':['nvidia-smi','-q','-x','-i',gpu],'raw_xml':xml([],gpu=gpu),'compute_pids':[],
            'started_monotonic':mono+index*.1,'finished_monotonic':mono+index*.1+.01,'mps_identity':pipe})
        value['singleton_queries'].append({'command':['nvidia-smi','-q','-x','-i',gpu],'raw_xml':xml([5001],gpu=gpu),'compute_pids':[5001],
            'started_monotonic':mono+.6+index*.1,'finished_monotonic':mono+.61+index*.1,'mps_identity':pipe,
            'process_snapshot_before':records,'process_snapshot_after':records})
    return value,q,spec,records

def test_qualified_namespace_singleton_association_replays_without_kernel_alias():
    a,q,s,_=association();result=c.audit_association(a,q,spec=s,gpu_uuid=GPU)
    assert result['compute_pid']==5001 and result['visible_pid']==21 and 'not kernel' in result['method']

@pytest.mark.parametrize('kind',['nonce','source','spec','uuid','runtime','birth','ancestry','changed_pid','extra','graphics','mps','qualification','stale','ack','sentinel'])
def test_context_association_rejects_tampering(kind):
    a,q,s,_=deepcopy(association());row=a['singleton_queries'][0]
    if kind=='nonce':a['request']['nonce']='0'*32
    elif kind=='source':a['request']['source_sha256']='0'*64
    elif kind=='spec':s['spec_sha256']='0'*64
    elif kind=='uuid':a['ready']['gpu_uuid']='GPU-wrong'
    elif kind=='runtime':a['ready']['runtime_identity']={'torch_version':'different'}
    elif kind=='birth':row['process_snapshot_after'][1]['start_ticks']+=1
    elif kind=='ancestry':row['process_snapshot_after'][1]['ppid']=99
    elif kind=='changed_pid':row.update(compute_pids=[6001],raw_xml=xml([6001]))
    elif kind=='extra':row.update(compute_pids=[5001,6001],raw_xml=xml([5001,6001]))
    elif kind=='graphics':row['raw_xml']=xml([5001],kind='G')
    elif kind=='mps':row['mps_identity']={**a['mps_identity'],'inode':99}
    elif kind=='qualification':q['complete']=False
    elif kind=='stale':row['started_monotonic']=1.
    elif kind=='ack':a['ack']['nonce']='0'*32
    else:a['ready']['sentinel_bytes']=0
    with pytest.raises((ValueError,KeyError)):c.audit_association(a,q,spec=s,gpu_uuid=GPU)


def sample(a,records,gpu=GPU):
    query={'command':['nvidia-smi','-q','-x','-i',gpu],'raw_xml':xml([5001],gpu=gpu),'compute_pids':[5001],
        'started_monotonic':210.,'finished_monotonic':210.1}
    return {'gpu':{'gpu_uuid':gpu},'compute_apps':[{'gpu_uuid':gpu,'pid':5001}],
        'sample_started_monotonic':209.9,'sample_finished_monotonic':210.2,'nvml_process_query':query,
        'mps_identity':a['mps_identity'],'context_association_sha256':c.canonical(a),'window_eligible':True,
        'process_snapshot_before':records,'process_snapshot':records}

@pytest.mark.parametrize('kind',['startup','raw','extra','afterbirth','mps','wrongassociation'])
def test_eligible_sample_raw_queries_and_live_brackets_replay_fail_closed(kind):
    a,q,s,records=association();x=sample(a,records);assert c.audit_sample(x,a)['observed_compute_pids']==[5001]
    if kind=='startup':x['window_eligible']=False
    elif kind=='raw':x['nvml_process_query']['raw_xml']=xml([9999])
    elif kind=='extra':x['compute_apps'].append({'gpu_uuid':GPU,'pid':9999})
    elif kind=='afterbirth':x['process_snapshot'][1]['start_ticks']+=1
    elif kind=='mps':x['mps_identity']={**a['mps_identity'],'inode':99}
    else:x['context_association_sha256']='0'*64
    with pytest.raises(ValueError):c.audit_sample(x,a)


def test_full_raw_row_replay_preserves_windows_updates_and_explicit_sentinel_overhead():
    spec,raw=evidence();runtime=spec['torch_version'];qualification=qualified('fixture-uuid',runtime)
    for index,launch in enumerate(raw['launches']):
        a,q,_,records=association(gpu='fixture-uuid',spec=spec,root_pid=launch['pid'],ticks=launch['pid_start_ticks'],
            mono=200.+index*100,wall=launch['launched_unix'],nonce=str(index+1)*32)
        assert q==qualification;launch['context_association']=a
        for row in [r for r in raw['samples'] if r['launch']==index]:
            start=200.+index*100+(row['sample_started_unix']-launch['launched_unix'])
            row.update(process_snapshot_before=records,process_snapshot=records,compute_apps=[{'gpu_uuid':'fixture-uuid','pid':5001}],
                sample_started_monotonic=start+.2,sample_finished_monotonic=start+.3,mps_identity=a['mps_identity'],
                context_association_sha256=c.canonical(a),window_eligible=True,nvml_process_query={
                    'command':['nvidia-smi','-q','-x','-i','fixture-uuid'],'raw_xml':xml([5001],gpu='fixture-uuid'),
                    'compute_pids':[5001],'started_monotonic':start+.2,'finished_monotonic':start+.25})
    raw['events'][0]['created_at']='1970-01-01T00:01:41.500000+00:00'
    raw['events'][1]['created_at']='1970-01-01T00:01:49.500000+00:00'
    raw['visibility_qualification']=qualification;raw['resources']['calibration_context_sentinel_bytes']=8
    assert c.audit_raw_evidence(spec,raw,gpu_uuid='fixture-uuid')['valid']
    retained=deepcopy(raw)
    row=calibrate.extract_row(spec,**raw)
    assert raw==retained
    assert all(t['corrected_seconds']>=0 and t['raw_seconds']==pytest.approx(t['corrected_seconds']+t['diagnostic_wait_seconds']) for t in row['calibration_diagnostic_timing'])
    assert row['phase_seconds']['startup']==pytest.approx(.9)
    assert row['successful_updates']==64 and row['windows']==raw['windows'] and row['context_instrumentation']['calibration_only_sentinel_bytes']==8
    assert 'no kernel PID mapping' in row['isolation_derivation']
    drained=deepcopy(raw);drain=deepcopy(drained['samples'][-1])
    drain.update(window_eligible=False,drain_classification='terminal_context_absent_or_zombie',compute_apps=[])
    drain['nvml_process_query'].update(raw_xml=xml([],gpu='fixture-uuid'),compute_pids=[])
    drain['process_snapshot_before']=drain['process_snapshot_before'][:1];drain['process_snapshot']=drain['process_snapshot'][:1]
    drain['worker'].update(trial_peak_allocated_mb=9999.,trial_peak_reserved_mb=10000.)
    drain['gpu']['used_mib']=15000.;drained['samples'].append(drain)
    replay=calibrate.extract_row(spec,**drained)
    assert replay['peak_allocated_mib']==row['peak_allocated_mib'] and replay['peak_device_used_mib']==row['peak_device_used_mib']
    missing=deepcopy(raw);missing.pop('visibility_qualification')
    with pytest.raises(ValueError,match='requires retained'):calibrate.extract_row(spec,**missing)
    bad=deepcopy(raw);bad['windows'][0]['started_at']=100.5
    with pytest.raises(ValueError,match='challenge'):calibrate.extract_row(spec,**bad)
    bad=deepcopy(raw);bad['launches'][1]['context_association']['request']['nonce']=bad['launches'][0]['context_association']['request']['nonce']
    with pytest.raises(ValueError,match='fresh context nonce'):calibrate.extract_row(spec,**bad)


def test_cleanup_reaps_owned_separate_session_orphan_and_preserves_unrelated(tmp_path):
    import os,subprocess,sys,time
    from types import SimpleNamespace
    path=tmp_path/'child.pid';release=tmp_path/'exit-parent'
    program="""import pathlib,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
while not pathlib.Path(sys.argv[2]).exists():time.sleep(.01)
"""
    parent=subprocess.Popen([sys.executable,'-c',program,str(path),str(release)])
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    child=None
    try:
        until=time.monotonic()+3
        while not path.exists() and time.monotonic()<until:time.sleep(.01)
        assert path.exists();child=int(path.read_text());root=c.visibility.birth(parent.pid);ready=c.visibility.birth(child)
        assert ready['ppid']==parent.pid
        release.touch();parent.wait(timeout=2)
        association={'root':{'pid':root['pid'],'start_ticks':root['start_ticks']},'ready':ready,'ownership_verified':True}
        c.cleanup_owned(SimpleNamespace(process=parent),snapshot=calibrate.process_snapshot,association=association)
        status=Path('/proc',str(child),'stat')
        assert not status.exists() or status.read_text().rsplit(')',1)[1].split()[0]=='Z'
        assert unrelated.poll()is None
    finally:
        if parent.poll()is None:parent.kill();parent.wait(timeout=2)
        if child is not None:
            try:os.kill(child,9)
            except ProcessLookupError:pass
        unrelated.kill();unrelated.wait(timeout=2)


def test_live_empty_visibility_rejected_but_evidenced_terminal_drain_noneligible():
    a,q,s,records=association();x=sample(a,records)
    x['compute_apps']=[];x['nvml_process_query'].update(compute_pids=[],raw_xml=xml([]))
    with pytest.raises(ValueError,match='empty'):c.audit_sample(x,a)
    x.update(window_eligible=False,drain_classification='terminal_context_absent_or_zombie')
    with pytest.raises(ValueError,match='still live'):c.audit_sample(x,a)
    x['process_snapshot_before']=records[:1];x['process_snapshot']=records[:1]
    assert c.audit_sample(x,a)['window_eligible']is False


def test_absolute_manifest_cutoff_bounds_context_proof_before_dispatch():
    common=dict(started=90.,job_started=95.,max_seconds=100.,job_timeout=100.,now_monotonic=100.,now_unix=1000.)
    assert c.bounded_context_deadline(**common,absolute_unix=1002.)==102.
    with pytest.raises(TimeoutError,match='before preparation'):c.bounded_context_deadline(**common,absolute_unix=999.)
