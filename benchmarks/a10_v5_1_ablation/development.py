"""Fixed, finite development diagnostics; never launches or retunes held-out work.

Six V5.1 episodes: short-mixed, backlog-six, published-pipeline x both modes. Every workload seed is shifted by
18000000; configuration/update budgets stay fixed. GPU work occurs only when a
human-authorized parent invokes this CLI; children own the existing GPU flock.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from .process_ownership import owned_descendants,cleanup_owned
from ..a10_feedback_latency.run import atomic_json,check_deadline,file_hash
from ..a10_resource_mechanisms import workloads
from ..a10_resource_mechanisms.profiles import SOURCE
from . import protocol
from ..a10_v5_ablation.calibrate_predictor import _verify_receipt
from .campaign import load,finite,stamp,latency_distribution,check_causal_preparation,profile_without_signature,audit_fifo_dispatch
from .fifo_episode import source_snapshot,validate_manifest,arm_flags

SEED_OFFSET=18000000


def planned_records():
    return [dict(kind='fifo_alpha',condition=condition,arm='v5_1',mode=mode,cap=None,output=f'{i:02d}-{condition}-{mode}',status='unattempted',exit_code=None,usable_receipts=0,expected_jobs=12 if condition=='backlog-six' else 6)
        for i,(condition,mode) in enumerate((c,m)for c in ('short-mixed','backlog-six','published-pipeline')for m in ('individual','group'))]


def development_manifest(manifest,record):
    if (manifest.get('profile_source') not in {SOURCE,'ml_predictor'} or any(str(k).startswith('development')for k in manifest)
            or manifest.get('condition',{}).get('name')!=record['condition']):
        raise ValueError('unchanged sealed measured condition required')
    output=deepcopy(manifest);output.update(development=True,
        development_seed_offset=SEED_OFFSET,development_scope='diagnostic only; not held-out; no automatic retuning')
    # development_seed_offset/scope are annotations for this runner, not physical
    # controls; episode.validate_manifest permits only the actual narrow flags.
    if record.get('kind')!='fifo_alpha'or record.get('arm')!='v5_1':raise ValueError('fixed V5.1 smoke design required')
    for spec in output['jobs']:
        for key in ('seed','train_pool_seed','validation_pool_seed'):spec[key]+=SEED_OFFSET
        spec['template_key']=workloads.template_key(spec)
        spec['spec_sha256']=workloads.spec_digest(spec)
        spec['expected_runtime_signature']=workloads.expected_signature(spec)
    if [s['template_key']for s in output['jobs']]!=[s['template_key']for s in manifest['jobs']]:
        raise ValueError('development seed shift changed calibrated configuration identity')
    validate_manifest(output,arm=record['arm'],mode=record['mode'])
    return output


def sealed_inputs(manifests,settings,calibration,source_commit,neural_calibration=None):
    from .campaign import sealed_bundle
    bundle=sealed_bundle(manifests,settings,calibration,neural_calibration)
    if bundle['plan']['source_commit']!=source_commit:raise ValueError('exact source commit required')
    inputs={name:bundle['manifests'][name]for name in ('short-mixed','backlog-six','published-pipeline')}
    if inputs['published-pipeline']['manifest']['profile_source']!='ml_predictor':raise ValueError('published smoke requires actual neural calibration')
    return dict(plan=bundle['plan'],source_commit=source_commit,settings=bundle['settings'],calibration=bundle['calibration'],table=bundle['table'],inputs=inputs,snapshots=bundle['snapshots'],hashes=bundle['hashes'])


def verify_inputs(bundle):
    for path,expected in bundle['hashes'].items():
        if file_hash(path)!=expected:raise ValueError('frozen development input changed: '+path)
    for condition,item in bundle['inputs'].items():
        if source_snapshot(item['manifest'],item['path'],source_commit=bundle['source_commit'])!=bundle['snapshots'][condition]:
            raise ValueError('actual source bytes changed during development')


def audit_episode(output,record,manifest,snapshot,table):
    """Development eligibility only. Reuse strict receipts, not held-out audit."""
    output=Path(output);errors=[];usable=0;result={};jobs={};profiles={};latencies={}
    try:
        validate_manifest(manifest,arm=record['arm'],mode=record['mode'])
        result=load(output/'result.json');receipts=result['receipts'];audits=load(output/'validity-audits.json')
        source=manifest['profile_source'];measured=source==SOURCE
        artifact=table['artifact_sha256']if measured else table['calibration_sha256']
        if (manifest.get('development')is not True or result.get('development_only')is not True
                or result.get('heldout_comparison_eligible')is not False
                or result.get('complete')is not True or result.get('predictor_comparison_eligible')is not True
                or result.get('arm')!=record['arm']or result.get('mode')!=record['mode']
                or result.get('profile_source')!=source or result.get('calibration_sha256')!=artifact
                or result.get('prediction_errors')or result.get('source_and_budget_audits')!=audits
                or len(receipts)!=len(manifest['jobs'])):
            raise ValueError('complete valid development-only result and all receipts required')
        alpha=load(output/'alpha-decisions.json')
        if alpha.get('events')!=result.get('alpha_events')or alpha.get('ordered_launch_events')!=result.get('ordered_launch_events'):raise ValueError('durable alpha/launch evidence mismatch')
        launches=alpha.get('ordered_launch_events',[])
        ids=[row.get('job_id')for row in launches]
        if len(ids)!=len(receipts)or len(set(ids))!=len(ids)or ids!=sorted(receipts,key=lambda j:receipts[j]['accepted_monotonic']):raise ValueError('V5.1 first dispatch is not exact accepted FIFO order')
        alpha_policy=audit_fifo_dispatch(output,result)
        expected_flags=arm_flags(record['arm'])
        if result.get('v5_flags')!=expected_flags:raise ValueError('development physical flags differ from fixed design')
        for path in ('source-pins-before.json','source-pins-after.json'):
            if load(output/path)!=snapshot:raise ValueError('development source/engine pins changed')
        for spec in manifest['jobs']:
            work=output/'jobs'/spec['name'];job=load(work/'job-final.json');jid=job['job_id'];receipt=receipts[jid]
            if jid in jobs or job.get('baseline_model_id')!=spec['name']:raise ValueError('duplicate/mismatched immutable job')
            _verify_receipt(spec,{'receipt':receipt,'final_audit':audits[spec['name']]},work)
            profile=result['prepared_numeric_profiles'][jid][str(spec['batch'])]
            from hashlib import sha256
            expected_artifact=artifact if measured else sha256((table['predictor_artifact_sha256']+artifact).encode()).hexdigest()
            if (result['prediction_sources'].get(jid)!=source or profile.get('artifact_sha256')!=expected_artifact
                    or profile.get('hardware')!='A10' or profile.get('batch_size')!=spec['batch']
                    or profile.get('job_signature')!=(job.get('packing',{}).get('signature')or jid)):
                raise ValueError('actual development profile identity mismatch')
            for field in ('step_cost','epoch_cost','avg_vram_mib','peak_vram_mib'):
                if not finite(profile.get(field))or profile[field]<=0:raise ValueError('nonfinite/nonpositive development profile '+field)
            if profile['peak_vram_mib']<profile['avg_vram_mib']:raise ValueError('profile memory order invalid')
            if measured:
                entry=table['entries'][spec['template_key']]
                expected={'step_cost':1000./entry['updates_per_second'],'epoch_cost':1000.*spec['steps_per_epoch']/entry['updates_per_second'],
                    'avg_vram_mib':entry['mean_allocated_mib'],'peak_vram_mib':entry['admission_reservation_mib']}
                if any(not math.isclose(profile[k],v,rel_tol=1e-9,abs_tol=1e-7)for k,v in expected.items()):raise ValueError('measured development profiles differ from table')
            elif profile.get('calibration')!='frozen_dedicated_a10_calibration_v1'or profile.get('calibration_sha256')!=artifact:
                raise ValueError('wrong fresh same-host neural calibration identity')
            if result['fixed_reference_memory_profiles'].get(spec['name'])!=profile:
                raise ValueError('FIFO memory guard used a different prepared profile')
            jobs[jid]=(spec,receipt,job);profiles[spec['name']]=profile_without_signature(result['prepared_numeric_profiles'][jid]);usable+=1
        request_groups={}
        for jid,(_,receipt,job)in jobs.items():
            raw=job['metadata']['feedback_request'];key=raw['request_id'];members=tuple(sorted(raw['required_job_ids']))
            identity=(members,raw['submitted_at'],raw.get('membership_finalized'),raw.get('completion_policy'))
            if key in request_groups and request_groups[key]!=identity:raise ValueError('inconsistent request identity')
            request_groups[key]=identity
        expected_n=len(jobs)if record['mode']=='individual'else 2
        if len(request_groups)!=expected_n:raise ValueError('complete request denominator missing')
        covered=[]
        for key,(members,submitted,sealed,mode)in request_groups.items():
            if sealed is not True or mode!=record['mode']or len(members)!=(1 if mode=='individual'else len(jobs)//2):
                raise ValueError('invalid request membership')
            covered.extend(members)
            value=max(receipts[j]['feedback_ready_monotonic']for j in members)-min(receipts[j]['accepted_monotonic']for j in members)
            if (not math.isclose(value,result['request_latencies_seconds'][key],abs_tol=1e-5)
                    or abs(stamp(submitted)-min(receipts[j]['accepted_unix']for j in members))>1e-5):
                raise ValueError('request clock differs from actual receipts')
            latencies[key]=value
        if len(covered)!=len(set(covered))or set(covered)!=set(jobs):raise ValueError('request jobs omitted/double counted')
        exits=load(output/'exit-observations.json')['completed_job_harness_observations']
        if exits!=result['harness_exit_observations']or set(exits)!=set(receipts):raise ValueError('observed exit coverage differs')
        for jid,observation in exits.items():
            if observation.get('returncode')!=0 or not receipts[jid]['accepted_monotonic']<=observation['observed_monotonic']<=receipts[jid]['feedback_ready_monotonic']:
                raise ValueError('successful observed process exit before usable feedback missing')
        check_causal_preparation(output,result,jobs)
    except (ValueError,KeyError,TypeError,OSError)as exc:errors.append(str(exc))
    return {'valid':not errors and usable==len(manifest['jobs']),'development_only':True,
        'heldout_comparison_eligible':False,'usable_receipts':usable,'errors':errors,
        'alpha_policy':locals().get('alpha_policy'),'request_latencies_seconds':latencies,'profiles_by_name':profiles,
        'episode_duration_seconds':result.get('episode_duration_seconds')}


def coverage(records):
    expected=planned_records()
    if len(records)!=6 or [r['output']for r in records]!=[r['output']for r in expected]:
        raise ValueError('every fixed development episode must remain in denominator')
    if any(r['expected_jobs']!=e['expected_jobs']for r,e in zip(records,expected)):
        raise ValueError('development job denominator changed')
    counts=Counter(r['status']for r in records)
    return {'planned_episodes':6,'planned_jobs':sum(r['expected_jobs']for r in records),
        'completed_episodes':counts['completed'],'failed_episodes':counts['failed'],
        'running_episodes':counts['running'],'unattempted_episodes':counts['unattempted'],
        'usable_receipts':sum(r['usable_receipts']for r in records if r['status']=='completed'),'observed_audited_receipts':sum(r['usable_receipts']for r in records),'failed_observed_receipts':sum(r['usable_receipts']for r in records if r['status']=='failed'),
        'unattempted_jobs':sum(r['expected_jobs']for r in records if r['status']=='unattempted')}


def episode_command(bundle,record,item,output):
    return [sys.executable,'-m','localml_scheduler.benchmarks.a10_v5_1_ablation.episode',
        '--manifest',str(item['path']),'--settings',str(bundle['settings']),'--calibration',str(bundle['inputs'][record['condition']]['calibration']),
        '--output',str(output/record['output']),'--source-commit',bundle['source_commit'],
        '--arm',record['arm'],'--mode',record['mode'],'--max-seconds',str(bundle['plan']['episode_max_seconds']),
        '--deadline-utc',bundle['plan']['deadline_utc']]


def execute(bundle,output,max_seconds):
    if not finite(max_seconds)or not 0<max_seconds<=3600:raise ValueError('finite development bound within3600seconds required')
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/'.a10-v5-development.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise FileExistsError('never overwrite or resume development output')
        verify_inputs(bundle);output.mkdir(exist_ok=False);records=planned_records();started=time.monotonic();active=None
        manifests={};snapshots={}
        inputs=output/'manifests';inputs.mkdir()
        for record in records:
            manifest=development_manifest(bundle['inputs'][record['condition']]['manifest'],record)
            path=inputs/(record['output']+'.json');atomic_json(path,manifest)
            manifests[record['output']]={'path':path,'manifest':manifest,'sha256':file_hash(path)}
            snapshots[record['output']]=source_snapshot(manifest,path,source_commit=bundle['source_commit'])
        atomic_json(output/'development-protocol-before.json',{'development_only':True,
            'source_commit':bundle['source_commit'],'input_sha256':bundle['hashes'],'order':records,
            'max_seconds':max_seconds,'episode_max_seconds':bundle['plan']['episode_max_seconds'],
            'seed_offset':SEED_OFFSET,'coverage':coverage(records),'automatic_heldout_launch':False,
            'automatic_retuning':False,'manifest_sha256':{k:v['sha256']for k,v in manifests.items()}})
        def save(status,error=None):
            atomic_json(output/'development-results.json',records);atomic_json(output/'coverage.json',coverage(records))
            atomic_json(output/'development-summary.json',{'activation_is_descriptive':True,'no_forced_acceptance':True,'coverage':coverage(records)})
            atomic_json(output/'status.json',{'status':status,'error':error,'elapsed_seconds':time.monotonic()-started,
                'development_only':True,'coverage':coverage(records)})
        def deadline():check_deadline(bundle['plan'],time.monotonic()-started,max_seconds)
        def interrupted(number,frame):raise InterruptedError('development signal; cleaning only owned episode child')
        previous={s:signal.getsignal(s)for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
        for sig in previous:signal.signal(sig,interrupted)
        save('running')
        try:
            for record in records:
                deadline();verify_inputs(bundle);active=record;item=manifests[record['output']]
                if file_hash(item['path'])!=item['sha256']:raise ValueError('frozen development manifest changed')
                child=None;owned={};record['status']='running';save('running')
                command=episode_command(bundle,record,item,output)
                record['command']=command
                try:
                    with (output/(record['output']+'.log')).open('x')as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            launched=time.monotonic()
                            atomic_json(output/'active-child.json',{'pid':child.pid,'command':command,'owned':True,'status':'running'})
                            while child.poll()is None:
                                owned.update(owned_descendants(child.pid))
                                deadline()
                                if time.monotonic()-launched>=bundle['plan']['episode_max_seconds']:raise TimeoutError('development episode bound reached')
                                time.sleep(.25)
                        finally:cleanup_owned(child,owned)
                    record['exit_code']=child.returncode
                    audit=audit_episode(output/record['output'],record,item['manifest'],snapshots[record['output']],bundle['inputs'][record['condition']]['table'])
                    record.update(audit=audit,usable_receipts=audit['usable_receipts'])
                    if child.returncode!=0 or not audit['valid']:raise RuntimeError('development episode failed/invalid: '+str(audit['errors']))
                    record['status']='completed';active=None;save('running')
                    atomic_json(output/'active-child.json',{'pid':child.pid,'owned':True,'status':'exited','exit_code':child.returncode})
                except BaseException as exc:
                    record.update(status='failed',error=repr(exc),exit_code=child.returncode if child is not None else None)
                    atomic_json(output/'active-child.json',{'pid':child.pid if child is not None else None,'owned':child is not None,
                        'status':'failed','exit_code':record['exit_code'],'error':repr(exc)})
                    raise
            deadline();verify_inputs(bundle)
            for item in manifests.values():
                if file_hash(item['path'])!=item['sha256']:raise ValueError('frozen development manifest changed after execution')
            save('complete')
            atomic_json(output/'development-complete.json',{'complete':True,'development_only':True,'coverage':coverage(records),
                'no_automatic_heldout_launch':True,'no_automatic_retuning':True})
            return records
        except BaseException as exc:
            if active is not None and active['status']not in ('completed','failed'):active.update(status='failed',error=repr(exc))
            save('failed',repr(exc));raise
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)


def replay(bundle,output):
    output=Path(output);records=load(output/'development-results.json');counts=coverage(records)
    if counts['completed_episodes']!=6 or counts['usable_receipts']!=48:raise ValueError('all six development episodes/48 receipts required')
    proof=load(output/'development-protocol-before.json');verify_inputs(bundle)
    if proof.get('source_commit')!=bundle['source_commit']or proof.get('seed_offset')!=SEED_OFFSET or proof.get('input_sha256')!=bundle['hashes']:raise ValueError('development freeze/input identity mismatch')
    audits={};hashes={}
    for record in records:
        if record['status']!='completed'or record['exit_code']!=0:raise ValueError('successful episode exit required')
        path=output/'manifests'/(record['output']+'.json');manifest=load(path)
        if file_hash(path)!=proof['manifest_sha256'][record['output']]or manifest!=development_manifest(bundle['inputs'][record['condition']]['manifest'],record):raise ValueError('development manifest differs from fixed seed-shift recipe')
        snap=source_snapshot(manifest,path,source_commit=bundle['source_commit'])
        audit=audit_episode(output/record['output'],record,manifest,snap,bundle['inputs'][record['condition']]['table'])
        if not audit['valid']:raise ValueError('saved development audit failed: '+str(audit['errors']))
        audits[record['output']]=audit
    for path in output.rglob('*'):
        if path.is_file()and path.suffix in {'.json','.jsonl','.log'}:hashes[str(path.relative_to(output))]=file_hash(path)
    return {'schema':'a10_v51_development_replay_v1','audit_passed':True,'source_commit':bundle['source_commit'],'development_episodes':6,'usable_receipts':48,'coverage':counts,'audits':audits,'development_artifact_sha256':hashes,'activation_is_descriptive':True,'heldout_comparison_eligible':False,'prerequisite_scope':'development only; exact-host CPU/isolation/calibration replay and package/GPU binding supplied separately by parent'}


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('manifests','settings','calibration','output','source-commit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--neural-calibration',required=True);parser.add_argument('--max-seconds',type=float,default=3600);parser.add_argument('--audit-only',action='store_true');parser.add_argument('--audit-output');args=parser.parse_args(argv)
    if args.audit_only:
        result=replay(sealed_inputs(args.manifests,args.settings,args.calibration,args.source_commit,args.neural_calibration),args.output)
        if not args.audit_output:raise ValueError('new audit output path required')
        target=Path(args.audit_output)
        if target.exists():raise FileExistsError('never overwrite replay')
        atomic_json(target,result);return result
    return execute(sealed_inputs(args.manifests,args.settings,args.calibration,args.source_commit,args.neural_calibration),args.output,args.max_seconds)


if __name__=='__main__':main()
