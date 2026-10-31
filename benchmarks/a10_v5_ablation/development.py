"""Fixed, finite development diagnostics; never launches or retunes held-out work.

Twelve episodes: caps1..4 x both modes on backlog-six, shadow x both modes,
combined smoke x both modes on short-mixed. Every workload seed is shifted by
9000000; configuration/update budgets stay fixed. GPU work occurs only when a
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

from ...execution.process_utils import terminate_process_tree
from ..a10_feedback_latency.run import atomic_json,check_deadline,file_hash
from ..a10_resource_mechanisms import workloads
from ..a10_resource_mechanisms.profiles import SOURCE
from . import protocol
from .calibrate_predictor import _verify_receipt
from .campaign import load,finite,stamp,latency_distribution,check_causal_preparation,profile_without_signature
from .episode import source_snapshot,validate_manifest,arm_flags

SEED_OFFSET=9000000


def planned_records():
    rows=[]
    for cap in (1,2,3,4):
        for mode in ('individual','group'):
            rows.append(dict(kind='fixed_cap',condition='backlog-six',arm='fixed_three',mode=mode,cap=cap))
    for kind,arm in [('shadow','forecast_only'),('smoke','combined')]:
        for mode in ('individual','group'):
            rows.append(dict(kind=kind,condition='short-mixed',arm=arm,mode=mode,cap=None))
    return [{**row,'output':f"{i:02d}-{row['kind']}-{row['mode']}"+(f"-cap{row['cap']}"if row['cap']else''),
        'status':'unattempted','exit_code':None,'usable_receipts':0,'expected_jobs':12 if row['condition']=='backlog-six'else 6}
        for i,row in enumerate(rows)]


def development_manifest(manifest,record):
    if (manifest.get('profile_source')!=SOURCE or any(str(k).startswith('development')for k in manifest)
            or manifest.get('condition',{}).get('name')!=record['condition']):
        raise ValueError('unchanged sealed measured condition required')
    output=deepcopy(manifest);output.update(development=True,
        development_seed_offset=SEED_OFFSET,development_scope='diagnostic only; not held-out; no automatic retuning')
    # development_seed_offset/scope are annotations for this runner, not physical
    # controls; episode.validate_manifest permits only the actual narrow flags.
    if record['kind']=='fixed_cap':output['development_fixed_cap']=record['cap']
    if record['kind']=='shadow':output['development_shadow']=True
    for spec in output['jobs']:
        for key in ('seed','train_pool_seed','validation_pool_seed'):spec[key]+=SEED_OFFSET
        spec['template_key']=workloads.template_key(spec)
        spec['spec_sha256']=workloads.spec_digest(spec)
        spec['expected_runtime_signature']=workloads.expected_signature(spec)
    if [s['template_key']for s in output['jobs']]!=[s['template_key']for s in manifest['jobs']]:
        raise ValueError('development seed shift changed calibrated configuration identity')
    validate_manifest(output,arm=record['arm'],mode=record['mode'])
    return output


def sealed_inputs(manifests,settings,calibration,source_commit):
    manifests,settings,calibration=map(lambda p:Path(p).resolve(),(manifests,settings,calibration))
    plan=load(manifests/'plan.json');declared=load(manifests/'protocol-before.json');protocol.validate_plan(plan)
    table=load(calibration);protocol.validate_table(table);check_deadline(plan)
    if (plan.get('status')!='frozen'or plan['source_commit']!=source_commit
            or declared.get('frozen_before_evaluation')is not True or declared.get('plan')!=plan
            or declared.get('plan_sha256')!=file_hash(manifests/'plan.json')
            or declared.get('settings_sha256')!=file_hash(settings)
            or declared.get('calibration_sha256')!=file_hash(calibration)
            or declared.get('calibration_artifact_sha256')!=table['artifact_sha256']):
        raise ValueError('frozen plan/settings/source/common calibration identity mismatch')
    paths=[manifests/'plan.json',manifests/'protocol-before.json',settings,calibration]
    selected={};snapshots={}
    for condition,count in [('short-mixed',6),('backlog-six',12)]:
        path=manifests/(condition+'.json');manifest=load(path);paths.append(path)
        if (declared.get('manifests',{}).get(condition)!=file_hash(path)
                or manifest.get('profile_source')!=SOURCE or manifest.get('study_plan_sha256')!=workloads.digest(plan)
                or manifest.get('calibration_sha256')!=table['artifact_sha256']or len(manifest['jobs'])!=count
                or manifest.get('experiment_source_commit')!=source_commit
                or manifest.get('condition',{}).get('name')!=condition):
            raise ValueError('exact sealed measured short/backlog condition required')
        current=source_snapshot(manifest,path,source_commit=source_commit)
        if (current!=declared['source_pins'][condition]
                or current['scheduler_python_sources']!=table['source_identity']['scheduler_python_sources']):
            raise ValueError('development source/engine differs from sealed common calibration')
        snapshots[condition]=current;selected[condition]={'path':path,'manifest':manifest}
    return dict(plan=plan,source_commit=source_commit,settings=settings,calibration=calibration,
        table=table,inputs=selected,snapshots=snapshots,hashes={str(p):file_hash(p)for p in paths})


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
        if (manifest.get('development')is not True or result.get('development_only')is not True
                or result.get('heldout_comparison_eligible')is not False
                or result.get('complete')is not True or result.get('predictor_comparison_eligible')is not True
                or result.get('arm')!=record['arm']or result.get('mode')!=record['mode']
                or result.get('profile_source')!=SOURCE or result.get('calibration_sha256')!=table['artifact_sha256']
                or result.get('prediction_errors')or result.get('source_and_budget_audits')!=audits
                or len(receipts)!=len(manifest['jobs'])):
            raise ValueError('complete valid development-only result and all receipts required')
        expected_flags=arm_flags(record['arm'],development_shadow=record['kind']=='shadow')
        if result.get('v5_flags')!=expected_flags:raise ValueError('development physical flags differ from fixed design')
        if record['kind']=='fixed_cap'and result.get('fixed_concurrency_cap')!=record['cap']:
            raise ValueError('physical fixed cap differs from declared development cap')
        if record['kind']=='shadow'and result.get('development_shadow')is not True:
            raise ValueError('actual shadow mode not enabled')
        for path in ('source-pins-before.json','source-pins-after.json'):
            if load(output/path)!=snapshot:raise ValueError('development source/engine pins changed')
        for spec in manifest['jobs']:
            work=output/'jobs'/spec['name'];job=load(work/'job-final.json');jid=job['job_id'];receipt=receipts[jid]
            if jid in jobs or job.get('baseline_model_id')!=spec['name']:raise ValueError('duplicate/mismatched immutable job')
            _verify_receipt(spec,{'receipt':receipt,'final_audit':audits[spec['name']]},work)
            profile=result['prepared_numeric_profiles'][jid][str(spec['batch'])]
            entry=table['entries'][spec['template_key']]
            if (result['prediction_sources'].get(jid)!=SOURCE or profile.get('artifact_sha256')!=table['artifact_sha256']
                    or profile.get('batch_size')!=spec['batch']or profile.get('job_signature')!=(job.get('packing',{}).get('signature')or jid)):
                raise ValueError('actual common development profile identity mismatch')
            expected={'step_cost':1000./entry['updates_per_second'],'epoch_cost':1000.*spec['steps_per_epoch']/entry['updates_per_second'],
                'avg_vram_mib':entry['mean_allocated_mib'],'peak_vram_mib':entry['admission_reservation_mib']}
            if any(not finite(profile.get(k))or not math.isclose(profile[k],v,rel_tol=1e-9,abs_tol=1e-7)for k,v in expected.items()):
                raise ValueError('actual development profiles differ from shared table')
            if record['kind']=='fixed_cap'and result['fixed_reference_memory_profiles'].get(spec['name'])!=profile:
                raise ValueError('fixed cap memory check used different profile')
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
        'request_latencies_seconds':latencies,'profiles_by_name':profiles,
        'episode_duration_seconds':result.get('episode_duration_seconds')}


def coverage(records):
    expected=planned_records()
    if len(records)!=12 or [r['output']for r in records]!=[r['output']for r in expected]:
        raise ValueError('every fixed development episode must remain in denominator')
    if any(r['expected_jobs']!=e['expected_jobs']for r,e in zip(records,expected)):
        raise ValueError('development job denominator changed')
    counts=Counter(r['status']for r in records)
    return {'planned_episodes':12,'planned_jobs':sum(r['expected_jobs']for r in records),
        'completed_episodes':counts['completed'],'failed_episodes':counts['failed'],
        'running_episodes':counts['running'],'unattempted_episodes':counts['unattempted'],
        'usable_receipts':sum(r['usable_receipts']for r in records),
        'unattempted_jobs':sum(r['expected_jobs']for r in records if r['status']=='unattempted')}


def fixed_cap_summary(records):
    rows=[];best={}
    for mode in ('individual','group'):
        selected=[r for r in records if r['kind']=='fixed_cap'and r['mode']==mode]
        complete=len(selected)==4 and all(r['status']=='completed'for r in selected)
        for r in selected:
            stats=latency_distribution(list(r['audit']['request_latencies_seconds'].values()))if r['status']=='completed'else None
            rows.append({'mode':mode,'cap':r['cap'],'status':r['status'],'request_distribution':stats})
        scores=[(row['request_distribution']['mean_seconds'],row['cap'])for row in rows if row['mode']==mode and row['request_distribution']]
        best[mode]=[cap for value,cap in scores if value==min(v for v,_ in scores)]if complete else None
    return {'development_only':True,'caps_by_mode':rows,'best_caps_by_mode':best,
        'automatic_retuning':False,'note':'Descriptive development selection only; fixed held-out arm remains fixed_three. Parent review required.'}


def shadow_samples(output):
    """Receipt correlations for the observed initial action, never counterfactual truth."""
    output=Path(output);result=load(output/'result.json');events=result.get('v5_events',[])[:]
    for path in output.glob('jobs/*/events.json'):events.extend(load(path))
    unique={json.dumps(e,sort_keys=True):e for e in events};events=list(unique.values())
    decided={e.get('payload',{}).get('trial_id'):e['payload'].get('decision')for e in events if e.get('event_type')=='cold_start_trial_decided'}
    rows=[];receipts=result.get('receipts',{})
    for event in events:
        if event.get('event_type')!='v5_continuation_forecast':continue
        p=event.get('payload',{});trial=p.get('trial_id');decision=decided.get(trial)
        action='admit'if decision=='accepted'else 'defer'if decision in ('rejected','inconclusive')else None
        snapshot=p.get('context',{}).get('snapshot_at')
        if snapshot is None:
            try:snapshot=stamp(event['created_at'])
            except (KeyError,ValueError,TypeError):snapshot=None
        for half,h in enumerate(p.get('halves',[])):
            for alternative in ('admit','defer'):
                predictions=h.get(alternative,{}).get('job_completions',{})
                future=[j for j,r in receipts.items()if finite(snapshot)and r.get('accepted_unix',0)>snapshot and j not in predictions]
                for jid,predicted in predictions.items():
                    observed=receipts.get(jid,{}).get('feedback_ready_unix');executed=alternative==action
                    usable=executed and finite(predicted)and finite(observed)and finite(snapshot)and observed>=snapshot and predicted>=snapshot
                    rows.append({'trial_id':trial,'half':half,'snapshot_unix':snapshot,'job_id':jid,
                        'alternative':alternative,'predicted_completion_unix':predicted,
                        'observed_initial_action':action,'initial_action_matches':executed,
                        'observed_receipt_completion_unix':observed if executed else None,
                        'executed_initial_action_error_seconds':observed-predicted if usable else None,
                        'future_cohort_changed':bool(future),'later_arrival_ids':future,
                        'observed_result_already_ready_at_snapshot':finite(observed)and finite(snapshot)and observed<snapshot,
                        'exact_future_continuation_verified':False,'unchosen_counterfactual_observed':False,
                        'note':'Observed initial action only; later online replanning and arrivals may differ from frozen reference. Unchosen alternative has no observed outcome.'})
    return rows


def execute(bundle,output,max_seconds):
    if not finite(max_seconds)or not 0<max_seconds<=28800:raise ValueError('finite development bound within eight hours required')
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
            atomic_json(output/'fixed-cap-summary.json',fixed_cap_summary(records))
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
                child=None;record['status']='running';save('running')
                command=[sys.executable,'-m','localml_scheduler.benchmarks.a10_v5_ablation.episode',
                    '--manifest',str(item['path']),'--settings',str(bundle['settings']),'--calibration',str(bundle['calibration']),
                    '--output',str(output/record['output']),'--source-commit',bundle['source_commit'],
                    '--arm',record['arm'],'--mode',record['mode'],'--max-seconds',str(bundle['plan']['episode_max_seconds']),
                    '--deadline-utc',bundle['plan']['deadline_utc']]
                record['command']=command
                try:
                    with (output/(record['output']+'.log')).open('x')as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            launched=time.monotonic()
                            atomic_json(output/'active-child.json',{'pid':child.pid,'command':command,'owned':True,'status':'running'})
                            while child.poll()is None:
                                deadline()
                                if time.monotonic()-launched>=bundle['plan']['episode_max_seconds']:raise TimeoutError('development episode bound reached')
                                time.sleep(.25)
                        finally:terminate_process_tree(child,timeout=15.)
                    record['exit_code']=child.returncode
                    audit=audit_episode(output/record['output'],record,item['manifest'],snapshots[record['output']],bundle['table'])
                    record.update(audit=audit,usable_receipts=audit['usable_receipts'])
                    if child.returncode!=0 or not audit['valid']:raise RuntimeError('development episode failed/invalid: '+str(audit['errors']))
                    if record['kind']=='shadow':atomic_json(output/(record['output']+'-shadow-samples.json'),shadow_samples(output/record['output']))
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


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('manifests','settings','calibration','output','source-commit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args(argv)
    return execute(sealed_inputs(args.manifests,args.settings,args.calibration,args.source_commit),args.output,args.max_seconds)


if __name__=='__main__':main()
