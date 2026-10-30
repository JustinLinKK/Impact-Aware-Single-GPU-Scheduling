"""Finite serial execution of a sealed resource-mechanism study; no cloud API.

The child is the existing run.py command and owns the shared GPU flock. This
controller never acquires a GPU or repairs, reseals, resumes, or overwrites work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time

from ...execution.process_utils import terminate_process_tree
from ..a10_feedback_latency.run import (
    RESOURCE_ARMS, atomic_json, check_deadline, file_hash, source_snapshot, with_hard_deadline, digest as runtime_digest,
)
from .profiles import SOURCE, SCHEMA
from .study import episode_order, validate_plan
from .workloads import digest


def load(path): return json.loads(Path(path).read_text())
def finite(value): return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def stamp(value): return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def sealed_bundle(directory, settings, calibration):
    directory,settings,calibration=map(lambda p:Path(p).resolve(),(directory,settings,calibration))
    protocol_path=directory/'protocol-before.json';plan_path=directory/'plan.json'
    protocol=load(protocol_path);plan=load(plan_path);validate_plan(plan)
    if plan.get('status')!='frozen': raise ValueError('candidate plan cannot launch; sealed plan.status must be frozen')
    if protocol.get('frozen_before_evaluation') is not True or protocol.get('plan')!=plan or protocol.get('plan_sha256')!=file_hash(plan_path):
        raise ValueError('sealed plan bytes/protocol identity mismatch')
    expected_cohorts = 48 if plan['iteration'] == 3 else 64
    if len(plan['conditions'])!=4:
        raise ValueError('exactly four predeclared conditions required')
    if plan['iteration'] == 2 and (set(plan['arms'])!=set(RESOURCE_ARMS) or len(plan['arms'])!=4):
        raise ValueError('iteration two requires exactly four resource arms')
    for key in ('episode_max_seconds','campaign_max_seconds','job_timeout_seconds'):
        if type(plan.get(key))is not int or not 0<plan[key]<=14400: raise ValueError('finite positive plan bounds of at most four hours required')
    if plan['episode_max_seconds']>plan['campaign_max_seconds']: raise ValueError('episode bound exceeds campaign bound')
    order=episode_order(plan)
    if (len(order)!=expected_cohorts or protocol.get('order')!=order
            or protocol.get('expected_cohorts')!=expected_cohorts
            or protocol.get('expected_jobs')!=6*expected_cohorts):
        raise ValueError('sealed balanced order has wrong full campaign denominator')
    names={c['name'] for c in plan['conditions']}
    if set(protocol.get('manifests',{}))!=names: raise ValueError('condition manifest coverage mismatch')
    if {p.name for p in directory.glob('*.json')}!={name+'.json' for name in names}|{'plan.json','protocol-before.json'}:
        raise ValueError('sealed bundle contains missing/extra JSON files')
    table=load(calibration)
    if (file_hash(calibration)!=protocol.get('calibration_sha256') or table.get('source')!=SOURCE or table.get('schema')!=SCHEMA
            or table.get('artifact_sha256')!=digest({k:v for k,v in table.items() if k!='artifact_sha256'})):
        raise ValueError('dedicated calibration file/table identity mismatch')
    manifests={};snapshots={}
    hashes={str(p):file_hash(p) for p in (protocol_path,plan_path,settings,calibration)}
    for name in names:
        path=directory/(name+'.json');manifest=load(path)
        if file_hash(path)!=protocol['manifests'][name]: raise ValueError('sealed condition manifest hash mismatch')
        if (manifest.get('study_plan_sha256')!=digest(plan) or manifest.get('profile_source')!=SOURCE
                or manifest.get('calibration_sha256')!=table['artifact_sha256'] or manifest.get('condition',{}).get('name')!=name
                or manifest.get('deadline_utc')!=plan['deadline_utc']): raise ValueError('condition does not bind frozen plan/table/deadline')
        specs=manifest['jobs']
        if len(specs)!=6 or len({s['name'] for s in specs})!=6 or {(s['role'],s['chain']) for s in specs}!={(role,i) for role in ('parent','child') for i in range(3)}:
            raise ValueError('six unique jobs in three declared chains required')
        for spec in specs:
            if (spec.get('epochs')!=2 or type(spec.get('steps_per_epoch'))is not int or spec['steps_per_epoch']<=0
                    or 2*spec['steps_per_epoch']>plan['maximum_updates'] or spec.get('template_key') not in table['entries']):
                raise ValueError('invalid per-job two-epoch calibrated budget/template')
        effective=with_hard_deadline(manifest,plan['deadline_utc'])
        snapshots[name]=source_snapshot(effective,path)
        manifests[name]={'path':path,'manifest':effective}
        hashes[str(path)]=file_hash(path)
    return {'plan':plan,'protocol':protocol,'manifests':manifests,'snapshots':snapshots,'hashes':hashes,
        'settings':settings,'calibration':calibration,'table':table,'order':order,'directory':directory}


def verify_bundle(bundle):
    validate_plan(bundle['plan'])
    for path,expected in bundle['hashes'].items():
        if file_hash(path)!=expected: raise ValueError('frozen plan/protocol/manifest/settings/calibration changed: '+path)
    for name,item in bundle['manifests'].items():
        if source_snapshot(item['manifest'],item['path'])!=bundle['snapshots'][name]: raise ValueError('frozen source/engine identity changed: '+name)


def profile_without_signature(value):
    if isinstance(value,dict):return {k:profile_without_signature(v) for k,v in value.items() if k!='job_signature'}
    if isinstance(value,list):return [profile_without_signature(v) for v in value]
    return value


def receipt_artifact_path(recorded,output,expected,remote_output=None):
    """Resolve one exact archived receipt path without rewriting its JSON."""
    output=Path(output);expected=Path(expected)
    if not isinstance(recorded,str) or not recorded or "\\" in recorded or "\0" in recorded:raise ValueError('unsafe receipt artifact path')
    if remote_output is not None:
        prefix=str(remote_output)
        for text in (prefix,recorded):
            path=PurePosixPath(text)
            if (not path.is_absolute() or text.startswith('//') or "\\" in text or "\0" in text
                    or any(part in {'.','..'} for part in text.split('/'))):raise ValueError('unsafe remote receipt path or cohort prefix')
        base=PurePosixPath(prefix)
        if len(base.parts)<2:raise ValueError('exact remote cohort prefix required')
        relative=PurePosixPath(recorded).relative_to(base)
        if not relative.parts:raise ValueError('receipt path must name an artifact beneath the remote cohort')
        actual=output.joinpath(*relative.parts)
    else:
        actual=Path(recorded)
        if '..' in actual.parts:raise ValueError('unsafe receipt traversal')
    if actual.resolve()!=expected.resolve() or not expected.resolve().is_relative_to(output.resolve()):raise ValueError('receipt path does not match expected owned cohort artifact')
    # Reject symlinks even when they happen to resolve back inside the cohort.
    relative=expected.relative_to(output)
    current=output
    if current.is_symlink():raise ValueError('symlink cohort root')
    for part in relative.parts:
        current=current/part
        if current.is_symlink():raise ValueError('symlink artifact path')
    return expected


def audit_episode(output, episode, manifest, snapshot, table, *, remote_output=None):
    """Saved JSON proofs only; variable budgets come from each sealed spec."""
    output=Path(output)
    errors=[];usable=0;profiles={};jobs={};result={};requests={}
    try:
        result=load(output/'result.json')
        if result.get('complete')is not True or result.get('predictor_comparison_eligible')is not True:
            errors.append('incomplete or predictor-comparison-ineligible episode')
        if result.get('arm')!=episode['arm'] or result.get('mode')!=episode['mode']: errors.append('episode arm/mode identity mismatch')
        if result.get('profile_source')!=SOURCE or result.get('calibration_sha256')!=table['artifact_sha256'] or result.get('prediction_errors'):
            errors.append('wrong template source/calibration or prediction errors')
        if load(output/'source-pins-before.json')!=snapshot or load(output/'source-pins-after.json')!=snapshot: errors.append('source pins changed or differ from sealed source')
        receipts=result.get('receipts',{})
        if len(receipts)!=6: errors.append('six required receipts absent')
        saved_audits=load(output/'validity-audits.json')
        if saved_audits!=result.get('source_and_budget_audits'):errors.append('saved checkpoint audit differs from result')
        for spec in manifest['jobs']:
            name=spec['name']
            try:
                job=load(output/'jobs'/name/'job-final.json');job_id=job['job_id'];receipt=receipts[job_id];returned=receipt['returned_result']
                if job.get('baseline_model_id')!=name or job_id in jobs: raise ValueError('missing/duplicate immutable job identity')
                jobs[job_id]=(spec,receipt,job)
                if load(output/'jobs'/name/'receipt.json')!=receipt: raise ValueError('durable receipt differs from result')
                expected=2*spec['steps_per_epoch'];identity=spec['expected_runtime_signature'];metadata=job['metadata']
                if (metadata.get('resource_workload_spec')!=spec or metadata.get('resource_template_key')!=spec['template_key']
                        or metadata.get('pilot_frozen_workload_sha256')!=runtime_digest(spec)
                        or json.loads(job['config']['env']['A10_PILOT_SPEC_JSON'])!=spec
                        or job['config']['runner_kwargs']['batch_size']!=spec['batch']):raise ValueError('immutable full workload specification differs from sealed manifest')
                artifacts=(('result',output/'jobs'/name/'working/shared_training/result.json'),('execution',output/'jobs'/name/'execution.json'))
                for kind,path in artifacts:
                    if (path.is_symlink() or not path.resolve().is_relative_to(output.resolve())
                            or receipt_artifact_path(receipt[kind+'_path'],output,path,remote_output).resolve()!=path.resolve()
                            or path.stat().st_size!=receipt[kind+'_bytes'] or file_hash(path)!=receipt[kind+'_sha256']):raise ValueError('saved '+kind+' artifact path/bytes/SHA mismatch')
                if load(artifacts[0][1])!=returned or load(artifacts[1][1]).get('exc_type')is not None:raise ValueError('saved returned result/execution does not match successful receipt')
                if (receipt.get('worker_exit_code')!=0 or receipt.get('expected_updates')!=expected or receipt.get('expected_epochs')!=2
                        or receipt.get('expected_runtime_signature')!=identity or receipt.get('expected_workload_sha256')!=runtime_digest(spec)
                        or returned.get('signature')!=identity or metadata.get('runtime_completion_identity')!=identity): raise ValueError('successful receipt identity/budget mismatch')
                if (returned.get('global_step')!=expected or returned.get('epochs')!=2 or len(returned.get('history',[]))!=2
                        or metadata.get('runtime_global_step')!=expected or metadata.get('runtime_total_optimizer_updates')!=expected
                        or metadata.get('runtime_finalization_complete')is not True):raise ValueError('per-job successful updates/two complete epochs not proven')
                budget=saved_audits[name]
                if (budget.get('global_step')!=expected or budget.get('all_adamw_steps')!=expected or budget.get('completed_epochs')!=2
                        or not budget.get('optimizer_parameter_states',0)>0):raise ValueError('checkpoint optimizer update proof mismatch')
                if receipt.get('auc_maximize')is not True or not finite(receipt.get('selected_auc')) or not 0<=receipt['selected_auc']<=1 or not finite(receipt.get('selected_validation_bce')) or receipt['selected_validation_bce']<0:
                    raise ValueError('invalid returned AUC/BCE')
                for epoch in returned['history']:
                    counts=[epoch['validation_negative_count'],epoch['validation_positive_count']]
                    if any(not finite(n) or n<=0 or not float(n).is_integer() for n in counts) or sum(counts)!=spec['validation_size']:raise ValueError('heldout class/budget proof mismatch')
                accepted,ready=receipt['accepted_monotonic'],receipt['feedback_ready_monotonic']
                if not all(finite(v) for v in (accepted,ready,receipt['jct_seconds'])) or ready<accepted or not math.isclose(ready-accepted,receipt['jct_seconds'],abs_tol=1e-5):raise ValueError('invalid usable-feedback clock')
                if abs(stamp(job['submitted_at'])-receipt['accepted_unix'])>1e-5:raise ValueError('acceptance wall clock mismatch')
                profile=result['prepared_numeric_profiles'][job_id][str(spec['batch'])]
                if (result['prediction_sources'].get(job_id)!=SOURCE or profile.get('profile_source')!=SOURCE or profile.get('calibration')!=SCHEMA
                        or profile.get('artifact_sha256')!=table['artifact_sha256'] or profile.get('hardware')!='A10'
                        or profile.get('batch_size')!=spec['batch'] or profile.get('job_signature')!=(job.get('packing',{}).get('signature') or job_id)):raise ValueError('wrong actual prepared numeric profile identity')
                for field in ('step_cost','epoch_cost','avg_vram_mib','peak_vram_mib'):
                    if not finite(profile.get(field)) or profile[field]<=0:raise ValueError('invalid numeric profile '+field)
                if profile['peak_vram_mib']<profile['avg_vram_mib']:raise ValueError('profile memory order invalid')
                if episode['arm']=='resource_fixed_three' and result['fixed_reference_memory_profiles'].get(name)!=profile:raise ValueError('fixed-three checked profile differs from actual prepared profile')
                profiles[name]=profile_without_signature(result['prepared_numeric_profiles'][job_id])
                request=metadata['feedback_request'];key=request['request_id'];members=tuple(sorted(request['required_job_ids']))
                value=(members,request['submitted_at'],request.get('membership_finalized'),request.get('completion_policy'))
                if key in requests and requests[key]['identity']!=value:raise ValueError('request membership differs among members')
                requests.setdefault(key,{'identity':value,'members':set()})['members'].add(job_id)
                usable+=1
            except (KeyError,TypeError,ValueError,OSError) as exc:errors.append(name+': '+str(exc))
        latencies={}
        if len(requests)!=(6 if episode['mode']=='individual' else 2):errors.append('wrong request-unit denominator')
        for key,request in requests.items():
            members,submitted,finalized,mode=request['identity']
            if (finalized is not True or mode!=episode['mode'] or set(members)!=request['members']
                    or len(members)!=(1 if mode=='individual' else 3)):raise ValueError('invalid finalized request membership')
            accepted=min(receipts[j]['accepted_monotonic'] for j in members)
            latency=max(receipts[j]['feedback_ready_monotonic'] for j in members)-accepted
            if not math.isclose(latency,result['request_latencies_seconds'][key],abs_tol=1e-5):raise ValueError('all-required-feedback latency differs from recorded request')
            if abs(stamp(submitted)-min(receipts[j]['accepted_unix'] for j in members))>1e-5:raise ValueError('request submission differs from first accepted member')
            latencies[key]=latency
        # The receipt gate waits for a successful observed exit, never COMPLETED alone.
        exits=load(output/'exit-observations.json')['completed_job_harness_observations']
        if exits!=result.get('harness_exit_observations') or set(exits)!=set(receipts):raise ValueError('exit observation coverage/identity mismatch')
        for job_id,observation in exits.items():
            receipt=receipts[job_id];observed=observation['observed_monotonic']
            if observation.get('returncode')!=0 or not finite(observed) or not receipt['accepted_monotonic']<=observed<=receipt['feedback_ready_monotonic']:raise ValueError('actual successful exit before feedback not proven')
        check_causal_preparation(output,result,jobs)
    except (KeyError,TypeError,ValueError,OSError) as exc:errors.append('episode evidence: '+str(exc))
    return {'valid':not errors and usable==6,'complete':result.get('complete')is True,'predictor_comparison_eligible':result.get('predictor_comparison_eligible')is True,'usable_receipts':usable,'errors':errors,'profiles_by_name':profiles,
        'request_latencies_seconds':latencies if 'latencies' in locals() else {},
        'request_unit':'one_job' if episode['mode']=='individual' else 'all_three_required_feedback_bracket',
        'episode_duration_seconds':result.get('episode_duration_seconds'),'job_jct_seconds':[r.get('jct_seconds') for r in result.get('receipts',{}).values()],
        'policy_activation':result.get('policy_activation'),'admission_policy_comparison_eligible':result.get('admission_policy_comparison_eligible')}


def check_causal_preparation(output,result,jobs):
    origin=result['episode_started_monotonic']
    arrivals=result['arrivals']
    if not finite(origin) or len(arrivals)!=6 or len({a['name'] for a in arrivals})!=6:raise ValueError('six unique causal arrivals and finite episode origin required')
    arrivals={a['name']:a for a in arrivals}
    specs={spec['name']:(job_id,spec,receipt,job) for job_id,(spec,receipt,job) in jobs.items()}
    if set(arrivals)!=set(specs):raise ValueError('causal arrival membership differs from saved jobs')
    parents={spec['chain']:receipt for _,spec,receipt,_ in specs.values() if spec['role']=='parent'}
    for name,(job_id,spec,receipt,job) in specs.items():
        accepted=receipt['accepted_monotonic'];row=arrivals[name]
        if spec['role']=='parent':due=0. if result['mode']=='group' else 2.*spec['chain']
        else:due=(max(r['feedback_ready_monotonic'] for r in parents.values()) if result['mode']=='group' else parents[spec['chain']]['feedback_ready_monotonic'])-origin+2.
        if (not all(finite(v) for v in (row['due_seconds'],row['accepted_seconds'],due))
                or abs(row['due_seconds']-due)>1e-5 or abs(row['accepted_seconds']-(accepted-origin))>1e-5
                or row['accepted_seconds']+1e-5<due):raise ValueError(name+': causal release/acceptance clock mismatch')
    timeline=load(output/'preparation-timeline.json')
    if timeline.get('episode_started_monotonic')!=origin or timeline.get('asynchronous')is not True or timeline.get('worker_stopped')is not True or timeline.get('failures'):raise ValueError('clean asynchronous preparation with same episode origin required')
    expected=set()
    for job_id,(spec,receipt,job) in jobs.items():
        for batch,profile in result['prepared_numeric_profiles'][job_id].items():
            expected.add((job_id,int(batch)))
            events=[e for e in timeline['events'] if e.get('job_id')==job_id and e.get('batch_size')==int(batch)]
            by_phase={phase:[e for e in events if e.get('phase')==phase] for phase in ('enqueue','start','end')}
            if len(events)!=3 or any(len(rows)!=1 for rows in by_phase.values()):raise ValueError('exactly one preparation enqueue/start/end per actual profile required')
            enqueue,start,end=(by_phase[k][0] for k in ('enqueue','start','end'))
            keys=('job_id','job_signature','batch_size','artifact_sha256','hardware','hardware_key','prediction_contract_sha256')
            if any(any(e.get(k)!=enqueue.get(k) for k in keys) for e in (start,end)):raise ValueError('preparation event identity changed')
            if any(end.get(k)!=profile.get(k) for k in ('job_signature','batch_size','artifact_sha256','hardware')):raise ValueError('preparation identity differs from actual prepared profile')
            if not end.get('hardware_key') or not end.get('prediction_contract_sha256'):raise ValueError('missing preparation contract/hardware identity')
            clocks=[receipt['accepted_monotonic'],enqueue['monotonic_seconds'],end['started_monotonic_seconds'],start['monotonic_seconds'],end['ended_monotonic_seconds'],end['monotonic_seconds']]
            if not all(finite(v) for v in clocks) or any(a>b for a,b in zip(clocks,clocks[1:])):raise ValueError('preparation starts/enqueues before acceptance or clocks unordered')
            if end.get('status')!='ready' or not math.isclose(end['wall_seconds'],clocks[-2]-clocks[2],abs_tol=1e-8):raise ValueError('preparation did not publish ready profile with valid wall interval')
    if {(e.get('job_id'),e.get('batch_size')) for e in timeline['events']}!=expected:raise ValueError('preparation event coverage differs from actual profile options')


def planned_records(order):
    return [{**episode,'output':f"{i:02d}-{episode['condition']}-r{episode['repeat']}-{episode['mode']}-{episode['arm']}",
        'status':'unattempted','exit_code':None,'usable_receipts':0} for i,episode in enumerate(order)]


def coverage(records, order):
    # These records are created from the full validated sealed order, including
    # every failed and unattempted cell; never pass a completed-only subset.
    identity=lambda row:tuple(row[key] for key in ('condition','repeat','mode','arm'))
    expected={identity(row) for row in order}
    if (len(order) not in (48,64) or len(expected)!=len(order)
            or len(records)!=len(order) or {identity(row) for row in records}!=expected):
        raise ValueError('coverage requires every cell of the full sealed order')
    planned=len(records)
    counts=Counter(r['status'] for r in records)
    return {'planned_cohorts':planned,'planned_jobs':6*planned,'completed_valid_cohorts':counts['completed'],
        'failed_cohorts':counts['failed'],'running_cohorts':counts['running'],'unattempted_cohorts':counts['unattempted'],
        'usable_receipts':sum(r.get('usable_receipts',0) for r in records),
        'attempted_planned_jobs':6*(planned-counts['unattempted']), 'unattempted_jobs':6*counts['unattempted'],
        'denominator_note':f'All {planned} planned cells / {6*planned} planned jobs remain; a failed cohort is never removed or replaced.'}


def comparisons(records,plan):
    rows=[]
    for condition in (c['name'] for c in plan['conditions']):
        for mode in plan['modes']:
            for arm in plan['arms']:
                selected=[r for r in records if (r['condition'],r['mode'],r['arm'])==(condition,mode,arm)]
                if len(selected)!=2 or {r['repeat'] for r in selected}!={1,2}:raise ValueError('paired denominator must be exactly two')
                good=[r for r in selected if r['status']=='completed'];complete=len(good)==2
                latencies=[v for r in good for v in r['audit']['request_latencies_seconds'].values()]
                row={'condition':condition,'mode':mode,'arm':arm,'planned_repeats':2,'valid_repeats':len(good),
                    'failed_repeats':sum(r['status']=='failed' for r in selected),'unattempted_repeats':sum(r['status']=='unattempted' for r in selected),
                    'repeat_outcomes':[{k:r[k] for k in ('repeat','output','status','exit_code','usable_receipts')} for r in selected],
                    'request_unit':'one_job' if mode=='individual' else 'all_three_required_feedback_bracket',
                    'planned_request_units':12 if mode=='individual' else 4,'observed_valid_request_units':len(latencies),
                    'request_latency_mean_seconds':sum(latencies)/len(latencies) if complete else None,
                    'request_latency_distribution':latency_distribution(latencies) if complete else None,
                    'request_distributions_by_repeat':{str(r['repeat']):latency_distribution(list(r['audit']['request_latencies_seconds'].values())) for r in good},
                    'job_jct_distributions_by_repeat':{str(r['repeat']):latency_distribution(r['audit']['job_jct_seconds']) for r in good},
                    'episode_duration_seconds_by_repeat':{str(r['repeat']):r['audit']['episode_duration_seconds'] for r in good}}
                rows.append(row)
    return rows


def latency_distribution(values):
    if not values or any(not finite(v) or v < 0 for v in values):
        raise ValueError('finite nonnegative observed latencies required')
    values=sorted(values); index=.95*(len(values)-1); low,high=math.floor(index),math.ceil(index)
    return {'n':len(values),'mean_seconds':sum(values)/len(values),
        'p95_seconds':values[low]+(index-low)*(values[high]-values[low]), 'max_seconds':values[-1],
        'p95_note':'linear sample interpolation; very small sample, not a production tail estimate'}


def final_plots(root, records, plan):
    """Plot each condition separately; no mean across resource/duration conditions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps, ImageDraw
    figure,axes=plt.subplots(len(plan['conditions']),2,figsize=(14,4*len(plan['conditions'])),squeeze=False)
    for condition_index,condition in enumerate(plan['conditions']):
        for mode_index,mode in enumerate(plan['modes']):
            axis=axes[condition_index][mode_index]
            for arm_index,arm in enumerate(plan['arms']):
                selected=[r for r in records if (r['condition'],r['mode'],r['arm'])==(condition['name'],mode,arm)]
                for record in selected:
                    value=latency_distribution(list(record['audit']['request_latencies_seconds'].values()))['mean_seconds']
                    axis.scatter([arm_index],[value]);axis.annotate('r'+str(record['repeat']),(arm_index,value),xytext=(4,3),textcoords='offset points',fontsize=8)
                axis.text(arm_index,.01,'valid 2/2',ha='center',transform=axis.get_xaxis_transform(),fontsize=8)
            axis.set_xticks(range(len(plan['arms'])),[a.replace('resource_','').replace('_','\n') for a in plan['arms']],fontsize=8)
            axis.set_title(condition['name']+' / '+mode)
            axis.set_ylabel('Average individual JCT (s)' if mode=='individual' else 'Average all-required bracket latency (s)')
            axis.grid(axis='y',alpha=.2)
    figure.suptitle('Separate sealed conditions; two observed repeats per arm/mode; no cross-condition mean')
    figure.tight_layout();figure.savefig(root/'combined-condition-comparisons.png',dpi=140);plt.close(figure)
    thumbnails=[]
    for record in records:
        path=root/record['output']/'cohort-measurements.png'
        canvas=Image.new('RGB',(600,530),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,5),record['output'],fill='black')
        if path.exists():
            with Image.open(path) as image:
                thumb=ImageOps.contain(image.convert('RGB'),(600,500));canvas.paste(thumb,(0,25))
        else:draw.text((10,50),'Per-episode plot unavailable; saved episode data retained.',fill='black')
        thumbnails.append(canvas)
    sheet=Image.new('RGB',(8*600,math.ceil(len(thumbnails)/8)*530),'white')
    for index,thumbnail in enumerate(thumbnails):sheet.paste(thumbnail,((index%8)*600,(index//8)*530))
    sheet.save(root/'combined-cohort-gantt-metrics.png')
    for condition in plan['conditions']:
        condition_records=[(r,t) for r,t in zip(records,thumbnails) if r['condition']==condition['name']]
        page=Image.new('RGB',(4*600,math.ceil(len(condition_records)/4)*530),'white')
        for index,(_,thumbnail) in enumerate(condition_records):page.paste(thumbnail,((index%4)*600,(index//4)*530))
        page.save(root/(condition['name']+'-cohort-gantt-metrics.png'))


def execute(bundle, root):
    """Execute once under an output-parent campaign flock; child owns GPU flock."""
    root=Path(root).resolve();root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'.a10-resource-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists(): raise FileExistsError('never overwrite/resume an existing campaign')
        verify_bundle(bundle)
        root.mkdir(exist_ok=False)
        started=time.monotonic();records=planned_records(bundle['order']);plan=bundle['plan'];profiles={};active=None
        frozen={'order':bundle['order'],'input_sha256':bundle['hashes'],'source_pins':bundle['snapshots'],
            'episode_max_seconds':plan['episode_max_seconds'],'campaign_max_seconds':plan['campaign_max_seconds'],'deadline_utc':plan['deadline_utc'],
            'planned_cohorts':len(records),'planned_jobs':6*len(records),'settings':str(bundle['settings']),'calibration':str(bundle['calibration']),
            'cache_policy':'fresh scheduler/predictor process and runtime per episode; charged dedicated calibration shared OS/data caches',
            'scope':'dedicated measured-template mechanism study; no neural-predictor accuracy or cross-condition mean claim'}
        atomic_json(root/'campaign-protocol-before.json',frozen)
        def save(status,error=None):
            atomic_json(root/'campaign-results.json',records)
            atomic_json(root/'coverage.json',coverage(records,bundle['order']))
            atomic_json(root/'comparisons.json',comparisons(records,plan))
            atomic_json(root/'status.json',{'status':status,'elapsed_seconds':time.monotonic()-started,'active_output':active['output'] if active else None,
                'error':error,'coverage':coverage(records,bundle['order'])})
        def deadline():
            check_deadline(plan,time.monotonic()-started,plan['campaign_max_seconds'])
        def interrupted(signum,frame): raise InterruptedError('resource campaign signal '+str(signum)+'; terminating only owned episode child')
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
        for sig in previous:signal.signal(sig,interrupted)
        save('running')
        try:
            for record in records:
                deadline();verify_bundle(bundle);active=record
                item=bundle['manifests'][record['condition']];output=root/record['output']
                if output.exists():raise FileExistsError('refuse duplicate cohort output')
                command=[sys.executable,'-m','localml_scheduler.benchmarks.a10_feedback_latency.run','run',
                    '--manifest',str(item['path']),'--settings',str(bundle['settings']),'--calibration',str(bundle['calibration']),
                    '--output',str(output),'--arm',record['arm'],'--mode',record['mode'],
                    '--deadline-utc',plan['deadline_utc'],'--max-seconds',str(plan['episode_max_seconds'])]
                record.update(status='running',command=command);save('running');child=None
                try:
                    with (root/(record['output']+'.log')).open('x') as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            launched=time.monotonic()
                            atomic_json(root/'active-child.json',{'pid':child.pid,'owned':True,'status':'running','command':command,
                                **{k:record[k] for k in ('condition','repeat','arm','mode','output')}})
                            while child.poll() is None:
                                deadline()
                                if time.monotonic()-launched>=plan['episode_max_seconds']:raise TimeoutError('sealed episode wall-clock bound reached')
                                time.sleep(.25)
                        finally:
                            # Signals remain enabled while the existing owned-tree helper terminates/reaps.
                            terminate_process_tree(child,timeout=15.)
                        record['exit_code']=child.returncode
                    audit=audit_episode(output,record,item['manifest'],bundle['snapshots'][record['condition']],bundle['table'])
                    record.update(audit=audit,usable_receipts=audit['usable_receipts'],result={key:audit.get(key) for key in ('complete','predictor_comparison_eligible')})
                    if record['exit_code']!=0 or not audit['valid']:raise RuntimeError('required episode failed/ineligible: '+json.dumps(audit['errors']))
                    common=profiles.setdefault(record['condition'],audit['profiles_by_name'])
                    if audit['profiles_by_name']!=common:raise ValueError('numeric profile identity changed within condition across arms/repeats')
                    record['status']='completed';active=None;save('running')
                    atomic_json(root/'active-child.json',{'pid':child.pid,'owned':True,'status':'exited','exit_code':child.returncode,'output':record['output']})
                except BaseException as exc:
                    record.update(status='failed',error=repr(exc),exit_code=child.returncode if child is not None else None)
                    atomic_json(root/'active-child.json',{'pid':child.pid if child is not None else None,'owned':child is not None,
                        'status':'failed','exit_code':record['exit_code'],'output':record['output'],'error':repr(exc)})
                    raise
            deadline();verify_bundle(bundle)
            if coverage(records,bundle['order'])['completed_valid_cohorts']!=len(records) or coverage(records,bundle['order'])['usable_receipts']!=6*len(records):raise ValueError('required complete campaign coverage absent')
            atomic_json(root/'campaign-source-pins-after.json',bundle['snapshots'])
            # Final-only comparisons: no partial completed-only rankings or mixed-condition means.
            try:
                final_plots(root,records,plan)
            except Exception as exc:
                atomic_json(root/'plot-error.json',{'error':repr(exc),'note':'completed scientific records retained; charts can be regenerated offline'})
            deadline();verify_bundle(bundle)
            save('complete')
            atomic_json(root/'campaign-complete.json',{'complete':True,'episodes':len(records),'receipts':6*len(records),'elapsed_seconds':time.monotonic()-started,
                'source_pins_after':bundle['snapshots'],'coverage':coverage(records,bundle['order'])})
            return records
        except BaseException as exc:
            if active is not None and active['status']=='running':active.update(status='failed',error=repr(exc))
            save('failed',repr(exc))
            atomic_json(root/'campaign-aborted.json',{'complete':False,'error':repr(exc),'elapsed_seconds':time.monotonic()-started,
                'coverage':coverage(records,bundle['order']),'failed_output':active['output'] if active else None})
            raise
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('manifests','settings','calibration','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    return execute(sealed_bundle(args.manifests,args.settings,args.calibration),args.output)


if __name__=='__main__':main()
