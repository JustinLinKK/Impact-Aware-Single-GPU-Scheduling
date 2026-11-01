"""Read-only canonical replay of every generation and matched freshness lease."""
import argparse
import json
from pathlib import Path
import time

from . import campaign
from .fresh_calibration import CONTRACT, freshness, matched_block, observation_range
from ..a10_feedback_latency.run import atomic_json, file_hash, source_snapshot
from ..a10_resource_mechanisms.calibrate import extract_row
from ..a10_resource_mechanisms.profiles import fit_templates
from ..a10_v5_ablation import calibrate_predictor as cp

def load(path):return json.loads(Path(path).read_text())

def validate_record_order(records,order):
    if len(records)!=len(order) or len(order)!=60:
        raise ValueError('exact sixty records required; no trailing or missing outcomes')
    expected=campaign.planned_records(order)
    for row,wanted in zip(records,expected):
        for key in ('host_id','condition','repeat','mode','arm','output','expected_jobs'):
            if row.get(key)!=wanted[key]:raise ValueError('record differs from frozen order/output: '+key)

def validate_generation_clock(generation,table,neural):
    oldest,newest=observation_range(table,neural)
    if (generation['oldest_observation_unix']!=oldest or generation['newest_observation_unix']!=newest
            or not newest<=generation['finished_unix']<=time.time()):
        raise ValueError('generation observation ledger differs from real timestamps')
    start=generation['started_unix']
    if generation['generation']=='initial':
        if start is not None:raise ValueError('initial generation is existing preparation')
    elif start is None or not start<=oldest:
        raise ValueError('refreshed generation must contain newly observed measurements')

def generation_audit(host, generation):
    table_path=Path(generation['calibration']);measured=table_path.parent
    manifest=load(host/'calibration-manifest.json');table=load(table_path)
    rows=load(measured/'calibration-rows.json');result=load(measured/'result.json')
    if not result.get('complete') or result.get('calibration')!=table or result.get('calibration_rows')!=rows or len(rows)!=8:
        raise ValueError('eight complete dedicated calibration rows required per generation')
    snapshot=load(measured/'source-snapshot.json')
    if snapshot!=load(measured/'source-snapshot-after.json') or snapshot!=source_snapshot(manifest,host/'calibration-manifest.json'):
        raise ValueError('calibration generation source drift')
    byid={row['job_signature']:row for row in rows}
    for spec in manifest['calibration']:
        work=measured/'jobs'/spec['name'];raw=load(work/'raw-calibration-evidence.json')
        if extract_row(spec,hardware_key=table['hardware_key'],gpu_uuid=table['gpu_uuid'],**raw)!=byid[spec['expected_runtime_signature']]:
            raise ValueError('raw calibration observation replay mismatch')
        cp._verify_receipt(spec,raw,work)
    if fit_templates(rows,hardware_key=table['hardware_key'],gpu_uuid=table['gpu_uuid'],source_identity=table['source_identity'])!=table:
        raise ValueError('dedicated template/phase fit mismatch')
    neural=Path(generation['neural_calibration']);proof=load(neural.parent/'conversion-proof.json')
    from ..a10_feedback_latency.run import STUDENT_SHA256
    source=load(host/'plan.json')['source_commit']
    selected={s['name']for s in manifest['calibration']if s.get('ballast_mib',0)==0}
    excluded={s['name']for s in manifest['calibration']if s.get('ballast_mib',0)!=0}
    if (proof.get('complete')is not True or proof.get('cpu_only')is not True
            or proof.get('source_commit')!=source or proof.get('predictor_artifact_sha256')!=STUDENT_SHA256
            or len(selected)!=6 or len(excluded)!=2 or set(proof['selected_no_ballast_names'])!=selected
            or set(proof['excluded_ballast_names'])!=excluded):
        raise ValueError('released S_A10 CPU-only conversion required')
    validate_generation_clock(generation,table,load(neural))
    if generation['generation']!='initial':
        parent=measured.parent
        if parent.name!=generation['generation']:raise ValueError('generation path identity mismatch')
        for stage in ('qualification','calibration','conversion','seal'):
            completed=load(parent/(stage+'-complete.json'));dispatch=load(parent/(stage+'-dispatch.json'))
            if (completed.get('exit_code')!=0 or not generation['started_unix']<=dispatch['started_unix']
                <=completed['finished_unix']<=generation['finished_unix']):
                raise ValueError('each refresh phase needs successful ordered exit evidence')
    for name,value in proof['inputs_sha256'].items():
        if file_hash(name)!=value:raise ValueError('conversion input changed')
    for name,value in proof['output_sha256'].items():
        if file_hash(neural.parent/name)!=value:raise ValueError('conversion output changed')
    for name,value in {**generation['input_sha256'],**generation['host_input_sha256']}.items():
        if file_hash(name)!=value:raise ValueError('generation pin changed')
    return dict(calibration_receipts=8,raw_replay_passed=True)

def replay(host):
    host=Path(host);root=host/'runs/campaign'
    generations=load(root/'calibration-generations.json');blocks=load(root/'block-calibration-bindings.json')
    records=load(root/'campaign-results.json');plan=load(host/'plan.json')
    if plan.get('calibration_refresh')!=CONTRACT:raise ValueError('frozen refresh contract required')
    order=campaign.episode_order(plan)
    validate_record_order(records,order)
    if len(generations)>CONTRACT['maximum_generations'] or len({g['generation']for g in generations})!=len(generations):
        raise ValueError('unique finite generation coverage required')
    bundles={};audits={}
    for g in generations:
        audits[g['generation']]=generation_audit(host,g)
        b=campaign.sealed_bundle(g['manifests'],host/'settings.json',g['calibration'],g['neural_calibration'])
        if b['plan']!=plan or b['snapshots']!=g['source_pins']:raise ValueError('generation source or plan mismatch')
        if bundles:
            first=next(iter(bundles.values()))
            for name,item in b['manifests'].items():
                if item['manifest']['jobs']!=first['manifests'][name]['manifest']['jobs']:
                    raise ValueError('global recipes, seeds or update budgets changed')
        bundles[g['generation']]=b
    if len(blocks)!=len(order)//3:raise ValueError('complete matched block denominator required')
    replayed=[]
    for index,block in enumerate(blocks):
        if block['block_index']!=index or block['first_record']!=3*index or block['status']!='completed':
            raise ValueError('complete ordered matched block required')
        if any(block[k]!=v for k,v in matched_block(order,3*index).items()):raise ValueError('block identity changed')
        b=bundles[block['generation']];neural=load(b['neural_calibration'])
        start=block['freshness_at_start']
        if freshness(b['table'],neural,now=start['checked_unix'],remaining_episodes=3,episode_seconds=plan['episode_max_seconds'])!=start or not start['fresh']:
            raise ValueError('block did not reserve complete fresh horizon')
        if not start['checked_unix']<=block['completed_unix']<start['expires_unix']-CONTRACT['margin_seconds']:
            raise ValueError('block completed outside freshness lease')
        if len(block['episode_checks'])!=3:raise ValueError('each arm needs independent remaining-horizon check')
        profiles=None
        for offset,check in enumerate(block['episode_checks']):
            ri=3*index+offset;record=records[ri]
            expected=freshness(b['table'],neural,now=check['checked_unix'],remaining_episodes=3-offset,episode_seconds=plan['episode_max_seconds'])
            if check!={'record_index':ri,**expected} or not expected['fresh']:
                raise ValueError('remaining block freshness check invalid')
            if record.get('calibration_generation')!=block['generation'] or record.get('calibration_block')!=index or record['status']!='completed' or record['exit_code']!=0:
                raise ValueError('arm differs from common block calibration or failed')
            item=b['manifests'][record['condition']];output=root/record['output'];result=load(output/'result.json')
            if not check['checked_unix']<=result['episode_started_unix']<=max(r['feedback_ready_unix']for r in result['receipts'].values())<=block['completed_unix']:
                raise ValueError('actual receipt interval lies outside admitted fresh block')
            audit=campaign.audit_episode(output,record,item['manifest'],b['snapshots'][record['condition']],item['table'])
            if not audit['valid'] or audit!=record['audit']:raise ValueError('independent episode replay mismatch')
            if profiles is None:profiles=audit['profiles_by_name']
            elif profiles!=audit['profiles_by_name']:raise ValueError('numeric profiles differ within matched block')
            replayed.append({**record,'audit':audit})
    coverage=campaign.coverage(replayed,order)
    if coverage['completed_valid_cohorts']!=60 or coverage['usable_receipts']!=432:raise ValueError('exact60/432 coverage required')
    return dict(audit_passed=True,audited_unix=time.time(),coverage=coverage,generation_audits=audits,
        matched_blocks=len(blocks),freshness_expired_during_comparison=False,
        refresh_wall_seconds=sum(g['finished_unix']-g['started_unix']for g in generations if g['started_unix']is not None),
        refresh_cost_note='inside campaign and resource billing; reported separately from per-request JCT',
        source_commit=plan['source_commit'],comparisons=campaign.comparisons(replayed,plan),
        paired_contrasts=campaign.paired_contrasts(replayed,plan))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('never overwrite audit evidence')
    result=replay(a.root);atomic_json(a.output,result)
    print(json.dumps({'audit_passed':True,'cohorts':60,'receipts':432,'generations':len(result['generation_audits'])}))
