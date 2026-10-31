"""CPU-only released S_A10 correction from the existing dedicated solo fold.

No calibration worker, GPU query, optimizer, neural-model fitting or weights write is
performed here. Fresh input-only predictions are paired with retained successful
solo windows, then the unchanged two-scalar legacy fit_calibration is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from ...config import SchedulerSettings
from ...domain import TrainingJob
from ...hardware import HardwareProfile
from ...prediction.ml_predictor import MLVramPredictor
from ..a10_feedback_latency.calibration import fit_calibration
from ..a10_feedback_latency.receipts import Submission, receive
from ..a10_feedback_latency.run import STUDENT_SHA256, atomic_json, build_job, file_hash, source_snapshot, digest as runtime_digest
from ..a10_resource_mechanisms import workloads
from ..a10_resource_mechanisms.calibrate import calibration_source_identity, extract_row
from ..a10_resource_mechanisms.profiles import fit_templates
from .campaign import receipt_artifact_path
from .protocol import neural_digest, validate_neural_table, validate_table


def load(path):
    return json.loads(Path(path).read_text())


def convert_row(spec, row, profile, raw):
    """Replay raw sample validity, then convert units/columns for legacy fitting.

    Window seconds / successful updates is the measured seconds-per-update;
    predicted step_cost remains milliseconds/update. Allocated memory is never
    relabeled reserved memory. Startup/resume use retained diagnostic-corrected
    phase times; no memory subtraction or additional uncertainty margin is made.
    """
    if spec.get('role')!='calibration' or spec.get('ballast_mib',0)!=0:
        raise ValueError('only dedicated no-ballast replicas may calibrate S_A10')
    if row.get('role')!='calibration' or row.get('complete')is not True:
        raise ValueError('complete dedicated row required')
    replay=extract_row(spec,hardware_key=row['hardware_key'],gpu_uuid=row['gpu_uuid'],**raw)
    if replay!=row:
        raise ValueError('saved row differs from raw source/sample/phase/update evidence')
    if (profile.get('artifact_sha256')!=STUDENT_SHA256 or profile.get('hardware')!='A10'
            or profile.get('batch_size')!=spec['batch'] or profile.get('kind')!='predicted_workload'
            or profile.get('calibration')!='uncalibrated_prior'):
        raise ValueError('fresh released S_A10 prior with exact physical batch required')
    for key in ('step_cost','epoch_cost','avg_vram_mib','peak_vram_mib'):
        v=profile.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError('invalid numeric CPU prediction: '+key)
    windows=row['windows'];steps=sum(w['steps'] for w in windows);seconds=sum(w['elapsed_seconds'] for w in windows)
    if steps<=0 or seconds<=0:
        raise ValueError('positive successful-update measurement required')
    phase=row['phase_seconds']
    return {'name':spec['name'],'role':'calibration','complete':True,
        'source_identity':deepcopy(row['source_identity']),
        'runtime_completion_identity':spec['expected_runtime_signature'],
        'parameter_count':spec['parameter_count'],'observed_at':row['observed_at'],
        'predicted_profile':deepcopy(profile),'windows':deepcopy(windows),
        'isolated_seconds_per_update':seconds/steps,'peak_reserved_mib':row['peak_reserved_mib'],
        'startup_seconds':phase['startup'],'resume_seconds':phase['resume'],
        'pause_seconds':phase['pause'],'delivery_seconds':phase['delivery'],
        'phase_history':deepcopy(raw['phase_history']),
        'receipt':deepcopy(raw['receipt']),'launches':deepcopy(raw['launches']),
        'raw_evidence_sha256':row['raw_evidence_sha256'],
        'timing_note':'pooled successful-update seconds/count; phase timing from replayed diagnostic-corrected dedicated row'}


def _verify_receipt(spec, raw, work):
    receipt=load(work/'receipt.json');job_payload=load(work/'job-final.json')
    if receipt!=raw['receipt']:
        raise ValueError('durable receipt differs from retained raw evidence')
    metadata=job_payload['metadata'];expected=2*spec['steps_per_epoch']
    if (spec['epochs']!=2 or raw['final_audit'].get('completed_epochs')!=2
            or raw['final_audit'].get('optimizer_parameter_states',0)<=0
            or metadata.get('resource_workload_spec')!=spec
            or metadata.get('pilot_frozen_workload_sha256')!=runtime_digest(spec)
            or json.loads(job_payload['config']['env']['A10_PILOT_SPEC_JSON'])!=spec
            or receipt.get('expected_updates')!=expected or receipt.get('expected_epochs')!=2
            or receipt.get('expected_workload_sha256')!=runtime_digest(spec)):
        raise ValueError('exact successful updates/two epochs/immutable workload not proven')
    for kind,expected_path in [('result',work/'working/shared_training/result.json'),('execution',work/'execution.json')]:
        receipt_artifact_path(receipt[kind+'_path'],work.parent.parent,expected_path)
    submission=Submission(**{f.name:receipt[f.name] for f in fields(Submission)})
    replay=receive(submission,TrainingJob.from_dict(job_payload),receipt['worker_exit_code'],
        monotonic=lambda:receipt['feedback_ready_monotonic'],wall_clock=lambda:receipt['feedback_ready_unix'])
    if replay!=receipt:
        raise ValueError('receipt artifacts/exit/final result do not replay identically')


def saved_hardware(result, table, manifest):
    """Use measured name/VRAM and actual hardware key; never probe a GPU.

    The retained calibration result does not record compute capability, driver
    or CUDA version. Those optional capture fields remain explicitly unknown.
    """
    gpu=result['hardware']
    if gpu.get('name')!='NVIDIA A10' or gpu.get('gpu_uuid')!=table['gpu_uuid']:
        raise ValueError('retained actual A10/GPU identity differs from table')
    memory=gpu.get('total_mib')
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<=0:
        raise ValueError('retained actual VRAM required')
    return HardwareProfile(hardware_key=table['hardware_key'],os_name='linux',
        gpu_name=gpu['name'],total_vram_mb=int(memory),compute_capability=None,cuda_runtime=None,
        torch_version=manifest['source_runtime']['torch_version'],driver_version=None)


@contextmanager
def _cpu_only():
    previous=os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        yield
    finally:
        if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=previous


def convert(settings_path, manifest_path, measured_directory, output, *, source_commit):
    """Create one exclusive CPU correction bundle; leave all inputs untouched."""
    settings_path,manifest_path,measured_directory,output=map(Path,(settings_path,manifest_path,measured_directory,output))
    if output.exists():raise FileExistsError('never overwrite/reuse predictor calibration output')
    if not re.fullmatch('[0-9a-f]{40}',str(source_commit)):
        raise ValueError('explicit source commit required')
    manifest=load(manifest_path);table=load(measured_directory/'calibration.json');validate_table(table)
    if manifest.get('experiment_source_commit')!=source_commit or manifest.get('jobs'):
        raise ValueError('explicit source commit and dedicated-only calibration manifest required')
    snapshot=load(measured_directory/'source-snapshot.json')
    if (snapshot!=load(measured_directory/'source-snapshot-after.json')
            or source_snapshot(manifest,manifest_path)!=snapshot
            or calibration_source_identity(snapshot,manifest)!=table['source_identity']):
        raise ValueError('actual source/runtime/hash snapshot differs from completed dedicated fold')
    result=load(measured_directory/'result.json');rows=load(measured_directory/'calibration-rows.json')
    audits=load(measured_directory/'validity-audits.json')
    if (result.get('complete')is not True or result.get('calibration')!=table
            or result.get('calibration_rows')!=rows or result.get('source_and_budget_audits')!=audits):
        raise ValueError('complete dedicated result/table/rows/audits required')
    specs=manifest['calibration']
    if (not specs or len({s['name'] for s in specs})!=len(specs) or len(rows)!=len(specs)
            or len({r['job_signature'] for r in rows})!=len(rows)):
        raise ValueError('all unique dedicated replicas required')
    by_signature={r['job_signature']:r for r in rows};raws={};pins={}
    def pin(path):
        pins[str(path.resolve())]=file_hash(path)
    for path in (settings_path,manifest_path,*[measured_directory/name for name in
            ('calibration.json','calibration-rows.json','result.json','validity-audits.json','source-snapshot.json','source-snapshot-after.json')]):pin(path)
    for spec in specs:
        work=measured_directory/'jobs'/spec['name'];raw=load(work/'raw-calibration-evidence.json')
        row=by_signature[spec['expected_runtime_signature']]
        if raw.get('visibility_qualification')is None:
            raise ValueError('retained fresh packaged qualification/context evidence required')
        replay=extract_row(spec,hardware_key=table['hardware_key'],gpu_uuid=table['gpu_uuid'],**raw)
        if replay!=row or audits.get(spec['name'])!=raw['final_audit']:
            raise ValueError('replica raw evidence/update/phase replay mismatch')
        _verify_receipt(spec,raw,work);raws[spec['name']]=raw
        for name in ('raw-calibration-evidence.json','receipt.json','job-final.json','execution.json','working/shared_training/result.json'):pin(work/name)
    # The complete source-compatible measured table must reconstruct before the
    # no-ballast subset is selected for the NN correction.
    if fit_templates(rows,hardware_key=table['hardware_key'],gpu_uuid=table['gpu_uuid'],source_identity=table['source_identity'])!=table:
        raise ValueError('complete dedicated pooled table does not replay')
    hardware=saved_hardware(result,table,manifest)
    settings=SchedulerSettings.from_dict(load(settings_path))
    if not settings.prediction.student_release_path or settings.prediction.test_override_enabled or settings.prediction.registry_path:
        raise ValueError('actual frozen released predictor required; no registry/test override')
    chosen=[s for s in specs if s.get('ballast_mib',0)==0]
    if not chosen:raise ValueError('no dedicated no-ballast calibration replicas')
    output.mkdir(parents=True,exist_ok=False)
    converted=[];profiles={}
    with _cpu_only(),TemporaryDirectory(prefix='a10-v5-predictor-',dir=output) as temporary:
        predictor=MLVramPredictor(settings.prediction,hardware)
        if not predictor.available or predictor.artifact_hash!=STUDENT_SHA256:
            raise ValueError('original released S_A10 artifact unavailable or SHA mismatch: '+str(predictor.unavailable_reason))
        for spec in chosen:
            job,_=build_job(spec,manifest_path=manifest_path,output=Path(temporary),manifest=manifest,memory_budget_mib=hardware.total_vram_mb*.9)
            profile=predictor.predict_profile_options(job,[spec['batch']])[spec['batch']]
            if profile.get('job_signature')!=(job.packing.signature or job.job_id):
                raise ValueError('CPU prediction belongs to a different reconstructed job')
            profiles[spec['name']]=profile
            converted.append(convert_row(spec,by_signature[spec['expected_runtime_signature']],profile,raws[spec['name']]))
    neural=fit_calibration(converted,hardware_key=table['hardware_key'],predictor_artifact_sha256=STUDENT_SHA256)
    neural.update(gpu_uuid=table['gpu_uuid'],measured_calibration_artifact_sha256=table['artifact_sha256'],
        experiment_source_commit=source_commit,source_runtime_sha256=workloads.digest(manifest['source_runtime']),
        cpu_profile_hardware=hardware.to_dict(),calibration_scope='only dedicated no-ballast replicas; fresh CPU S_A10 predictions; no evaluation outcomes')
    neural['calibration_sha256']=neural_digest(neural);validate_neural_table(neural,table)
    if source_snapshot(manifest,manifest_path)!=snapshot:
        raise ValueError('actual source bytes changed during CPU prediction')
    for path,expected in pins.items():
        if file_hash(path)!=expected:raise ValueError('dedicated input changed during CPU conversion: '+path)
    atomic_json(output/'calibration.json',neural)
    atomic_json(output/'converted-rows.json',converted);atomic_json(output/'cpu-profiles.json',profiles)
    atomic_json(output/'conversion-proof.json',{'complete':True,'cpu_only':True,'source_commit':source_commit,
        'source_commit_note':'caller-declared commit; actual source SHA snapshot verified; no clean Git claim',
        'source_snapshot':snapshot,'inputs_sha256':pins,'selected_no_ballast_names':[s['name'] for s in chosen],
        'excluded_ballast_names':[s['name'] for s in specs if s.get('ballast_mib',0)!=0],
        'predictor_artifact_sha256':STUDENT_SHA256,'neural_calibration_sha256':neural['calibration_sha256'],
        'output_sha256':{name:file_hash(output/name) for name in ('calibration.json','converted-rows.json','cpu-profiles.json')}})
    return neural


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('settings','manifest','measured-calibration','output','source-commit'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    return convert(args.settings,args.manifest,args.measured_calibration,args.output,source_commit=args.source_commit)


if __name__=='__main__':main()
