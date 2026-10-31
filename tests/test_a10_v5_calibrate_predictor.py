"""Pure saved-evidence fixtures; no A10 measurement or neural inference."""
from copy import deepcopy
import pytest
from localml_scheduler.benchmarks.a10_resource_mechanisms.calibrate import extract_row
from localml_scheduler.benchmarks.a10_feedback_latency.calibration import fit_calibration
from localml_scheduler.benchmarks.a10_v5_ablation import calibrate_predictor as bridge
from localml_scheduler.benchmarks.a10_v5_ablation import protocol
from localml_scheduler.tests.test_a10_resource_calibration import evidence


def fixture():
    spec,raw=evidence();host={k:raw.pop(k) for k in ('hardware_key','gpu_uuid')}
    row=extract_row(spec,**host,**raw)
    profile={'kind':'predicted_workload','artifact_sha256':bridge.STUDENT_SHA256,
        'hardware':'A10','batch_size':spec['batch'],'calibration':'uncalibrated_prior',
        'job_signature':'fixture','step_cost':100.,'epoch_cost':3200.,
        'avg_vram_mib':100.,'peak_vram_mib':800.}
    return spec,row,profile,raw


def test_column_units_raw_memory_and_phase_conversion_hand_checked():
    spec,row,profile,raw=fixture();before=deepcopy((spec,row,profile,raw))
    converted=bridge.convert_row(spec,row,profile,raw)
    assert converted['isolated_seconds_per_update']==.25 #4sec/16updates
    assert converted['predicted_profile']['step_cost']==100. #milliseconds retained
    assert converted['peak_reserved_mib']==1200. #reserved, not900allocated
    assert converted['startup_seconds']==converted['resume_seconds']==.5
    assert converted['pause_seconds']==pytest.approx(1.8)
    assert converted['delivery_seconds']==1.
    correction=fit_calibration([converted],hardware_key=row['hardware_key'],predictor_artifact_sha256=bridge.STUDENT_SHA256)
    assert correction['wall_multiplier']==pytest.approx(2.5)
    assert correction['peak_add_mib']==400.
    correction.update(gpu_uuid=row['gpu_uuid'],measured_calibration_artifact_sha256='a'*64)
    correction['calibration_sha256']=protocol.neural_digest(correction)
    protocol.validate_neural_table(correction,{'hardware_key':row['hardware_key'],'gpu_uuid':row['gpu_uuid'],'artifact_sha256':'a'*64})
    assert (spec,row,profile,raw)==before


@pytest.mark.parametrize('mutation',['sample','row','phase','updates','window','host'])
def test_raw_validity_replayed_before_fit(mutation):
    spec,row,profile,raw=fixture()
    if mutation=='sample':raw['samples'][1]['compute_apps'][0]['pid']=999999
    elif mutation=='row':row['peak_reserved_mib']+=1
    elif mutation=='phase':raw['phase_history'][0]['hardware_key']='other'
    elif mutation=='updates':raw['final_audit']['all_adamw_steps']-=1
    elif mutation=='window':raw['windows'][0]['steps']=0
    elif mutation=='host':row['gpu_uuid']='other-GPU'
    with pytest.raises((ValueError,KeyError)):bridge.convert_row(spec,row,profile,raw)


@pytest.mark.parametrize('mutation',['ballast','role','complete','artifact','hardware','batch','nonfinite'])
def test_original_solo_prior_only(mutation):
    spec,row,profile,raw=fixture()
    if mutation=='ballast':spec['ballast_mib']=1
    elif mutation=='role':spec['role']='parent'
    elif mutation=='complete':row['complete']=False
    elif mutation=='artifact':profile['artifact_sha256']='a'*64
    elif mutation=='hardware':profile['hardware']='A100'
    elif mutation=='batch':profile['batch_size']+=1
    elif mutation=='nonfinite':profile['step_cost']=float('nan')
    with pytest.raises(ValueError):bridge.convert_row(spec,row,profile,raw)


def test_saved_hardware_never_probes_or_invents_missing_fields(monkeypatch):
    import torch
    def no_gpu(*a,**kw):raise AssertionError('no GPU access')
    monkeypatch.setattr(torch.cuda,'get_device_properties',no_gpu)
    monkeypatch.setattr(torch.cuda,'is_available',no_gpu)
    hw=bridge.saved_hardware({'hardware':{'name':'NVIDIA A10','gpu_uuid':'fixture','total_mib':24000}},
        {'hardware_key':'saved-host','gpu_uuid':'fixture'},{'source_runtime':{'torch_version':'saved'}})
    assert hw.hardware_key=='saved-host' and hw.total_vram_mb==24000
    assert hw.compute_capability is hw.cuda_runtime is hw.driver_version is None


def test_cpu_visibility_context_restores_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','original')
    with bridge._cpu_only():assert bridge.os.environ['CUDA_VISIBLE_DEVICES']==''
    assert bridge.os.environ['CUDA_VISIBLE_DEVICES']=='original'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(RuntimeError):
        with bridge._cpu_only():raise RuntimeError('fixture')
    assert 'CUDA_VISIBLE_DEVICES' not in bridge.os.environ


def test_output_exclusive_before_input_or_predictor_access(tmp_path):
    output=tmp_path/'exists';output.mkdir()
    with pytest.raises(FileExistsError):bridge.convert('absent','absent','absent',output,source_commit='a'*40)
    assert not list(output.iterdir())


def test_complete_bridge_with_real_raw_replay_receipts_and_owned_cpu_job_dirs(tmp_path,monkeypatch):
    """Model outputs mocked; raw lifecycle and receipt validators stay real."""
    from dataclasses import fields
    from localml_scheduler.config import SchedulerSettings
    from localml_scheduler.domain import JobStatus
    from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import fit_templates
    from localml_scheduler.benchmarks.a10_resource_mechanisms.calibrate import calibration_source_identity
    from localml_scheduler.benchmarks.a10_feedback_latency.receipts import receive
    from localml_scheduler.benchmarks.a10_feedback_latency.run import build_job, submission
    from localml_scheduler.tests.test_a10_context_isolation import instrumented_evidence
    import json
    def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
    spec,raw=instrumented_evidence();host={k:raw.pop(k) for k in ('hardware_key','gpu_uuid')}
    measured=tmp_path/'measured';measured.mkdir();manifest_path=tmp_path/'manifest.json'
    manifest={'jobs':[],'calibration':[spec],'job_timeout_seconds':600,'experiment_source_commit':'a'*40,
        'source_runtime':bridge.workloads.source_metadata(),'calibration_identity_schema':'resource_solo_runtime_v1'}
    write(manifest_path,manifest)
    job,work=build_job(spec,manifest_path=manifest_path,output=measured,manifest=manifest,memory_budget_mib=20000.)
    job.status=JobStatus.COMPLETED;job.finished_at='1970-01-01T00:01:58+00:00'
    job.metadata.update(runtime_global_step=64,runtime_total_optimizer_updates=64,
        runtime_finalization_complete=True,runtime_completion_identity=spec['expected_runtime_signature'])
    returned={'global_step':64,'epochs':2,'signature':spec['expected_runtime_signature'],
        'metric_name':'negative_validation_auc','selected_metric':-.6,
        'history':[{'epoch':i,'validation_auc':.6,'negative_validation_auc':-.6,'validation_bce':.7,
                    'validation_negative_count':128,'validation_positive_count':128} for i in (1,2)]}
    result_path=work/'working/shared_training/result.json';result_path.parent.mkdir(parents=True)
    write(result_path,returned);write(work/'execution.json',{'exc_type':None})
    frozen=submission(spec,job,work,1000.,100.)
    raw['receipt']=receive(frozen,job,0,monotonic=lambda:1019.,wall_clock=lambda:119.)
    raw['final_audit'].update(completed_epochs=2,optimizer_parameter_states=2)
    row=extract_row(spec,**host,**raw)
    write(work/'raw-calibration-evidence.json',raw);write(work/'receipt.json',raw['receipt']);write(work/'job-final.json',job.to_dict())
    snapshot={'fixture':'not an actual source claim'}
    monkeypatch.setattr(bridge,'source_snapshot',lambda *a:snapshot)
    identity=calibration_source_identity(snapshot,manifest)
    table=fit_templates([row],hardware_key=host['hardware_key'],gpu_uuid=host['gpu_uuid'],source_identity=identity)
    write(measured/'calibration.json',table);write(measured/'calibration-rows.json',[row])
    write(measured/'validity-audits.json',{spec['name']:raw['final_audit']})
    result={'complete':True,'calibration':table,'calibration_rows':[row],
        'source_and_budget_audits':{spec['name']:raw['final_audit']},
        'hardware':{'name':'NVIDIA A10','gpu_uuid':host['gpu_uuid'],'total_mib':24000.}}
    write(measured/'result.json',result)
    for name in ('source-snapshot.json','source-snapshot-after.json'):write(measured/name,snapshot)
    settings=tmp_path/'settings.json';write(settings,{'prediction':{'mode':'ml_predictor','ml':{'student_release_path':'/fixture-original-release'}}})
    predictions=[]
    class FakeCPUStudent:
        available=True;artifact_hash=bridge.STUDENT_SHA256;unavailable_reason=None
        def __init__(self,settings,hardware):assert bridge.os.environ['CUDA_VISIBLE_DEVICES']==''
        def predict_profile_options(self,job,batches):
            predictions.append(job)
            return {spec['batch']:{'kind':'predicted_workload','artifact_sha256':bridge.STUDENT_SHA256,
                'hardware':'A10','batch_size':spec['batch'],'calibration':'uncalibrated_prior',
                'job_signature':job.packing.signature or job.job_id,'step_cost':100.,'epoch_cost':3200.,
                'avg_vram_mib':100.,'peak_vram_mib':800.}}
    monkeypatch.setattr(bridge,'MLVramPredictor',FakeCPUStudent)
    originals={str(p):p.read_bytes() for p in measured.rglob('*') if p.is_file()}
    output=tmp_path/'correction';neural=bridge.convert(settings,manifest_path,measured,output,source_commit='a'*40)
    assert len(predictions)==1 and neural['wall_multiplier']==pytest.approx(2.5)
    assert neural['measured_calibration_artifact_sha256']==table['artifact_sha256']
    assert neural['gpu_uuid']==host['gpu_uuid']
    assert all(bridge.Path(p).read_bytes()==raw for p,raw in originals.items())
    assert not any(p.is_dir() for p in output.iterdir()) #owned temp jobs removed
    proof=json.loads((output/'conversion-proof.json').read_text())
    assert proof['complete'] and proof['cpu_only'] and proof['selected_no_ballast_names']==[spec['name']]
    with pytest.raises(FileExistsError):bridge.convert(settings,manifest_path,measured,output,source_commit='a'*40)
