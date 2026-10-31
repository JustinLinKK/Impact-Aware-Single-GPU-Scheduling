"""Public parser and real released S_A10 source-capture regressions; CPU only."""
from copy import deepcopy
import json
import math
import os
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from localml_scheduler.benchmarks.a10_v5_ablation import job_builder as module
from localml_scheduler.benchmarks.a10_v5_ablation.protocol import make_spec
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import resource_config
from localml_scheduler.benchmarks.a10_feedback_latency.run import STUDENT_SHA256,build_job as baseline_build_job
from localml_scheduler.prediction.source_contract import WorkloadContract,ContractError,import_source_module,load_entry,build_adapter_batch
from localml_scheduler.prediction.ml_predictor import MLVramPredictor,released_source_request
from localml_scheduler.config import PredictionSettings
from localml_scheduler.hardware import HardwareProfile

RELEASE=Path('/mnt/c/Users/Justin/Documents/Codex/2026-10-27/ok/work/a10-latency-pilot/predictor-release')


def spec(config=None):
    return make_spec(config or resource_config('tiny'),name='sealed',role='calibration',chain=None,
        epochs=2,steps_per_epoch=1024,seed=731000,train_pool_seed=741000,validation_pool_seed=741001,
        source_path=workloads.HERE/'candidate.py')


def job(tmp_path,config=None):
    sealed=spec(config);manifest={'job_timeout_seconds':600,'calibration':[sealed],'jobs':[]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result,work=module.build_job(sealed,manifest_path=path,output=tmp_path/'v5',manifest=manifest,memory_budget_mib=20000.)
    return sealed,manifest,path,result,work


def test_v5_contract_fixes_real_parser_rejection_without_changing_executable_spec(tmp_path):
    sealed,manifest,path,current,work=job(tmp_path)
    old,_=baseline_build_job(sealed,manifest_path=path,output=tmp_path/'baseline',manifest=manifest,memory_budget_mib=20000.)
    request={'batch_size':sealed['batch'],'precision':sealed['precision'],'raw_steps':sealed['steps_per_epoch']}
    with pytest.raises(ContractError,match='workload_contract.version must equal'):
        WorkloadContract.parse(old.metadata['perfseer_model']['workload_contract'],request=request,metadata={})
    contract=current.metadata['perfseer_model']['workload_contract'];parsed=WorkloadContract.parse(contract,request=request,metadata={})
    assert parsed.mode=='candidate_adapter'and parsed.entry=='CandidateAdapter'
    assert contract['adapter']['context']==workloads.adapter_context(sealed)
    for key,value in workloads.recipe(sealed).prediction_training().items():assert contract['training'][key]==value
    assert contract['training']['backend']==module.TRAINING_BACKEND
    assert current.config.env['A10_PILOT_SPEC_JSON']==old.config.env['A10_PILOT_SPEC_JSON']
    assert current.metadata['resource_workload_spec']==old.metadata['resource_workload_spec']==sealed
    assert current.max_epochs==old.max_epochs==2 and current.max_steps==old.max_steps==2048
    assert current.config.runner_kwargs['batch_size']==old.config.runner_kwargs['batch_size']==sealed['batch']
    assert json.loads(current.config.env['A10_PILOT_SPEC_JSON'])['expected_runtime_signature']==sealed['expected_runtime_signature']
    assert released_source_request(current,sealed['batch'])['source_path']==sealed['source']
    from localml_scheduler.adapters.shared_training import _check_runtime_recipe
    workload=workloads.make_workload(sealed,tmp_path/'unchanged-workload')
    _check_runtime_recipe(workload,SimpleNamespace(job=current))


def test_contract_loads_actual_adapter_shapes_loss_and_exact_optimizer_on_cpu(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    sealed,_,_,current,_=job(tmp_path)
    raw=current.metadata['perfseer_model']['workload_contract']
    request={'batch_size':sealed['batch'],'precision':sealed['precision'],'raw_steps':sealed['steps_per_epoch']}
    contract=WorkloadContract.parse(raw,request=request,metadata={})
    imported=import_source_module(sealed['source'],import_roots=raw['import_roots'],entry=contract.entry)
    model,adapter,details=load_entry(imported,contract,torch=torch,request=request)
    batch,context=build_adapter_batch(adapter,model,details,torch=torch,max_input_numel=contract.max_input_numel,optimizer_spec=contract.optimizer)
    assert type(adapter)is workloads.CandidateAdapter
    assert batch['features'].shape==(sealed['batch'],sealed['input_features'])and batch['target'].shape==(sealed['batch'],1)
    assert batch['features'].dtype==batch['target'].dtype==torch.float32
    assert batch['features'].device.type==batch['target'].device.type=='cpu'
    loss=adapter.training_step(model,batch,context)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(model(batch['features']).float(),batch['target'].float())
    assert torch.equal(loss,expected)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation',['steps','source','signature'])
def test_bad_spec_cannot_be_hidden_by_metadata_wrapper(tmp_path,mutation):
    sealed=spec()
    if mutation=='steps':sealed['steps_per_epoch']+=1
    elif mutation=='source':sealed['source_sha256']='a'*64
    else:sealed['expected_runtime_signature']='b'*64
    with pytest.raises(ValueError):module.build_job(sealed,manifest_path=tmp_path/'manifest.json',output=tmp_path,
        manifest={'job_timeout_seconds':600},memory_budget_mib=20000.)


def test_both_v5_routes_import_same_amended_builder():
    from localml_scheduler.benchmarks.a10_v5_ablation import episode,calibrate_predictor
    assert episode.build_job is module.build_job
    assert calibrate_predictor.build_job is module.build_job


@pytest.mark.parametrize('label',['light','medium','compute'])
def test_actual_released_s_a10_cpu_source_capture_on_frozen_config(tmp_path,monkeypatch,label):
    supplied=os.environ.get('A10_V5_TEST_RELEASE')
    if not supplied:pytest.skip('set A10_V5_TEST_RELEASE for actual released inference CPU preflight')
    release=Path(supplied)
    assert (release/'checkpoints/S_A10/student.ts').is_file(),'explicit supplied public S_A10 release missing'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    blueprint=json.loads((Path(module.__file__).with_name('blueprint.json')).read_text())
    sealed,_,_,current,_=job(tmp_path,blueprint['configurations'][label])
    # Match the retained calibration bridge: optional compute/driver fields are
    # unknown rather than synthesized from the GPU product name.
    hardware=HardwareProfile('CPU-ONLY-contract-amendment','linux','NVIDIA A10',23028,None,None,torch.__version__)
    settings=PredictionSettings(mode='ml_predictor',student_release_path=str(release),
        student_model_path=str(release/'checkpoints/S_A10'),conversion_timeout_seconds=180,cpu_threads=4)
    predictor=MLVramPredictor(settings,hardware)
    assert predictor.available,predictor.unavailable_reason
    assert predictor.artifact_hash==STUDENT_SHA256
    profile=predictor.predict_profile_options(current,[sealed['batch']])[sealed['batch']]
    assert profile['artifact_sha256']==STUDENT_SHA256 and profile['hardware']=='A10'
    assert profile['batch_size']==sealed['batch']and profile['epoch_cost']==pytest.approx(profile['step_cost']*1024)
    assert all(math.isfinite(profile[k])and profile[k]>0 for k in ('step_cost','epoch_cost','avg_vram_mib','peak_vram_mib'))
    assert predictor._released.last_capture_audits[sealed['batch']]['numeric_parity']is True
    assert not torch.cuda.is_initialized()
