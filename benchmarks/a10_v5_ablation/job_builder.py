"""V5-only prediction-contract amendment; the executable workload stays intact.

The historical builder's synthetic metadata contains only adapter.context and
training. That is enough for its shared-runner consistency check but is not a
valid source-capture contract. Supply the complete public candidate_adapter v1
contract here for both V5 calibration and evaluation; never guess a module loss
or bypass the released predictor's contract validation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ...adapters.shared_training import CONTRACT as TRAINING_BACKEND
from ...domain import BatchResolution
from ...prediction.source_contract import CONTRACT_VERSION,WorkloadContract
from ..a10_feedback_latency.run import build_job as baseline_build_job
from ..a10_resource_mechanisms import workloads


def workload_contract(spec):
    recipe=workloads.recipe(spec)
    context=workloads.adapter_context(spec)
    source_root=Path(workloads.__file__).resolve().parents[3]
    contract={'version':CONTRACT_VERSION,'mode':'candidate_adapter','entry':'CandidateAdapter',
        'import_roots':[str(source_root)],'constructor_args':[],'constructor_kwargs':{},
        'adapter':{'context':context,'scenario':{'batch_size':recipe.batch_size},
            'batch_spec':{'features':{'shape':['$batch',context['input_features']],'dtype':'float32'},
                          'target':{'shape':['$batch',1],'dtype':'float32'}}},
        'loss':{'kind':'candidate_adapter_training_step'},
        'training':{**recipe.prediction_training(),'backend':TRAINING_BACKEND}}
    # Validate using the actual public parser, not a wrapper-specific imitation.
    WorkloadContract.parse(contract,request={'batch_size':recipe.batch_size,
        'precision':recipe.precision,'raw_steps':recipe.steps_per_epoch},metadata={})
    return contract


def build_job(spec,*,manifest_path,output,manifest,memory_budget_mib):
    if spec.get('workload_kind')!='synthetic_resource_control':
        raise ValueError('V5 builder requires the explicitly sealed synthetic workload')
    if (spec.get('spec_sha256')!=workloads.spec_digest(spec)
            or spec.get('expected_runtime_signature')!=workloads.expected_signature(spec)):
        raise ValueError('frozen source/spec/runtime signature mismatch')
    job,work=baseline_build_job(spec,manifest_path=manifest_path,output=output,
        manifest=manifest,memory_budget_mib=memory_budget_mib)
    if BatchResolution.resolved_batch_size(job)!=spec['batch']:
        raise ValueError('physical batch differs from frozen workload')
    metadata=deepcopy(job.metadata['perfseer_model'])
    metadata.update(source_path=str((Path(manifest_path).parent/spec['source']).resolve()),
        workload_contract=workload_contract(spec))
    job.metadata['perfseer_model']=metadata
    return job,work
