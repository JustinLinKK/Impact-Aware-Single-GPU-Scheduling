from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import pytest
from localml_scheduler.domain import TrainingJob
from localml_scheduler.scheduler.experimental_v5 import update_state, V5Options
from localml_scheduler.scheduler.parallel_preparation import IsolatedPredictionBackend, ParallelPreparedPredictor
from localml_scheduler.benchmarks.a10_feedback_latency.calibration import RecordingPredictor


def job(name='fixture'):
    j=TrainingJob.create('fixture:runner','fixture','/tmp/unused',job_id=name)
    j.max_steps=64;j.max_epochs=2;j.config.max_epochs=2
    j.metadata.update(runtime_steps_per_epoch=32,runtime_total_optimizer_updates=64,runtime_planned_epochs=2)
    return j


def test_zero_updates_means_finishing_then_receipt_done_not_unknown():
    j=job();j.metadata['runtime_global_step']=64
    assert update_state(j)==('FINISHING',0)
    j.metadata['feedback_ready_at']='2026-10-31T00:00:00Z'
    assert update_state(j)==('DONE',0)
    j.metadata['runtime_global_step']=65
    assert update_state(j)==('UNKNOWN',None)


def test_missing_resumed_update_counter_is_unknown_not_zero_work():
    j=job();j.latest_checkpoint_path='/tmp/checkpoint'
    assert update_state(j)==('UNKNOWN',None)
    j.metadata['runtime_global_step']=3
    assert update_state(j)==('TRAINING',61)


def test_budget_identity_mismatch_never_becomes_valid_slope_work():
    j=job();j.metadata.update(runtime_global_step=3,runtime_total_optimizer_updates=63)
    assert update_state(j)==('UNKNOWN',None)


@pytest.mark.parametrize('kwargs',[{'continuation':1},{'forecast_cpu_ms':51},{'forecast_max_events':129},{'bracket_probe_seconds':float('nan')}])
def test_invalid_options_reject(kwargs):
    with pytest.raises(ValueError):V5Options(**kwargs)


def test_parallel_preparation_rejects_shared_mutable_engine():
    with pytest.raises(ValueError,match='process-isolated'):
        ParallelPreparedPredictor(SimpleNamespace(),workers=2)


def predict_fixture(spec,payload,batches):
    time.sleep(.04)
    return {b:dict(kind='predicted_workload',artifact_sha256='pinned',hardware='A10',
        batch_size=b,job_signature=payload['packing']['signature']or payload['job_id'],
        step_cost=10.,epoch_cost=320.,avg_vram_mib=100.,peak_vram_mib=150.,busy_indicator=20.)for b in batches},'ml_predictor',None


def test_two_async_consumers_publish_complete_profiles_and_never_infer_on_read():
    # Inert executor seam tests the parent coordination, not GPU or neural math.
    executor=ThreadPoolExecutor(max_workers=2)
    backend=IsolatedPredictionBackend(SimpleNamespace(artifact_hash='pinned',available=True),
        kind='ml_predictor',settings_dict={},calibration={},hardware=SimpleNamespace(hardware_key='h'),
        gpu_uuid='gpu',_executor=executor,_predict=predict_fixture)
    recording=RecordingPredictor(backend,lambda *args:None)
    prep=ParallelPreparedPredictor(recording,workers=2,hardware='A10')
    a,b=job('a'),job('b')
    try:
        assert not prep.ready(a,1)
        assert prep.enqueue(a,1) and prep.enqueue(b,1)
        deadline=time.monotonic()+2
        while not (prep.ready(a,1)and prep.ready(b,1))and time.monotonic()<deadline:time.sleep(.005)
        assert prep.ready(a,1)and prep.ready(b,1),prep.failures
        assert prep.predict_profile_options(a,[1])[1]['step_cost']==10.
        starts=[e for e in prep.events if e['phase']=='start']
        ends=[e for e in prep.events if e['phase']=='end']
        assert len(starts)==len(ends)==2
        assert max(e['monotonic_seconds']for e in starts)<min(e['monotonic_seconds']for e in ends)
    finally:
        assert prep.stop(timeout=2.)
