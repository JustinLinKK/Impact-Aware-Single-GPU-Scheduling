from copy import deepcopy
from types import SimpleNamespace

import pytest

from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import (
    SOURCE, MeasuredTemplatePredictor, duration_matched_budget, fit_templates,
)
from localml_scheduler.prediction import JobPredictionError
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import make_spec, resource_config


def spec():
    return make_spec(resource_config('tiny'), name='test', role='parent', chain=0,
        epochs=2, steps_per_epoch=64, seed=41, train_pool_seed=101, validation_pool_seed=211,
        source_path=workloads.HERE / 'candidate.py')


def key():
    return spec()['template_key']


def row():
    return dict(role='calibration', complete=True, hardware_key='host', gpu_uuid='gpu',
        job_signature='calibration-job', solo_process_isolation_verified=True,
        template_key=key(), successful_updates=32, observed_at='2026-10-30T09:00:00Z',
        windows=[dict(start_step=2,end_step=10,steps=8,elapsed_seconds=2.,excludes_nonupdate_phases=True),
                 dict(start_step=12,end_step=28,steps=16,elapsed_seconds=2.,excludes_nonupdate_phases=True)],
        peak_reserved_mib=1200., peak_allocated_mib=1000., peak_device_used_mib=1400.,
        steady_gpu_busy_percent=[20.,40.], steady_allocated_mib=[700.,900.],
        phase_seconds={k:1. for k in ('startup','resume','pause','validation','checkpoint','finalization','delivery')})


def fit(rows=None):
    return fit_templates(rows or [row()], hardware_key='host', gpu_uuid='gpu', source_identity='source')


def test_rate_is_total_successful_updates_over_total_window_time():
    entry=fit()['entries'][key()]
    assert entry['updates_per_second']==6.  #24 /4; not unweighted mean of wall times.
    assert entry['mean_allocated_mib']==800.
    assert entry['peak_reserved_mib']==1200.
    assert entry['admission_reservation_mib']==1528.
    assert entry['gpu_busy_percent']==30.
    budget=duration_matched_budget(entry,10.)
    assert budget['total_updates']==64 and budget['steps_per_epoch']==32
    assert not budget['clipped']
    assert duration_matched_budget(entry,1000.,maximum_updates=128)['clipped']


@pytest.mark.parametrize('mutation', [
    lambda r:r.update(role='evaluation'),
    lambda r:r.update(gpu_uuid='other'),
    lambda r:r.update(solo_process_isolation_verified=False),
    lambda r:r['windows'][1].update(start_step=9,steps=19),
    lambda r:r['windows'][0].update(elapsed_seconds=float('nan')),
    lambda r:r['windows'][0].update(excludes_nonupdate_phases=False),
    lambda r:r.update(steady_allocated_mib=[1001.]),
])
def test_invalid_or_leaked_calibration_is_rejected(mutation):
    value=row();mutation(value)
    with pytest.raises(ValueError):fit([value])


def test_provider_is_explicit_and_rejects_unknown_configuration_or_host():
    table=fit();hardware=SimpleNamespace(hardware_key='host')
    predictor=MeasuredTemplatePredictor(table,hardware,gpu_uuid='gpu')
    job=SimpleNamespace(job_id='eval',packing=SimpleNamespace(signature='eval-signature'),
        metadata={'resource_template_key':key(),'resource_workload_spec':spec()})
    profile=predictor.predict_profile_options(job,[32])[32]
    assert profile['profile_source']==SOURCE and predictor.last_sources['eval']==SOURCE
    assert profile['step_cost']==1000/6 and profile['epoch_cost']==64*1000/6
    assert profile['avg_vram_mib']==800 and profile['peak_vram_mib']==1528
    with pytest.raises(JobPredictionError):predictor.predict_profile_options(job,[64])
    job.metadata['resource_template_key']='b'*64
    with pytest.raises(JobPredictionError):predictor.predict_profile_options(job,[32])
    with pytest.raises(ValueError):MeasuredTemplatePredictor(table,hardware,gpu_uuid='other')
    changed=deepcopy(table);changed['entries'][key()]['updates_per_second']=1000
    with pytest.raises(ValueError):MeasuredTemplatePredictor(changed,hardware,gpu_uuid='gpu')


@pytest.mark.parametrize('mutation', [
    lambda j:j.metadata['resource_workload_spec'].update(width=256),
    lambda j:j.metadata['resource_workload_spec'].update(steps_per_epoch=128),
    lambda j:j.metadata['resource_workload_spec'].update(expected_runtime_signature='stale'),
    lambda j:j.metadata.update(resource_template_key='b'*64),
])
def test_same_job_mutation_cannot_reuse_a_measured_template(mutation):
    from localml_scheduler.benchmarks.a10_feedback_latency.async_preparation import PreparedPredictor
    underlying=MeasuredTemplatePredictor(fit(),SimpleNamespace(hardware_key='host'),gpu_uuid='gpu')
    job=SimpleNamespace(job_id='eval',packing=SimpleNamespace(signature='unchanged'),
        metadata={'resource_template_key':key(),'resource_workload_spec':spec()},
        config=SimpleNamespace())
    prepared=PreparedPredictor(underlying,asynchronous=False,profile_source=SOURCE)
    try:
        prepared.enqueue(job,32)
        assert prepared.ready(job,32)
        mutation(job)
        assert not prepared.ready(job,32)
        with pytest.raises(JobPredictionError):prepared.predict_profile_options(job,[32])
        with pytest.raises(JobPredictionError):underlying.predict_profile_options(job,[32])
    finally:
        prepared.stop()
