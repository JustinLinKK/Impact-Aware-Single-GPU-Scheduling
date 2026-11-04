"""Scientific and operational invariants for a preparation-only package."""
from copy import deepcopy
import math
import pytest

from localml_scheduler.benchmarks.a10_v5_1_1.prepare import (
    build_plan, hardware_prior, validate_preparation,
)


@pytest.mark.parametrize('kind,cohorts,jobs', [
    ('screening', 48, 360), ('confirmation', 80, 576),
    ('confirmation_with_v5', 120, 864), ('development', 24, 192),
])
def test_declared_denominators_and_nonexecution(kind, cohorts, jobs):
    plan = build_plan(kind)
    report = validate_preparation(plan)
    assert report['denominators']['cohorts'] == cohorts
    assert report['denominators']['jobs'] == jobs
    assert report['valid_preparation'] and not report['launch_ready']
    assert not plan['execution_enabled'] and plan['pending_gates']


def test_history_persists_across_conditions_but_not_independent_traces():
    rows = [r for r in build_plan('screening')['rows'] if r['role'] == 'evaluation']
    namespaces = {r['history_namespace'] for r in rows}
    assert len(namespaces) == 4
    for namespace in namespaces:
        group = [r for r in rows if r['history_namespace'] == namespace]
        assert len(group) == 10
        assert {r['cycle'] for r in group} == {1, 2}
        assert len({r['condition'] for r in group}) == 5
        assert len({(r['mode'], r['trace_repeat']) for r in group}) == 1


def test_paired_jobs_identical_and_no_arm_history_leak():
    plan = build_plan('confirmation_with_v5')
    for i in range(0, len(plan['rows']), 3):
        block = plan['rows'][i:i + 3]
        assert {r['arm'] for r in block} == {'fixed_three', 'combined', 'v5_1_1'}
        assert all(r['jobs'] == block[0]['jobs'] for r in block)
        assert len({(r['mode'], r['condition'], r['cycle'], r['trace_repeat']) for r in block}) == 1
        assert all(r['history_namespace'] is None for r in block if r['arm'] != 'v5_1_1')


def test_learning_cycles_and_development_do_not_reuse_seeds():
    screen = build_plan('screening')
    seeds = [j['seed'] for r in screen['rows'] if r['jobs'] for j in r['jobs']]
    dev = [j['seed'] for r in build_plan('development')['rows'] for j in r['jobs']]
    assert len(seeds) == len(set(seeds)) == 288
    assert len(dev) == len(set(dev)) == 192
    assert set(seeds).isdisjoint(dev)


def test_anchors_are_not_independent_candidate_repeats_or_fabricated_bindings():
    plan = build_plan('screening')
    anchors = [r for r in plan['rows'] if r['role'] == 'stability_anchor']
    assert len(anchors) == 8
    assert all(r['baseline_manifest_sha256'] is None and r['jobs'] is None for r in anchors)
    assert all(r['history_namespace'] is None for r in anchors)
    assert plan['claim'] == 'historical_screen_only'


@pytest.mark.parametrize('mutation', ['execution', 'future_history', 'cap', 'drop_job', 'import_history', 'clock'])
def test_forged_preparation_rejected(mutation):
    plan = deepcopy(build_plan('confirmation'))
    if mutation == 'execution': plan['execution_enabled'] = True
    if mutation == 'future_history': plan['history']['future_observations_allowed'] = True
    if mutation == 'cap': plan['budget']['cap_usd'] = 31
    if mutation == 'drop_job': plan['rows'][0]['jobs'].pop()
    if mutation == 'import_history': plan['rows'][0]['history_import'] = ['old-evaluation']
    if mutation == 'clock': plan['calibration']['timestamps_may_be_redated'] = True
    with pytest.raises(ValueError): validate_preparation(plan)


@pytest.mark.parametrize('model,expected', [
    ('NVIDIA A10', 3), ('NVIDIA A100-SXM4-40GB', 5), ('Tesla A100-PCIE-80GB', 5),
    ('H100', 8), ('NVIDIA H100 80GB HBM3', 8), ('nvidia h100 pcie', 8),
    ('NVIDIA A10G', 3), ('NVIDIA H1000', 3), ('NVIDIA A100-SXM4-80GB MIG 1g.10gb', 3),
    ('unknown', 3),
])
def test_family_start_takes_precedence_over_precision_formula(model, expected):
    result = hardware_prior(gpu_name=model)
    assert result['alpha'] == expected
    assert not result['validated_optimum']
    if expected > 3:
        assert hardware_prior(gpu_name=model, device_tflops=19.5, a10_tflops=31.2,
                              bandwidth_gbs=1935, arithmetic_path_verified=True)['alpha'] == expected


def test_arithmetic_path_changes_hardware_prior():
    assert hardware_prior(device_tflops=125, a10_tflops=125, bandwidth_gbs=600,
                          arithmetic_path_verified=True)['alpha'] == 3
    a100_tensor = hardware_prior(device_tflops=312, a10_tflops=125, bandwidth_gbs=1935,
                                arithmetic_path_verified=True)
    assert a100_tensor['alpha'] == 5 and not a100_tensor['validated_optimum']
    assert hardware_prior(device_tflops=19.5, a10_tflops=31.2, bandwidth_gbs=1935,
                          arithmetic_path_verified=True)['alpha'] == 3
    assert hardware_prior()['alpha'] == 3


@pytest.mark.parametrize('invalid', [0, -1, math.nan, math.inf, True, None])
def test_no_fabricated_hardware_capacity(invalid):
    with pytest.raises(ValueError):
        hardware_prior(device_tflops=invalid, a10_tflops=125, bandwidth_gbs=600,
                       arithmetic_path_verified=True)


def test_finite_headroom_and_same_budget_period():
    plan = build_plan('confirmation')
    limits = plan['limits']
    assert sum(limits[k] for k in ['campaign_seconds', 'full_preparation_review_seconds',
                                  'recovery_seconds', 'cleanup_margin_seconds']) <= limits['maximum_new_lease_seconds']
    assert plan['budget']['cutover_unix'] == 1793488910.6600087
    assert plan['budget']['cap_usd'] == 30
    assert not plan['budget']['resource_reservation_created']
    assert plan['calibration']['block_arms'] == 2
    assert plan['calibration']['phase_ttl_seconds'] == 14400


def test_workload_change_requires_a_new_protocol():
    plan = build_plan('confirmation')
    changed = deepcopy(plan['blueprint'])
    changed['configurations']['light']['width'] += 1
    with pytest.raises(ValueError): build_plan('confirmation', changed)
    dev = build_plan('development')
    assert dev['development_shape_overrides']['width_by_configuration']['light'] == 288
    assert plan['development_shape_overrides'] is None
