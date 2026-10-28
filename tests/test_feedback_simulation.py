"""Outcome regression plus independent work/memory accounting for the fluid model."""
import pytest

from localml_scheduler.benchmarks.feedback_v4.simulate import CASES, POLICIES, simulate, verify


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('policy', POLICIES)
def test_five_jobs_complete_without_lost_trials_or_memory_overcommit(policy, case):
    result=simulate(policy,case)
    verify(result)
    assert result['metrics']['finished_jobs']==5
    assert result['metrics']['p95_jct_minutes']==result['metrics']['makespan_minutes']
    if policy.startswith('v4_'):
        for decision in result['decisions']:
            evidence=decision.get('evidence')
            if evidence:
                assert evidence['admission_policy']=='derived_solo_slope_feedback_v4'
                assert evidence['error'] is None


@pytest.mark.parametrize('policy,case,mean,makespan', [
    ('fixed_three_fifo','reference',84.3888888889,97.2222222222),
    ('v3_slowdown50','reference',70.158,93.17),
    ('v4_individual','reference',53.5357777778,78.8744444444),
    ('fixed_three_fifo','slow_C_stale_prediction',88.6296296296,98.7037037037),
    ('v3_slowdown50','slow_C_stale_prediction',74.392,98.56),
    ('v4_individual','slow_C_stale_prediction',67.4391111111,148.3911111111),
    ('v4_individual','slow_C_accurate_prediction',65.8,97.),
])
def test_reproduces_prior_five_job_reference_and_counterexample(policy,case,mean,makespan):
    result=simulate(policy,case)
    assert result['metrics']['mean_jct_minutes']==pytest.approx(mean)
    assert result['metrics']['makespan_minutes']==pytest.approx(makespan)


def test_mixed_feedback_can_regress_despite_improved_job_mean_and_makespan():
    v4=simulate('v4_mixed','slow_C_accurate_prediction')['metrics']
    v3=simulate('v3_slowdown50','slow_C_accurate_prediction')['metrics']
    assert v4['mean_jct_minutes']<v3['mean_jct_minutes']
    assert v4['makespan_minutes']<v3['makespan_minutes']
    assert v4['mixed_mean_feedback_minutes']>v3['mixed_mean_feedback_minutes']
