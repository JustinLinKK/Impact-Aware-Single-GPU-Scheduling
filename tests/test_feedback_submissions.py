import pytest

from localml_scheduler.benchmarks.feedback_v4.submission_scenarios import (
    CASES, JOBS, POLICIES, SCENARIOS, simulate, verify,
)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('policy', POLICIES)
def test_closed_loop_completes_identical_work_and_obeys_feedback_dependencies(case, scenario, policy):
    r=simulate(scenario,policy,case)
    verify(r)
    assert r['metrics']['finished_jobs']==10
    assert r['metrics']['request_count']==(2 if scenario=='batch_barrier' else 10)
    for decision in r['decisions']:
        evidence=decision.get('evidence')
        if evidence:
            assert evidence['error'] is None
            for half in evidence['halves']:
                for action in ('defer','admit'):
                    assert all(r['arrivals'][j]<=decision['time']+1e-8 for j in half[action]['job_completions'])
    if scenario=='sequential_pipeline':
        assert r['metrics']['average_request_feedback_minutes']==pytest.approx(r['metrics']['average_job_jct_minutes'])
        assert len(set(r['arrivals'].values()))==10
        assert r['arrivals']['B1']<r['completion']['A1']  # Sequential does not mean one outstanding.
    else:
        values=[max(r['completion'][j+str(n)] for j in JOBS)-r['arrivals']['A'+str(n)] for n in (1,2)]
        assert r['metrics']['average_request_feedback_minutes']==pytest.approx(sum(values)/2)


def test_strictly_single_outstanding_has_no_packing_benefit():
    runs=[simulate('sequential_pipeline',p,strict_single=True) for p in POLICIES]
    for r in runs:
        assert r['arrivals']==runs[0]['arrivals']
        assert r['completion']==runs[0]['completion']
        assert r['metrics']==runs[0]['metrics']
        assert r['metrics']['trial_elapsed_minutes']==0


def test_stale_prediction_can_improve_individual_average_but_worsen_tail_and_workflow():
    fifo,v4=[simulate('sequential_pipeline',p,'slow_C_stale_prediction')['metrics'] for p in ('fixed_three_fifo','v4_request_feedback')]
    assert v4['average_request_feedback_minutes']<fifo['average_request_feedback_minutes']
    assert v4['p95_request_feedback_minutes']>fifo['p95_request_feedback_minutes']
    assert v4['workflow_makespan_minutes']>fifo['workflow_makespan_minutes']


def test_request_average_is_not_job_average_under_a_batch_barrier():
    r=simulate('batch_barrier','v4_request_feedback')
    assert r['metrics']['average_request_feedback_minutes']>r['metrics']['average_job_jct_minutes']


@pytest.mark.parametrize('case', CASES)
def test_bracket_optimization_leaves_pipeline_schedules_identical(case):
    old,new=[simulate('sequential_pipeline',p,case) for p in ('v4_previous','v4_request_feedback')]
    for key in ('completion','arrivals','metrics','segments','progress','events'):
        assert old[key]==new[key]
