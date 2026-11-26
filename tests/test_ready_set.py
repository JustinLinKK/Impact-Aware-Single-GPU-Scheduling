"""CPU-only contract tests; synthetic costs are not performance evidence."""
import copy
import math
from unittest.mock import patch
import pytest
from localml_scheduler.scheduler import ready_set as rs
from localml_scheduler.scheduler.joint_resource_dispatch import allocate_counts


def profile(signature, duration=100., sensitivity=.5, gpu=.5, memory=2000.):
    return dict(status='estimated',source_sha256=signature,
                cpu_profiles={str(c):dict(runtime_seconds=duration*(6/c)**sensitivity,
                    planning_memory_mib=memory,gpu_busy_fraction=gpu,
                    cpu_pressure_lower_bound=.4) for c in (2,3,4,6)})


def fixture(joint=False):
    reports={str(i):profile(str(i),duration=40+20*i,gpu=.1+.15*i) for i in range(5)}
    return rs.Costs(dict(profiles=[],co_runs=[]),reports,joint_cpu=joint)


def choose(ready, remaining=None, costs=None, **kwargs):
    return rs.choose(ready,remaining or {},costs or fixture(),
                     accepted={k:0. for k in ready},now=1.,used_mib=600.,
                     capacity_mib=14672.,**kwargs)


def test_only_arrived_jobs_can_launch():
    d=choose(['1','3'])
    assert set(d['admit'])<= {'1','3'}
    assert not d['future_arrivals_used']


def test_residents_cannot_be_preempted_to_make_a_better_pair():
    d=choose(['2','3','4'],{'0':.8,'1':.5})
    assert {'0','1'} <= set(d['cpu_counts'])
    assert sum(d['cpu_counts'].values())==6 and len(d['cpu_counts'])<=3


def test_non_job_memory_reservation_cannot_disappear():
    reports={k:profile(k,memory=4782.) for k in ['a','b','c']}
    c=rs.Costs(dict(profiles=[],co_runs=[]),reports)
    d=choose(['c'],{'a':.5,'b':.5},costs=c,policy='fifo',external_floor_mib=1024.)
    assert not d['admit']


def test_unknown_resident_keeps_exclusive_execution():
    c=fixture();c.reports['u']=dict(status='unknown',source_sha256='u')
    d=choose(['0','1'],{'u':.5},costs=c)
    assert not d['admit']


def test_unknown_tail_does_not_turn_known_pool_into_exclusive_fifo():
    c=fixture();c.reports['u']=dict(status='unknown',source_sha256='u')
    d=choose(['0','1','2','3','4','u'],costs=c)
    assert d['forecasts'] and d['unknown_deferred']==['u']
    assert 'u' not in d['admit']


def test_age_override_requires_explicit_finite_scope():
    with pytest.raises(ValueError):choose(['0'],max_wait=None)
    assert choose(['0'],max_wait=None,finite_cohort=True)['fairness_mode']=='finite_cohort_drain'


def test_open_queue_protects_aged_unknown_job():
    c=fixture();c.reports['u']=dict(status='unknown',source_sha256='u')
    d=rs.choose(['0','u'],{},c,accepted={'0':200.,'u':0.},now=220.,used_mib=600.,capacity_mib=14672.)
    assert d['admit']==['u'] and d['cpu_counts']=={'u':6}


def test_resource_diversity_sees_novel_tail_signature():
    reports={str(i):profile(str(i)) for i in range(20)}
    reports['tail']=profile('tail',duration=400.,gpu=.95)
    c=rs.Costs(dict(profiles=[],co_runs=[]),reports)
    selected=rs.representatives(list(reports),c,8)
    assert {'0','1','2','tail'}<=set(selected) and len(selected)==8


def test_duplicate_source_instances_do_not_collapse():
    reports={'a':profile('same'),'b':profile('same')}
    c=rs.Costs(dict(profiles=[],co_runs=[]),reports)
    rates,evidence=c.rates(['a','b'])
    assert set(rates)=={'a','b'} and 'extrapolated' in evidence


def test_joint_cpu_partitions_do_not_increase_total_cpu():
    c=fixture(joint=True)
    assert list(c.allocations(['a','b']))==[{'a':3,'b':3},{'a':2,'b':4},{'a':4,'b':2}]
    assert list(c.allocations(['a','b','c']))==[{'a':2,'b':2,'c':2}]


def test_actual_rate_cache_separates_asymmetric_allocations():
    c=fixture(joint=True)
    a=c.rates(['0','1'],{'0':2,'1':4})[0]
    b=c.rates(['0','1'],{'0':4,'1':2})[0]
    assert a!=b and a==c.rates(['0','1'],{'0':2,'1':4})[0]


def test_reallocation_preserves_disjoint_ownership():
    plan=allocate_counts(range(6),{'a':[0,1,2],'b':[3,4,5]},{'a':4,'b':2})
    assert len(plan['a'])==4 and len(plan['b'])==2
    assert not set(plan['a'])&set(plan['b']) and set(plan['a']+plan['b'])==set(range(6))


def test_completed_better_option_is_retained_when_later_rollout_times_out():
    values=iter([1e6]*3+[0.]*3)
    def forecast(*args,**kwargs):
        try:return next(values)
        except StopIteration:raise TimeoutError('fixture planning limit')
    with patch.object(rs,'rollout',side_effect=forecast):
        d=choose(['0','1','2','3','4'])
    assert d['planning_budget_exhausted'] and len(d['forecasts'])==2
    assert d['admit']==d['forecasts'][1]['admit']


def test_finite_forecast_drains_all_distinct_instances():
    c=fixture(joint=True);ready=['0','1','2','3','4'];work={};done=set()
    for _ in range(10):
        d=choose(ready,work,costs=c,max_wait=None,finite_cohort=True)
        before=set(work)|set(d['admit'])
        rates,_=c.rates(list(work)+d['admit'],d['cpu_counts'])
        ready,work,dt=rs.advance(ready,work,d['admit'],rates)
        assert math.isfinite(dt) and dt>0
        done.update(before-set(work))
        if not ready and not work:break
    assert done=={'0','1','2','3','4'}
