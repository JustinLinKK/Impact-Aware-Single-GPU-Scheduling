"""A FIFO control must not silently reorder when only a later job fits."""
from localml_scheduler.scheduler.ready_set import Costs, choose


def test_fifo_cannot_bypass_an_infeasible_head_when_idle():
    reports = {k: dict(status='estimated',source_sha256=k,
        cpu_profiles={str(c):dict(runtime_seconds=100.,planning_memory_mib=memory,
                                 gpu_busy_fraction=.5,cpu_pressure_lower_bound=.4)
                      for c in (2,3,4,6)}) for k,memory in [('head',20000.),('tail',1000.)]}
    costs = Costs(dict(profiles=[],co_runs=[]),reports)
    options = dict(accepted={'head':0.,'tail':1.},now=2.,used_mib=600.,capacity_mib=14670.)
    for mode in ('fifo','v511'):
        assert choose(['head','tail'],{},costs,policy=mode,**options)['admit']==[]
    assert choose(['head','tail'],{},costs,policy='lookahead_drain',**options)['admit']==['tail']
