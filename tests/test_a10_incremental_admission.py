from copy import deepcopy

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.policy import conservative_admission
from localml_scheduler.benchmarks.a10_resource_mechanisms.policy import incremental_admission
from localml_scheduler.scheduler.feedback_objective import CompletionCosts, FeedbackRequest


@pytest.mark.parametrize('mode', ['individual', 'group'])
def test_historical_reserve_changes_only_posttrial_margin(mode):
    requests=([FeedbackRequest(j,'workflow','individual',(j,),0.) for j in ('a','b')]
              if mode=='individual' else [FeedbackRequest('r','workflow','group',('a','b'),0.)])
    args=dict(remaining={'a':30.,'b':30.},solo={'a':1.,'b':1.},incumbents=('a',),
        candidate='b',queue=['b'],pre_halves={'a':[1.,1.]},
        packed_halves={'a':[.9,.9],'b':[.9,.9]},requests=requests,completed={},now=0.,
        costs={j:CompletionCosts(startup=1.,finish=1.,delivery=.2,pause=1.,resume=1.) for j in ('a','b')})
    original=deepcopy(args)
    old=conservative_admission(**args,measured_reserve_seconds=40.)
    new=incremental_admission(**args,measured_reserve_seconds=40.)
    assert old['decision']=='rejected' and new['decision']=='accepted'
    assert old['required_gain_seconds']==41. and new['required_gain_seconds']==1.
    assert old['worst_gain_seconds']==new['worst_gain_seconds']
    assert [s['solo_hypothesis'] for s in old['scenarios']]==[s['solo_hypothesis'] for s in new['scenarios']]
    assert args==original  #remaining updates and future costs were not rewritten.


def test_missing_trial_cost_evidence_still_fails_closed():
    assert incremental_admission(measured_reserve_seconds=0.)['decision']=='inconclusive'
