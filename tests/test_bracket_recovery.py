from copy import deepcopy

import pytest

from localml_scheduler.config import FeedbackLatencySettings
from localml_scheduler.scheduler.feedback_objective import FeedbackRequest
from localml_scheduler.tests.test_feedback_objective import example


def test_recovery_regret_uses_bracket_maxima_and_reports_nominal_tradeoff():
    # Predicted B solo=1/s, measured packed=.4/s. Nominal bracket cost:
    # defer=101, admit=101.25. No recovery: defer=102.5, admit=101.25.
    r=example('group',bracket_admission_mode='recovery_regret')
    assert r['decision']=='accepted'
    assert r['decision_basis']=='bracket_recovery_regret_v1'
    for half in r['halves']:
        assert half['gain_seconds']==-.25
        assert half['no_recovery_scenario']['gain_seconds']==1.25
        assert half['admit_max_regret_seconds']==.25
        assert half['defer_max_regret_seconds']==1.25
        assert half['decision_score_seconds']==1.
    assert example('group',bracket_admission_mode='recovery_regret',min_gain_seconds=1.001)['decision']=='rejected'
    assert example('group',bracket_admission_mode='nominal')['decision']=='rejected'


def test_two_independent_halves_still_must_agree():
    r=example('group',bracket_admission_mode='recovery_regret',
              packed_halves={'a':[.5,.001],'b':[.4,.001]})
    assert r['decision']=='inconclusive' and r['error'] is None


def test_optimism_scenario_does_not_change_predictions_or_claim_solo_measurement():
    solo={'a':1.,'b':1.}
    frozen=deepcopy(solo)
    r=example('group',solo=solo,bracket_admission_mode='recovery_regret')
    assert solo==frozen
    scenario=r['halves'][0]['no_recovery_scenario']
    assert scenario['candidate_solo_updates_per_second']==.4
    assert scenario['source']=='hypothesis_from_current_packed_rate_not_a_solo_measurement'


def test_individual_and_mixed_requests_keep_the_nominal_policy():
    assert example(bracket_admission_mode='recovery_regret')==example()
    requests=[FeedbackRequest('bracket','w','group',('a','c'),0),
              FeedbackRequest('b','w','individual',('b',),0)]
    args=dict(requests=requests,remaining={'a':100.,'b':1.,'c':1.},
              solo={'a':1.,'b':1.,'c':1.},queue=['b','c'])
    assert example(**args,bracket_admission_mode='recovery_regret')==example(**args)


def test_nonoptimistic_prediction_collapses_to_one_scenario():
    args=dict(solo={'a':1.,'b':.4})
    nominal=example('group',**args)
    robust=example('group',**args,bracket_admission_mode='recovery_regret')
    assert robust['decision']==nominal['decision']
    assert robust['halves'][0]['decision_score_seconds']==nominal['halves'][0]['gain_seconds']


@pytest.mark.parametrize('change', [dict(solo={'a':1,'b':0}),dict(queue=[]),
    dict(remaining={'a':100,'b':0}),dict(packed_halves={'a':[.5,.5],'b':[float('nan'),.4]})])
def test_invalid_bracket_evidence_remains_inconclusive(change):
    assert example('group',**change,bracket_admission_mode='recovery_regret')['decision']=='inconclusive'


def test_bracket_mode_is_explicit_validated_and_reversible():
    assert FeedbackLatencySettings().bracket_admission_mode=='recovery_regret'
    assert FeedbackLatencySettings(bracket_admission_mode='nominal').to_dict()['bracket_admission_mode']=='nominal'
    with pytest.raises(ValueError):
        FeedbackLatencySettings(bracket_admission_mode='unknown')
    assert example('group',bracket_admission_mode='unknown')['decision']=='inconclusive'
