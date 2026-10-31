"""Hand-worked event timelines; no GPU, scheduler state or synthetic speed oracle."""
from copy import deepcopy

import pytest

from localml_scheduler.scheduler.feedback_objective import CompletionCosts, FeedbackRequest
from localml_scheduler.scheduler.continuation_forecast import (
    compare_continuations, compare_feedback_admission, forecast_continuation,
    reference_continuation,
)


def rollout(remaining, initial, queue=(), **kw):
    return forecast_continuation(remaining, initial, {j: 1. for j in remaining}, queue, **kw)


def test_concurrent_finish_refills_before_pack_drains():
    # a ends at2, c refills immediately; b has8 updates left. c ends5,
    # b receives3 measured updates during2..5 then finishes solo at10.
    r = rollout({'a': 2, 'b': 10, 'c': 3}, ('a', 'b'), ['c'], concurrency_limit=2,
                evidence={('a', 'b'): {'a': 1, 'b': 1}, ('b', 'c'): {'b': 1, 'c': 1}},
                memory={'a': 4, 'b': 4, 'c': 4}, memory_capacity=8)
    assert r['status'] == 'complete', r
    assert r['completions'] == {'a': 2, 'c': 5, 'b': 10}
    assert next(e['at'] for e in r['events'] if e['job'] == 'c' and e['event'] == 'start') == 2


def test_held_vram_until_release_delivery_overlaps_new_training():
    # a finishes training2, holds all VRAM3 more seconds, delivers10 later.
    # b starts5 and finishes9; a's caller receives at15.
    r = rollout({'a': 2, 'b': 4}, ('a',), ['b'],
        memory={'a': 8, 'b': 8}, memory_capacity=8,
        costs={'a': CompletionCosts(finish=3, delivery=10), 'b': CompletionCosts()})
    assert r['status'] == 'complete', r
    assert r['completions'] == {'b': 9, 'a': 15}
    assert [e for e in r['events'] if e['event'] == 'gpu_release' and e['job'] == 'a'][0]['at'] == 5
    assert [e for e in r['events'] if e['event'] == 'start'][0]['at'] == 5


def test_finish_while_other_job_starting_and_startup_progress_is_zero():
    r = rollout({'a': 2, 'b': 3}, ('a', 'b'),
        phase_state={'b': {'phase': 'starting', 'startup': 4}},
        costs={'a': CompletionCosts(delivery=10), 'b': CompletionCosts()})
    assert r['completions'] == {'b': 7, 'a': 12}
    assert [p['members'] for p in r['phases']] == [['a'], [], ['b'], []]


def test_finish_is_concurrent_not_serialized_and_zero_is_valid():
    r = rollout({'a': 0, 'b': 0}, ('a', 'b'),
        costs={'a': CompletionCosts(finish=3, delivery=7), 'b': CompletionCosts(finish=5)})
    assert r['status'] == 'complete'
    assert r['completions'] == {'b': 5, 'a': 10}


def test_unknown_surviving_subgroup_is_inconclusive_not_parent_rates():
    r = rollout({'a': 1, 'b': 4, 'c': 4}, ('a', 'b', 'c'),
        evidence={('a', 'b', 'c'): {'a': 1, 'b': 1, 'c': 1}})
    assert r['status'] == 'inconclusive'
    assert "unknown exact training group ('b', 'c')" == r['reason']
    assert set(r['completions']) == {'a'}
    assert all(p['source'] != 'assumed_parent_rates' for p in r['phases'])


def test_solo_predictions_are_never_used_as_packed_rates():
    r = rollout({'a': 1, 'b': 1}, ('a', 'b'))
    assert r['status'] == 'inconclusive' and not r['phases']


@pytest.mark.parametrize('record,reason', [
    ({'rates': {'a': 2}, 'identity': 'old'}, 'identity mismatch'),
    ({'rates': {'a': 2}, 'identity': 'recipe', 'observed_at': 1, 'max_age_seconds': 3}, 'stale evidence'),
    ({'rates': {'a': 2}, 'identity': 'recipe', 'valid': False, 'observed_at': 10, 'max_age_seconds': 0}, 'invalid evidence'),
])
def test_evidence_identity_age_and_validity(record, reason):
    r = rollout({'a': 2}, ('a',), evidence={('a',): record},
                context={'evidence_identity': 'recipe', 'snapshot_at': 10})
    assert r['status'] == 'inconclusive' and reason in r['reason']


def test_matching_measured_solo_precedes_predictor_and_records_provenance():
    r = rollout({'a': 6}, ('a',), evidence={('a',): {'rates': {'a': 2},
        'identity': 'recipe', 'observed_at': 8, 'max_age_seconds': 3}},
        context={'evidence_identity': 'recipe', 'snapshot_at': 10})
    assert r['completions'] == {'a': 3}
    assert r['phases'][0]['source'] == 'matching_exact_group_measurement'
    assert r['phases'][0]['identity'] == 'recipe'


def test_protected_job_drains_and_prevents_younger_refill():
    r = rollout({'a': 5, 'old': 2, 'young': 1}, ('a',), ['young', 'old'],
        protected=('old',), memory={'a': 2, 'old': 5, 'young': 2}, memory_capacity=6)
    assert r['status'] == 'complete', r
    assert r['completions'] == {'a': 5, 'old': 7, 'young': 8}
    assert [(e['job'], e['at']) for e in r['events'] if e['event'] == 'start'] == [('old', 5), ('young', 7)]


def test_oversized_protected_cannot_deadlock_other_work_and_stable_priority():
    pick = reference_continuation(['young', 'p2', 'huge', 'p1'], (),
        protected=('huge', 'p1', 'p2'), memory={'young': 1, 'p2': 1, 'huge': 9, 'p1': 1},
        memory_capacity=3, concurrency_limit=2, placement_allowed=lambda group: True)
    assert pick['selected'] == ('p1', 'p2')
    assert pick['blocked'][0] == {'job': 'huge', 'reason': 'solo_resource_or_compatibility_blocked'}


def test_reference_fallback_obeys_exact_compatibility_and_retry_restrictions():
    pick = reference_continuation(['old', 'young'], ('a',), protected=('old',),
        compatible=lambda group: 'young' not in group,
        placement_allowed=lambda group: len(group) == 1)
    assert pick['selected'] == ()
    assert pick['blocked'][0]['reason'] == 'protected_drain_to_legal_slot'


@pytest.mark.parametrize('kw,reason', [
    ({'candidate_count': 7}, 'six candidate limit'),
    ({'max_events': 1}, 'event limit'),
    ({'max_events': 129}, 'event limit'),
    ({'cpu_budget_ms': 51}, 'exceeds 50'),
])
def test_hard_limits_are_explicit_inconclusive(kw, reason):
    r = rollout({'a': 1}, ('a',), **kw)
    assert r['status'] == 'inconclusive' and reason in r['reason']


def test_actual_foreground_timeout_inert_clock():
    ticks = iter([0., 0.001, 0.051, 0.052])
    r = rollout({'a': 1}, ('a',), _clock=lambda: next(ticks))
    assert r['status'] == 'inconclusive' and r['reason'] == 'foreground CPU time limit'
    assert r['cpu_ms'] == 52


def args(policy='individual', **kw):
    req = ([FeedbackRequest(j, 'w', 'individual', (j,), 0) for j in ('a', 'b')]
           if policy == 'individual' else [FeedbackRequest('round', 'w', 'group', ('a', 'b'), 0)])
    a = dict(remaining={'a': 100., 'b': 1.}, solo={'a': 1., 'b': 1.},
        incumbents=('a',), candidate='b', queue=['b'],
        pre_halves={'a': [1., 1.]}, packed_halves={'a': [.5, .5], 'b': [.4, .4]},
        requests=req, completed={}, now=0.)
    a.update(kw)
    return a


def test_adapter_retains_v4_request_rule_halves_and_gain_threshold():
    r = compare_feedback_admission(**args())
    assert r['decision'] == 'accepted', r
    assert r['halves'][0]['gain_seconds'] == 48.625
    assert compare_feedback_admission(**args('group'))['decision'] == 'rejected'
    assert compare_feedback_admission(**args(min_gain_seconds=48.625))['decision'] == 'accepted'
    assert compare_feedback_admission(**args(packed_halves={'a': [.5, .001], 'b': [.4, .001]}))['decision'] == 'inconclusive'


def test_current_remaining_credits_trial_progress_and_preserves_original_clock():
    a = args(remaining={'a': 90., 'b': 1.}, now=10.)
    before = deepcopy(a)
    r = compare_feedback_admission(**a)
    h = r['halves'][0]
    assert h['defer']['job_completions'] == {'a': 100, 'b': 101}
    assert h['admit']['job_completions'] == {'a': 101.25, 'b': 12.5}
    assert h['admit']['requests'][0]['submitted_at'] == 0
    assert a == before


def test_future_pause_does_not_freeze_incumbent_or_charge_sunk_trial():
    r = compare_feedback_admission(**args(costs={'a': CompletionCosts(delivery=500),
        'b': CompletionCosts(pause=2, resume=3)}))
    h = r['halves'][0]
    assert h['defer']['job_completions'] == {'a': 600, 'b': 104}
    assert h['admit']['job_completions']['b'] == 2.5


def test_bracket_recovery_rule_is_original_regret_not_margin_stack():
    from localml_scheduler.scheduler.feedback_objective import compare_feedback_admission as legacy
    a = args('group', bracket_admission_mode='recovery_regret')
    r, old = compare_feedback_admission(**a), legacy(**a)
    assert r['decision'] == old['decision']
    for h, old_h in zip(r['halves'], old['halves']):
        assert h['decision_score_seconds'] == old_h['decision_score_seconds']
        assert h['no_recovery_scenario']['source'] == 'hypothesis_from_current_packed_rate_not_a_solo_measurement'


def test_zero_candidate_preserves_finishing_instead_of_pausing():
    r = compare_feedback_admission(**args(remaining={'a': 10, 'b': 0},
        costs={'a': CompletionCosts(), 'b': CompletionCosts(finish=2, delivery=4, pause=99)},
        memory={'a': 1, 'b': 1}, memory_capacity=2))
    assert r['error'] is None, r
    for h in r['halves']:
        assert h['admit']['job_completions']['b'] == h['defer']['job_completions']['b'] == 6
        assert not any(e['event'] == 'pause_release' for e in h['defer_trace']['events'])


def test_complete_cohort_not_silently_truncated_to_candidate_bound():
    remaining = {f'j{i}': 1 for i in range(9)}
    r = rollout(remaining, (), list(remaining), concurrency_limit=1, candidate_count=6)
    assert r['status'] == 'complete' and len(r['completions']) == 9
    a = args(requests=[FeedbackRequest('a', 'w', 'individual', ('a',), 0)])
    assert compare_feedback_admission(**a)['decision'] == 'inconclusive'


def test_group_bootstrap_scores_real_alternatives_without_preadd_halves():
    # Three equal jobs, pair/triple measurements genuinely supplied. Keep triple
    # at2updates/s => all finish1. Defer explicitly checkpoint b,c then solo FIFO
    # => a2,b4,c6. No fabricated incumbents/pre-add evidence.
    remaining = dict.fromkeys(('a', 'b', 'c'), 2)
    r = compare_continuations(remaining=remaining, solo=dict.fromkeys(remaining, 1),
        requests=[FeedbackRequest('round', 'w', 'group', tuple(remaining), 0)],
        completed={}, now=0, alternatives={
            'admit': {'initial': tuple(remaining), 'queue': []},
            'defer': {'initial': ('a',), 'queue': ['b', 'c'], 'concurrency_limit': 1}},
        evidence_halves=[{tuple(remaining): dict.fromkeys(remaining, 2)}] * 2)
    assert r['decision'] == 'accepted', r
    assert r['halves'][0]['gain_seconds'] == 5


def test_missing_refill_evidence_blocks_unknown_group_and_drains_to_solo():
    reqs = [FeedbackRequest(j, 'w', 'individual', (j,), 0) for j in ('a', 'b', 'c')]
    r = compare_feedback_admission(**args(remaining={'a': 10, 'b': 1, 'c': 1},
        solo={'a': 1, 'b': 1, 'c': 1}, queue=['b', 'c'], requests=reqs))
    assert r['error'] is None, r
    trace = r['halves'][0]['admit_trace']
    assert trace['status'] == 'complete'
    assert any(p['blocked_unknown'] for p in trace['placement_checks'])


def test_membership_generation_and_freshness_stamps_are_required_when_context_supplied():
    context = {'evidence_identity': {'hardware': 'A10', 'recipe': 'r'},
               'membership_generation': 4, 'snapshot_at': 10}
    record = {'rates': {'a': 2}, 'identity': context['evidence_identity'],
              'membership_generation': 3, 'observed_at': 10, 'max_age_seconds': 0}
    r = rollout({'a': 2}, ('a',), evidence={('a',): record}, context=context)
    assert 'membership generation mismatch' in r['reason']
    record['membership_generation'] = 4
    del record['observed_at']
    r = rollout({'a': 2}, ('a',), evidence={('a',): record}, context=context)
    assert 'freshness missing' in r['reason']
    record['observed_at'] = 10
    assert rollout({'a': 2}, ('a',), evidence={('a',): record}, context=context)['completions'] == {'a': 1}


def test_adapter_composes_caller_and_deferred_unchanged_pack_restrictions():
    calls = []
    def allowed(group):
        calls.append(group)
        return group != ('a', 'c')
    reqs = [FeedbackRequest(j, 'w', 'individual', (j,), 0) for j in ('a', 'b', 'c')]
    r = compare_feedback_admission(**args(remaining={'a': 10, 'b': 1, 'c': 1},
        solo={'a': 1, 'b': 1, 'c': 1}, queue=['b', 'c'], requests=reqs,
        packed_halves={'a': [1, 1], 'b': [1, 1]},
        evidence={('a', 'c'): {'a': 1, 'c': 1}}, placement_allowed=allowed))
    assert r['error'] is None, r
    assert ('a', 'c') in calls
    for half in r['halves']:
        for alternative in ('admit', 'defer'):
            trace = half[alternative + '_trace']
            c_start = next(e['at'] for e in trace['events'] if e['job'] == 'c' and e['event'] == 'start')
            assert c_start >= trace['completions']['a']
        b_start = next(e['at'] for e in half['defer_trace']['events'] if e['job'] == 'b' and e['event'] == 'resume')
        assert b_start >= half['defer_trace']['completions']['a']


def test_bootstrap_cleanup_pauses_all_then_preserves_original_fifo_and_progress():
    jobs = ('a', 'b', 'c')
    r = compare_continuations(remaining={'a': 3, 'b': 2, 'c': 1},
        solo=dict.fromkeys(jobs, 1), requests=[FeedbackRequest('g', 'w', 'group', jobs, 0)],
        completed={}, now=7,
        costs={j: CompletionCosts(pause=1, resume=2) for j in jobs},
        alternatives={
            'admit': {'initial': jobs, 'queue': []},
            'defer': {'initial': jobs, 'queue': [], 'queue_order': jobs,
                      'phase_state': {j: {'phase': 'pausing', 'pause': 1} for j in jobs},
                      'resumed': jobs, 'concurrency_limit': 1}},
        evidence_halves=[{jobs: dict.fromkeys(jobs, 1), ('a', 'b'): {'a': 1, 'b': 1}}] * 2)
    assert r['error'] is None, r
    assert r['halves'][0]['defer']['job_completions'] == {'a': 13, 'b': 17, 'c': 20}
    assert r['halves'][0]['admit']['job_completions'] == {'a': 10, 'b': 9, 'c': 8}


def test_default_fallback_does_not_invent_packed_evidence():
    assert reference_continuation(['b'], ('a',))['selected'] == ()
    assert reference_continuation(['a', 'b'], ())['selected'] == ('a',)


def test_missing_solo_estimate_can_use_matching_measured_singleton():
    r = forecast_continuation({'a': 3}, (), {}, ['a'], {('a',): {'a': 3}})
    assert r['status'] == 'complete' and r['completions'] == {'a': 1}


def test_compare_bound_counts_all_halves_in_one_foreground_budget():
    ticks = [0.]
    def clock():
        ticks[0] += .003
        return ticks[0]
    r = compare_feedback_admission(**args(), _clock=clock)
    assert r['decision'] == 'inconclusive'
    assert 'time limit' in r['error']
