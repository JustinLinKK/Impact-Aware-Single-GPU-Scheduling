"""The causal ablation covers every common-profile treatment in paired order."""
from collections import Counter
from localml_scheduler.benchmarks.a10_feedback_latency.campaign import episode_order
from localml_scheduler.benchmarks.a10_feedback_latency.run import ARMS, CAUSAL_ARMS


def test_causal_order_contains_complete_balanced_comparison():
    rows = episode_order(CAUSAL_ARMS)
    assert len(rows) == 20
    assert Counter((r['arm'], r['mode']) for r in rows) == {
        (arm, mode): 2 for arm in CAUSAL_ARMS for mode in ('individual', 'group')}
    first = [(r['mode'], r['arm']) for r in rows if r['repeat'] == 1]
    second = [(r['mode'], r['arm']) for r in rows if r['repeat'] == 2]
    assert second == first[::-1]
    assert len(episode_order()) == 16
    assert set(r['arm'] for r in episode_order()) == set(ARMS)
