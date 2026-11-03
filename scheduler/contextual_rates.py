"""Bounded empirical rate envelopes; no offline interference model or fitting."""
import math
import time
from .workload_fingerprint import structural_distances, peer_distance, log_distance


def quantile(values, q):
    values = sorted(values)
    total = sum(w for _, w in values)
    at = 0.
    for x, weight in values:
        at += weight
        if at >= total * q: return x
    return values[-1][0]


def reference_generation(profile):
    # The released student weights can stay fixed while measured scale/phase
    # calibration changes. Its weight hash is not that reference generation.
    return profile.get('calibration_sha256') or profile.get('artifact_sha256')


def estimate(store, fingerprint, peers, profile, *, now, deadline, allow_old=False):
    """Every returned reference is available before `now` and trace-local.

    Raw rates are explicitly re-expressed against the target's present solo
    reference. This preserves measurements rather than transplanting a stale
    calibrated residual. Exact matches still receive the 20% empirical margin.
    """
    reference = 1000. / profile['step_cost']
    exact, approximate, rejected = [], [], {}
    candidates = store.candidates(now, deadline=deadline)
    # Exact evidence already dominates approximate matches below. Identify it
    # before expensive graph distances, preserving eligibility and envelopes.
    exact_candidates = []
    for row in candidates:
        if time.perf_counter() >= deadline: raise TimeoutError('history matching elapsed budget')
        if (fingerprint['exact'] == row['fingerprint']['exact'] and peers['key'] == row['context']['key']
                and (now - row['observed_unix'] <= 14400 or allow_old)):
            exact_candidates.append(row)
    for row in exact_candidates or candidates:
        if time.perf_counter() >= deadline:
            raise TimeoutError('history matching elapsed budget')
        age = now - row['observed_unix']
        if age > 14400 and not allow_old:
            rejected['older_than_phase_ttl'] = rejected.get('older_than_phase_ttl', 0) + 1
            continue
        try:
            graph, shape = structural_distances(fingerprint, row['fingerprint'])
            solo = log_distance([reference, profile['peak_vram_mib']],
                [row['reference_rate'], row['reference_memory']])
            peer = peer_distance(peers, row['context'])
        except (ValueError, KeyError):
            rejected['incompatible'] = rejected.get('incompatible', 0) + 1
            continue
        distances = [graph, shape, solo, peer]
        match = fingerprint['exact'] == row['fingerprint']['exact'] and peers['key'] == row['context']['key']
        if not match and row['reference_generation'] != reference_generation(profile):
            rejected['approximate_cross_generation_without_reference'] = rejected.get(
                'approximate_cross_generation_without_reference', 0) + 1
            continue
        if not match and (max(distances) > .25 or sum(distances) / 4 > .15):
            rejected['distance'] = rejected.get('distance', 0) + 1
            continue
        similarity = 1. if match else 1. - sum(distances) / 4
        value = {'row': row, 'distances': distances, 'distance': sum(distances) / 4,
                 'weight': similarity / (1. + age / 3600.)}
        (exact if match else approximate).append(value)
    selected = exact or approximate
    selected.sort(key=lambda x: (x['distance'], -x['row']['observed_unix'], x['row']['aggregate_key']))
    # Never give multiple aggregates from one episode multiple independent votes.
    unique = {}
    for value in selected:
        unique.setdefault(value['row']['episode'], value)
    selected = list(unique.values())[:16]
    if not selected or (not exact and len(selected) < 3):
        raise ValueError('insufficient compatible independent episode history')
    residuals = [(math.log(v['row']['raw_rate'] / (reference if exact else v['row']['reference_rate'])),
                  v['weight']) for v in selected]
    mid = reference * math.exp(quantile(residuals, .5))
    lower = min(mid * .8, reference * math.exp(quantile(residuals, .1)) * .8)
    upper = max(mid * 1.2, reference * math.exp(quantile(residuals, .9)) * 1.2)
    return {'rate': mid, 'lower': lower, 'upper': upper, 'exact': bool(exact),
        'episodes': len(selected), 'rejected': rejected,
        'basis': ('exact raw observed rates re-expressed against present reference' if exact else
                  'similar-workload residuals under the same calibration generation') + '; empirical, not calibrated confidence',
        'neighbors': [{'aggregate_key': v['row']['aggregate_key'], 'episode': v['row']['episode'],
            'observed_unix': v['row']['observed_unix'], 'distances': v['distances'],
            'weight': v['weight'], 'original_reference_generation': v['row']['reference_generation'],
            'original_reference_rate': v['row']['reference_rate'], 'present_reference_rate': reference,
            'raw_rate': v['row']['raw_rate']} for v in selected]}
