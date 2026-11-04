"""Prepare deterministic manifests; never execute, authorize, or reserve a GPU.

This is a protocol specification, not the V5.1.1 scheduler implementation.
Run as `python -m localml_scheduler.benchmarks.a10_v5_1_1.prepare --output DIR`.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from ...scheduler.hardware_concurrency import hardware_prior

BASE_SOURCE = '053f0a412e2060b1ba17b25226e4082205e1ae40'
BASE_BLUEPRINT = Path(__file__).parents[1] / 'a10_v5_1_ablation' / 'blueprint.json'
MODES = ('individual', 'group')
KINDS = ('screening', 'confirmation', 'confirmation_with_v5', 'development')
PENDING_GATES = (
    'v511_runtime_integration', 'source_package_and_execution_closure_review',
    'history_causality_and_fifo_replay', 'full_cpu_and_real_predictor_checks',
    'current_inventory_and_original_lease_headroom', 'live_same_period_budget',
    'exact_host_qualification_and_fresh_calibration', 'canonical_development_review',
    'baseline_compatibility_and_anchor_binding', 'block_width_freshness_adapter',
    'exclusive_dispatch_binding',
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _row(condition, mode, repeat, cycle, arm, index, *, development=False):
    base = 50_000_000 if development else 40_000_000
    seed = base + repeat * 100_000 + MODES.index(mode) * 10_000 + cycle * 1_000 + index * 100
    jobs = []
    for chain, label in enumerate(condition['classes']):
        for role_index, role in enumerate(('parent', 'child')):
            jobs.append({'chain': chain, 'role': role, 'configuration': label,
                         'seed': seed + 2 * chain + role_index, 'epochs': 2,
                         'successful_updates': 2 * condition['steps_per_epoch'][label]})
    trace = f'{"development" if development else "evaluation"}-{mode}-r{repeat}-{arm}'
    return {'id': f'{trace}-c{cycle}-{condition["name"]}', 'role': 'development' if development else 'evaluation',
            'arm': arm, 'mode': mode, 'trace_repeat': repeat, 'cycle': cycle,
            'condition': condition['name'], 'trace_id': trace,
            'history_namespace': trace if arm == 'v5_1_1' else None,
            'history_import': [], 'jobs': jobs,
            'profile_source': condition.get('profile_source', 'measured'),
            'phase': 'learning' if cycle == 1 else 'reuse',
            'seed_scope': 'prospective_disjoint_recipe_reuse',
            'shape_variant_development_fixture': bool(development and repeat == 2)}


def build_plan(kind, blueprint=None):
    if kind not in KINDS:
        raise ValueError('unknown preparation protocol')
    frozen = json.loads(BASE_BLUEPRINT.read_text())
    b = frozen if blueprint is None else deepcopy(blueprint)
    if b != frozen:
        raise ValueError('established workload blueprint must remain unchanged')
    development = kind == 'development'
    selected = [c for c in b['conditions'] if not development or c['name'] in
                {'short-mixed', 'backlog-six', 'published-pipeline'}]
    arms = ['v5_1_1'] if kind in {'screening', 'development'} else ['fixed_three', 'v5_1_1']
    if kind == 'confirmation_with_v5':
        arms.append('combined')
    rows = []
    for repeat in (1, 2):
        ordered = selected if repeat == 1 else list(reversed(selected))
        for mode in MODES:
            for cycle in (1, 2):
                for offset, condition in enumerate(ordered):
                    stable_index = next(i for i, c in enumerate(b['conditions']) if c['name'] == condition['name'])
                    shift = (repeat - 1 + MODES.index(mode) + cycle - 1 + offset) % len(arms)
                    for arm in arms[shift:] + arms[:shift]:
                        rows.append(_row(condition, mode, repeat, cycle, arm, stable_index,
                                         development=development))
    if kind == 'screening':
        anchors = []
        for stage in ('before', 'after'):
            for mode in MODES:
                for name in ('short-mixed', 'backlog-six'):
                    c = next(c for c in b['conditions'] if c['name'] == name)
                    anchors.append({'id': f'anchor-{stage}-{mode}-{name}', 'role': 'stability_anchor',
                        'stage': stage, 'arm': 'fixed_three', 'mode': mode, 'condition': name,
                        'trace_id': None, 'history_namespace': None, 'history_import': [],
                        'jobs': None, 'expected_jobs': 2 * len(c['classes']),
                        'seed_scope': 'bind_exact_original_anchor_manifest_before_dispatch',
                        'baseline_manifest_sha256': None})
        rows = [r for r in anchors if r['stage'] == 'before'] + rows + [r for r in anchors if r['stage'] == 'after']
    counts = Counter()
    job_counts = Counter()
    for row in rows:
        counts[row['role']] += 1
        job_counts[row['role']] += len(row['jobs']) if row['jobs'] is not None else row['expected_jobs']
    plan = {
        'schema': 'a10_v511_preparation_v1', 'version': '5.1.1', 'kind': kind,
        'status': 'design_prepared_runtime_not_implemented', 'execution_enabled': False,
        'base_source': BASE_SOURCE, 'blueprint_sha256': digest(b), 'blueprint': b,
        'development_shape_overrides': {'repeat': 2, 'width_by_configuration':
            {'light': 288, 'resident': 288, 'medium': 832, 'compute': 1920}}
            if development else None,
        'arms': arms, 'modes': list(MODES), 'independent_trace_repeats_per_mode': 2,
        'cycles_per_trace': 2, 'fifo': 'strict_acceptance_sequence',
        'history': {'cold_start': True, 'persist_across_episodes': True,
                    'reset_between_modes_repeats_and_arms': True, 'future_observations_allowed': False},
        'execution_order': 'paired_episode_interleaving' if kind.startswith('confirmation') else 'sequential_trace_episodes',
        'calibration': {'phase_ttl_seconds': 14400, 'margin_seconds': 300,
                        'block_arms': len(arms), 'refresh_between_blocks_only': True,
                        'timestamps_may_be_redated': False},
        'limits': {'job_seconds': 1800, 'episode_seconds': 3600,
                   'campaign_seconds': 70200 if kind.startswith('confirmation') else (5400 if development else 43200),
                   'full_preparation_review_seconds': 14400, 'recovery_seconds': 900,
                   'cleanup_margin_seconds': 900, 'maximum_new_lease_seconds': 86400},
        'budget': {'period': 2, 'cutover_unix': 1793488910.6600087, 'cap_usd': 30.,
                   'trigger_usd': 29.5, 'additional_period_created': False,
                   'authority_relative_to_task_root': 'work/a10-v5-redesign/budget.json',
                   'resource_reservation_created': False},
        'denominators': {'cohorts': len(rows), 'jobs': sum(job_counts.values()),
                         'cohorts_by_role': dict(counts), 'jobs_by_role': dict(job_counts)},
        'pending_gates': list(PENDING_GATES),
        'claim': 'historical_screen_only' if kind == 'screening' else ('development_only' if development else 'prospective_fresh_paired_full_policy'),
        'rows': rows,
    }
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_preparation(plan):
    if plan != build_plan(plan.get('kind'), plan.get('blueprint')):
        raise ValueError('prepared manifest differs from the deterministic sealed design')
    if plan['execution_enabled'] or not plan['pending_gates']:
        raise ValueError('CPU preparation cannot authorize GPU execution')
    ids = [r['id'] for r in plan['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate cohort identity')
    namespaces = {}
    for row in plan['rows']:
        if row['history_import']:
            raise ValueError('primary history must start empty')
        if row.get('history_namespace'):
            owner = (row['arm'], row['mode'], row['trace_repeat'])
            old = namespaces.setdefault(row['history_namespace'], owner)
            if old != owner:
                raise ValueError('history leaked across independent traces')
    return {'valid_preparation': True, 'launch_ready': False,
            'pending_gates': plan['pending_gates'], 'plan_sha256': plan['plan_sha256'],
            'denominators': plan['denominators']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for kind in KINDS:
        plan = build_plan(kind)
        reports[kind] = validate_preparation(plan)
        (args.output / f'{kind}.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    (args.output / 'validation.json').write_text(json.dumps(reports, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v['denominators'] for k, v in reports.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
