"""Source-bound V5.1.1 screening/development manifests, never cloud operations."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re

from .prepare import build_plan, validate_preparation, digest
from ..a10_v5_1_ablation import protocol as base
from ..a10_resource_mechanisms import workloads
from ..a10_resource_mechanisms.profiles import SOURCE
from ..a10_feedback_latency.run import atomic_json, file_hash

AMENDMENT = {
    'schema': 'v511_execution_detail_v1',
    'new_policy_thresholds_fit_to_results': False,
    'warm_initial_prior': 'one_adjacent_step_from_same_ordered_prepared_three_job_prefix_with_prior_two_windows_at_actual_occupancy;4h_max;no_unresolved_exploration',
    'demotion_rule': 'factor_1.5_rate_miss_in_two_consecutive_valid_windows',
    'memory_margin': 'max_256MiB_or_10pct_plus_predicted_peak_or_larger_measured_startup',
    'fingerprint_capture': 'isolated_CPU_real_training_graph_after_acceptance_charged_to_request',
    'incomplete_dependency_cache': 'no_cache_hit; frozen_harness_execution_closure_required',
    'development_calibration_jobs': 16, 'evaluation_calibration_jobs_per_generation': 8,
    'extra_development_calibration_reason': 'repeat_two_width_variants_need_their_own_real_templates',
    'full_preparation_review_seconds': 21600,
    'development_campaign_seconds': 7200,
    'development_bound_reason': 'retain full next-episode headroom across all24 development cohorts',
    'episode_bounds_unchanged': True, 'weights_workloads_evaluation_seeds_unchanged': True,
}


def execution_plan(kind, *, source_commit, host_id, deadline_utc, anchors=None):
    if kind not in ('screening', 'development') or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('explicit supported screen/development and source required')
    prepared = build_plan(kind); validate_preparation(prepared)
    if not host_id or datetime.fromisoformat(deadline_utc.replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('host and finite timezone-aware cutoff required')
    anchors = deepcopy(anchors or {})
    expected = {r['id'] for r in prepared['rows'] if r['role'] == 'stability_anchor'}
    if set(anchors) != expected:
        raise ValueError('bind every fresh anchor to actual reviewed original recipe')
    for key, value in anchors.items():
        row = next(r for r in prepared['rows'] if r['id'] == key)
        if (not re.fullmatch('[0-9a-f]{64}', value.get('manifest_sha256', ''))
                or not re.fullmatch('[0-9a-f]{64}', value.get('terminal_audit_sha256', ''))
                or len(value.get('jobs', [])) != row['expected_jobs']):
            raise ValueError('reviewed terminal baseline manifest/audit and jobs required')
        for job in value['jobs']:
            if job.get('spec_sha256') != workloads.spec_digest(job):
                raise ValueError('anchor original workload identity changed')
    plan = {'schema': 'a10_v511_execution_v1', 'prepared': prepared,
        'source_commit': source_commit, 'host_id': host_id, 'deadline_utc': deadline_utc,
        'anchors': anchors, 'execution_amendment': AMENDMENT,
        'bounds': {**prepared['limits'], 'full_preparation_review_seconds': 21600,
                   'campaign_seconds': 7200 if kind == 'development' else 43200},
        'stage': kind, 'status': 'frozen_requires_parent_development_gate'}
    plan['sha256'] = digest(plan)
    return plan


def validate_plan(plan):
    expected = execution_plan(plan['stage'], source_commit=plan['source_commit'], host_id=plan['host_id'],
                              deadline_utc=plan['deadline_utc'], anchors=plan['anchors'])
    if plan != expected:
        raise ValueError('execution plan differs from frozen deterministic specification')
    return plan


def calibration_plan(plan, *, variant=False, generation=0):
    validate_plan(plan)
    if type(generation) is not int or not 0 <= generation < 64:
        raise ValueError('bounded calibration generation required')
    b = deepcopy(plan['prepared']['blueprint'])
    # Disjoint measured data. None of these seeds appear in traces or anchors.
    b['seed_base'] = 60_000_000 + generation * 1000 + (500_000 if variant else 0)
    if variant:
        if plan['stage'] != 'development': raise ValueError('variants are development only')
        for label, width in plan['prepared']['development_shape_overrides']['width_by_configuration'].items():
            b['configurations'][label]['width'] = width
    return base.candidate_plan(b, blueprint_sha256=digest(b), source_commit=plan['source_commit'],
        host_id=plan['host_id'], deadline_utc=plan['deadline_utc'], job_timeout_seconds=1800,
        episode_max_seconds=3600, campaign_max_seconds=plan['bounds']['campaign_seconds'], calibration_max_seconds=3600)


def row_manifest(plan, row, measured, neural, *, history_root):
    validate_plan(plan); base.validate_table(measured); base.validate_neural_table(neural, measured)
    if row not in plan['prepared']['rows']: raise ValueError('row absent from frozen trace')
    variant = row.get('shape_variant_development_fixture', False)
    configs = deepcopy(plan['prepared']['blueprint']['configurations'])
    if variant:
        for label, width in plan['prepared']['development_shape_overrides']['width_by_configuration'].items():
            configs[label]['width'] = width
    condition = next(c for c in plan['prepared']['blueprint']['conditions'] if c['name'] == row['condition'])
    specs = []
    if row['role'] == 'stability_anchor':
        for original in plan['anchors'][row['id']]['jobs']:
            spec = deepcopy(original)
            # Paths relocate, while complete executable recipe and seeds stay
            # identical. Hashes that bind paths are recomputed transparently.
            spec['source'] = str(workloads.HERE / 'candidate.py')
            spec['source_sha256'] = workloads.file_hash(spec['source'])
            spec['spec_sha256'] = workloads.spec_digest(spec)
            spec['expected_runtime_signature'] = workloads.expected_signature(spec)
            specs.append(spec)
    else:
        for job in row['jobs']:
            specs.append(base.make_spec(configs[job['configuration']],
                name=f"{job['role']}-{job['chain']}-{job['configuration']}", role=job['role'], chain=job['chain'],
                epochs=2, steps_per_epoch=job['successful_updates'] // 2, seed=job['seed'],
                train_pool_seed=job['seed'] + 10_000_000, validation_pool_seed=job['seed'] + 20_000_000,
                source_path=workloads.HERE / 'candidate.py'))
    # Arrival loop must enqueue all parents before any child is eligible.
    specs.sort(key=lambda j: (j['role'] != 'parent', j['chain']))
    if any(s['template_key'] not in measured['entries'] for s in specs):
        raise ValueError('fresh matching calibrated configuration missing')
    source = 'ml_predictor' if row.get('profile_source') == 'ml_predictor' else SOURCE
    result = dict(jobs=specs, calibration=[], profile_source=source, deadline_utc=plan['deadline_utc'],
        job_timeout_seconds=1800, prune_completed_weights=True, experiment_source_commit=plan['source_commit'],
        condition=condition, source_runtime=workloads.source_metadata(), study_plan_sha256=plan['sha256'],
        blueprint_sha256=plan['prepared']['blueprint_sha256'],
        calibration_sha256=neural['calibration_sha256'] if source == 'ml_predictor' else measured['artifact_sha256'],
        evaluation_budget_source='explicit frozen prospective trace, not observed host performance',
        scope='V5.1.1 historical screening; fresh anchors do not make historical outcomes paired repeats')
    if row['arm'] == 'v5_1_1':
        path = Path(history_root).resolve() / row['history_namespace']
        result['v511_trace'] = {'episode': row['id'], 'history_namespace': row['history_namespace'],
                              'history_path': str(path), 'history_import': []}
    if plan['stage'] == 'development':
        result.update(development=True, development_scope='separate diagnostic seeds, no outcome tuning')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['calibration-manifest'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', action='store_true')
    parser.add_argument('--generation', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('no calibration manifest overwrite')
    plan = json.loads(args.plan.read_text())
    atomic_json(args.output, base.calibration_manifest(calibration_plan(plan, variant=args.variant, generation=args.generation)))


if __name__ == '__main__': main()
