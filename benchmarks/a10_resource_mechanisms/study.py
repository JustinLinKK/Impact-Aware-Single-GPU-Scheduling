"""Freeze designed-workload budgets from dedicated calibration before evaluation.

No GPU work is performed by this planner. Runtime-bound specifications are
generated on the evaluation host so their actual torch/source identities match.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from ..a10_feedback_latency.run import RESOURCE_ARMS, atomic_json, check_deadline, file_hash
from . import workloads
from .manifest import make_spec
from .profiles import SOURCE, SCHEMA, duration_matched_budget


def validate_plan(plan):
    if plan.get('schema') != 'a10_resource_study_v1':
        raise ValueError('explicit resource study schema required')
    if plan.get('iteration') == 3:
        from .heldout import validate_heldout_plan
        validate_heldout_plan(plan)
    elif plan.get('iteration') != 2:
        raise ValueError('only declared resource iterations two and three are supported')
    check_deadline(plan)
    configs, conditions = plan['configurations'], plan['conditions']
    if not configs or not conditions or len({c['name'] for c in conditions}) != len(conditions):
        raise ValueError('unique named conditions required')
    if (not plan['arms'] or len(set(plan['arms'])) != len(plan['arms'])
            or not set(plan['arms']) <= set(RESOURCE_ARMS)
            or plan.get('repeats') != 2 or plan.get('modes') != ['individual', 'group']):
        raise ValueError('two balanced repeats, both modes and explicit supported arms required')
    for condition in conditions:
        if Path(condition['name']).name != condition['name'] or condition['name'] in {'.', '..'}:
            raise ValueError('condition name must be a directory component')
        if len(condition['classes']) != 3 or not set(condition['classes']) <= set(configs):
            raise ValueError('each condition requires three declared configurations')
        if condition['target_training_seconds'] <= 0:
            raise ValueError('positive calibration-based target duration required')
    if plan['calibration_replicas'] < 2 or plan['calibration_steps_per_epoch'] < 1024:
        raise ValueError('multiple sustained calibration replicas required')
    return plan


def _spec(plan, config, *, name, role, chain, steps_per_epoch, index, calibration):
    # Same configuration seeds/budgets may differ, but calibration and evaluation
    # have disjoint initialization/data seed ranges. No evaluation metric enters.
    offset = plan['seed_base'] + (0 if calibration else 100000)
    return make_spec(config, name=name, role=role, chain=chain, epochs=2,
        steps_per_epoch=steps_per_epoch, seed=offset + index,
        train_pool_seed=offset + 10000 + 2*index,
        validation_pool_seed=offset + 10001 + 2*index,
        source_path=workloads.HERE / 'candidate.py')


def calibration_manifest(plan):
    validate_plan(plan)
    rows = []
    labels = plan.get('configuration_order', list(plan['configurations']))
    for label in labels:
        config = plan['configurations'][label]
        for replica in range(plan['calibration_replicas']):
            rows.append(_spec(plan, config, name=f'{label}-cal-{replica}', role='calibration',
                chain=None, steps_per_epoch=plan['calibration_steps_per_epoch'],
                index=len(rows), calibration=True))
    return dict(jobs=[], calibration=rows, profile_source=SOURCE,
        deadline_utc=plan['deadline_utc'], job_timeout_seconds=plan['job_timeout_seconds'],
        prune_completed_weights=True, study_plan_sha256=workloads.digest(plan),
        source_runtime=workloads.source_metadata(),
        scope='synthetic training controls; resource labels are hypotheses until measured')


def evaluation_manifests(plan, table):
    validate_plan(plan)
    if (table.get('schema') != SCHEMA or table.get('source') != SOURCE
            or table.get('artifact_sha256') != workloads.digest(
                {k:v for k,v in table.items() if k != 'artifact_sha256'})):
        raise ValueError('intact dedicated calibration table required')
    output = {}
    for ci, condition in enumerate(plan['conditions']):
        jobs, decisions = [], {}
        for role in ('parent', 'child'):
            for chain, label in enumerate(condition['classes']):
                name = f'{role}-{chain}-{label}'
                index = ci*6 + len(jobs)
                prototype = _spec(plan, plan['configurations'][label], name=name, role=role,
                    chain=chain, steps_per_epoch=32, index=index, calibration=False)
                entry = table['entries'][prototype['template_key']]
                budget = duration_matched_budget(entry, condition['target_training_seconds'],
                    epochs=2, maximum_updates=plan['maximum_updates'])
                spec = _spec(plan, plan['configurations'][label], name=name, role=role,
                    chain=chain, steps_per_epoch=budget['steps_per_epoch'], index=index, calibration=False)
                if spec['template_key'] != prototype['template_key']:
                    raise ValueError('work budget unexpectedly changed template configuration')
                jobs.append(spec)
                decisions[name] = {**budget, 'configuration_label': label,
                    'template_key': prototype['template_key'], 'calibration_artifact': table['artifact_sha256']}
        output[condition['name']] = dict(jobs=jobs, calibration=[], profile_source=SOURCE,
            deadline_utc=plan['deadline_utc'], job_timeout_seconds=plan['job_timeout_seconds'],
            prune_completed_weights=True, study_plan_sha256=workloads.digest(plan),
            calibration_sha256=table['artifact_sha256'], condition=deepcopy(condition),
            budget_decisions=decisions, source_runtime=workloads.source_metadata(),
            scope='same-configuration measured template scheduler controls; not neural accuracy')
    return output


def episode_order(plan):
    validate_plan(plan)
    conditions = [c['name'] for c in plan['conditions']]
    return [dict(condition=condition, repeat=repeat+1, mode=mode, arm=arm)
        for repeat in range(2)
        for condition in (conditions if repeat == 0 else list(reversed(conditions)))
        for mode in (plan['modes'] if repeat == 0 else list(reversed(plan['modes'])))
        for arm in (plan['arms'] if repeat == 0 else list(reversed(plan['arms'])))]


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['calibration-manifest', 'evaluation-manifests'])
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.output.exists():
        raise FileExistsError('never overwrite a sealed manifest bundle')
    if args.stage == 'calibration-manifest':
        atomic_json(args.output, calibration_manifest(plan))
    else:
        if args.calibration is None:
            raise ValueError('--calibration required')
        table = json.loads(args.calibration.read_text())
        manifests = evaluation_manifests(plan, table)
        args.output.mkdir(parents=True)
        # Preserve the exact plan bytes for independent protocol verification.
        (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
        pins = {}
        for name, manifest in manifests.items():
            path = args.output / (name + '.json')
            atomic_json(path, manifest)
            pins[name] = file_hash(path)
        order = episode_order(plan)
        atomic_json(args.output / 'protocol-before.json', dict(order=order,
            manifests=pins, plan=plan, plan_sha256=file_hash(args.plan),
            calibration_sha256=file_hash(args.calibration), expected_cohorts=len(order),
            expected_jobs=6*len(order), frozen_before_evaluation=True))


if __name__ == '__main__':
    main()
