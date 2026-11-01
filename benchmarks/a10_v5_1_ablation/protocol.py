"""CPU-only freeze for three-arm, per-host matched V5.1 comparisons.

Evaluation update budgets are explicit GLOBAL blueprint inputs, never inferred
from host rates. Dedicated solo calibration only supplies common profiles and
phase costs. Neither this module nor a candidate plan authorizes GPU execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from hashlib import sha256
from pathlib import Path
import re

from ..a10_feedback_latency.run import atomic_json, check_deadline, file_hash, STUDENT_SHA256
from ..a10_resource_mechanisms import workloads
from ..a10_resource_mechanisms.profiles import SOURCE, SCHEMA

ARMS = ('fixed_three', 'combined', 'v5_1')
SCHEMA_PLAN = 'a10_v5_1_ablation_v1'
BLUEPRINT_FIELDS = ('configurations', 'conditions', 'seed_base', 'modes', 'repeats',
                    'arms', 'calibration_replicas', 'calibration_steps_per_epoch', 'maximum_updates')


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_plan(plan):
    if plan.get('schema') != SCHEMA_PLAN:
        raise ValueError('explicit v5.1 three-arm schema required')
    if (tuple(plan.get('arms', ())) != ARMS or plan.get('modes') != ['individual', 'group']
            or type(plan.get('repeats')) is not int or plan['repeats'] != 2):
        raise ValueError('all three exact arms and both modes in every repeated block required')
    if not re.fullmatch('[0-9a-f]{40}', str(plan.get('source_commit', ''))):
        raise ValueError('explicit source commit required')
    if not isinstance(plan.get('host_id'), str) or not plan['host_id'].strip():
        raise ValueError('explicit host identity label required; hosts are never pooled')
    blueprint = plan.get('blueprint')
    if not isinstance(blueprint, dict) or not _sha(plan.get('blueprint_sha256')):
        raise ValueError('frozen global blueprint and byte SHA required')
    for key in BLUEPRINT_FIELDS:
        if plan.get(key) != blueprint.get(key):
            raise ValueError('global blueprint changed: ' + key)
    configs, conditions = plan['configurations'], plan['conditions']
    if not isinstance(configs, dict) or not configs or not conditions:
        raise ValueError('declared configurations and conditions required')
    # JSON objects have no semantic key order. Seed assignment follows the
    # separately frozen list, including after canonical sort_keys serialization.
    order = plan.get('configuration_order')
    if (not isinstance(order, list) or any(not isinstance(k, str) for k in order)
            or len(order) != len(configs) or set(order) != set(configs)):
        raise ValueError('explicit calibration seed order required')
    if len({c['name'] for c in conditions}) != len(conditions):
        raise ValueError('condition names must be unique')
    for condition in conditions:
        if any(str(k).startswith('development') for k in condition):
            raise ValueError('development controls are forbidden in held-out protocol')
        name = condition['name']
        if not isinstance(name, str) or Path(name).name != name or name in {'', '.', '..'}:
            raise ValueError('condition must be an owned directory component')
        classes = condition['classes']
        if len(classes) not in (3, 6) or not set(classes) <= set(configs):
            raise ValueError('each condition requires three or six declared chains')
        steps = condition.get('steps_per_epoch')
        if isinstance(steps,dict) and set(steps)!=set(classes):
            raise ValueError('per-configuration budget mapping must exactly cover condition labels')
        for label in classes:
            budget=steps_for(condition,label)
            if type(budget)is not int or budget<=0 or 2*budget>plan['maximum_updates']:
                raise ValueError('explicit fixed global two-epoch budget required')
        if profile_source(condition) not in {SOURCE,'ml_predictor'}:
            raise ValueError('unsupported explicit profile source')
        if profile_source(condition)=='ml_predictor' and any(configs[label].get('ballast_mib',0)!=0 for label in classes):
            raise ValueError('neural pipeline control requires no-ballast configurations')
        if 'target_training_seconds' in condition:
            raise ValueError('host-duration matched evaluation budgets are forbidden')
    if (type(plan['calibration_replicas']) is not int or plan['calibration_replicas'] < 2
            or type(plan['calibration_steps_per_epoch']) is not int or plan['calibration_steps_per_epoch'] < 1024
            or type(plan['maximum_updates']) is not int or plan['maximum_updates'] < 2
            or type(plan['seed_base']) is not int or plan['seed_base'] < 0):
        raise ValueError('sustained replicated calibration and explicit budget/seed bounds required')
    limits={'job_timeout_seconds':3600,'episode_max_seconds':3600,
            'campaign_max_seconds':70200,'calibration_max_seconds':7200}
    for key,upper in limits.items():
        if type(plan.get(key)) is not int or not 0 < plan[key] <= upper:
            raise ValueError('finite positive operational bound exceeded: ' + key)
    if plan['episode_max_seconds'] > plan['campaign_max_seconds']:
        raise ValueError('episode bound exceeds campaign bound')
    cutoff = datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00'))
    if cutoff.tzinfo is None:
        raise ValueError('timezone-aware deadline required')
    return plan


def steps_for(condition,label):
    value=condition.get('steps_per_epoch')
    return value.get(label) if isinstance(value,dict) else value


def profile_source(condition):
    source=condition.get('profile_source','measured')
    return SOURCE if source=='measured' else source


def neural_digest(table):
    return sha256(json.dumps({k:v for k,v in table.items() if k!='calibration_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate_neural_table(table, measured_table):
    if (table.get('schema')!='a10_dedicated_calibration_v1'
            or table.get('calibration_sha256')!=neural_digest(table)
            or table.get('predictor_artifact_sha256')!=STUDENT_SHA256
            or table.get('hardware_key')!=measured_table['hardware_key']
            or table.get('gpu_uuid')!=measured_table['gpu_uuid']
            or table.get('measured_calibration_artifact_sha256')!=measured_table['artifact_sha256']):
        raise ValueError('neural calibration must bind released predictor and the same host/dedicated fold')
    return table


def candidate_plan(blueprint, *, blueprint_sha256, source_commit, host_id, deadline_utc,
                   job_timeout_seconds, episode_max_seconds, campaign_max_seconds,
                   calibration_max_seconds):
    plan = {k: deepcopy(blueprint[k]) for k in BLUEPRINT_FIELDS}
    plan.update(schema=SCHEMA_PLAN, status='candidate_pending_explicit_execution_freeze',
        blueprint=deepcopy(blueprint), blueprint_sha256=blueprint_sha256,
        source_commit=source_commit, host_id=host_id, configuration_order=list(blueprint['configurations']),
        deadline_utc=deadline_utc, job_timeout_seconds=job_timeout_seconds,
        episode_max_seconds=episode_max_seconds, campaign_max_seconds=campaign_max_seconds,
        calibration_max_seconds=calibration_max_seconds)
    return validate_plan(plan)


def make_spec(config, *, name, role, chain, epochs, steps_per_epoch, seed,
              train_pool_seed, validation_pool_seed, source_path):
    """Baseline spec construction with the sole extension chain IDs 0..5.

    The shared recipe, adapter, template, workload SHA and actual runtime
    signature validators remain unchanged. Chain number is not a recipe limit.
    """
    if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('name must be one owned directory component')
    if role not in {'calibration', 'parent', 'child'}:
        raise ValueError('invalid role')
    if role in {'parent', 'child'} and (type(chain) is not int or not 0 <= chain <= 5):
        raise ValueError('evaluation chain must be 0..5')
    if role == 'calibration' and chain is not None:
        raise ValueError('calibration has no evaluation chain')
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    forbidden = {'target_training_seconds', 'predicted_training_seconds', 'clipped',
                 'selection_source', 'budget_decision', 'outcome', 'policy_result'}
    if forbidden.intersection(config):
        raise ValueError('budget/outcome annotations belong outside workload specifications')
    spec = {**deepcopy(config), 'name': name, 'role': role, 'chain': chain,
            'epochs': epochs, 'steps_per_epoch': steps_per_epoch, 'seed': seed,
            'train_pool_seed': train_pool_seed, 'validation_pool_seed': validation_pool_seed,
            'source': str(source), 'source_sha256': workloads.file_hash(source),
            'model_family': config.get('model_family', workloads.MODEL_FAMILY),
            'workload_kind': 'synthetic_resource_control', **workloads.source_metadata()}
    for key in ('template_key', 'spec_sha256', 'expected_runtime_signature'):
        spec.pop(key, None)
    count = workloads.parameter_count(spec.get('width', 128), spec.get('input_features', 128))
    spec['parameter_count'] = spec['trainable_parameter_count'] = count
    workloads.recipe(spec)
    workloads.adapter_context(spec)
    spec['template_key'] = workloads.template_key(spec)
    spec['spec_sha256'] = workloads.spec_digest(spec)
    spec['expected_runtime_signature'] = workloads.expected_signature(spec)
    return spec


def _spec(plan, config, *, name, role, chain, steps_per_epoch, index, calibration):
    offset = plan['seed_base'] + (0 if calibration else 100000)
    return make_spec(config, name=name, role=role, chain=chain, epochs=2,
        steps_per_epoch=steps_per_epoch, seed=offset + index,
        train_pool_seed=offset + 10000 + 2 * index, validation_pool_seed=offset + 10001 + 2 * index,
        source_path=workloads.HERE / 'candidate.py')


def calibration_manifest(plan):
    validate_plan(plan)
    rows = []
    for label in plan['configuration_order']:
        for replica in range(plan['calibration_replicas']):
            rows.append(_spec(plan, plan['configurations'][label], name=f'{label}-cal-{replica}',
                role='calibration', chain=None, steps_per_epoch=plan['calibration_steps_per_epoch'],
                index=len(rows), calibration=True))
    return dict(jobs=[], calibration=rows, profile_source=SOURCE,
        deadline_utc=plan['deadline_utc'], job_timeout_seconds=plan['job_timeout_seconds'],
        prune_completed_weights=True, study_plan_sha256=workloads.digest(plan),
        experiment_source_commit=plan['source_commit'], blueprint_sha256=plan['blueprint_sha256'],
        source_runtime=workloads.source_metadata(), calibration_identity_schema='resource_solo_runtime_v1',
        scope='dedicated solo measured-template calibration common to all three arms on one host')


def validate_table(table, *, runtime=None):
    runtime = workloads.source_metadata() if runtime is None else runtime
    identity = table.get('source_identity') or {}
    if (table.get('schema') != SCHEMA or table.get('source') != SOURCE
            or table.get('artifact_sha256') != workloads.digest({k:v for k,v in table.items() if k != 'artifact_sha256'})
            or not table.get('hardware_key') or not table.get('gpu_uuid')
            or identity.get('calibration_identity_schema') != 'resource_solo_runtime_v1'
            or identity.get('source_runtime') != runtime or identity.get('torch_version') != runtime['torch_version']):
        raise ValueError('intact same-source/runtime dedicated solo table required')
    return table


def evaluation_manifests(plan, table, neural_table=None):
    validate_plan(plan)
    validate_table(table)
    if any(profile_source(c)=='ml_predictor' for c in plan['conditions']):
        if neural_table is None:raise ValueError('explicit --neural-calibration required for neural conditions')
        validate_neural_table(neural_table,table)
    output = {}
    index = 0
    for condition in plan['conditions']:
        jobs = []
        for role in ('parent', 'child'):
            for chain, label in enumerate(condition['classes']):
                spec = _spec(plan, plan['configurations'][label], name=f'{role}-{chain}-{label}',
                    role=role, chain=chain, steps_per_epoch=steps_for(condition,label),
                    index=index, calibration=False)
                index += 1
                if spec['template_key'] not in table['entries']:
                    raise ValueError('missing common calibration template: ' + label)
                jobs.append(spec)
        source=profile_source(condition)
        artifact=table['artifact_sha256'] if source==SOURCE else neural_table['calibration_sha256']
        output[condition['name']] = dict(jobs=jobs, calibration=[], profile_source=source,
            deadline_utc=plan['deadline_utc'], job_timeout_seconds=plan['job_timeout_seconds'],
            prune_completed_weights=True, study_plan_sha256=workloads.digest(plan),
            experiment_source_commit=plan['source_commit'], blueprint_sha256=plan['blueprint_sha256'],
            calibration_sha256=artifact, calibration_file_scope='one host; common to every arm',
            condition=deepcopy(condition), source_runtime=workloads.source_metadata(),
            evaluation_budget_source='explicit global blueprint; independent of host rates',
            scope='synthetic scheduler mechanism controls; no neural accuracy or counterfactual trace claim')
    return output


def episode_order(plan):
    validate_plan(plan)
    # Every contiguous block has all three arms on the same host/condition/mode.
    # Reverse ordering in alternate repeats, retaining paired cell identity.
    rows = []
    for repeat in range(1, plan['repeats'] + 1):
        reverse = repeat % 2 == 0
        conditions = list(reversed(plan['conditions'])) if reverse else plan['conditions']
        modes = list(reversed(plan['modes'])) if reverse else plan['modes']
        arms = list(reversed(plan['arms'])) if reverse else plan['arms']
        for condition in conditions:
            for mode in modes:
                block = f"{plan['host_id']}:{condition['name']}:{mode}:r{repeat}"
                for arm in arms:
                    rows.append(dict(host_id=plan['host_id'], block=block,
                        condition=condition['name'], repeat=repeat, mode=mode, arm=arm,
                        chains=len(condition['classes']), expected_jobs=2 * len(condition['classes'])))
    return rows


def seal_evaluation_bundle(plan_path, calibration_path, settings_path, output, neural_calibration=None):
    """Seal actual runtime/spec/source bytes before any policy evaluation."""
    from .episode import source_snapshot
    plan_path, calibration_path, settings_path, output = map(Path, (plan_path, calibration_path, settings_path, output))
    plan, table = json.loads(plan_path.read_text()), json.loads(calibration_path.read_text())
    validate_plan(plan)
    if plan.get('status') != 'frozen':
        raise ValueError('explicitly frozen plan required')
    check_deadline(plan)
    neural_table=json.loads(Path(neural_calibration).read_text()) if neural_calibration is not None else None
    manifests = evaluation_manifests(plan, table, neural_table)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    pins, snapshots = {}, {}
    for name, manifest in manifests.items():
        path = output / (name + '.json')
        atomic_json(path, manifest)
        pins[name] = file_hash(path)
        snapshots[name] = source_snapshot(manifest, path, source_commit=plan['source_commit'])
    order = episode_order(plan)
    atomic_json(output / 'protocol-before.json', dict(schema='a10_v5_1_protocol_v1',
        plan=plan, plan_sha256=file_hash(output / 'plan.json'), blueprint_sha256=plan['blueprint_sha256'],
        settings_sha256=file_hash(settings_path), calibration_sha256=file_hash(calibration_path),
        calibration_artifact_sha256=table['artifact_sha256'],
        neural_calibration_sha256=file_hash(neural_calibration) if neural_calibration is not None else None,
        neural_calibration_artifact_sha256=neural_table['calibration_sha256'] if neural_table is not None else None,
        source_runtime=workloads.source_metadata(),
        source_runtime_sha256=workloads.digest(workloads.source_metadata()), source_pins=snapshots,
        source_commit=plan['source_commit'], host_id=plan['host_id'], order=order, manifests=pins,
        expected_cohorts=len(order), expected_jobs=sum(e['expected_jobs'] for e in order),
        frozen_before_evaluation=True))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['calibration-manifest', 'evaluation-manifests'])
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--neural-calibration',type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.output.exists(): raise FileExistsError('never overwrite a sealed protocol')
    if args.stage == 'calibration-manifest':
        validate_plan(plan)
        if plan.get('status') != 'frozen': raise ValueError('explicitly frozen plan required')
        check_deadline(plan)
        atomic_json(args.output, calibration_manifest(plan))
    else:
        if args.calibration is None or args.settings is None:
            raise ValueError('--calibration and --settings required')
        seal_evaluation_bundle(args.plan, args.calibration, args.settings, args.output, args.neural_calibration)


if __name__ == '__main__': main()
