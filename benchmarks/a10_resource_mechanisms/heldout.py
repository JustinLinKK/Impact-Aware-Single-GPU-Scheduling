"""Predeclared third-iteration inputs; policy selection is always explicit.

This module only validates or constructs a candidate plan. It cannot launch,
select a policy from scores, or turn a candidate into a frozen deployment.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BLUEPRINT_SHA = 'a95a277a9c5969c9ec698d7bd85c67322f87818f523b4928abf3bee8e17962ce'
SELECTABLE_REVISED = {'resource_revised', 'resource_incremental'}


def blueprint():
    raw = (HERE / 'iteration-3-blueprint.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BLUEPRINT_SHA:
        raise ValueError('predeclared held-out blueprint changed')
    return json.loads(raw)


def validate_selection(selection):
    if not isinstance(selection, dict) or selection.get('arm') not in SELECTABLE_REVISED:
        raise ValueError('explicit supported revised policy selection required')
    if (selection.get('iteration2_valid_cohorts') != 64
            or selection.get('iteration2_usable_receipts') != 384
            or not isinstance(selection.get('rationale'), str)
            or not selection['rationale'].strip()
            or not re.fullmatch('[0-9a-f]{64}', str(selection.get('iteration2_review_sha256', '')))
            or not re.fullmatch('[0-9a-f]{40}', str(selection.get('selected_source_commit', '')))):
        raise ValueError('reviewed complete iteration2 and explicit source/rationale required')
    return selection


def validate_heldout_plan(plan):
    declared = blueprint()
    if plan.get('iteration') != 3 or plan.get('heldout_blueprint_sha256') != BLUEPRINT_SHA:
        raise ValueError('third iteration must bind its exact predeclared blueprint')
    for key in ('configurations', 'conditions', 'seed_base', 'modes', 'repeats',
                'calibration_replicas', 'calibration_steps_per_epoch', 'maximum_updates'):
        if plan.get(key) != declared[key]:
            raise ValueError('held-out inputs changed: ' + key)
    if plan.get('configuration_order') != list(declared['configurations']):
        raise ValueError('explicit predeclared calibration seed order required')
    selected = validate_selection(plan.get('policy_selection'))
    if plan.get('arms') != ['resource_v4', selected['arm'], 'resource_fixed_three']:
        raise ValueError('third iteration requires v4, selected revision and fixed-three')
    return plan


def candidate_plan(selection, *, deadline_utc, job_timeout_seconds,
                   episode_max_seconds, campaign_max_seconds):
    """Caller supplies a reviewed selection and finite future execution bounds.

    A review hash is an identity, not independent proof of its contents. The
    deployment handoff must verify that report, source and raw evidence later.
    """
    selected = validate_selection(selection)
    declared = blueprint()
    plan = {key: deepcopy(declared[key]) for key in ('configurations', 'conditions',
        'seed_base', 'modes', 'repeats', 'calibration_replicas',
        'calibration_steps_per_epoch', 'maximum_updates')}
    for value in (job_timeout_seconds, episode_max_seconds, campaign_max_seconds):
        if type(value) is not int or not 0 < value <= 14400:
            raise ValueError('explicit finite positive operational bounds required')
    if episode_max_seconds > campaign_max_seconds:
        raise ValueError('episode bound exceeds campaign bound')
    cutoff = datetime.fromisoformat(str(deadline_utc).replace('Z', '+00:00'))
    if cutoff.tzinfo is None or cutoff <= datetime.now(timezone.utc):
        raise ValueError('explicit future timezone-aware cutoff required')
    plan.update(schema='a10_resource_study_v1', iteration=3,
        status='candidate_protocol_pending_execution_freeze',
        heldout_blueprint_sha256=BLUEPRINT_SHA,
        configuration_order=list(declared['configurations']),
        policy_selection=deepcopy(selected),
        arms=['resource_v4', selected['arm'], 'resource_fixed_three'],
        deadline_utc=deadline_utc, job_timeout_seconds=job_timeout_seconds,
        episode_max_seconds=episode_max_seconds, campaign_max_seconds=campaign_max_seconds,
        measurement_scope=declared['scope'])
    return validate_heldout_plan(plan)
