from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads, study
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE, SCHEMA


def plan():
    value = json.loads((workloads.HERE/'iteration-2-plan.json').read_text())
    value['deadline_utc'] = (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    return value


def table(manifest):
    # CPU-only fixture values, never presented as measured hardware evidence.
    value = dict(schema=SCHEMA, source=SOURCE,
        entries={s['template_key']: dict(updates_per_second=10.) for s in manifest['calibration']})
    value['artifact_sha256'] = workloads.digest(value)
    return value


def test_all_cells_and_calibration_only_budget_selection_are_frozen():
    p = plan()
    calibration = study.calibration_manifest(p)
    manifests = study.evaluation_manifests(p, table(calibration))
    assert len(calibration['calibration']) == 8
    assert all(s['epochs']*s['steps_per_epoch'] == 4096 for s in calibration['calibration'])
    assert len({s['template_key'] for s in calibration['calibration']}) == 4
    order = study.episode_order(p)
    assert len(order) == 64 and len(order)*6 == 384
    cells = [(r['condition'],r['mode'],r['arm']) for r in order]
    assert cells[:32] == list(reversed(cells[32:]))
    assert len(set(cells)) == 32
    calibration_seeds = {s[k] for s in calibration['calibration'] for k in ('seed','train_pool_seed','validation_pool_seed')}
    for name, manifest in manifests.items():
        target = manifest['condition']['target_training_seconds']
        assert len(manifest['jobs']) == 6
        for s in manifest['jobs']:
            assert s['epochs'] == 2 and s['epochs']*s['steps_per_epoch'] == target*10
            assert not any(s[k] in calibration_seeds for k in ('seed','train_pool_seed','validation_pool_seed'))
            assert s['template_key'] in table(calibration)['entries']
            assert s['expected_runtime_signature'] == workloads.expected_signature(s)
        assert all(d['selection_source'].startswith('dedicated calibration') for d in manifest['budget_decisions'].values())


def test_changed_configuration_requires_fresh_calibration():
    p = plan(); calibration = study.calibration_manifest(p)
    changed = deepcopy(p); changed['configurations']['tiny']['width'] = 192
    with pytest.raises(KeyError):study.evaluation_manifests(changed,table(calibration))
    bad = table(calibration);bad['entries'][next(iter(bad['entries']))]['updates_per_second'] = 99
    with pytest.raises(ValueError):study.evaluation_manifests(p,bad)


def test_planner_does_not_accept_incomplete_or_unbalanced_comparison():
    p = plan();p['modes']=['group']
    with pytest.raises(ValueError):study.episode_order(p)
