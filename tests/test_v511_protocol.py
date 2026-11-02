"""Prospective recipe, calibration, finite-bound and evidence replay regressions."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from localml_scheduler.benchmarks.a10_v5_1_1 import protocol as p, campaign as c
from localml_scheduler.benchmarks.a10_v5_1_1.replay import history_audit
from localml_scheduler.tests.test_v511_runtime import fingerprint
from localml_scheduler.scheduler.fifo_experience import ExperienceStore
from localml_scheduler.scheduler.workload_fingerprint import context


def plan():
    return p.execution_plan('development', source_commit='a'*40, host_id='host-v511',
                            deadline_utc='2099-01-01T00:00:00Z')


def test_execution_roundtrip_preserves_all_development_recipes():
    value = json.loads(json.dumps(plan(), sort_keys=True))
    assert p.validate_plan(value) == value
    assert value['prepared']['denominators']['cohorts'] == 24
    assert value['prepared']['denominators']['jobs'] == 192
    assert len({r['history_namespace'] for r in value['prepared']['rows']}) == 4


@pytest.mark.parametrize('mutation', ['threshold', 'seed', 'order', 'cutoff', 'budget'])
def test_execution_mutations_fail_closed(mutation):
    value = deepcopy(plan())
    if mutation == 'threshold': value['execution_amendment']['demotion_rule'] = 'tuned'
    elif mutation == 'seed': value['prepared']['rows'][0]['jobs'][0]['seed'] += 1
    elif mutation == 'order': value['prepared']['rows'].reverse()
    elif mutation == 'cutoff': value['deadline_utc'] = '2098-01-01T00:00:00Z'
    else: value['bounds']['campaign_seconds'] += 1
    with pytest.raises(ValueError): p.validate_plan(value)


def test_screen_requires_every_terminal_reviewed_anchor():
    with pytest.raises(ValueError, match='every fresh anchor'):
        p.execution_plan('screening', source_commit='a'*40, host_id='one', deadline_utc='2099-01-01T00:00:00Z')


def test_separate_width_calibration_and_disjoint_generation_seeds():
    values = [p.calibration_plan(plan(), variant=v, generation=g) for v in (False, True) for g in (0, 1)]
    manifests = [p.base.calibration_manifest(value) for value in values]
    seeds = [{row['seed'] for row in m['calibration']} for m in manifests]
    assert all(len(x) == 8 for x in seeds)
    assert len(set.union(*seeds)) == 32
    widths = [{x['width'] for x in m['calibration']} for m in manifests]
    assert widths[0] != widths[2]
    for generation in (-1, 64, True):
        with pytest.raises(ValueError): p.calibration_plan(plan(), generation=generation)


def test_campaign_counts_failed_and_unattempted_jobs_without_raw_receipt_promotion():
    rows = [dict(status=s, expected_jobs=6, audit={'valid': True}, usable_receipts=6)
            for s in ('completed', 'failed', 'unattempted', 'running')]
    result = c.coverage(rows)
    assert result == dict(planned_cohorts=4, planned_jobs=24, completed_valid_cohorts=1,
        usable_receipts=6, failed_cohorts=1, failed_planned_jobs=6, running_cohorts=1,
        unattempted_cohorts=1, unattempted_jobs=6)


def test_child_will_not_dispatch_without_full_phase_headroom(tmp_path, monkeypatch):
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan()))
    settings = tmp_path/'settings.json'; settings.write_text('{}')
    x = c.Runner(path, settings, tmp_path/'runs')
    x.cutoff = 100
    monkeypatch.setattr(c.time, 'time', lambda: 90)
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *a, **k: pytest.fail('no dispatch allowed'))
    with pytest.raises(TimeoutError): x.child('trial', 'unknown', [], seconds=20)
    settings.write_text('{"changed":true}')
    with pytest.raises(ValueError): x.guard()


def test_append_log_replay_binds_actual_counters_and_rejects_tampering(tmp_path):
    store = ExperienceStore('trace', path=tmp_path)
    fp = fingerprint(); peers = context([], resident_count=1)
    store.observe(episode='one', fingerprint=fp, context=peers, updates=4, seconds=1,
        observed_unix=102, reference_rate=4, reference_memory=100, reference_generation='g',
        job_id='j', accepted_unix=100, window_start_unix=101,
        counter_proof={'start_step':4,'end_step':8,'stable_identity_sha256':'identity'})
    store.checkpoint()
    result = {'receipts': {'j': {'expected_updates':64,'accepted_unix':100,'feedback_ready_unix':103}},
        'alpha_events':[{'kind':'valid_windows','halves':[dict(start=1,end=2,rates={'j':4},
            start_steps={'j':4},end_steps={'j':8},stable_identity_sha256='identity')]}]}
    episodes={'one':{'result':result,'offset':100}}
    assert history_audit(tmp_path,'trace',episodes)['observations'] == 1
    result['alpha_events'][0]['halves'][0]['end_steps']['j']=9
    with pytest.raises(ValueError,match='actual stable counter'): history_audit(tmp_path,'trace',episodes)


def test_all_manifest_rows_roundtrip_with_actual_base_and_variant_template_keys(tmp_path):
    from localml_scheduler.tests.test_a10_v5_campaign import neural_table
    from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SCHEMA,SOURCE
    from localml_scheduler.benchmarks.a10_v5_1_1.episode import validate_manifest
    value=plan()
    for variant in (False,True):
        manifest=p.base.calibration_manifest(p.calibration_plan(value,variant=variant))
        table={'schema':SCHEMA,'source':SOURCE,'hardware_key':'fixture','gpu_uuid':'GPU-test',
            'source_identity':{'calibration_identity_schema':'resource_solo_runtime_v1',
              'torch_version':p.workloads.source_metadata()['torch_version'],
              'source_runtime':p.workloads.source_metadata(),'scheduler_python_sources':{'fixture.py':'a'*64}},
            'entries':{s['template_key']:{'updates_per_second':10.,'mean_allocated_mib':100.,
              'admission_reservation_mib':200.,'phase_seconds':dict.fromkeys(
               ('startup','pause','resume','checkpoint','validation','finalization','delivery'),1.)}
              for s in manifest['calibration']}}
        table['artifact_sha256']=p.digest(table);neural=neural_table(table)
        for row in value['prepared']['rows']:
            if row.get('shape_variant_development_fixture',False)!=variant:continue
            result=p.row_manifest(value,row,table,neural,history_root=tmp_path)
            validate_manifest(result,arm=row['arm'],mode=row['mode'])
            assert len(result['jobs'])==len(row['jobs'])
            assert all(j['template_key'] in table['entries'] for j in result['jobs'])
            assert {(s['role'],s['chain']):s['seed'] for s in result['jobs']}=={
                (s['role'],s['chain']):s['seed'] for s in row['jobs']}
