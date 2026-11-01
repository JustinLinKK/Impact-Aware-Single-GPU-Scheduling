from datetime import datetime, timezone
from copy import deepcopy
import json

import pytest

from localml_scheduler.benchmarks.a10_v5_1_ablation.fresh_calibration import (
    CONTRACT, FreshCalibration, freshness, matched_block, observation_range,
)

def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

def tables(oldest=1000, newest=1010):
    return ({'entries': {'a': {'observed_at': iso(oldest)}, 'b': {'observed_at': iso(newest)}}},
            {'observed_at': iso(newest)})

@pytest.mark.parametrize('age,expected', [(0, True), (3299, True), (3300, False), (14401, False)])
def test_full_block_requires_real_four_hour_coverage(age, expected):
    table, neural=tables(1000,1000)
    assert freshness(table,neural,now=1000+age,remaining_episodes=3,episode_seconds=3600)['fresh'] is expected

def test_remaining_block_does_not_restart_full_three_hour_reservation():
    table, neural=tables(1000,1000)
    assert freshness(table,neural,now=4500,remaining_episodes=2,episode_seconds=3600)['fresh']
    assert not freshness(table,neural,now=4500,remaining_episodes=3,episode_seconds=3600)['fresh']

def test_oldest_template_and_actual_neural_observation_control_expiry():
    table, neural=tables()
    neural['observed_at']=iso(900)
    assert observation_range(table,neural)==(900,1010)
    neural['conversion_finished_unix']=999999
    assert observation_range(table,neural)==(900,1010)

@pytest.mark.parametrize('stamp', ['2026-11-01T00:00:00', None, 'invalid'])
def test_unknown_or_naive_clocks_are_rejected(stamp):
    table,neural=tables();neural['observed_at']=stamp
    with pytest.raises((ValueError,TypeError)):observation_range(table,neural)

def test_future_evidence_cannot_be_fresh():
    table,neural=tables(1000,2000)
    assert not freshness(table,neural,now=1500,remaining_episodes=1,episode_seconds=3600)['fresh']

def order():
    return [dict(host_id='one',condition='backlog',mode='individual',repeat=1,arm=arm)
            for arm in ('fixed_three','combined','v5_1')]

def test_block_is_exactly_three_matched_arms():
    assert matched_block(order(),0)==dict(host_id='one',condition='backlog',mode='individual',repeat=1)

@pytest.mark.parametrize('mutation', ['mode','repeat','arm','partial'])
def test_unmatched_or_partial_blocks_rejected(mutation):
    rows=order()
    if mutation=='partial':rows.pop()
    else:rows[-1][mutation]=rows[0][mutation] if mutation=='arm' else 'other'
    with pytest.raises(ValueError):matched_block(rows,0)

def stub(tmp_path,monkeypatch,*,now):
    table,neural=tables(1000,1000)
    path=tmp_path/'neural.json';path.write_text(json.dumps(neural))
    x=object.__new__(FreshCalibration)
    x.bundle={'table':table,'neural_calibration':path}
    x.original={'plan':{'episode_max_seconds':3600,'deadline_utc':iso(100000)}}
    x.campaign_expires_unix=100000
    x.root=tmp_path;x.blocks=[];x.generations=[{'generation':'initial'}]
    x.active={'freshness_at_start':{'expires_unix':15400},'episode_checks':[]}
    x.blocks.append(x.active)
    monkeypatch.setattr('localml_scheduler.benchmarks.a10_v5_1_ablation.fresh_calibration.time.time',lambda:now)
    return x

def test_mid_block_expiry_fails_without_refresh_or_retimestamp(tmp_path,monkeypatch):
    x=stub(tmp_path,monkeypatch,now=15100)
    x._refresh=lambda:pytest.fail('mid-block refresh is forbidden')
    before=deepcopy(x.bundle['table'])
    with pytest.raises(TimeoutError):x.before_episode(1)
    with pytest.raises(TimeoutError):x.guard()
    assert x.bundle['table']==before

def test_end_marker_requires_live_freshness(tmp_path,monkeypatch):
    x=stub(tmp_path,monkeypatch,now=14000)
    x.end_episode(2)
    assert x.active['status']=='completed'
    assert x.active['completed_unix']==14000

def test_new_generation_rejects_old_measurements_before_new_capture(tmp_path,monkeypatch):
    x=stub(tmp_path,monkeypatch,now=2000)
    with pytest.raises(ValueError,match='new real observations'):
        x._publish_generation(x.bundle,'g01',1500)

def test_policy_ttl_contract_stays_four_hours():
    assert CONTRACT['ttl_seconds']==14400
    assert CONTRACT['evaluation_feedback_used'] is False
    assert CONTRACT['refresh_only_between_blocks'] is True

def test_package_shaped_raw_settings_use_the_actual_episode_override(tmp_path,monkeypatch):
    from localml_scheduler.benchmarks.a10_feedback_latency.run import make_settings
    raw={'gpu_scheduler':{'mode':'parallel_time_aware'}}
    path=tmp_path/'settings.json';path.write_text(json.dumps(raw))
    monkeypatch.setattr(FreshCalibration,'_publish_generation',lambda *args:None)
    monkeypatch.setattr(FreshCalibration,'_verify_inputs',lambda *args:{})
    FreshCalibration({'plan':{'calibration_refresh':CONTRACT},'settings':path,
        'directory':tmp_path/'sealed-manifests'},tmp_path,lambda:None,9999999999.)
    actual=make_settings(raw,tmp_path/'actual',{'total_mib':23028.})
    assert actual.gpu_scheduler.feedback_latency.phase_model_ttl_seconds==CONTRACT['ttl_seconds']

def test_reused_generation_reserves_host_headroom_before_another_arm(tmp_path,monkeypatch):
    x=stub(tmp_path,monkeypatch,now=2000)
    x.original['plan']['deadline_utc']=iso(9000)
    with pytest.raises(TimeoutError,match='campaign/host'):x.before_episode(1)

def test_refresh_child_cannot_launch_after_source_drift(tmp_path,monkeypatch):
    x=stub(tmp_path,monkeypatch,now=2000)
    x.campaign_deadline=lambda:None
    def changed():raise ValueError('source drift')
    x._verify_inputs=changed
    monkeypatch.setattr('localml_scheduler.benchmarks.a10_v5_1_ablation.fresh_calibration.subprocess.Popen',
                        lambda *a,**k:pytest.fail('must verify before execution'))
    with pytest.raises(ValueError,match='source drift'):
        x._run(tmp_path,'calibration','unused',[],10)

@pytest.mark.parametrize('mutation', ['trailing','missing','reordered','output'])
def test_final_replay_rejects_extra_or_misbound_records(mutation):
    from localml_scheduler.benchmarks.a10_v5_1_ablation.fresh_replay import validate_record_order
    from localml_scheduler.benchmarks.a10_v5_1_ablation.campaign import planned_records
    rows=[]
    for i in range(20):
        rows.extend([{**r,'condition':str(i),'expected_jobs':6}for r in order()])
    records=planned_records(rows)
    validate_record_order(records,rows)
    if mutation=='trailing':records.append(records[-1])
    elif mutation=='missing':records.pop()
    elif mutation=='reordered':records[0],records[1]=records[1],records[0]
    else:records[0]['output']='wrong-cohort'
    with pytest.raises(ValueError):validate_record_order(records,rows)

@pytest.mark.parametrize('mutation', ['redate','oldest','future'])
def test_final_replay_rejects_refreshed_timestamp_laundering(mutation):
    from localml_scheduler.benchmarks.a10_v5_1_ablation.fresh_replay import validate_generation_clock
    import time
    table,neural=tables(1000,1010)
    g=dict(generation='g01',started_unix=900,finished_unix=1020,
           oldest_observation_unix=1000,newest_observation_unix=1010)
    validate_generation_clock(g,table,neural)
    if mutation=='redate':g['started_unix']=1001
    elif mutation=='oldest':g['oldest_observation_unix']=1001
    else:g['finished_unix']=time.time()+1000
    with pytest.raises(ValueError):validate_generation_clock(g,table,neural)
