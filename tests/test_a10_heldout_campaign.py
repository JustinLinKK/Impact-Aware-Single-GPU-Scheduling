"""CPU-only held-out controller fixtures; no real selection or GPU results."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from localml_scheduler.benchmarks.a10_resource_mechanisms import calibrate, campaign, heldout, study
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE, SCHEMA
from localml_scheduler.tests.test_a10_resource_campaign import fake_execution, write


def fixture_plan(arm='resource_incremental'):
    selection = dict(arm=arm, iteration2_valid_cohorts=64, iteration2_usable_receipts=384,
        rationale='SYNTHETIC fixture only; no policy has been selected for the experiment',
        iteration2_review_sha256='a'*64, selected_source_commit='b'*40)
    return heldout.candidate_plan(selection,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
        job_timeout_seconds=600, episode_max_seconds=600, campaign_max_seconds=14400)


def sealed_fixture(tmp_path, monkeypatch, arm='resource_incremental'):
    plan=fixture_plan(arm);plan['status']='frozen'  # Synthetic fixture, not deployment.
    # Serialized JSON key sorting must not alter predeclared calibration seeds.
    plan=json.loads(json.dumps(plan,sort_keys=True))
    calibration=study.calibration_manifest(plan)
    table={'source':SOURCE,'schema':SCHEMA,
        'source_identity':calibrate.calibration_source_identity({'fixture_only':True}, calibration),
        'entries':{spec['template_key']:{'updates_per_second':10.} for spec in calibration['calibration']}}
    table['artifact_sha256']=campaign.digest(table)
    directory=tmp_path/'sealed';directory.mkdir()
    settings=tmp_path/'settings.json';calibration_path=tmp_path/'calibration.json'
    write(settings,{});write(calibration_path,table);write(directory/'plan.json',plan)
    pins={}
    for name,manifest in study.evaluation_manifests(plan,table).items():
        path=directory/(name+'.json');write(path,manifest);pins[name]=campaign.file_hash(path)
    write(directory/'protocol-before.json',dict(plan=plan,
        plan_sha256=campaign.file_hash(directory/'plan.json'),manifests=pins,
        calibration_sha256=campaign.file_hash(calibration_path),order=study.episode_order(plan),
        expected_cohorts=48,expected_jobs=288,frozen_before_evaluation=True))
    monkeypatch.setattr(campaign,'source_snapshot',lambda manifest,path:{'fixture_only':True,'manifest_sha256':campaign.file_hash(path)})
    return campaign.sealed_bundle(directory,settings,calibration_path)


def test_candidate_cannot_choose_a_policy_or_launch(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='selection required'):
        heldout.candidate_plan(None,deadline_utc=None,job_timeout_seconds=600,
            episode_max_seconds=600,campaign_max_seconds=14400)
    value=sealed_fixture(tmp_path,monkeypatch)
    plan=fixture_plan();write(value['directory']/'plan.json',plan)
    with pytest.raises(ValueError,match='must be frozen'):
        campaign.sealed_bundle(value['directory'],value['settings'],value['calibration'])


@pytest.mark.parametrize('key,value',[
    ('iteration2_valid_cohorts',63),('iteration2_usable_receipts',383),
    ('iteration2_review_sha256',None),('selected_source_commit',None),
    ('rationale',''),('arm','resource_fixed_three')])
def test_selection_needs_complete_review_and_explicit_revision(key,value):
    plan=fixture_plan();plan['policy_selection'][key]=value
    with pytest.raises(ValueError):study.validate_plan(plan)


@pytest.mark.parametrize('mutation',['configuration','mixture','duration','seed','order','arms'])
def test_runtime_plan_cannot_change_predeclared_inputs(mutation):
    plan=fixture_plan()
    if mutation=='configuration':plan['configurations']['resident']['ballast_mib']=8192
    elif mutation=='mixture':plan['conditions'][0]['classes']=['tiny','resident','compute']
    elif mutation=='duration':plan['conditions'][0]['target_training_seconds']=12
    elif mutation=='seed':plan['seed_base']=41000
    elif mutation=='order':plan['configuration_order']=sorted(plan['configurations'])
    else:plan['arms'].append('resource_revised')
    with pytest.raises(ValueError):study.validate_plan(plan)


def test_sorted_json_preserves_calibration_seed_order_and_actual_budget_semantics():
    plan=json.loads(json.dumps(fixture_plan(),sort_keys=True))
    calibration_manifest=study.calibration_manifest(plan)
    calibration=calibration_manifest['calibration']
    assert [(row['name'],row['seed']) for row in calibration]==[
        (f'{label}-cal-{replica}',91000+2*index+replica)
        for index,label in enumerate(('tiny','resident','compute','both')) for replica in range(2)]
    table=dict(schema=SCHEMA,source=SOURCE,
        source_identity=calibrate.calibration_source_identity({'fixture_only':True}, calibration_manifest),
        entries={row['template_key']:{'updates_per_second':10.} for row in calibration})
    table['artifact_sha256']=campaign.digest(table)
    manifests=study.evaluation_manifests(plan,table)
    assert {2*job['steps_per_epoch'] for manifest in manifests.values() for job in manifest['jobs']}=={184,552}
    assert all(job['epochs']==2 for manifest in manifests.values() for job in manifest['jobs'])
    assert len(study.episode_order(plan))==48


def test_calibration_identity_keeps_raw_snapshot_and_binds_runtime_for_replay():
    manifest=study.calibration_manifest(fixture_plan())
    snapshot={'fixture_only':True,'manifest_sha256':'c'*64}
    identity=calibrate.calibration_source_identity(snapshot,manifest)
    assert snapshot=={'fixture_only':True,'manifest_sha256':'c'*64}
    assert identity['torch_version']==manifest['source_runtime']['torch_version']
    assert identity['source_runtime']==manifest['source_runtime']
    # Verification on a different host must replay retained, externally verified
    # remote metadata rather than substitute that host's local Torch version.
    remote=deepcopy(manifest)
    remote['source_runtime']['torch_version']='SYNTHETIC-REMOTE-RUNTIME'
    for spec in remote['calibration']:spec['torch_version']='SYNTHETIC-REMOTE-RUNTIME'
    replay=calibrate.calibration_source_identity(snapshot,remote,
        runtime_metadata=remote['source_runtime'])
    assert replay['torch_version']=='SYNTHETIC-REMOTE-RUNTIME'
    with pytest.raises(ValueError,match='actual source/runtime differ'):
        calibrate.calibration_source_identity(snapshot,remote)


@pytest.mark.parametrize('mutation',['schema','manifest_runtime','spec_runtime','spec_source','empty_specs'])
def test_calibration_runtime_mismatch_rejects_before_gpu_work(mutation):
    manifest=study.calibration_manifest(fixture_plan())
    if mutation=='schema':manifest['calibration_identity_schema']='unknown'
    elif mutation=='manifest_runtime':manifest['source_runtime']['torch_version']='mismatched'
    elif mutation=='spec_runtime':manifest['calibration'][0]['torch_version']='mismatched'
    elif mutation=='spec_source':manifest['calibration'][0]['runner_sha256']='0'*64
    else:manifest['calibration']=[]
    with pytest.raises(ValueError):calibrate.calibration_source_identity({'fixture_only':True},manifest)


def test_legacy_calibration_table_representation_is_preserved():
    snapshot={'scheduler_python_sources':{'synthetic.py':'d'*64}}
    actual=calibrate.calibration_source_identity(snapshot,{})
    assert actual==snapshot
    actual['scheduler_python_sources']['synthetic.py']='changed-copy'
    assert snapshot['scheduler_python_sources']['synthetic.py']=='d'*64


@pytest.mark.parametrize('mutation',['missing','torch','source'])
def test_heldout_budget_generation_rejects_table_from_another_runtime(mutation):
    plan=fixture_plan();manifest=study.calibration_manifest(plan)
    identity=calibrate.calibration_source_identity({'fixture_only':True},manifest)
    if mutation=='missing':identity={}
    elif mutation=='torch':identity['torch_version']='mismatched'
    else:identity['source_runtime']['runner_sha256']='0'*64
    table=dict(schema=SCHEMA,source=SOURCE,source_identity=identity,entries={})
    table['artifact_sha256']=campaign.digest(table)
    with pytest.raises(ValueError,match='actual source/runtime'):
        study.evaluation_manifests(plan,table)


@pytest.mark.parametrize('arm',['resource_revised','resource_incremental'])
def test_serial_heldout_controller_keeps48_cells288_jobs_and_correct_request_units(tmp_path,monkeypatch,arm):
    value=sealed_fixture(tmp_path,monkeypatch,arm);launches,cleaned=fake_execution(monkeypatch)
    records=campaign.execute(value,tmp_path/'run')
    assert len(launches)==len(cleaned)==len(records)==48
    proof=campaign.load(tmp_path/'run/campaign-complete.json')
    assert proof['complete'] and proof['episodes']==48 and proof['receipts']==288
    assert proof['coverage']['planned_jobs']==288 and proof['coverage']['unattempted_jobs']==0
    rows=campaign.load(tmp_path/'run/comparisons.json')
    assert len(rows)==24 and all(row['valid_repeats']==2 for row in rows)
    assert {row['planned_request_units'] for row in rows}=={4,12}
    assert all(row['request_latency_distribution']['n']==row['planned_request_units'] for row in rows)
    with pytest.raises(FileExistsError):campaign.execute(value,tmp_path/'run')
    assert len(launches)==48


def test_failed_heldout_cell_preserves_every_unattempted_denominator(tmp_path,monkeypatch):
    value=sealed_fixture(tmp_path,monkeypatch);launches,_=fake_execution(monkeypatch,exit_code=7)
    with pytest.raises(RuntimeError):campaign.execute(value,tmp_path/'run')
    proof=campaign.load(tmp_path/'run/coverage.json')
    assert len(launches)==1 and proof['planned_cohorts']==48 and proof['planned_jobs']==288
    assert proof['failed_cohorts']==1 and proof['unattempted_cohorts']==47
    assert not (tmp_path/'run/campaign-complete.json').exists()
    assert all(row['request_latency_mean_seconds'] is None for row in campaign.load(tmp_path/'run/comparisons.json'))


def test_coverage_cannot_reinterpret_partial64_cell_campaign_as_complete48_cell_campaign():
    plan=json.loads((study.workloads.HERE/'iteration-2-plan.json').read_text())
    plan['deadline_utc']=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    order=study.episode_order(plan);records=campaign.planned_records(order)
    with pytest.raises(ValueError,match='full sealed order'):campaign.coverage(records[:48],order)
    assert campaign.coverage(records,order)['planned_jobs']==384


def test_predeclared_blueprint_is_revalidated_between_episodes(tmp_path,monkeypatch):
    value=sealed_fixture(tmp_path,monkeypatch)
    changed=tmp_path/'changed-blueprint';changed.mkdir()
    (changed/'iteration-3-blueprint.json').write_bytes(
        (heldout.HERE/'iteration-3-blueprint.json').read_bytes()+b' ')
    monkeypatch.setattr(heldout,'HERE',changed)
    with pytest.raises(ValueError,match='blueprint changed'):campaign.verify_bundle(value)


def test_heldout_enriched_runtime_identity_composes_with_qualified_raw_replay(monkeypatch):
    from localml_scheduler.tests.test_a10_context_isolation import instrumented_evidence
    from localml_scheduler.benchmarks.a10_resource_mechanisms import context_isolation
    from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import fit_templates
    spec,raw=instrumented_evidence()
    manifest=study.calibration_manifest(fixture_plan())
    assert {'context_isolation.py','visibility_qualification.py'} <= set(manifest['source_runtime']['package_pins'])
    # A small synthetic row exercises composition, not a held-out measurement.
    row=calibrate.extract_row(spec,**raw)
    snapshot={'fixture_only':True,'scheduler_python_sources':{'saved.py':'c'*64}}
    retained_snapshot=deepcopy(snapshot);retained_raw=deepcopy(raw)
    saved_runtime=deepcopy(manifest['source_runtime'])
    context_isolation.qualification_gate(raw['visibility_qualification'],'fixture-uuid',runtime=saved_runtime)
    assert context_isolation.audit_raw_evidence(spec,raw,gpu_uuid='fixture-uuid')['valid']
    monkeypatch.setattr(study.workloads,'source_metadata',lambda:pytest.fail('offline replay must use saved remote metadata'))
    identity=calibrate.calibration_source_identity(snapshot,manifest,runtime_metadata=saved_runtime)
    table=fit_templates([row],hardware_key='fixture-hardware',gpu_uuid='fixture-uuid',source_identity=identity)
    replay=calibrate.calibration_source_identity(snapshot,manifest,runtime_metadata=saved_runtime)
    assert fit_templates([row],hardware_key='fixture-hardware',gpu_uuid='fixture-uuid',source_identity=replay)==table
    assert table['source_identity']['calibration_identity_schema']=='resource_solo_runtime_v1'
    assert table['source_identity']['source_runtime']==saved_runtime
    assert raw==retained_raw and snapshot==retained_snapshot
    wrong_runtime={**saved_runtime,'torch_version':'SYNTHETIC-WRONG-RUNTIME'}
    with pytest.raises(ValueError,match='actual runtime changed'):
        context_isolation.qualification_gate(raw['visibility_qualification'],'fixture-uuid',runtime=wrong_runtime)
    with pytest.raises(ValueError,match='actual source/runtime differ'):
        calibrate.calibration_source_identity(snapshot,manifest,runtime_metadata=wrong_runtime)
