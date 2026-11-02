"""Scientific invariants for graph-conditioned, trace-local FIFO experience."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from localml_scheduler.scheduler.workload_fingerprint import make_fingerprint, context, digest, structural_distances
from localml_scheduler.scheduler.fifo_experience import ExperienceStore
from localml_scheduler.scheduler.contextual_rates import estimate
from localml_scheduler.scheduler.adaptive_fifo_v511 import AdaptiveFIFO511Controller
from localml_scheduler.tests.test_adaptive_fifo import job, profile, SAMPLE, launch


def fingerprint(size=100, gpu='gpu-1', op='mm', stratum='measured'):
    summary = {'training_graph': True, 'graph_sha256': digest([size, op]),
        'operations': {op: 1.}, 'phases': {'forward': .4, 'backward': .6},
        'size_bins': {'10': 1.}, 'structure': [size, size * 2, 10, 20],
        'sizes': [size * 100, size * 200]}
    return make_fingerprint(summary, dict(batch=1, accumulation=1, precision='fp32_ieee',
        optimizer={'name': 'AdamW'}, data_path={'profile_source': stratum}, backend='cuda_process',
        ballast_mib=0.), hardware=gpu, runtime='runtime-1', artifact='frozen', source='source')


def add(store, episode='episode-1', *, fp=None, peers=None, at=100., rate=10., generation='g1'):
    fp = fp or fingerprint()
    peers = peers or context([], resident_count=1)
    store.observe(episode=episode, fingerprint=fp, context=peers, updates=rate * 2,
        seconds=2., observed_unix=at, reference_rate=10., reference_memory=100.,
        reference_generation=generation, job_id=episode + '-job', accepted_unix=at - 10,
        window_start_unix=at - 2.)


def query(store, fp=None, peers=None, now=110., seconds=1.):
    return estimate(store, fp or fingerprint(), peers or context([], resident_count=1),
                    {'step_cost': 100., 'peak_vram_mib': 100., 'artifact_sha256': 'g1'}, now=now,
                    deadline=time.perf_counter() + seconds)


def test_graph_structure_not_equal_solo_numbers_establishes_similarity():
    a, b = fingerprint(), fingerprint(op='convolution')
    assert a['exact'] != b['exact']
    assert structural_distances(a, b)[0] > .25
    store = ExperienceStore('trace')
    for i in range(4): add(store, str(i), fp=a)
    with pytest.raises(ValueError): query(store, b)


def test_history_survives_empty_and_restart_with_hash_chain(tmp_path):
    store = ExperienceStore('trace', path=tmp_path)
    add(store); store.append({'kind': 'control', 'state': {'pending': {'target': 4}}}); store.checkpoint()
    restored = ExperienceStore('trace', path=tmp_path)
    assert restored.parent == store.parent and restored.control == store.control
    assert query(restored)['rate'] == pytest.approx(10.)
    with pytest.raises(ValueError, match='namespace'): ExperienceStore('other-trace', path=tmp_path)
    data = (tmp_path / 'experience.jsonl').read_text().replace('"updates": 20.0', '"updates": 200.0')
    (tmp_path / 'experience.jsonl').write_text(data)
    with pytest.raises(ValueError, match='hash chain'): ExperienceStore('trace', path=tmp_path)


def test_polling_windows_never_become_independent_episode_votes():
    store = ExperienceStore('trace')
    for i in range(100): add(store, at=100. + i)
    assert len(store.rows) == 1
    assert query(store, now=200.)['episodes'] == 1
    near = fingerprint(size=105)
    with pytest.raises(ValueError, match='independent'): query(store, near, now=200.)
    add(store, 'episode-2', at=199.); add(store, 'episode-3', at=199.)
    result = query(store, near, now=200.)
    assert result['episodes'] == 3 and result['lower'] <= result['rate'] * .8
    assert result['upper'] >= result['rate'] * 1.2


@pytest.mark.parametrize('change', ['gpu', 'stratum', 'phase', 'occupancy'])
def test_hard_compatibility_and_peer_phase_gates(change):
    store = ExperienceStore('trace'); add(store)
    fp, peers = fingerprint(), context([], resident_count=1)
    if change == 'gpu': fp = fingerprint(gpu='gpu-2')
    if change == 'stratum': fp = fingerprint(stratum='published')
    if change == 'phase': peers = context([], resident_count=2, starting=1)
    if change == 'occupancy': peers = context([fp], resident_count=2)
    with pytest.raises(ValueError): query(store, fp, peers)


def test_future_and_stale_observations_cannot_authorize_a_change():
    store = ExperienceStore('trace'); add(store, at=1000.)
    with pytest.raises(ValueError): query(store, now=999.)
    with pytest.raises(ValueError): query(store, now=15401.)
    with pytest.raises(TimeoutError): query(store, now=1001., seconds=-1.)


def test_reference_refresh_reexpresses_raw_measurements_without_redating():
    store = ExperienceStore('trace'); add(store, at=100.)
    before = deepcopy(next(iter(store.rows.values())))
    value = estimate(store, fingerprint(), context([], resident_count=1),
        {'step_cost': 50., 'peak_vram_mib': 100.}, now=110., deadline=time.perf_counter() + 1)
    assert value['rate'] == pytest.approx(10.)
    assert value['neighbors'][0]['present_reference_rate'] == 20.
    assert value['neighbors'][0]['original_reference_rate'] == 10.
    assert next(iter(store.rows.values())) == before


def test_neighbors_bounded_and_episode_balanced():
    store = ExperienceStore('trace')
    for i in range(30): add(store, str(i), at=100. + i)
    assert len(query(store, now=140.)['neighbors']) == 16


def setup_controller(monkeypatch, *, n=4, store=None):
    from localml_scheduler.scheduler import adaptive_fifo
    monkeypatch.setattr(adaptive_fifo, 'time', SimpleNamespace(monotonic=lambda: 0.,
                        time=time.time, perf_counter=time.perf_counter))
    c = AdaptiveFIFO511Controller(store=store, wall_offset=0.)
    jobs = [job(chr(97 + i), ordinal=i + 1, budget=10000) for i in range(n)]
    profiles = {j.job_id: {**profile(j), 'v511_fingerprint': fingerprint()} for j in jobs}
    c.observe(now=0., queued_jobs=jobs, running_jobs=[], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={})
    return c, jobs, profiles


def observe(c, jobs, profiles, now, n=3):
    c.observe(now=now, queued_jobs=jobs[n:], running_jobs=jobs[:n], profiles=profiles,
              gpu_sample=SAMPLE, completed_job_ids={})


def test_exploration_independent_of_missing_future_rate_forecast(monkeypatch):
    c, jobs, profiles = setup_controller(monkeypatch)
    launch(c, jobs)
    observe(c, jobs, profiles, 10.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 11.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 12.)
    assert c.admission_limit == 4, c.events
    assert c._trace_pending and len(c._tried) == 1
    decisions = [e for e in c.events if e['kind'] == 'decision']
    assert decisions[-1]['reason'] == 'productive_head_exploration_no_latency_guarantee'
    assert decisions[-1]['failures']  # no invented future rates
    c.observe(now=13., queued_jobs=[], running_jobs=[], profiles={},
              gpu_sample=SAMPLE, completed_job_ids={j.job_id: 12.5 for j in jobs})
    assert c.admission_limit == 4 and c._trace_pending and c.store.rows


def test_strict_fifo_and_pending_exploration_checkpoint(monkeypatch, tmp_path):
    store = ExperienceStore('trace', path=tmp_path)
    c, jobs, profiles = setup_controller(monkeypatch, store=store)
    with pytest.raises(ValueError, match='head'): c.on_launch(jobs[1].job_id, 0.)
    c.on_launch(jobs[0].job_id, 0.)
    c._trace_pending = {'target': 4, 'episode': 'standalone'}
    c._save_control(); state = c.export_state(); store.checkpoint()
    restored = AdaptiveFIFO511Controller.restore_state(state, store=ExperienceStore('trace', path=tmp_path))
    assert restored._trace_pending == c._trace_pending and restored._launched == c._launched
    with pytest.raises(ValueError): restored.on_launch(jobs[0].job_id, 1.)


def test_stale_phase_cannot_explore_even_with_fresh_rates(monkeypatch):
    c, jobs, profiles = setup_controller(monkeypatch)
    launch(c, jobs)
    for j in jobs: j.metadata['feedback_phase_model']['observed_at'] = '2000-01-01T00:00:00+00:00'
    observe(c, jobs, profiles, 10.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 11.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 12.)
    assert c.admission_limit == 3 and not c._tried


def test_only_observed_compatible_prefix_can_supply_warm_initial_target(monkeypatch):
    c, jobs, profiles = setup_controller(monkeypatch)
    launch(c, jobs)
    observe(c, jobs, profiles, 10.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 11.)
    for j in jobs[:3]: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 12.)
    assert c.admission_limit == 4 and c._trace_pending
    # An unmeasured speculative increase does not become a warm prior.
    next_c, _, _ = setup_controller(monkeypatch, store=c.store)
    assert next_c.admission_limit == 3
    c.on_launch(jobs[3].job_id, 12.1)
    jobs[3].status = jobs[0].status
    observe(c, jobs, profiles, 13., n=4)
    for j in jobs: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 14., n=4)
    for j in jobs: j.metadata['runtime_global_step'] += 20
    observe(c, jobs, profiles, 15., n=4)
    assert c._trace_pending is None
    warm = AdaptiveFIFO511Controller(store=c.store, episode='episode-2', wall_offset=20.)
    fresh_jobs = [job('new-' + str(i), ordinal=i + 1) for i in range(4)]
    fresh_profiles = {j.job_id: {**profile(j), 'v511_fingerprint': fingerprint()} for j in fresh_jobs}
    warm.observe(now=0., queued_jobs=fresh_jobs, running_jobs=[], profiles=fresh_profiles,
                 gpu_sample=SAMPLE, completed_job_ids={})
    assert warm.admission_limit == 4
    assert warm.events[-1]['reason'] == 'compatible_initial_prefix_prior'


def test_actual_training_graph_fingerprint_capture(tmp_path, monkeypatch):
    # Explicit release dependency makes this a real CPU qualification, never a
    # skipped substitute in the frozen operations test selection.
    from localml_scheduler.tests import test_a10_v5_job_builder as original
    from localml_scheduler.config import PredictionSettings
    from localml_scheduler.hardware import HardwareProfile
    from localml_scheduler.prediction.ml_predictor import MLVramPredictor
    from localml_scheduler.prediction.fingerprint_preparation import FingerprintPredictor
    import torch
    release = Path(os.environ['A10_V5_TEST_RELEASE'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    blueprint = json.loads(Path(original.module.__file__).with_name('blueprint.json').read_text())
    sealed, _, _, current, _ = original.job(tmp_path, blueprint['configurations']['light'])
    hardware = HardwareProfile('CPU-ONLY-graph', 'linux', 'NVIDIA A10', 23028, None, None, torch.__version__)
    predictor = MLVramPredictor(PredictionSettings(mode='ml_predictor', student_release_path=str(release),
        student_model_path=str(release / 'checkpoints/S_A10'), conversion_timeout_seconds=180, cpu_threads=4), hardware)
    prepared = FingerprintPredictor(predictor, predictor._released, gpu_uuid='cpu-fixture',
        runtime_identity={'torch': torch.__version__}, backend='cuda_process', profile_source='ml_predictor',
        source_closure={'test_fixture.py': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()})
    value = prepared.predict_profile_options(current, [sealed['batch']])[sealed['batch']]
    fp = value['v511_fingerprint']
    assert fp['graph']['training_graph'] and fp['graph']['phases']['backward'] > 0
    assert fp['graph']['structure'][0] > 1 and not torch.cuda.is_initialized()
    assert value['v511_reservation_mib'] > value['peak_vram_mib']
    assert prepared.predict_profile_options(current, [sealed['batch']])[sealed['batch']] == value
    expected = 1 if fp['source_provenance']['complete'] else 2
    assert len(prepared.capture_events) == expected


def test_unchanged_student_weights_do_not_hide_new_reference_calibration():
    store=ExperienceStore('trace')
    for i in range(3):add(store,str(i),generation='old-calibration')
    target={'step_cost':100.,'peak_vram_mib':100.,'artifact_sha256':'same-student',
            'calibration_sha256':'new-calibration'}
    with pytest.raises(ValueError,match='independent'):
        estimate(store,fingerprint(size=105),context([],resident_count=1),target,
                 now=110.,deadline=time.perf_counter()+1.)
    exact=estimate(store,fingerprint(),context([],resident_count=1),target,
                   now=110.,deadline=time.perf_counter()+1.)
    assert exact['rate']==pytest.approx(10.)
