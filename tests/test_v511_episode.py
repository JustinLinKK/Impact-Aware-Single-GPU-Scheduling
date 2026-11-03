"""Actual SQLite/receipt harness with injected CPU workers, not A10 evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from localml_scheduler.benchmarks.a10_v5_1_1 import episode as f
from localml_scheduler.benchmarks.a10_v5_1_1.audit import audit_episode
from localml_scheduler.tests import test_a10_v5_episode as original
from localml_scheduler.tests.test_v511_runtime import fingerprint


class GraphFixture:
    def __init__(self, predictor, released, **kwargs):
        self.predictor = predictor
        self.capture_events = []
        self.failures, self._results = [], {}
    def __getattr__(self, name): return getattr(self.predictor, name)
    def augment(self, job, batch, profile): return profile
    def stop(self, timeout=3.): return True
    def predict_profile_options(self, job, batches):
        values = self.predictor.predict_profile_options(job, batches)
        for p in values.values():
            p['v511_fingerprint'] = fingerprint()
            p['v511_reservation_mib'] = p['peak_vram_mib'] + 256.
        return values


@pytest.mark.parametrize('mode', ['individual', 'group'])
@pytest.mark.parametrize('development', [False, True])
def test_new_harness_fifo_receipt_audit_and_history_gate(tmp_path, monkeypatch, mode, development):
    from localml_scheduler.prediction import fingerprint_preparation
    monkeypatch.setattr(original, 'e', f)
    harness = original.harness.__wrapped__(monkeypatch)
    snapshot_fixture = f.source_snapshot
    monkeypatch.setattr(f, 'source_snapshot', lambda *args, **kwargs: {
        **snapshot_fixture(*args, **kwargs), 'scheduler_python_sources': {'cpu_fixture.py': 'a' * 64}})
    monkeypatch.setattr(fingerprint_preparation, 'FingerprintPredictor', GraphFixture)
    def factory(settings, **kw):
        service = harness[0](settings, **kw)
        service.planner.estimator.ml_predictor = SimpleNamespace(_released=None)
        return service
    manifest = original.manifest()
    namespace = 'dev-' + mode if development else 'screen-' + mode
    manifest['v511_trace'] = {'episode': namespace + '-1', 'history_namespace': namespace,
        'history_path': str(tmp_path / 'history' / namespace)}
    if development: manifest['development'] = True
    table = original.calibration_fixture(manifest['jobs'][0]['template_key'], original.H)
    manifest['calibration_sha256'] = table['artifact_sha256']
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    output = tmp_path / 'episode'; output.mkdir()
    result = f.run_episode(manifest, path, {}, output, table, arm='v5_1_1', mode=mode,
        max_seconds=35, source_commit='a' * 40, service_factory=factory, gpu_reader=lambda: dict(original.GPU))
    assert result['complete'] and len(result['receipts']) == 6
    assert result['history_namespace'] == namespace
    assert (tmp_path / 'history' / namespace / 'index.json').exists()
    snapshot = f.source_snapshot(manifest, path, 'a' * 40)
    audit = audit_episode(output, {'arm': 'v5_1_1', 'mode': mode}, manifest, snapshot, table,
                          development=development)
    assert audit['valid'], audit['errors']
    if development:
        assert not audit_episode(output, {'arm': 'v5_1_1', 'mode': mode}, manifest, snapshot, table)['valid']
    before = result['ordered_launch_events']
    assert [x['job_id'] for x in before] == sorted(result['receipts'],
        key=lambda j: result['receipts'][j]['accepted_monotonic'])
