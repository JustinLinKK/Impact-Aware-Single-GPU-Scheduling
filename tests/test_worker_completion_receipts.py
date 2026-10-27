"""Replay completion requires real process success, not just a store flag."""
from unittest.mock import Mock

import pytest

from localml_scheduler.domain import JobStatus
from localml_scheduler.execution.executor import WorkerProcessHandle
from localml_scheduler.scheduler.supervisor import ManagedWorker, PlacementGroupHandle, WorkerSupervisor
from localml_scheduler.tests.test_cold_start_trials import setup


@pytest.mark.parametrize('monitor_via_store', [True, False])
@pytest.mark.parametrize('exit_code', [0, 7])
def test_observed_exit_code_survives_reaping_but_not_shutdown(tmp_path, monitor_via_store, exit_code):
    settings, store, _, _ = setup(tmp_path)
    supervisor = WorkerSupervisor(settings, store=store)
    process = Mock(); process.poll.return_value = None
    handle = WorkerProcessHandle('a', process, tmp_path / 'out', tmp_path / 'err', monitor_via_store)
    supervisor._groups['g'] = PlacementGroupHandle(
        'g', 'stack_anchor', 'cuda_process', workers={'a': ManagedWorker('a', handle, 'solo')})
    store.update_job('a', status=JobStatus.COMPLETED)
    assert supervisor.poll() == []
    assert supervisor.observed_exit_code('a') is None
    assert not supervisor.has_reaped_process('a')
    process.poll.return_value = exit_code
    assert supervisor.poll()[0].returncode == exit_code
    assert supervisor.has_reaped_process('a')
    assert not supervisor.has_live_process('a')
    assert supervisor.observed_exit_code('a') == exit_code
    assert WorkerSupervisor(settings, store=store).observed_exit_code('a') is None
    supervisor.shutdown()
    assert supervisor.observed_exit_code('a') is None


def test_redispatch_revokes_exit_success_before_launch_even_if_launch_raises(tmp_path):
    settings, store, _, _ = setup(tmp_path)
    backend = Mock(); backend.launch.side_effect = RuntimeError('launch failed')
    supervisor = WorkerSupervisor(settings, store=store, backends={'cuda_process': backend})
    supervisor.placement_for = Mock(return_value=Mock(can_run=True))
    supervisor._record_process_exit('a', 0)
    with pytest.raises(RuntimeError, match='launch failed'):
        supervisor.dispatch([store.get_job('a')], mode='stack_anchor', backend_name='cuda_process')
    assert not supervisor.has_reaped_process('a')
    assert supervisor.observed_exit_code('a') is None
