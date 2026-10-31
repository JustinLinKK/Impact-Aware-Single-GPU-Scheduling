"""Real spawn workers/capture children, CPU-only; no GPU or predictor model load."""
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from types import SimpleNamespace
from threading import Thread
import json,os,subprocess,sys,time
import pytest
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import TrainingJob
from localml_scheduler.scheduler.parallel_preparation import IsolatedPredictionBackend,ParallelPreparedPredictor
from localml_scheduler.scheduler import owned_prediction_processes as owned


def _hang_prediction(spec,payload,batches):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'],start_new_session=True)
    path=Path(spec[2]['child_path'])
    path.write_text(json.dumps({'worker':owned.identity(os.getpid()),'child':owned.identity(child.pid),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}))
    time.sleep(90)


def backend(tmp_path,conversion=.2,allowance=5.):
    settings=SchedulerSettings(runtime_root=tmp_path/'runtime',prediction={'ml':{'source_conversion_timeout_seconds':conversion}})
    result=IsolatedPredictionBackend(SimpleNamespace(artifact_hash='pinned',available=True,model_id='fixture'),
        kind='ml_predictor',settings_dict=settings.to_dict(),calibration={'child_path':str(tmp_path/'child.json')},
        hardware=SimpleNamespace(hardware_key='CPU-ONLY'),gpu_uuid='CPU-ONLY',workers=1,
        model_init_allowance_seconds=allowance,_predict=_hang_prediction)
    return result


def job():return TrainingJob.create('fixture:runner','fixture','/tmp/unused',max_epochs=2)


def wait_file(path,thread=None):
    end=time.monotonic()+12
    while not path.exists()and time.monotonic()<end:
        if thread is not None and not thread.is_alive():break
        time.sleep(.02)
    assert path.exists(),'spawned worker did not create its bounded CPU child'
    return json.loads(path.read_text())


def assert_not_live(row):
    actual=owned.identity(row['pid'])
    assert actual is None or actual['start_ticks']!=row['start_ticks']or actual['state']in ('Z','X')


def test_future_timeout_kills_owned_pool_and_capture_child_preserves_unrelated(tmp_path):
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])
    b=backend(tmp_path);started=time.monotonic()
    try:
        with pytest.raises(FutureTimeout):b.predict_profile_options(job(),[1])
        assert time.monotonic()-started<9
        identities=wait_file(tmp_path/'child.json')
        assert identities['cuda_visible_devices']==''
        assert_not_live(identities['worker']);assert_not_live(identities['child'])
        assert unrelated.poll()is None
        assert b.cleanup_report and b.cleanup_report['remaining_live_pids']==[]
        assert {identities['worker']['pid'],identities['child']['pid']}<={r['pid']for r in b.cleanup_report['owned_identities']}
    finally:
        b.stop();unrelated.terminate();unrelated.wait(timeout=3)


def test_parallel_stop_terminates_backend_before_joining_blocked_consumer(tmp_path):
    b=backend(tmp_path,conversion=30,allowance=5.)
    # Real process executor + real preparation consumer; valid profile is never
    # fabricated because this fixture intentionally hangs before returning any.
    prep=ParallelPreparedPredictor(b,workers=1,hardware='A10',hardware_key='CPU-ONLY')
    j=job();j.metadata['perfseer_model']={}
    try:
        prep.enqueue(j,1);identities=wait_file(tmp_path/'child.json')
        started=time.monotonic();assert prep.stop(timeout=3.)
        assert time.monotonic()-started<3.3 and not prep.worker_alive
        assert_not_live(identities['worker']);assert_not_live(identities['child'])
        assert b.cleanup_report['remaining_live_pids']==[]
    finally:prep.stop(timeout=3.)


def test_wrapper_deadline_comes_from_frozen_capture_plus_explicit_model_init(tmp_path):
    b=backend(tmp_path,conversion=180,allowance=60)
    try:
        assert b.conversion_timeout_seconds==180 and b.timeout_seconds==240
        assert b._spec[1]['prediction']['ml']['source_conversion_timeout_seconds']==180
    finally:assert b.stop()
    with pytest.raises(ValueError,match='future bound must equal'):
        IsolatedPredictionBackend(SimpleNamespace(),kind='ml_predictor',settings_dict=SchedulerSettings().to_dict(),calibration={},hardware=None,gpu_uuid='fixture',timeout_seconds=1)


def test_reused_birth_is_not_killed(tmp_path):
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])
    class Executor:
        _processes={}
        def shutdown(self,**kwargs):pass
    current=owned.identity(unrelated.pid);wrong={unrelated.pid:{**current,'start_ticks':current['start_ticks']-1}}
    try:
        report=owned.cleanup(Executor(),wrong,timeout=.1)
        assert unrelated.poll()is None and report['remaining_live_pids']==[]
    finally:unrelated.terminate();unrelated.wait(timeout=3)


def test_repeated_termination_cannot_interrupt_bounded_cleanup_or_change_handler():
    import signal
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(*args):raise InterruptedError('would interrupt owned cleanup')
    class Executor:
        _processes={}
        def shutdown(self,**kwargs):os.kill(os.getpid(),signal.SIGTERM)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        assert owned.cleanup(Executor(),{},timeout=.1)['remaining_live_pids']==[]
        assert signal.getsignal(signal.SIGTERM)is interrupted
    finally:signal.signal(signal.SIGTERM,previous)


def cached_incomplete_backend(tmp_path,monkeypatch,*,terminal_after=None,terminal_state="Z",reused=False):
    """Deterministic process-birth observations, without spawning or signaling."""
    from localml_scheduler.scheduler import parallel_preparation
    clock=SimpleNamespace(now=10.)
    fake_time=SimpleNamespace(monotonic=lambda:clock.now,sleep=lambda delay:setattr(clock,'now',clock.now+delay))
    monkeypatch.setattr(owned,'time',fake_time)
    monkeypatch.setattr(parallel_preparation,'time',fake_time)
    expected={'pid':12345,'ppid':os.getpid(),'state':'S','start_ticks':111}
    calls=[]
    def cleanup(executor,known,*,timeout):
        calls.append(timeout)
        return {'owned_identities':[dict(expected)],'remaining_live_pids':[expected['pid']],
            'elapsed_seconds':0.,'terminal_states':['absent','Z','X']}
    monkeypatch.setattr(owned,'cleanup',cleanup)
    def identity(pid):
        assert pid==expected['pid'],'re-observation must not inspect an unrelated process'
        if reused:return {**expected,'start_ticks':222}
        if terminal_after is not None and clock.now>=terminal_after:
            return None if terminal_state is None else {**expected,'state':terminal_state}
        return dict(expected)
    monkeypatch.setattr(owned,'identity',identity)
    monkeypatch.setattr(owned.os,'kill',lambda *_:pytest.fail('cached cleanup re-observation must never signal a PID'))
    settings=SchedulerSettings(runtime_root=tmp_path/'runtime')
    b=IsolatedPredictionBackend(SimpleNamespace(artifact_hash='frozen',available=True),kind='ml_predictor',
        settings_dict=settings.to_dict(),calibration={},hardware=SimpleNamespace(hardware_key='cpu'),
        gpu_uuid='cpu',_executor=SimpleNamespace())
    b._known_owned={expected['pid']:dict(expected)}
    assert b.stop(timeout=0.)is False
    assert calls==[0.]
    return b,clock,calls


@pytest.mark.parametrize("terminal_state", [None,"Z","X"])
def test_zero_time_stop_then_delayed_terminal_birth_refreshes_cached_incomplete_report(tmp_path,monkeypatch,terminal_state):
    b,clock,calls=cached_incomplete_backend(tmp_path,monkeypatch,terminal_after=10.03,terminal_state=terminal_state)
    assert b.stop(timeout=.2)
    assert clock.now>=10.03 and clock.now-10<=.2
    assert b.cleanup_report['remaining_live_pids']==[]
    assert b.cleanup_report['reobservations'][-1]['observed_monotonic_seconds']==clock.now
    assert calls==[0.]  # Existing TERM/KILL cleanup is not repeated.
    previous_clock=clock.now;previous_report=b.cleanup_report
    assert b.stop(timeout=.2)
    assert b.cleanup_report is previous_report and clock.now==previous_clock


def test_persistent_live_owned_birth_remains_failed_within_each_callers_budget(tmp_path,monkeypatch):
    b,clock,calls=cached_incomplete_backend(tmp_path,monkeypatch)
    started=clock.now
    assert not b.stop(timeout=.05)
    assert clock.now-started<=.05+1e-9
    assert b.cleanup_report['remaining_live_pids']==[12345]
    assert b.cleanup_report['reobservations'][-1]['elapsed_seconds']<=.05+1e-9
    assert calls==[0.]


def test_reused_pid_is_not_signaled_and_does_not_keep_the_old_owned_birth_live(tmp_path,monkeypatch):
    b,clock,calls=cached_incomplete_backend(tmp_path,monkeypatch,reused=True)
    assert b.stop(timeout=.1)
    assert b.cleanup_report['remaining_live_pids']==[]
    assert clock.now==10. and calls==[0.]


def test_backend_stop_state_lock_wait_consumes_its_existing_deadline(tmp_path):
    # A busy prediction-state lock must not turn a bounded stop into an
    # unbounded lock acquisition before the cleanup budget even starts.
    settings=SchedulerSettings(runtime_root=tmp_path/'runtime')
    b=IsolatedPredictionBackend(SimpleNamespace(artifact_hash='frozen',available=True),kind='ml_predictor',
        settings_dict=settings.to_dict(),calibration={},hardware=SimpleNamespace(hardware_key='cpu'),
        gpu_uuid='cpu',_executor=SimpleNamespace())
    b._lock.acquire()
    try:
        started=time.monotonic()
        assert b.stop(timeout=.02)is False
        assert time.monotonic()-started<.2
        assert b.cleanup_report is None
    finally:b._lock.release()
