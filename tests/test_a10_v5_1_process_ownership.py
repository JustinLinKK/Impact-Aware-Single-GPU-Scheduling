"""Real CPU-only ownership regressions for the NEW comparison controller."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from localml_scheduler.benchmarks.a10_v5_1_ablation import process_ownership as own


def test_cached_births_clean_separate_session_orphan_preserve_unrelated(tmp_path):
 evidence=tmp_path/'child.json'
 script='import subprocess,sys,time,json\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)\nPath('+repr(str(evidence))+').write_text(json.dumps({"pid":p.pid}))\ntime.sleep(.4)\n'
 parent=subprocess.Popen([sys.executable,'-c',script],start_new_session=True)
 unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
 child_pid=None
 try:
  deadline=time.monotonic()+3
  while not evidence.exists()and time.monotonic()<deadline:time.sleep(.01)
  child_pid=json.loads(evidence.read_text())['pid'];cached=own.owned_descendants(parent.pid)
  assert child_pid in cached
  parent.wait(timeout=3);own.cleanup_owned(parent,cached)
  current=own.process_identity(child_pid)
  assert current is None or current['state']in {'Z','X'}
  assert unrelated.poll()is None
 finally:
  for p in (parent,unrelated):
   if p.poll()is None:p.kill();p.wait()
  if child_pid:
   current=own.process_identity(child_pid)
   if current and current['state']not in {'Z','X'}:os.kill(child_pid,signal.SIGKILL)


def test_repeated_signals_ignored_only_during_bounded_cleanup(monkeypatch):
 signals=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP);before={s:signal.getsignal(s)for s in signals};seen=[]
 def cleanup(child,**kw):
  for s in signals:signal.raise_signal(s)
  seen.append(True)
 monkeypatch.setattr(own,'terminate_owned',cleanup)
 own.cleanup_owned(None,{})
 assert seen==[True] and all(signal.getsignal(s)==before[s]for s in signals)
