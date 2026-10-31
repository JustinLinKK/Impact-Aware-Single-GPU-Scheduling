"""Birth-bound cleanup of an executor's observed CPU workers/descendants only."""
from pathlib import Path
import os,signal,time
from threading import current_thread,main_thread


def identity(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text();tail=text[text.rfind(')')+2:].split()
        return {'pid':pid,'ppid':int(tail[1]),'state':tail[0],'start_ticks':int(tail[19])}
    except (FileNotFoundError,ProcessLookupError):return None


def remember(executor, known):
    roots=[]
    for process in (getattr(executor,'_processes',None) or {}).values():
        row=identity(process.pid)
        if row and row['ppid']==os.getpid() and process.is_alive():
            previous=known.get(row['pid'])
            if previous is None or previous['start_ticks']==row['start_ticks']:
                known[row['pid']]=row;roots.append(row['pid'])
    rows={}
    for path in Path('/proc').glob('[0-9]*'):
        row=identity(int(path.name))
        if row:rows[row['pid']]=row
    owned={pid for pid,row in known.items()if pid in rows and rows[pid]['start_ticks']==row['start_ticks']}
    while True:
        more={pid for pid,row in rows.items()if row['ppid']in owned}-owned
        if not more:break
        for pid in more:known[pid]=rows[pid]
        owned|=more
    return roots


def live(known):
    result=[]
    for pid,previous in known.items():
        row=identity(pid)
        if row and row['start_ticks']==previous['start_ticks']and row['state']not in ('Z','X'):result.append(pid)
    return result


def _cleanup(executor, known, *, timeout=2.):
    started=time.monotonic();deadline=started+max(0.,timeout)
    roots=remember(executor,known)
    processes=list((getattr(executor,'_processes',None)or{}).values())
    executor.shutdown(wait=False,cancel_futures=True)
    def freeze_tree():
        # Freeze exact owned births before the last ancestry scan, closing the
        # ordinary fork-between-snapshot-and-parent-termination race.
        frozen=set()
        while time.monotonic()<deadline:
            remember(executor,known)
            new=set(live(known))-frozen
            if not new:return
            for pid in new:
                current=identity(pid)
                if current and current['start_ticks']==known[pid]['start_ticks']:
                    try:os.kill(pid,signal.SIGSTOP)
                    except ProcessLookupError:pass
            frozen|=new
    freeze_tree()
    # Descendants first; then their exact parent worker births. No process-group,
    # device-wide or name-based kill is used, so unrelated children remain alone.
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if sig==signal.SIGKILL:freeze_tree()
        for pid in sorted(known,key=lambda pid:pid in roots):
            current=identity(pid)
            if current and current['start_ticks']==known[pid]['start_ticks']and current['state']not in ('Z','X'):
                try:
                    os.kill(pid,sig)
                    if sig==signal.SIGTERM:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
        until=min(deadline,time.monotonic()+max(0.,timeout)/3)
        while time.monotonic()<until and live(known):time.sleep(.01)
    for process in processes:process.join(timeout=max(0.,deadline-time.monotonic()))
    return {'owned_identities':list(known.values()),'remaining_live_pids':live(known),
        'elapsed_seconds':time.monotonic()-started,'terminal_states':['absent','Z','X'],
        'scope':'Exact observed pool-worker/descendant PID births; unrelated processes untouched.'}


def cleanup(executor, known, *, timeout=2.):
    # A second TERM/INT/HUP must not strand a worker frozen during owned cleanup.
    previous={}
    if current_thread()is main_thread():
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            previous[number]=signal.getsignal(number);signal.signal(number,signal.SIG_IGN)
    try:return _cleanup(executor,known,timeout=timeout)
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
