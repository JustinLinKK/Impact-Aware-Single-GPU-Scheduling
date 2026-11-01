"""Frozen ownership-aware subprocess cleanup reused from reviewed iteration3."""
import os,signal,subprocess,time
from pathlib import Path

def process_identity(pid,proc=Path('/proc')):
    try:
        fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'state':fields[0],'ppid':int(fields[1]),'pgrp':int(fields[2]),'session':int(fields[3]),'start_ticks':int(fields[19])}
    except (OSError,ValueError,IndexError):return None

def owned_descendants(pid,proc=Path('/proc')):
    snapshot={}
    for path in proc.glob('[0-9]*/stat'):
        identity=process_identity(int(path.parent.name),proc)
        if identity:snapshot[identity['pid']]=identity
    owned={int(pid)};changed=True
    while changed:
        children={i for i,row in snapshot.items() if row['ppid'] in owned}
        changed=not children<=owned;owned.update(children)
    return {i:snapshot[i] for i in owned if i in snapshot}

def same_live(identity,proc):
    current=process_identity(identity['pid'],proc)
    return bool(current and current['start_ticks']==identity['start_ticks'] and current['state'] not in {'Z','X'})

def terminate_owned(child,*,proc=Path('/proc'),identities=None):
    """Clean live/reparented owned identities even after the direct child exits."""
    snapshot=dict(identities or {})
    parent_alive=child.poll()is None
    if parent_alive:snapshot.update(owned_descendants(child.pid,proc))
    parent=snapshot.get(child.pid)
    def signal_parent(sig):
        if parent and not same_live(parent,proc):return
        try:
            if parent and parent['pgrp']==child.pid and parent['session']==child.pid:os.killpg(child.pid,sig)
            else:os.kill(child.pid,sig)
        except ProcessLookupError:pass
    parent_error=None
    if parent_alive:
        try:
            signal_parent(signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                signal_parent(signal.SIGKILL);child.wait(timeout=5)
        except BaseException as exc:parent_error=exc
    descendants=[row for pid,row in snapshot.items() if pid!=child.pid]
    for sig,seconds in ((signal.SIGTERM,3.),(signal.SIGKILL,2.)):
        survivors=[row for row in descendants if same_live(row,proc)]
        for row in survivors:
            # Revalidate immediately before signaling: never target a reused PID.
            if same_live(row,proc):
                try:os.kill(row['pid'],sig)
                except ProcessLookupError:pass
        until=time.monotonic()+seconds
        while any(same_live(row,proc) for row in survivors) and time.monotonic()<until:time.sleep(.05)
    if any(same_live(row,proc) for row in descendants):raise TimeoutError('owned descendant survived bounded TERM/KILL cleanup; identities retained')
    if parent_error is not None:raise parent_error


def cleanup_owned(child,identities):
    """Repeated termination signals cannot strand owned children during cleanup."""
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        for s in previous:signal.signal(s,signal.SIG_IGN)
        terminate_owned(child,identities=identities)
    finally:
        for s,h in previous.items():signal.signal(s,h)
