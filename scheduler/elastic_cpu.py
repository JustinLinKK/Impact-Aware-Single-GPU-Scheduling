"""Redistribute an existing CPU envelope, without modifying candidate programs.

Minimum reservations determine admission capacity. Spare cores are shared
work-conservingly. Existing core ownership is retained where possible. Apply
shrinks before expansions or new launches. This is per-thread affinity, not an
atomic cgroup quota; fork/thread races are reconciled by repeated observation.
"""
from __future__ import annotations
import os
import psutil


def allocation(pool, current, pending=(), minimum=2):
    pool = tuple(sorted(set(pool)))
    keys = list(current) + list(pending)
    if minimum < 1 or len(set(keys)) != len(keys):
        raise ValueError('invalid reservation or duplicate job')
    if len(keys) * minimum > len(pool):
        raise ValueError('physical CPU reservations exhausted')
    if not keys:
        return {}
    seen = set()
    for key, cores in current.items():
        values = set(cores)
        if not values <= set(pool) or seen & values:
            raise ValueError('invalid or overlapping prior allocation')
        seen |= values
    q, r = divmod(len(pool), len(keys))
    targets = {k:q+(i<r) for i,k in enumerate(keys)}
    result = {k:list(sorted(current.get(k,())))[:targets[k]] for k in keys}
    free = sorted(set(pool)-set(c for cs in result.values() for c in cs))
    for key in keys:
        need = targets[key]-len(result[key])
        result[key].extend(free[:need]); del free[:need]
        result[key].sort()
    assert not free
    return result


def apply_owned(item, cpus):
    """Only touch a birth-verified child session and its same-session threads."""
    root = item['p']
    if root.poll() is not None:
        return dict(exited=True,threads=[])
    process = psutil.Process(root.pid)
    if process.create_time() != item['birth'] or os.getpgid(root.pid) != root.pid:
        raise RuntimeError('owned session identity changed')
    target = set(cpus)
    if not target:
        raise ValueError('empty affinity')
    changed = []; observed = set()
    for _ in range(3):
        try:
            members = [process] + process.children(recursive=True)
        except psutil.NoSuchProcess:
            break
        for member in members:
            try:
                birth = member.create_time()
                if os.getpgid(member.pid) != root.pid:
                    raise RuntimeError('descendant escaped owned process group')
                for thread in member.threads():
                    # Recheck the process before accessing a potentially stale TID.
                    if not member.is_running() or member.create_time() != birth:
                        break
                    try:
                        before = os.sched_getaffinity(thread.id)
                        if before != target:
                            os.sched_setaffinity(thread.id,target)
                            changed.append(dict(pid=member.pid,birth=birth,tid=thread.id,before=sorted(before)))
                        observed.add(thread.id)
                    except ProcessLookupError:
                        continue
            except (psutil.NoSuchProcess,ProcessLookupError):
                continue
    return dict(exited=root.poll() is not None,threads=sorted(observed),changed=changed,cpus=sorted(target))


def reconcile(active, plan):
    records = []
    # Shrink before expand; a pending child is launched only after this returns.
    keys = sorted(active,key=lambda k:len(plan[k])-len(active[k]['cpus']))
    for key in keys:
        item = active[key]
        record = apply_owned(item,plan[key])
        if item['cpus'] != plan[key] or record.get('changed'):
            records.append(dict(id=key,previous=list(item['cpus']),**record))
        item['cpus'] = list(plan[key])
    return records
