"""Real acceptance order, independent of manifest rank and precheck finish order."""
def ready_order(accepted_sequence, reports, active, done, *, fifo):
    if len(set(accepted_sequence.values())) != len(accepted_sequence):
        raise ValueError('duplicate acceptance sequence')
    if any(type(v) is not int or v <= 0 for v in accepted_sequence.values()):
        raise ValueError('invalid acceptance sequence')
    if not (set(reports) | set(active) | set(done)) <= set(accepted_sequence):
        raise ValueError('non-arrived task state')
    pending=sorted((k for k in accepted_sequence if k not in active and k not in done),
                   key=accepted_sequence.__getitem__)
    ready=[]
    for key in pending:
        if key in reports:
            ready.append(key)
        elif fifo:
            break
    return ready
