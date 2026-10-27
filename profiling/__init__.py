"""Batch-probing helpers, imported lazily for fresh runner processes."""

__all__ = ["run_batch_probe_preflight", "runtime_profile_for_job"]


def __getattr__(name):
    # runner_protocol imports runtime_probe; loading batch_probe here would
    # re-enter runner_protocol before its protocol classes have been defined.
    if name == "run_batch_probe_preflight":
        from .batch_probe import run_batch_probe_preflight
        return run_batch_probe_preflight
    if name == "runtime_profile_for_job":
        from .runtime_probe import runtime_profile_for_job
        return runtime_profile_for_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
