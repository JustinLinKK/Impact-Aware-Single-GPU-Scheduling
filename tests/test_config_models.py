from pathlib import Path

from localml_scheduler.config.models import SchedulerConfig


def test_cache_address_uses_fixed_short_socket_dir_for_deep_runtime_root() -> None:
    """Deep runtime roots must not inherit a too-long TMPDIR for AF_UNIX sockets."""
    settings = SchedulerConfig(runtime_root=Path("/tmp/") / ("run-" * 30))

    address = settings.cache_address()

    assert isinstance(address, str)
    assert address.startswith("/tmp/localml-scheduler-")
    assert len(address.encode()) < 100
