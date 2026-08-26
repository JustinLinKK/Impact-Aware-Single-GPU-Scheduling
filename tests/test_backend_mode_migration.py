from __future__ import annotations

import json
import sqlite3

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import RuntimeProfile
from localml_scheduler.migrations import migrate_backend_mode_v2
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def test_backend_mode_migration_is_dry_run_first_and_idempotent(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    store = SQLiteStateStore(settings)
    mps_profile = RuntimeProfile.create(
        signature="mps-model",
        hardware_key="gpu",
        backend_name="mps_process",
        resolved_batch_size=8,
        strategy="epoch_1",
        metadata={"cuda_mps_pipe_directory": "/tmp/mps"},
    )
    retired_profile = RuntimeProfile.create(
        signature="retired-model",
        hardware_key="gpu",
        backend_name="cuda_process",
        resolved_batch_size=4,
        strategy="epoch_1",
    )
    canonical_profile = RuntimeProfile.create(
        signature="canonical-model",
        hardware_key="gpu",
        backend_name="cuda_process",
        resolved_batch_size=2,
        strategy="epoch_1",
    )
    store.upsert_runtime_profile(mps_profile)
    store.upsert_runtime_profile(retired_profile)
    store.upsert_runtime_profile(canonical_profile)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute(
            "UPDATE runtime_profiles SET backend_name='mps' WHERE profile_key=?",
            (mps_profile.profile_key,),
        )
        connection.execute(
            "UPDATE runtime_profiles SET backend_name='stream' WHERE profile_key=?",
            (retired_profile.profile_key,),
        )
        connection.execute(
            "UPDATE runtime_profiles SET profile_key='legacy-v1-key' WHERE profile_key=?",
            (canonical_profile.profile_key,),
        )
        connection.commit()
    store.log_event("historical", payload={"backend": "stream", "current": "mps_process"})

    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        "packing_backend: mps\nhistorical_backend: stream_mps\n",
        encoding="utf-8",
    )
    cache_path = settings.runtime_root / "cache_meta" / "legacy.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"backend":"cuda_stream"}', encoding="utf-8")

    dry = migrate_backend_mode_v2(
        settings, dry_run=True, config_paths=[config_path]
    )
    assert dry["profiles"]["mps"] == 1
    assert dry["profiles"]["stream"] == 1
    assert dry["config_references"]["mps"] == 1
    assert dry["config_references"]["stream_mps"] == 1
    assert dry["config_references"]["stream"] == 0
    assert dry["cache_entries"]["cuda_stream"] == 1
    assert dry["events"]["stream"] == 1
    assert dry["events"]["mps"] == 0
    assert dry["schema_v2_rekey_rows"] == 1
    assert dry["would_change_rows"] == 3

    applied = migrate_backend_mode_v2(settings, dry_run=False)
    assert applied["changed_rows"] == 3
    with sqlite3.connect(settings.db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT backend_name, metadata_json FROM runtime_profiles ORDER BY signature"
        ).fetchall()
    by_backend = {row["backend_name"]: json.loads(row["metadata_json"] or "{}") for row in rows}
    assert by_backend["mps_process"]["original_backend_identifier"] == "mps"
    assert by_backend["stream"]["retired_backend"] is True
    assert by_backend["stream"]["selectable"] is False

    repeated = migrate_backend_mode_v2(settings, dry_run=False)
    assert repeated["changed_rows"] == 0
