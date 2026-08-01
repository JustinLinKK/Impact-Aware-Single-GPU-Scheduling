from __future__ import annotations

import json
import sqlite3

from localml_scheduler.adapters.mlevolve import (
    build_branch_profile_key,
    build_model_family_profile_key,
    normalize_branch_name,
)
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import BatchProbeProfile
from localml_scheduler.storage import BranchProfileStore, StateStore
from localml_scheduler.storage.models import BRANCH_PROFILE_SCHEMA_STATEMENTS, PROFILE_TABLE_NAMES


def _table_exists(db_path, table_name: str) -> bool:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return row is not None


def _count_rows(db_path, table_name: str) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0])


def test_state_store_creates_dedicated_branch_profile_database(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    store = StateStore(settings)

    assert settings.branch_profile_db_path == tmp_path / "db" / "branch_profile.sqlite3"
    assert settings.db_path.exists()
    assert settings.branch_profile_db_path.exists()
    assert _table_exists(settings.db_path, "jobs")
    assert all(not _table_exists(settings.db_path, table) for table in PROFILE_TABLE_NAMES)
    assert all(_table_exists(settings.branch_profile_db_path, table) for table in PROFILE_TABLE_NAMES)
    assert store.branch_profile_store.db_path == settings.branch_profile_db_path


def test_legacy_batch_probe_table_is_migrated(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    settings.ensure_runtime_layout()
    with sqlite3.connect(settings.branch_profile_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE batch_probe_profiles (
                probe_key TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                device_type TEXT NOT NULL,
                shape_signature TEXT NOT NULL,
                batch_param_name TEXT NOT NULL,
                resolved_batch_size INTEGER NOT NULL,
                peak_vram_mb INTEGER,
                memory_total_mb INTEGER,
                target_budget_mb INTEGER,
                observations INTEGER NOT NULL DEFAULT 1,
                last_job_id TEXT,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            )
            """
        )
        connection.commit()

    StateStore(settings)

    with sqlite3.connect(settings.branch_profile_db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(batch_probe_profiles)")}
    assert {"profile_namespace", "hardware_key", "search_mode", "contract_version"} <= columns


def test_profile_writes_land_in_branch_database_only(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    store = StateStore(settings)
    profile = BatchProbeProfile(
        probe_key="branch-profile:resnet50",
        model_key="branch-profile:resnet50",
        device_type="cuda-test",
        shape_signature="mlevolve-branch-shape:test",
        batch_param_name="batch_size",
        resolved_batch_size=32,
        peak_vram_mb=2048,
        memory_total_mb=8192,
        target_budget_mb=6144,
        metadata={"branch_name": "resnet50"},
    )

    store.upsert_batch_probe_profile(profile)

    restored = store.get_batch_probe_profile("branch-profile:resnet50")
    assert restored is not None
    assert restored.resolved_batch_size == 32
    assert restored.metadata["branch_name"] == "resnet50"
    assert _count_rows(settings.branch_profile_db_path, "batch_probe_profiles") == 1
    assert not _table_exists(settings.db_path, "batch_probe_profiles")


def test_compatible_profile_lookup_excludes_legacy_rows(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    store = StateStore(settings)
    namespace = "branch-profile:resnet18"
    store.upsert_batch_probe_profile(
        BatchProbeProfile(
            probe_key="legacy-branch-only",
            model_key="resnet18",
            device_type="test-gpu",
            shape_signature="shape-a",
            batch_param_name="batch_size",
            resolved_batch_size=8,
        )
    )
    store.upsert_batch_probe_profile(
        BatchProbeProfile(
            probe_key="concrete-v2",
            model_key="resnet18",
            device_type="test-gpu",
            shape_signature="shape-a",
            batch_param_name="batch_size",
            resolved_batch_size=16,
            profile_namespace=namespace,
            hardware_key="gpu-a",
            search_mode="power_of_two",
            contract_version=2,
        )
    )

    compatible = store.get_compatible_batch_probe_profile(
        profile_namespace=namespace,
        hardware_key="gpu-a",
        shape_signature="shape-a",
        search_mode="power_of_two",
        contract_version=2,
    )

    assert compatible is not None
    assert compatible.probe_key == "concrete-v2"
    assert store.get_compatible_batch_probe_profile(
        profile_namespace=namespace,
        hardware_key="gpu-b",
        shape_signature="shape-a",
        search_mode="power_of_two",
        contract_version=2,
    ) is None


def test_branch_store_imports_legacy_scheduler_profiles_once(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path)
    settings.ensure_runtime_layout()

    with sqlite3.connect(settings.db_path) as connection:
        for statement in BRANCH_PROFILE_SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO batch_probe_profiles(
                probe_key, model_key, device_type, shape_signature, batch_param_name,
                resolved_batch_size, peak_vram_mb, memory_total_mb, target_budget_mb,
                observations, last_job_id, updated_at, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-probe",
                "branch-profile:resnet50",
                "cuda-test",
                "shape-resnet50",
                "batch_size",
                16,
                1024,
                8192,
                6144,
                1,
                "legacy-job",
                "2026-01-01T00:00:00+00:00",
                json.dumps({"legacy": True}),
            ),
        )
        connection.commit()

    store = BranchProfileStore(settings)
    assert store.get_batch_probe_profile("legacy-probe") is not None
    assert _count_rows(settings.branch_profile_db_path, "batch_probe_profiles") == 1

    with sqlite3.connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO batch_probe_profiles(
                probe_key, model_key, device_type, shape_signature, batch_param_name,
                resolved_batch_size, observations, updated_at, metadata_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-probe-late",
                "branch-profile:resnet50",
                "cuda-test",
                "shape-resnet50",
                "batch_size",
                64,
                1,
                "2026-01-02T00:00:00+00:00",
                "{}",
            ),
        )
        connection.commit()

    second_store = BranchProfileStore(settings)
    assert second_store.get_batch_probe_profile("legacy-probe-late") is None
    assert _count_rows(settings.branch_profile_db_path, "batch_probe_profiles") == 1


def test_branch_identity_is_task_independent_and_canonicalizes_resnet_variants() -> None:
    assert normalize_branch_name("resnet50_v2") == "resnet50"
    assert normalize_branch_name("resnet50_augmented") == "resnet50"
    assert normalize_branch_name("timm.create_model('resnet50d')") == "resnet50"

    key_a = build_model_family_profile_key(task_id="exp-a", model_family="ResNet50_v2")
    key_b = build_model_family_profile_key(task_id="exp-b", model_family="resnet50_augmented")
    key_c = build_branch_profile_key("resnet50d")

    assert key_a == key_b == key_c == "branch-profile:resnet50"
