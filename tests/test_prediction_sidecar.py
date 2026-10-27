from pathlib import Path
import pytest

from localml_scheduler.domain import TrainingJob
from localml_scheduler.prediction.job_contract import prediction_metadata, write_prediction_sidecar
from localml_scheduler.prediction.released_student import steps_for_job


def test_framework_neutral_sidecar_and_stale_source(tmp_path):
    source = tmp_path / "generated.py"
    source.write_text("# agent candidate version 1\n")
    job = TrainingJob.create("agent:train", "candidate", str(source), runner_kwargs={"batch_size": 4})
    write_prediction_sidecar(source, {"dataset_size": 100, "precision": "fp32"})
    assert prediction_metadata(job)["source_path"] == str(source)
    assert steps_for_job(job, 4) == 25
    source.write_text("# agent candidate version 2\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        prediction_metadata(job)


def test_explicit_metadata_wins_without_combining_recipes(tmp_path):
    source = tmp_path / "generated.py"
    source.write_text("# candidate\n")
    write_prediction_sidecar(source, {"dataset_size": 100, "precision": "bf16"})
    job = TrainingJob.create("agent:train", "candidate", str(source),
                             metadata={"perfseer_model": {"dataset_size": 20}})
    assert prediction_metadata(job) == {"dataset_size": 20}


def test_sidecar_cannot_substitute_another_source(tmp_path):
    source = tmp_path / "generated.py"
    source.write_text("# candidate\n")
    write_prediction_sidecar(source, {"source_path": str(tmp_path / "surrogate.py")})
    job = TrainingJob.create("agent:train", "candidate", str(source))
    with pytest.raises(ValueError, match="own source file"):
        prediction_metadata(job)


def test_explicit_step_cap_and_nested_accumulation():
    job = TrainingJob.create("agent:train", "candidate", "/unused.py", runner_kwargs={"batch_size": 4},
        metadata={"perfseer_model": {"dataset_size": 1000, "steps_per_epoch": 8}})
    assert steps_for_job(job, 4) == 8
    with pytest.raises(ValueError, match="another physical batch"):
        steps_for_job(job, 8)
    job.metadata["perfseer_model"] = {"dataset_size": 100,
        "workload_contract": {"training": {"gradient_accumulation_steps": 2}}}
    assert steps_for_job(job, 4) == 13
    job.metadata["perfseer_model"] = {"dataset_size": 8920, "steps_per_epoch_by_batch": {"4": 8}}
    assert steps_for_job(job, 4) == 8
    with pytest.raises(ValueError, match="no entry"):
        steps_for_job(job, 8)
