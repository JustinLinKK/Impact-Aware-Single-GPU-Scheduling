"""Verify installed students against the public release's content manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hardware_matches(name, trained):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(trained) + r"(?![A-Za-z0-9])", str(name), re.I) is not None


def load_verified_student_files(release_root, model_path, hardware_name):
    """Use a trusted release checkout, never an arbitrary unverified pickle.

    RELEASE.json binds the asset manifest; each extracted file is checked against
    its published archive member. This also supports later additive releases
    without hard-coding a single device's TorchScript hash into the scheduler.
    """
    release_root = Path(release_root)
    release = json.loads((release_root / "RELEASE.json").read_text())
    manifest_path = release_root / "assets/manifest.json"
    if digest(manifest_path) != release.get("assets_manifest_sha256"):
        raise ValueError("release asset manifest hash mismatch")
    members = {}
    for asset in json.loads(manifest_path.read_text())["assets"]:
        for member in asset["members"]:
            parts = Path(member["path"]).parts
            if len(parts) == 4 and parts[:2] == ("PerfSeer-release-v1", "checkpoints"):
                model_id, filename = parts[2:]
                if model_id.startswith(("S_", "TS_")) and model_id in release["models"]:
                    key = (model_id, filename)
                    if key in members and members[key] != member["sha256"]:
                        raise ValueError(f"ambiguous release member: {key}")
                    members[key] = member["sha256"]
    if model_path:
        root = Path(model_path).expanduser().resolve()
    else:
        device = next((name for name in ("A10G", "A100", "A10") if hardware_matches(hardware_name, name)), None)
        if device is None:
            raise ValueError(f"no released student trained for {hardware_name}")
        root = release_root / "extracted/PerfSeer-release-v1/checkpoints" / f"S_{device}"
    artifact_hash = digest(root / "student.ts")
    matches = [model for (model, filename), value in members.items()
               if filename == "student.ts" and value == artifact_hash]
    if len(matches) != 1:
        raise ValueError("student artifact hash is absent or ambiguous in the public release manifest")
    model_id = matches[0]
    for filename in ("student.ts", "normalization.pt", "architecture.pt"):
        if digest(root / filename) != members.get((model_id, filename)):
            raise ValueError(f"student {filename.removesuffix('.pt')} hash mismatch")
    trained = model_id.split("_")[1]
    if not hardware_matches(hardware_name, trained):
        raise ValueError(f"student is trained for {trained}, not {hardware_name}")
    return root, model_id, trained, artifact_hash
