"""Reproducible experiment manifests and content fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import RuntimeConfig, assert_allowed_path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(project_root: Path) -> tuple[str, dict[str, str]]:
    """Hash package sources and packaging metadata, excluding tests and outputs."""

    project_root = project_root.resolve(strict=True)
    candidates = [project_root / "pyproject.toml"]
    candidates.extend(sorted((project_root / "src").rglob("*.py")))
    files: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        files[relative] = file_sha256(path)
    if not files:
        raise ValueError("source tree contains no package files")
    return canonical_sha256(files), files


def _identity_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Remove storage/telemetry settings that do not define learned weights."""

    result = json.loads(json.dumps(raw))
    result.pop("architecture_spec", None)
    result.pop("run_root", None)
    result.pop("logging", None)
    checkpoint = result.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("every_optimizer_steps", None)
        checkpoint.pop("keep_last", None)
        checkpoint.pop("save_at_pass_end", None)
    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("root", None)
        dataset.pop("manifest", None)
    return result


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: int
    created_at_utc: str
    experiment_fingerprint: str
    architecture_id: str
    architecture_spec_sha256: str
    operational_config_sha256: str
    dataset_manifest_sha256: str
    source_tree_sha256: str
    source_files: Mapping[str, str]
    resolved_config: Mapping[str, Any]
    architecture_spec: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_experiment_manifest(
    config: RuntimeConfig,
    *,
    project_root: str | os.PathLike[str],
) -> ExperimentManifest:
    """Build a manifest without reading dataset payloads or unrelated trees."""

    config.require_resolved()
    raw = config.as_dict()
    dataset_manifest = Path(raw["dataset"]["authoritative_manifest"]).resolve(strict=True)
    assert_allowed_path(dataset_manifest.as_posix(), "dataset.authoritative_manifest")
    source_hash, source_files = source_tree_fingerprint(Path(project_root))
    spec_hash = file_sha256(config.spec_path)
    data_hash = file_sha256(dataset_manifest)
    config_hash = canonical_sha256(raw)
    identity = {
        "architecture_spec_sha256": spec_hash,
        "dataset_manifest_sha256": data_hash,
        "source_tree_sha256": source_hash,
        "config": _identity_config(raw),
    }
    experiment_fingerprint = canonical_sha256(identity)
    return ExperimentManifest(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment_fingerprint=experiment_fingerprint,
        architecture_id=str(config.spec["architecture_id"]),
        architecture_spec_sha256=spec_hash,
        operational_config_sha256=config_hash,
        dataset_manifest_sha256=data_hash,
        source_tree_sha256=source_hash,
        source_files=source_files,
        resolved_config=raw,
        architecture_spec=dict(config.spec),
    )


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON in the destination directory and publish it with os.replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_experiment_manifest(path: Path, manifest: ExperimentManifest) -> None:
    atomic_write_json(path, manifest.as_dict())
