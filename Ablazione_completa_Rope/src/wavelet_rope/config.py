"""Strict loading for the operational config and fixed architecture spec."""

from __future__ import annotations

import json
import hashlib
import math
import os
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ConfigError, RequiredValueError

REQUIRED = "REQUIRED"
FORBIDDEN_WORK_ROOT = PurePosixPath("/work/u10970284/airway ATM26 project")
EXPECTED_ARCHITECTURE_ID = "WR-Agent-4B-Wildcard-v1"
EXPECTED_SPEC_VERSION = "0.3"
EXPECTED_ATTENTION_BACKEND = "EFFICIENT_ATTENTION"  # CUDA/A40 lane (Astra 14/09 §2)
# Backend fixed per GPU family after the 14/09 gate: on ROCm the memory-efficient kernel rejects
# Q/K != V and > 256 (sdp_utils.cpp:132), FlexAttention with Q/K zero-padded to 512 is exact and passed
# every required case (RISULTATI-GATE-BACKEND-20260914.md). The user fixed FLEX_PAD512 for MI300X.
EXPECTED_BACKEND_BY_GPU_FAMILY = {"A40": "EFFICIENT_ATTENTION", "MI300X": "FLEX_PAD512"}
EXPECTED_PLATFORM_BY_GPU_FAMILY = {"A40": "cuda", "MI300X": "rocm"}
EXPECTED_DATA_REPO = "nz00shuuuu/AnomalyXL"
EXPECTED_DATA_REVISION = "df363a467f855e55aaad414b0bdd58590829df9a"
EXPECTED_DATA_FILES = {
    "train": {
        "relative_path": "anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet",
        "rows": 4488,
        "sha256": "b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d",
        "manifest_key": "precise/train",
    },
    "validation": {
        "relative_path": "anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/val.parquet",
        "rows": 520,
        "sha256": "90b805b8f40e203b76873a9cf3a0f986e21700c9185c74acc33c44a183b1d977",
        "manifest_key": "precise/val",
    },
    "test": {
        "relative_path": "anomalyXL/datasets/anomalyxl-precise/data.parquet",
        "rows": 800,
        "sha256": "7b37269d106f70ba7bee5feb504c00059bc880217a56a8279a2631c8bf5e1524",
        "manifest_key": "precise/test",
    },
}
FORBIDDEN_FIT_ROW_COUNTS = (723, 77)


def _collect_required(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if value == REQUIRED:
        found.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_collect_required(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_collect_required(child, f"{prefix}[{index}]"))
    return found


def _lookup(config: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing configuration field: {dotted_path}")
        current = current[part]
    return current


def _expect_type(
    config: Mapping[str, Any],
    dotted_path: str,
    expected: type | tuple[type, ...],
    *,
    allow_required: bool,
) -> Any:
    value = _lookup(config, dotted_path)
    if allow_required and value == REQUIRED:
        return value
    if isinstance(value, bool) and expected is not bool:
        raise ConfigError(f"{dotted_path} has invalid boolean value")
    if not isinstance(value, expected):
        expected_name = (
            "/".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ConfigError(f"{dotted_path} must be {expected_name}, got {type(value).__name__}")
    return value


def _expect_positive(config: Mapping[str, Any], dotted_path: str, *, allow_required: bool) -> None:
    value = _expect_type(config, dotted_path, (int, float), allow_required=allow_required)
    if value == REQUIRED:
        return
    if not math.isfinite(float(value)) or value <= 0:
        raise ConfigError(f"{dotted_path} must be finite and greater than zero")


def _expect_positive_int(
    config: Mapping[str, Any], dotted_path: str, *, allow_required: bool
) -> None:
    value = _expect_type(config, dotted_path, int, allow_required=allow_required)
    if value != REQUIRED and value <= 0:
        raise ConfigError(f"{dotted_path} must be greater than zero")


def _normalise_posix_path(value: str) -> PurePosixPath | None:
    if not value.startswith("/"):
        return None
    parts: list[str] = []
    for part in PurePosixPath(value).parts:
        if part in ("/", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath("/", *parts)


def assert_allowed_path(value: str, field: str) -> None:
    """Reject the project tree that the user explicitly placed out of scope."""

    normalised = _normalise_posix_path(value)
    if normalised is None:
        return
    if normalised == FORBIDDEN_WORK_ROOT or FORBIDDEN_WORK_ROOT in normalised.parents:
        raise ConfigError(f"{field} points inside the forbidden work tree")


def _assert_verified_scratch_path(value: str, field: str) -> None:
    assert_allowed_path(value, field)
    normalised = _normalise_posix_path(value)
    if normalised is None:
        return
    disallowed_roots = (PurePosixPath("/home"), PurePosixPath("/work/home"))
    if any(root == normalised or root in normalised.parents for root in disallowed_roots):
        raise ConfigError(f"{field} must not use a home or shared work cache")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_authoritative_data_manifest(raw: Mapping[str, Any]) -> None:
    manifest_path = Path(_lookup(raw, "dataset.authoritative_manifest")).resolve(strict=True)
    normalised_suffix = "integrity/artifacts/va1/RUN_MANIFEST.json"
    if not manifest_path.as_posix().endswith(normalised_suffix):
        raise ConfigError(
            "dataset.authoritative_manifest must be integrity/artifacts/va1/RUN_MANIFEST.json"
        )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError("authoritative dataset manifest is unreadable") from error
    if document.get("qwen_model_id") != "Qwen/Qwen3.5-4B":
        raise ConfigError("authoritative manifest has the wrong Qwen model id")
    if document.get("qwen_revision") != _lookup(raw, "backbone_model_revision"):
        raise ConfigError("authoritative manifest has the wrong Qwen revision")
    data = document.get("data_manifest")
    if not isinstance(data, dict):
        raise ConfigError("authoritative manifest has no data_manifest")
    if data.get("repo") != EXPECTED_DATA_REPO or data.get("revision") != EXPECTED_DATA_REVISION:
        raise ConfigError("authoritative manifest has the wrong AnomalyXL pin")
    files = data.get("files")
    if not isinstance(files, dict):
        raise ConfigError("authoritative manifest has no dataset file table")
    dataset_root = Path(_lookup(raw, "dataset.root"))
    for split, expected in EXPECTED_DATA_FILES.items():
        item = files.get(expected["manifest_key"])
        if not isinstance(item, dict):
            raise ConfigError(f"authoritative manifest is missing {split}")
        for field in ("path", "rows", "sha256"):
            expected_field = "relative_path" if field == "path" else field
            if item.get(field) != expected[expected_field]:
                raise ConfigError(f"authoritative manifest has wrong {split} {field}")
        payload = dataset_root / expected["relative_path"]
        if not payload.is_file():
            raise ConfigError(f"authoritative {split} parquet is missing")
        if _sha256_file(payload) != expected["sha256"]:
            raise ConfigError(f"authoritative {split} parquet checksum mismatch")


def _validate_shape(raw: Mapping[str, Any], *, allow_required: bool) -> None:
    _expect_type(raw, "schema_version", int, allow_required=allow_required)
    _expect_type(raw, "phase", str, allow_required=allow_required)
    _expect_type(raw, "smoke_only", bool, allow_required=allow_required)
    _expect_type(raw, "architecture_spec", str, allow_required=allow_required)
    _expect_type(raw, "run_root", str, allow_required=allow_required)
    _expect_type(raw, "seed", int, allow_required=allow_required)
    _expect_type(raw, "dataset.root", str, allow_required=allow_required)
    _expect_type(raw, "dataset.authoritative_manifest", str, allow_required=allow_required)
    _expect_type(raw, "dataset.repo", str, allow_required=allow_required)
    _expect_type(raw, "dataset.revision", str, allow_required=allow_required)
    for split in EXPECTED_DATA_FILES:
        _expect_type(raw, f"dataset.{split}_relative_path", str, allow_required=allow_required)
        _expect_positive_int(raw, f"dataset.{split}_rows", allow_required=allow_required)
        _expect_type(raw, f"dataset.{split}_sha256", str, allow_required=allow_required)
    forbidden_counts = _expect_type(
        raw, "dataset.forbidden_fit_row_counts", list, allow_required=allow_required
    )
    if forbidden_counts != list(FORBIDDEN_FIT_ROW_COUNTS):
        raise ConfigError("dataset.forbidden_fit_row_counts must be [723, 77]")
    _expect_type(raw, "backbone.transformers_version", str, allow_required=allow_required)
    _expect_type(raw, "backbone.tokenizer_revision", str, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.root", str, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.verification_manifest", str, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.cache_scope", str, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.allow_shared_cache", bool, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.local_files_only", bool, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.allow_download", bool, allow_required=allow_required)
    _expect_type(raw, "backbone.snapshot.quantization", str, allow_required=allow_required)
    _expect_positive(raw, "normalization.mad_epsilon", allow_required=allow_required)
    _expect_type(raw, "runtime.gpu_family", str, allow_required=allow_required)
    _expect_type(raw, "runtime.world_size", int, allow_required=allow_required)
    _expect_type(raw, "runtime.attention_backend", str, allow_required=allow_required)
    _expect_type(raw, "runtime.platform", str, allow_required=allow_required)
    _expect_type(raw, "runtime.attention_qk_pad_to", int, allow_required=allow_required)
    _expect_type(raw, "runtime.torch_build", str, allow_required=allow_required)
    _expect_type(raw, "runtime.accelerator_runtime", str, allow_required=allow_required)
    _expect_type(raw, "runtime.compute_dtype", str, allow_required=allow_required)
    _expect_type(raw, "runtime.gradient_checkpointing", bool, allow_required=allow_required)
    _expect_type(raw, "runtime.requires_cuda_gate", bool, allow_required=allow_required)
    _expect_type(raw, "runtime.requires_real_qwen_smoke", bool, allow_required=allow_required)
    _expect_type(raw, "optimizer.name", str, allow_required=allow_required)
    _expect_positive(raw, "optimizer.beta1", allow_required=allow_required)
    _expect_positive(raw, "optimizer.beta2", allow_required=allow_required)
    _expect_positive(raw, "optimizer.epsilon", allow_required=allow_required)
    _expect_type(raw, "optimizer.lr_scheduler", str, allow_required=allow_required)
    _expect_positive(raw, "optimizer.p3_frontend_learning_rate", allow_required=allow_required)
    _expect_positive(raw, "optimizer.p3_lora_learning_rate", allow_required=allow_required)
    _expect_type(raw, "training.segment_loss_reduction", str, allow_required=allow_required)
    _expect_positive_int(raw, "checkpoint.every_optimizer_steps", allow_required=allow_required)
    _expect_positive_int(raw, "checkpoint.keep_last", allow_required=allow_required)
    _expect_type(raw, "checkpoint.save_at_pass_end", bool, allow_required=allow_required)
    _expect_type(raw, "checkpoint.optimizer_boundary_only", bool, allow_required=allow_required)
    _expect_type(raw, "evaluation.selection_metric", str, allow_required=allow_required)
    _expect_type(raw, "evaluation.selection_tiebreaker", str, allow_required=allow_required)
    _expect_positive_int(raw, "logging.every_optimizer_steps", allow_required=allow_required)

    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    if raw.get("phase") not in {"P0", "P1", "P2", "P3"}:
        raise ConfigError("phase must be one of P0, P1, P2 or P3")
    if raw.get("phase") == "P0" and raw.get("smoke_only") is not True:
        raise ConfigError("P0 must set smoke_only=true")
    if raw.get("phase") != "P0" and raw.get("smoke_only") is not False:
        raise ConfigError("P1/P2/P3 must set smoke_only=false")
    gpu_family = _lookup(raw, "runtime.gpu_family")
    if gpu_family not in EXPECTED_BACKEND_BY_GPU_FAMILY:
        raise ConfigError("runtime.gpu_family must be one of A40 (CUDA lane) or MI300X (ROCm lane)")
    expected_backend = EXPECTED_BACKEND_BY_GPU_FAMILY[gpu_family]
    if _lookup(raw, "runtime.attention_backend") != expected_backend:
        raise ConfigError(
            f"runtime.attention_backend must be {expected_backend} for gpu_family {gpu_family}"
        )
    if _lookup(raw, "runtime.platform") != EXPECTED_PLATFORM_BY_GPU_FAMILY[gpu_family]:
        raise ConfigError(
            f"runtime.platform must be {EXPECTED_PLATFORM_BY_GPU_FAMILY[gpu_family]} for gpu_family {gpu_family}"
        )
    if expected_backend == "FLEX_PAD512" and _lookup(raw, "runtime.attention_qk_pad_to") != 512:
        raise ConfigError("runtime.attention_qk_pad_to must be 512 for the FLEX_PAD512 backend")
    if _lookup(raw, "runtime.compute_dtype") != "bf16":
        raise ConfigError("runtime.compute_dtype must be bf16")
    if _lookup(raw, "runtime.requires_cuda_gate") is not True:
        raise ConfigError("runtime.requires_cuda_gate must be true")
    if _lookup(raw, "runtime.requires_real_qwen_smoke") is not True:
        raise ConfigError("runtime.requires_real_qwen_smoke must be true")
    snapshot_rules = {
        "backbone.snapshot.cache_scope": "run_local_verified_scratch",
        "backbone.snapshot.allow_shared_cache": False,
        "backbone.snapshot.local_files_only": True,
        "backbone.snapshot.allow_download": False,
        "backbone.snapshot.quantization": "none",
    }
    for field, expected in snapshot_rules.items():
        if _lookup(raw, field) != expected:
            raise ConfigError(f"{field} must be {expected!r}")
    boundary_only = _lookup(raw, "checkpoint.optimizer_boundary_only")
    if boundary_only is not True:
        raise ConfigError("checkpoint.optimizer_boundary_only must be true")
    world_size = _lookup(raw, "runtime.world_size")
    if world_size != REQUIRED and world_size not in (1, 2):
        raise ConfigError("runtime.world_size must be 1 or 2")
    for beta_name in ("optimizer.beta1", "optimizer.beta2"):
        beta = _lookup(raw, beta_name)
        if beta != REQUIRED and not 0.0 < float(beta) < 1.0:
            raise ConfigError(f"{beta_name} must be between zero and one")

    expected_dataset_fields = {
        "dataset.repo": EXPECTED_DATA_REPO,
        "dataset.revision": EXPECTED_DATA_REVISION,
    }
    for split, expected in EXPECTED_DATA_FILES.items():
        expected_dataset_fields[f"dataset.{split}_relative_path"] = expected["relative_path"]
        expected_dataset_fields[f"dataset.{split}_rows"] = expected["rows"]
        expected_dataset_fields[f"dataset.{split}_sha256"] = expected["sha256"]
    for field, expected in expected_dataset_fields.items():
        if _lookup(raw, field) != expected:
            raise ConfigError(f"{field} does not match the authoritative VA1 manifest")

    for field in ("run_root", "dataset.root", "dataset.authoritative_manifest"):
        value = _lookup(raw, field)
        if value != REQUIRED:
            assert_allowed_path(value, field)
    snapshot_root = _lookup(raw, "backbone.snapshot.root")
    if snapshot_root != REQUIRED:
        _assert_verified_scratch_path(snapshot_root, "backbone.snapshot.root")
    snapshot_manifest = _lookup(raw, "backbone.snapshot.verification_manifest")
    if snapshot_manifest != REQUIRED:
        assert_allowed_path(snapshot_manifest, "backbone.snapshot.verification_manifest")


def _validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("architecture_id") != EXPECTED_ARCHITECTURE_ID:
        raise ConfigError("unexpected architecture_id in prototype.spec.json")
    if spec.get("spec_version") != EXPECTED_SPEC_VERSION:
        raise ConfigError("unexpected architecture spec version")
    if spec.get("architecture_choices_fixed") is not True:
        raise ConfigError("architecture spec is not marked fixed")
    temporal = _lookup(spec, "backbone.temporal_rope")
    expected = {
        "factorized_qk_dim_inference": 352,
        "factorized_qk_dim_curriculum": 384,
        "value_head_dim": 256,
        "attention_scale": 0.0625,
        "single_joint_softmax": True,
    }
    for key, value in expected.items():
        if temporal.get(key) != value:
            raise ConfigError(f"architecture spec has unexpected temporal_rope.{key}")


def _backbone_revision(spec: Mapping[str, Any]) -> str:
    revision = _lookup(spec, "backbone.revision")
    if not isinstance(revision, str):
        raise ConfigError("architecture backbone revision must be a string")
    return revision


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated operational values plus the immutable architecture spec."""

    source_path: Path
    spec_path: Path
    raw: Mapping[str, Any]
    spec: Mapping[str, Any]
    unresolved_fields: tuple[str, ...]

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved_fields

    def require_resolved(self) -> None:
        if self.unresolved_fields:
            joined = ", ".join(self.unresolved_fields)
            raise RequiredValueError(f"Unresolved REQUIRED fields: {joined}")

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.raw))


def load_runtime_config(
    path: str | os.PathLike[str],
    *,
    allow_required: bool = False,
    validate_paths: bool = True,
) -> RuntimeConfig:
    """Load TOML without silently supplying any unresolved design decision."""

    source_path = Path(path).resolve(strict=True)
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    unresolved = tuple(sorted(_collect_required(raw)))
    if unresolved and not allow_required:
        joined = ", ".join(unresolved)
        raise RequiredValueError(f"Unresolved REQUIRED fields: {joined}")
    _validate_shape(raw, allow_required=allow_required)

    spec_value = _lookup(raw, "architecture_spec")
    if spec_value == REQUIRED:
        raise ConfigError("architecture_spec cannot be REQUIRED")
    spec_path = Path(spec_value)
    if not spec_path.is_absolute():
        spec_path = source_path.parent / spec_path
    spec_path = spec_path.resolve(strict=True)
    assert_allowed_path(spec_path.as_posix(), "architecture_spec")
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    _validate_spec(spec)
    raw["backbone_model_revision"] = _backbone_revision(spec)

    if validate_paths and not unresolved:
        dataset_root = Path(_lookup(raw, "dataset.root"))
        dataset_manifest = Path(_lookup(raw, "dataset.authoritative_manifest"))
        run_root = Path(_lookup(raw, "run_root"))
        snapshot_root = Path(_lookup(raw, "backbone.snapshot.root"))
        snapshot_manifest = Path(_lookup(raw, "backbone.snapshot.verification_manifest"))
        if not dataset_root.is_absolute() or not dataset_root.is_dir():
            raise ConfigError("dataset.root must be an existing absolute directory")
        if not dataset_manifest.is_absolute() or not dataset_manifest.is_file():
            raise ConfigError("dataset.authoritative_manifest must be an existing absolute file")
        if not run_root.is_absolute():
            raise ConfigError("run_root must be absolute")
        if not snapshot_root.is_absolute() or not snapshot_root.is_dir():
            raise ConfigError("backbone.snapshot.root must be an existing absolute directory")
        if not snapshot_manifest.is_absolute() or not snapshot_manifest.is_file():
            raise ConfigError(
                "backbone.snapshot.verification_manifest must be an existing absolute file"
            )
        if raw["phase"] != "P0":
            _validate_authoritative_data_manifest(raw)

    return RuntimeConfig(
        source_path=source_path,
        spec_path=spec_path,
        raw=MappingProxyType(deepcopy(raw)),
        spec=MappingProxyType(deepcopy(spec)),
        unresolved_fields=unresolved,
    )
