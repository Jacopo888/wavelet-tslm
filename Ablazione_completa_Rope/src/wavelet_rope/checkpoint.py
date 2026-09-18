"""Atomic checkpoints with strict fingerprinted resume semantics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import CheckpointError, FingerprintMismatchError
from .manifest import atomic_write_json, canonical_json_bytes, file_sha256

CHECKPOINT_PATTERN = re.compile(r"^step_(\d{12})$")
REQUIRED_ARTIFACTS = frozenset(
    {"model.safetensors", "optimizer.pt", "schedulers.pt", "rng.pt"}
)
REQUIRED_STATE_FIELDS = frozenset(
    {
        "experiment_fingerprint",
        "phase",
        "pass_index",
        "optimizer_step",
        "sample_cursor",
        "micro_step_in_accumulation",
        "wildcard_lambda",
    }
)


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    optimizer_step: int
    trainer_state: Mapping[str, Any]


class AtomicCheckpointStore:
    """Publish immutable checkpoint directories after every file is durable."""

    def __init__(self, root: str | os.PathLike[str], experiment_fingerprint: str):
        self.root = Path(root)
        if not re.fullmatch(r"[0-9a-f]{64}", experiment_fingerprint):
            raise ValueError("experiment_fingerprint must be a lowercase SHA-256 hex digest")
        self.experiment_fingerprint = experiment_fingerprint

    def _validate_state(self, state: Mapping[str, Any]) -> int:
        missing = sorted(REQUIRED_STATE_FIELDS - state.keys())
        if missing:
            raise CheckpointError(f"trainer_state is missing: {', '.join(missing)}")
        if state["experiment_fingerprint"] != self.experiment_fingerprint:
            raise FingerprintMismatchError("trainer_state experiment fingerprint mismatch")
        step = state["optimizer_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise CheckpointError("optimizer_step must be a non-negative integer")
        if state["micro_step_in_accumulation"] != 0:
            raise CheckpointError("checkpoints are only valid at optimizer boundaries")
        value = state["wildcard_lambda"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CheckpointError("wildcard_lambda must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise CheckpointError("wildcard_lambda must be between zero and one")
        return step

    def save(
        self,
        *,
        trainer_state: Mapping[str, Any],
        artifact_writers: Mapping[str, Callable[[Path], None]],
    ) -> CheckpointRef:
        """Save serialized model/trainer artifacts and atomically publish a step."""

        step = self._validate_state(trainer_state)
        names = set(artifact_writers)
        missing = sorted(REQUIRED_ARTIFACTS - names)
        if missing:
            raise CheckpointError(f"artifact writers are missing: {', '.join(missing)}")
        for name in names:
            if Path(name).name != name or name in {"trainer_state.json", "COMPLETE"}:
                raise CheckpointError(f"invalid checkpoint artifact name: {name}")

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"step_{step:012d}"
        if destination.exists():
            raise CheckpointError(f"checkpoint already exists: {destination.name}")
        temporary = self.root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            artifact_hashes: dict[str, str] = {}
            for name in sorted(artifact_writers):
                artifact_path = temporary / name
                artifact_writers[name](artifact_path)
                if not artifact_path.is_file():
                    raise CheckpointError(f"writer did not create artifact: {name}")
                artifact_hashes[name] = file_sha256(artifact_path)

            state = dict(trainer_state)
            state["artifacts"] = artifact_hashes
            atomic_write_json(temporary / "trainer_state.json", state)
            state_hash = hashlib.sha256(canonical_json_bytes(state) + b"\n").hexdigest()
            complete = {
                "schema_version": 1,
                "experiment_fingerprint": self.experiment_fingerprint,
                "optimizer_step": step,
                "trainer_state_sha256": state_hash,
            }
            atomic_write_json(temporary / "COMPLETE", complete)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self.validate(destination)

    def validate(self, path: str | os.PathLike[str]) -> CheckpointRef:
        checkpoint = Path(path)
        match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if not match or not checkpoint.is_dir():
            raise CheckpointError(f"invalid checkpoint directory: {checkpoint}")
        state_path = checkpoint / "trainer_state.json"
        complete_path = checkpoint / "COMPLETE"
        if not state_path.is_file() or not complete_path.is_file():
            raise CheckpointError(f"checkpoint is incomplete: {checkpoint.name}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"checkpoint metadata is unreadable: {checkpoint.name}") from error

        step = self._validate_state(state)
        if step != int(match.group(1)) or complete.get("optimizer_step") != step:
            raise CheckpointError("checkpoint step metadata does not match its directory")
        if complete.get("experiment_fingerprint") != self.experiment_fingerprint:
            raise FingerprintMismatchError("checkpoint COMPLETE fingerprint mismatch")
        expected_state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
        if complete.get("trainer_state_sha256") != expected_state_hash:
            raise CheckpointError("trainer_state checksum mismatch")
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
            raise CheckpointError("checkpoint artifact index is incomplete")
        for name, expected_hash in artifacts.items():
            if Path(name).name != name:
                raise CheckpointError("checkpoint artifact index contains an invalid name")
            artifact_path = checkpoint / name
            if not artifact_path.is_file() or file_sha256(artifact_path) != expected_hash:
                raise CheckpointError(f"checkpoint artifact checksum mismatch: {name}")
        return CheckpointRef(path=checkpoint, optimizer_step=step, trainer_state=state)

    def latest_complete(self) -> CheckpointRef | None:
        """Return the highest valid complete checkpoint; ignore temp/incomplete dirs."""

        if not self.root.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match and path.is_dir() and (path / "COMPLETE").is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        _, latest = max(candidates, key=lambda item: item[0])
        return self.validate(latest)

    def resolve_resume(self, resume: str | os.PathLike[str] | None) -> CheckpointRef | None:
        """Resolve none/auto/an explicit checkpoint under the same experiment."""

        if resume is None or str(resume).lower() == "none":
            return None
        if str(resume).lower() == "auto":
            return self.latest_complete()
        return self.validate(Path(resume).resolve(strict=True))

def inspect_foreign_checkpoint(path: str | os.PathLike[str]) -> CheckpointRef:
    """Validate a COMPLETE checkpoint directory WITHOUT requiring the current experiment fingerprint.

    Used to evaluate a run with a newer tool tree, or to initialise a new phase from another experiment
    (P2 from P1).  Artifact checksums and the COMPLETE marker are still verified; the foreign fingerprint is
    returned in ``trainer_state`` so callers can record it explicitly.
    """

    checkpoint = Path(path)
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if not match or not checkpoint.is_dir():
        raise CheckpointError(f"invalid checkpoint directory: {checkpoint}")
    state_path = checkpoint / "trainer_state.json"
    complete_path = checkpoint / "COMPLETE"
    if not state_path.is_file() or not complete_path.is_file():
        raise CheckpointError(f"checkpoint is incomplete: {checkpoint.name}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    step = int(match.group(1))
    if state.get("optimizer_step") != step or complete.get("optimizer_step") != step:
        raise CheckpointError("checkpoint step metadata does not match its directory")
    if complete.get("experiment_fingerprint") != state.get("experiment_fingerprint"):
        raise FingerprintMismatchError("checkpoint COMPLETE and trainer_state disagree on the fingerprint")
    if complete.get("trainer_state_sha256") != hashlib.sha256(state_path.read_bytes()).hexdigest():
        raise CheckpointError("trainer_state checksum mismatch")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise CheckpointError("checkpoint artifact index is incomplete")
    for name, expected_hash in artifacts.items():
        artifact_path = checkpoint / name
        if not artifact_path.is_file() or file_sha256(artifact_path) != expected_hash:
            raise CheckpointError(f"checkpoint artifact checksum mismatch: {name}")
    return CheckpointRef(path=checkpoint, optimizer_step=step, trainer_state=state)


def latest_complete_dir(root: str | os.PathLike[str]) -> Path | None:
    """Highest step directory carrying a COMPLETE marker, without fingerprint checks."""

    root = Path(root)
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if CHECKPOINT_PATTERN.fullmatch(p.name) and (p / "COMPLETE").is_file()]
    return max(candidates, key=lambda p: int(CHECKPOINT_PATTERN.fullmatch(p.name).group(1))) if candidates else None
