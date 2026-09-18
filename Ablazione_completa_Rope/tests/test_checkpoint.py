from __future__ import annotations

from pathlib import Path

import pytest

from wavelet_rope.checkpoint import AtomicCheckpointStore
from wavelet_rope.errors import CheckpointError, FingerprintMismatchError

FINGERPRINT = "a" * 64


def _state(step: int, **updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "experiment_fingerprint": FINGERPRINT,
        "phase": "P1",
        "pass_index": 0,
        "optimizer_step": step,
        "sample_cursor": step * 4,
        "micro_step_in_accumulation": 0,
        "wildcard_lambda": 0.5,
    }
    state.update(updates)
    return state


def _writers(payload: bytes = b"state"):
    names = ("model.safetensors", "optimizer.pt", "schedulers.pt", "rng.pt")
    return {name: lambda path, name=name: path.write_bytes(payload + name.encode()) for name in names}


def test_atomic_save_and_auto_resume_choose_highest_complete(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoints", FINGERPRINT)
    first = store.save(trainer_state=_state(4), artifact_writers=_writers())
    assert first.optimizer_step == 4
    incomplete = store.root / "step_000000000006"
    incomplete.mkdir()
    store.save(trainer_state=_state(8), artifact_writers=_writers(b"new"))
    resumed = store.resolve_resume("auto")
    assert resumed is not None
    assert resumed.optimizer_step == 8
    assert not list(store.root.glob(".*.tmp-*"))


def test_checkpoint_rejects_partial_accumulation(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path, FINGERPRINT)
    with pytest.raises(CheckpointError, match="optimizer boundaries"):
        store.save(
            trainer_state=_state(4, micro_step_in_accumulation=1),
            artifact_writers=_writers(),
        )


def test_resume_rejects_different_experiment(tmp_path: Path) -> None:
    first = AtomicCheckpointStore(tmp_path, FINGERPRINT)
    checkpoint = first.save(trainer_state=_state(4), artifact_writers=_writers())
    other = AtomicCheckpointStore(tmp_path, "b" * 64)
    with pytest.raises(FingerprintMismatchError):
        other.resolve_resume(checkpoint.path)


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path, FINGERPRINT)
    checkpoint = store.save(trainer_state=_state(4), artifact_writers=_writers())
    (checkpoint.path / "optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        store.resolve_resume("auto")



def test_pass_end_can_supersede_the_periodic_checkpoint_of_the_same_step(tmp_path: Path) -> None:
    """A pass boundary that is a multiple of the checkpoint interval must not crash the run.

    ``va1r`` died at step 4488 -- four times the 1122-step interval -- because the periodic save had already
    written that step and the pass-end save raised.  The trainer now removes the superseded directory and
    rewrites it with the new pass bookkeeping; this pins both halves of that contract.
    """

    import shutil

    store = AtomicCheckpointStore(tmp_path / "checkpoints", FINGERPRINT)
    periodic = store.save(trainer_state=_state(4488), artifact_writers=_writers())
    assert periodic.trainer_state["pass_index"] == 0

    with pytest.raises(CheckpointError, match="already exists"):
        store.save(trainer_state=_state(4488, pass_index=1, sample_cursor=0), artifact_writers=_writers())

    shutil.rmtree(store.root / f"step_{4488:012d}")
    rewritten = store.save(trainer_state=_state(4488, pass_index=1, sample_cursor=0), artifact_writers=_writers())

    assert rewritten.optimizer_step == 4488
    assert rewritten.trainer_state["pass_index"] == 1
    assert rewritten.trainer_state["sample_cursor"] == 0
    assert store.resolve_resume("auto").trainer_state["pass_index"] == 1
    assert not list(store.root.glob(".*.tmp-*"))
