"""Atomic run-directory lease used as the final guard against duplicate workers."""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ActiveRunError, FingerprintMismatchError
from .manifest import atomic_write_json


class RunLease:
    """Own an atomic ACTIVE.lock directory; stale leases require explicit review."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        experiment_fingerprint: str,
        *,
        job_id: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.experiment_fingerprint = experiment_fingerprint
        self.job_id = job_id or os.environ.get("PBS_JOBID")
        self.lock_dir = self.run_dir / "state" / "ACTIVE.lock"
        self.token = uuid.uuid4().hex
        self._held = False

    def acquire(self) -> Mapping[str, Any]:
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_dir.mkdir()
        except FileExistsError as error:
            owner = self.inspect()
            raise ActiveRunError(f"run already has an active lease: {owner}") from error
        owner = {
            "schema_version": 1,
            "experiment_fingerprint": self.experiment_fingerprint,
            "token": self.token,
            "job_id": self.job_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            atomic_write_json(self.lock_dir / "owner.json", owner)
        except BaseException:
            self.lock_dir.rmdir()
            raise
        self._held = True
        return owner

    def inspect(self) -> Mapping[str, Any]:
        owner_path = self.lock_dir / "owner.json"
        if not owner_path.is_file():
            return {"status": "present_without_owner_metadata"}
        try:
            return json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "present_with_unreadable_owner_metadata"}

    def release(self) -> None:
        if not self._held:
            return
        owner = self.inspect()
        if owner.get("experiment_fingerprint") != self.experiment_fingerprint:
            raise FingerprintMismatchError("active lease experiment fingerprint changed")
        if owner.get("token") != self.token:
            raise ActiveRunError("active lease ownership token changed")
        (self.lock_dir / "owner.json").unlink()
        self.lock_dir.rmdir()
        self._held = False

    def __enter__(self) -> "RunLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

