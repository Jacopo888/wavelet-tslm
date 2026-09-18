#!/usr/bin/env python3
"""Plan or idempotently submit the single-GPU P0 backend gate on Polimi."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path("/work/u10970284/wavelet-tslm-rope-ablation")
QSUB = Path("/opt/pbs/bin/qsub")
QSELECT = Path("/opt/pbs/bin/qselect")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RESOURCE_SPEC = {
    "queue": "gpu",
    "select": "1:ncpus=4:ngpus=1:mem=16gb",
    "walltime": "00:20:00",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_sha256(root: Path) -> str:
    ignored_parts = {".git", ".pytest_cache", "__pycache__"}
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def submission_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".p0-submit.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def active_job_ids(job_name: str) -> list[str]:
    completed = subprocess.run(
        [str(QSELECT), "-u", getpass.getuser(), "-N", job_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "qselect failed")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def append_submission(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--command",
        required=True,
        help="single-line P0 gate command; no default is intentionally provided",
    )
    parser.add_argument("--gate-kind", choices=("kernel", "full-smoke"), required=True)
    parser.add_argument(
        "--scratch-root",
        required=True,
        help="absolute compute-node scratch path or the explicit token pbs-tmpdir",
    )
    parser.add_argument("--scratch-lifetime-hours", type=float, required=True)
    parser.add_argument("--environment-mode", choices=("readonly", "staged"), required=True)
    parser.add_argument("--environment-source", default="")
    parser.add_argument("--environment-install-gib", type=float, default=0.0)
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--snapshot-revision", default="")
    parser.add_argument("--stage-data-gib", type=float, default=0.0)
    parser.add_argument("--expected-output-gib", type=float, default=1.0)
    parser.add_argument("--required-inodes", type=int, default=100000)
    parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="perform qsub; without this flag only print the immutable plan",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="allow another attempt after a failed or ambiguous prior submission",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise SystemExit("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    if not args.command or len(args.command) > 4096 or any(c in args.command for c in "\r\n\0"):
        raise SystemExit("--command must be one non-empty line of at most 4096 bytes")
    if args.scratch_root != "pbs-tmpdir" and not Path(args.scratch_root).is_absolute():
        raise SystemExit("--scratch-root must be absolute or pbs-tmpdir")
    numeric_values = (
        args.scratch_lifetime_hours,
        args.environment_install_gib,
        args.stage_data_gib,
        args.expected_output_gib,
        args.required_inodes,
    )
    if any(value < 0 for value in numeric_values):
        raise SystemExit("storage sizes, lifetime, and inode count must be non-negative")
    if args.scratch_lifetime_hours < (20 / 60):
        raise SystemExit("scratch lifetime must cover the 20-minute PBS walltime")
    if args.environment_mode == "readonly" and (
        not args.environment_source or args.environment_install_gib != 0
    ):
        raise SystemExit("readonly environment requires a source and zero install GiB")
    if args.environment_mode == "staged" and args.environment_install_gib <= 0:
        raise SystemExit("staged environment requires a positive install-size inventory")
    if args.gate_kind == "kernel" and (args.snapshot or args.snapshot_revision):
        raise SystemExit("kernel gate must not use a model snapshot")
    if args.gate_kind == "full-smoke" and (not args.snapshot or not args.snapshot_revision):
        raise SystemExit("full-smoke requires a shared snapshot path and exact revision")

    root = args.project_root.resolve()
    if root != PROJECT_ROOT:
        raise SystemExit(f"project root must be exactly {PROJECT_ROOT}")
    code_root = root / "code"
    pbs_script = code_root / "pbs" / "p0_backend_gate.pbs"
    if not code_root.is_dir() or not pbs_script.is_file():
        raise SystemExit("synchronize the code tree before planning or submitting")
    for executable in (QSELECT, QSUB):
        if not executable.is_file():
            raise SystemExit(f"missing OpenPBS executable: {executable}")

    job_name = f"wrp0-{sha256_bytes(args.run_id.encode('utf-8'))[:8]}"
    command_sha = sha256_bytes(args.command.encode("utf-8"))
    request = {
        "schema_version": 1,
        "kind": "p0_backend_gate",
        "run_id": args.run_id,
        "job_name": job_name,
        "command": args.command,
        "command_sha256": command_sha,
        "gate_kind": args.gate_kind,
        "storage": {
            "scratch_root_spec": args.scratch_root,
            "scratch_lifetime_seconds": int(args.scratch_lifetime_hours * 3600),
            "environment_mode": args.environment_mode,
            "environment_source": args.environment_source,
            "environment_install_bytes": int(args.environment_install_gib * 1024**3),
            "snapshot": args.snapshot,
            "snapshot_revision": args.snapshot_revision,
            "stage_data_bytes": int(args.stage_data_gib * 1024**3),
            "expected_output_bytes": int(args.expected_output_gib * 1024**3),
            "required_inodes": args.required_inodes,
        },
        "code_tree_sha256": code_tree_sha256(code_root),
        "pbs_script_sha256": sha256_file(pbs_script),
        "resources": RESOURCE_SPEC,
    }
    plan = {"action": "submit" if args.submit else "dry_run", "request": request}

    run_dir = root / "runs" / args.run_id
    request_file = run_dir / "request.json"
    state_file = run_dir / "state.json"
    existing_request = read_json(request_file)
    if existing_request is not None and existing_request != request:
        raise SystemExit(
            "run-id already has a different immutable request; choose a new run-id"
        )

    active = active_job_ids(job_name)
    if active:
        plan.update({"action": "deduplicated", "active_job_ids": active})
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    state = read_json(state_file)
    if state and state.get("status") == "succeeded":
        plan.update({"action": "already_succeeded", "state": state})
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.submit:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    latest_submission = read_json(run_dir / "submitted.json")
    if latest_submission and not args.retry_failed:
        raise SystemExit(
            "a prior submission is no longer active and did not succeed; inspect state/logs "
            "and pass --retry-failed only when a retry is intended"
        )

    root.mkdir(parents=False, exist_ok=True)
    with submission_lock(root):
        active = active_job_ids(job_name)
        if active:
            plan.update({"action": "deduplicated", "active_job_ids": active})
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        run_dir.mkdir(mode=0o770, parents=False, exist_ok=True)
        existing_request = read_json(request_file)
        if existing_request is None:
            atomic_json(request_file, request)
        elif existing_request != request:
            raise SystemExit("immutable request changed while acquiring submit lock")

        history_file = run_dir / "submissions.jsonl"
        attempt = 1
        if history_file.exists():
            attempt += sum(1 for line in history_file.read_text(encoding="utf-8").splitlines() if line)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stdout_path = root / "logs" / f"{args.run_id}.a{attempt}.{stamp}.out"
        stderr_path = root / "logs" / f"{args.run_id}.a{attempt}.{stamp}.err"
        environment = os.environ.copy()
        environment.update(
            {
                "WR_PROJECT_ROOT": str(root),
                "WR_RUN_ID": args.run_id,
                "WR_ATTEMPT": str(attempt),
                "P0_COMMAND": args.command,
                "WR_P0_GATE_KIND": args.gate_kind,
                "WR_SCRATCH_ROOT_SPEC": args.scratch_root,
                "WR_SCRATCH_LIFETIME_SECONDS": str(int(args.scratch_lifetime_hours * 3600)),
                "WR_ENV_MODE": args.environment_mode,
                "WR_READONLY_ENV_ROOT": args.environment_source,
                "WR_ENV_INSTALL_BYTES": str(int(args.environment_install_gib * 1024**3)),
                "WR_SHARED_SNAPSHOT": args.snapshot,
                "WR_SNAPSHOT_REVISION": args.snapshot_revision,
                "WR_STAGE_DATA_BYTES": str(int(args.stage_data_gib * 1024**3)),
                "WR_EXPECTED_OUTPUT_BYTES": str(int(args.expected_output_gib * 1024**3)),
                "WR_REQUIRED_SCRATCH_INODES": str(args.required_inodes),
            }
        )
        qsub_command = [
            str(QSUB),
            "-N",
            job_name,
            "-v",
            "WR_PROJECT_ROOT,WR_RUN_ID,WR_ATTEMPT,P0_COMMAND,WR_P0_GATE_KIND,"
            "WR_SCRATCH_ROOT_SPEC,WR_SCRATCH_LIFETIME_SECONDS,WR_ENV_MODE,"
            "WR_READONLY_ENV_ROOT,WR_ENV_INSTALL_BYTES,WR_SHARED_SNAPSHOT,"
            "WR_SNAPSHOT_REVISION,WR_STAGE_DATA_BYTES,WR_EXPECTED_OUTPUT_BYTES,"
            "WR_REQUIRED_SCRATCH_INODES",
            "-o",
            str(stdout_path),
            "-e",
            str(stderr_path),
            str(pbs_script),
        ]
        completed = subprocess.run(
            qsub_command, check=False, capture_output=True, text=True, env=environment
        )
        if completed.returncode != 0:
            raise SystemExit(completed.stderr.strip() or "qsub failed")
        job_id = completed.stdout.strip()
        if not job_id:
            raise SystemExit("qsub succeeded without returning a job id")
        submission = {
            "schema_version": 1,
            "attempt": attempt,
            "job_id": job_id,
            "job_name": job_name,
            "submitted_at_utc": utc_now(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        atomic_json(run_dir / "submitted.json", submission)
        append_submission(history_file, submission)
        plan.update({"action": "submitted", "submission": submission})
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"submit_run.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
