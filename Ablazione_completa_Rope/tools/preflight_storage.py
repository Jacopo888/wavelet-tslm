#!/usr/bin/env python3
"""Fail-closed compute-node storage preflight for Polimi P0 gates."""

from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GIB = 1024**3
MINIMUM_FREE_BYTES = 25 * GIB
MEMORY_FILESYSTEMS = {"tmpfs", "ramfs", "devtmpfs"}


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--job-scratch", type=Path, required=True)
    parser.add_argument("--gate-kind", choices=("kernel", "full-smoke"), required=True)
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--snapshot-revision", default="")
    parser.add_argument("--environment-mode", choices=("readonly", "staged"), required=True)
    parser.add_argument("--environment-source", default="")
    parser.add_argument("--environment-install-bytes", type=nonnegative_int, required=True)
    parser.add_argument("--stage-data-bytes", type=nonnegative_int, required=True)
    parser.add_argument("--expected-output-bytes", type=nonnegative_int, required=True)
    parser.add_argument("--required-inodes", type=nonnegative_int, required=True)
    parser.add_argument("--required-lifetime-seconds", type=nonnegative_int, required=True)
    parser.add_argument("--declared-lifetime-seconds", type=nonnegative_int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def inventory_bytes(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    seen: set[tuple[int, int]] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += stat.st_size
        count += 1
    return total, count


def verify_snapshot(path: Path, revision: str) -> dict[str, Any]:
    if not revision or path.name != revision:
        raise RuntimeError("shared snapshot directory must be named by the exact pinned revision")
    required = ("config.json", "tokenizer_config.json", "tokenizer.json")
    missing = [name for name in required if not (path / name).is_file()]
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise RuntimeError("snapshot weight index has no shards")
        missing.extend(name for name in shards if not (path / name).is_file())
    elif not list(path.glob("*.safetensors")):
        missing.append("*.safetensors or model.safetensors.index.json")
    if missing:
        raise RuntimeError(f"incomplete shared snapshot; missing: {', '.join(missing)}")
    size, files = inventory_bytes(path)
    if size <= 0:
        raise RuntimeError("shared snapshot inventory is empty")
    return {"path": str(path), "revision": revision, "bytes": size, "files": files}


def find_mount(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["findmnt", "-J", "-b", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "findmnt failed")
    filesystems = json.loads(completed.stdout).get("filesystems", [])
    if len(filesystems) != 1:
        raise RuntimeError("could not identify exactly one scratch filesystem")
    return {key: str(value) for key, value in filesystems[0].items()}


def quota_probe(kind: str, identity: str, source: str) -> dict[str, Any]:
    flag = "-u" if kind == "user" else "-g"
    completed = subprocess.run(
        ["quota", f"--filesystem={source}", "-w", "-v", flag, identity],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    if completed.returncode != 0 and "no quota enabled" not in combined.lower():
        raise RuntimeError(f"{kind} quota query failed: {combined}")
    remaining: int | None = None
    for line in completed.stdout.splitlines():
        columns = line.split()
        if len(columns) < 4 or columns[0] != source:
            continue
        used = int(columns[1].rstrip("*")) * 1024
        hard = int(columns[3].rstrip("*")) * 1024
        remaining = None if hard == 0 else max(0, hard - used)
        break
    return {
        "kind": kind,
        "identity": identity,
        "query_returncode": completed.returncode,
        "hard_remaining_bytes": remaining,
        "output": combined,
    }


def atomic_report(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    scratch_root = args.scratch_root.resolve(strict=True)
    job_scratch = args.job_scratch.resolve(strict=True)
    for forbidden in (Path("/work").resolve(), Path("/home").resolve()):
        if is_within(scratch_root, forbidden):
            raise RuntimeError(f"scratch cannot be under {forbidden}")
    if not is_within(job_scratch, scratch_root) or job_scratch == scratch_root:
        raise RuntimeError("job scratch must be a child of the verified scratch root")
    if not os.access(scratch_root, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError("scratch root is not readable, writable, and searchable")

    mount = find_mount(scratch_root)
    fstype = mount.get("fstype", "").lower()
    if fstype in MEMORY_FILESYSTEMS:
        raise RuntimeError(f"scratch filesystem {fstype!r} is memory-backed")
    if Path(mount.get("target", "/")).resolve() == Path("/"):
        raise RuntimeError("scratch resolves to the node root filesystem, not a dedicated mount")

    probe = job_scratch / f".write-probe-{os.getpid()}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"wavelet-rope-p0\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        probe.unlink(missing_ok=True)

    snapshot: dict[str, Any] | None = None
    snapshot_bytes = 0
    if args.gate_kind == "full-smoke":
        if not args.snapshot:
            raise RuntimeError("full-smoke requires a complete shared snapshot")
        snapshot_path = Path(args.snapshot).resolve(strict=True)
        if is_within(snapshot_path, Path("/work")) or is_within(snapshot_path, Path("/home")):
            raise RuntimeError("full-smoke snapshot must be in a shared read-only cache")
        if os.access(snapshot_path, os.W_OK):
            raise RuntimeError("shared snapshot must be read-only to this job")
        snapshot = verify_snapshot(snapshot_path, args.snapshot_revision)
        snapshot_bytes = int(snapshot["bytes"])
    elif args.snapshot or args.snapshot_revision:
        raise RuntimeError("kernel gate must not stage or inspect model weights")

    environment_source: str | None = None
    environment_bytes = args.environment_install_bytes
    if args.environment_mode == "readonly":
        if not args.environment_source:
            raise RuntimeError("readonly environment mode requires --environment-source")
        environment_path = Path(args.environment_source).resolve(strict=True)
        if not os.access(environment_path, os.R_OK | os.X_OK):
            raise RuntimeError("read-only environment is not accessible")
        if os.access(environment_path, os.W_OK):
            raise RuntimeError("declared read-only environment is writable")
        if environment_bytes != 0:
            raise RuntimeError("read-only environment requires environment-install-bytes=0")
        environment_source = str(environment_path)
    elif environment_bytes <= 0:
        raise RuntimeError("staged environment requires a positive byte inventory")

    required_bytes = max(
        MINIMUM_FREE_BYTES,
        2 * snapshot_bytes
        + environment_bytes
        + args.stage_data_bytes
        + args.expected_output_bytes
        + 5 * GIB,
    )
    stats = os.statvfs(scratch_root)
    filesystem_available = stats.f_bavail * stats.f_frsize
    available_inodes = stats.f_favail
    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    source = mount.get("source", "")
    user_quota = quota_probe("user", user, source)
    group_quota = quota_probe("group", group, source)
    available_limits = [filesystem_available]
    for quota in (user_quota, group_quota):
        remaining = quota["hard_remaining_bytes"]
        if remaining is not None:
            available_limits.append(int(remaining))
    effective_available = min(available_limits)

    if effective_available < required_bytes:
        raise RuntimeError(
            f"scratch has {effective_available} effective bytes; {required_bytes} required"
        )
    if available_inodes < args.required_inodes:
        raise RuntimeError(
            f"scratch has {available_inodes} inodes; {args.required_inodes} required"
        )
    if args.declared_lifetime_seconds < args.required_lifetime_seconds:
        raise RuntimeError("declared scratch lifetime is shorter than PBS walltime")

    report = {
        "schema_version": 1,
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_kind": args.gate_kind,
        "scratch_root": str(scratch_root),
        "job_scratch": str(job_scratch),
        "mount": mount,
        "filesystem_available_bytes": filesystem_available,
        "effective_available_bytes": effective_available,
        "available_inodes": available_inodes,
        "required_inodes": args.required_inodes,
        "required_bytes": required_bytes,
        "formula": "max(25GiB,2*S_snapshot+S_env+S_data+S_output+5GiB)",
        "components": {
            "snapshot_bytes": snapshot_bytes,
            "environment_install_bytes": environment_bytes,
            "stage_data_bytes": args.stage_data_bytes,
            "expected_output_bytes": args.expected_output_bytes,
        },
        "snapshot": snapshot,
        "environment_mode": args.environment_mode,
        "environment_source": environment_source,
        "user_quota": user_quota,
        "group_quota": group_quota,
        "declared_lifetime_seconds": args.declared_lifetime_seconds,
        "required_lifetime_seconds": args.required_lifetime_seconds,
    }
    atomic_report(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"preflight_storage.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
