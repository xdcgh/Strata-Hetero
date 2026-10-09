"""Plan and validate the run47 CPU-pool matrix without launching Strata.

The production executor is intentionally not wired here. A later root-reviewed
coordinator can consume the serialized recipes and reuse the run41 guarded stages.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


EXPECTED_ARMS = {
    ("w10-all", 10, "all"), ("w04-all", 4, "all"), ("w06-all", 6, "all"),
    ("w16-all", 16, "all"), ("w06-auto", 6, "auto"),
}
RUN_ID_RE = re.compile(r"^20261009-47-(?:w10-all|w04-all|w06-all|w16-all|w06-auto)-(?:quality-on|performance-off)$")


class SweepError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_windows_argv(argv: Iterable[str]) -> str:
    """Serialize an argv list for Windows receipts; never use shell=True."""
    values = list(argv)
    if any(not isinstance(x, str) or "\0" in x for x in values):
        raise SweepError("argv must contain NUL-free strings")
    return subprocess.list2cmdline(values)


def validate_create_time(value: str) -> str:
    """Validate, but preserve, a full-precision UTC CreateTime string."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SweepError("CreateTime must remain an ISO UTC string ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SweepError("invalid CreateTime string") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise SweepError("CreateTime must be UTC")
    # Do not normalize fractions: PID-reuse checks compare the original text.
    return value


def _reject_constant(value: str) -> None:
    raise SweepError(f"non-finite JSON constant: {value}")


def _save_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Keep any partial evidence; never unlink on a write failure.
        raise


def read_last_complete_jsonl_record(
    path: Path,
    *,
    max_tail_bytes: int = 262_144,
    max_record_bytes: int = 65_536,
    timestamp_field: str = "observed_at_utc",
    owner_field: str | None = "owner_pid",
    expected_owner_pid: int | None = None,
    record_type_field: str | None = "record_type",
    expected_record_type: str | None = None,
    max_age_seconds: float | None = 5.0,
    now_utc: dt.datetime | None = None,
    partial_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Read one bounded newline-complete JSONL row; preserve a trailing fragment."""
    if max_record_bytes < 1 or max_tail_bytes < max_record_bytes:
        raise SweepError("invalid JSONL size bound")
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        if length == 0:
            raise SweepError("JSONL file is empty")
        take = min(length, max_tail_bytes + 1)
        offset = length - take
        stream.seek(offset)
        buf = stream.read(take)
        if len(buf) != take:
            raise SweepError("short read of JSONL tail")

    start = 0
    if offset:
        boundary = buf.find(b"\n")
        if boundary < 0:
            raise SweepError("no line boundary within bounded JSONL tail")
        start = boundary + 1
    last_lf = buf.rfind(b"\n")
    if last_lf < start:
        fragment = buf[start:]
        fragment_path = None
        if fragment and partial_evidence_dir is not None:
            digest = sha256_bytes(fragment)
            fragment_path = partial_evidence_dir / f"{path.name}.partial.{digest}.bin"
            _save_exclusive(fragment_path, fragment)
        raise SweepError("no complete newline-terminated JSONL row; trailing fragment preserved")

    trailing = buf[last_lf + 1:] if last_lf < len(buf) - 1 else b""
    previous_lf = buf.rfind(b"\n", start, last_lf)
    row_start = previous_lf + 1 if previous_lf >= start else start
    row_bytes = buf[row_start:last_lf]
    if row_bytes.endswith(b"\r"):
        row_bytes = row_bytes[:-1]
    if not row_bytes:
        raise SweepError("last complete JSONL row is empty")
    if len(row_bytes) > max_record_bytes:
        raise SweepError("last complete JSONL row exceeds bound")

    partial_path = None
    if trailing and partial_evidence_dir is not None:
        digest = sha256_bytes(trailing)
        partial_path = partial_evidence_dir / f"{path.name}.partial.{digest}.bin"
        if partial_path.exists():
            raise SweepError("partial evidence target already exists; refusing overwrite")
        _save_exclusive(partial_path, trailing)

    try:
        text = row_bytes.decode("utf-8", errors="strict")
        row = json.loads(text, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, SweepError) as exc:
        raise SweepError(f"last complete JSONL row is malformed: {exc}") from exc
    if not isinstance(row, dict):
        raise SweepError("JSONL row must be an object")

    observed = row.get(timestamp_field) if timestamp_field else None
    age = None
    if timestamp_field:
        if not isinstance(observed, str):
            raise SweepError("timestamp field is missing or not a string")
        try:
            stamp = dt.datetime.fromisoformat(observed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SweepError("timestamp is not ISO format") from exc
        if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
            raise SweepError("timestamp must be UTC")
        now = now_utc or dt.datetime.now(dt.timezone.utc)
        age = (now - stamp).total_seconds()
        if age < -2:
            raise SweepError("timestamp is in the future")
        if max_age_seconds is not None and age > max_age_seconds:
            raise SweepError("last complete JSONL row is stale")

    owner = row.get(owner_field) if owner_field else None
    if owner_field and (type(owner) is not int or (expected_owner_pid is not None and owner != expected_owner_pid)):
        raise SweepError("owner field is missing, malformed, or mismatched")
    record_type = row.get(record_type_field) if record_type_field else None
    if record_type_field and not isinstance(record_type, str):
        raise SweepError("record-type field is missing or malformed")
    if expected_record_type is not None and record_type != expected_record_type:
        raise SweepError("record-type mismatch")
    return {
        "record": row,
        "raw_record": text,
        "record_sha256": sha256_bytes(row_bytes),
        "file_length_bytes": length,
        "record_start_offset": offset + row_start,
        "record_end_offset": offset + last_lf,
        "age_seconds": age,
        "owner_pid": owner,
        "record_type": record_type,
        "trailing_partial_bytes": len(trailing),
        "trailing_partial_sha256": sha256_bytes(trailing) if trailing else None,
        "trailing_partial_path": str(partial_path) if partial_path else None,
    }


def owned_cleanup_order(snapshot: Iterable[dict[str, Any]], owners: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exact-identity cleanup order; refuse a reused PID and never stops anything."""
    process_rows = list(snapshot)
    ordered = []
    for role in ("engine", "server", "launcher"):
        expected = owners[role]
        validate_create_time(expected["create_utc"])
        found = [p for p in process_rows if p.get("pid") == expected["pid"]]
        if not found:
            ordered.append({"role": role, "pid": expected["pid"], "status": "already_absent"})
            continue
        if len(found) != 1:
            raise SweepError(f"ambiguous PID for {role}")
        actual = found[0]
        for field in ("exe", "create_utc", "parent_pid"):
            if actual.get(field) != expected.get(field):
                raise SweepError(f"PID identity mismatch for {role}: {field}")
        if expected.get("command_contains") and expected["command_contains"] not in actual.get("command_line", ""):
            raise SweepError(f"command mismatch for {role}")
        ordered.append({"role": role, "pid": expected["pid"], "status": "exact_match_stop_child_first"})
    return ordered


def _phase_commands(arm: dict[str, Any], phase_name: str, repo: Path) -> list[dict[str, Any]]:
    phase = arm[phase_name]
    run_id = phase["run_id"]
    if not RUN_ID_RE.fullmatch(run_id):
        raise SweepError("arm RunId is outside the prepared allowlist")
    return [{
        "run_id": run_id,
        "config": phase["path"] + r"\config.json",
        "identity": phase["path"] + r"\identity.json",
        "phase": phase_name,
        "token_capture": phase_name == "quality_on",
        "launch_template": str(repo / "bench/hetero/20261009-41-hetero41-direct-quality/launch-owned.ps1"),
        "quality_template": str(repo / "bench/hetero/20261009-41-hetero41-direct-quality/run-quality-after-ready.ps1"),
        "startup_template": str(repo / "bench/hetero/20261009-41-hetero41-direct-quality/startup-watch.ps1"),
    }]


def build_plan(matrix_path: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve(strict=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_for_root_review_not_launched":
        raise SweepError("matrix is not a prepared plan")
    if len(matrix.get("arms", [])) != 5:
        raise SweepError("matrix must contain exactly five worker-pool arms")
    plan = []
    for phase_name in ("quality_on", "performance_off"):
        for arm in matrix["arms"]:
            phase = arm[phase_name]
            config_path = Path(phase["path"]) / "config.json"
            identity_path = Path(phase["path"]) / "identity.json"
            if sha256_bytes(config_path.read_bytes()) != phase["config_sha256"]:
                raise SweepError(f"config hash mismatch: {config_path}")
            if sha256_bytes(identity_path.read_bytes()) != phase["identity_sha256"]:
                raise SweepError(f"identity hash mismatch: {identity_path}")
            plan.extend(_phase_commands(arm, phase_name, matrix_path.parents[2]))
    return {
        "schema_version": 1,
        "status": "plan_only_no_hardware_or_process_access",
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_bytes(matrix_path.read_bytes()),
        "quality_phase_then_performance_phase": plan,
        "execution_enabled": False,
        "execution_requires_root_review_adapter": True,
    }


def validate_root_authorization(matrix_path: Path, authorization_path: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve(strict=True)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_hash = sha256_bytes(matrix_path.read_bytes())
    if authorization.get("status") != "root_reviewed_authorized":
        raise SweepError("authorization status is not root_reviewed_authorized")
    if authorization.get("matrix_sha256") != expected_hash:
        raise SweepError("authorization does not bind this exact matrix hash")
    arms = authorization.get("arms")
    expected_arms = {row["arm"] for row in json.loads(matrix_path.read_text())["arms"]}
    if not isinstance(arms, list) or set(arms) != expected_arms:
        raise SweepError("authorization arm set does not exactly match the matrix")
    nonce = authorization.get("nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise SweepError("authorization nonce must be 32 lowercase hex digits")
    return authorization


def execute_plan(plan: dict[str, Any], authorization: dict[str, Any], adapter: Any) -> list[Any]:
    """Only injected review/test adapters can execute; the CLI never supplies one."""
    if not adapter or not callable(getattr(adapter, "run_phase", None)):
        raise SweepError("no reviewed run41 execution adapter is installed")
    results = []
    for phase in plan["quality_phase_then_performance_phase"]:
        if phase["run_id"].split("-")[2] not in authorization["arms"]:
            raise SweepError("RunId is not authorized")
        result = adapter.run_phase(phase)
        results.append(result)
        if result.get("status") != "pass":
            raise SweepError("stop-first: adapter phase failed; later arms are not started")
    return results


def invoke_host_fake_cli(fake_cli: Path, args: list[str], fixture_root: Path) -> dict[str, Any]:
    """Run an actual temporary fake CLI, bounded to the test fixture root."""
    root = fixture_root.resolve(strict=True)
    target = fake_cli.resolve(strict=True)
    if root not in target.parents or target.suffix.casefold() != ".py":
        raise SweepError("fake CLI must be a Python script beneath the isolated fixture root")
    completed = subprocess.run([sys.executable, str(target), *args], cwd=str(root),
                               shell=False, capture_output=True, text=True, timeout=10)
    if completed.returncode != 0:
        raise SweepError("host fake CLI failed")
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(Path(args.matrix))
        if args.execute:
            if not args.authorization:
                raise SweepError("execution requires a separate root authorization receipt")
            validate_root_authorization(Path(args.matrix), Path(args.authorization))
            # No production adapter is intentionally wired in this preparation build.
            raise SweepError("production run41 adapter is not installed; no process was launched")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "rejected_no_execution", "error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
