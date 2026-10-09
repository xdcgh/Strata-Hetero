"""Root-reviewable planner/guards for the prepared CPU worker-pool sweep.

Default CLI mode is plan-only and performs no device or network access.
Explicit execution uses the pinned run41 PowerShell lifecycle scripts after
validating a root authorization bound to the matrix and adapter source hashes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Iterable


EXPECTED_ARMS = {
    ("w10-all", 10, "all"), ("w04-all", 4, "all"), ("w06-all", 6, "all"),
    ("w16-all", 16, "all"), ("w06-auto", 6, "auto"),
}
ARM_EXECUTION_ORDER = ("w10-all", "w04-all", "w06-all", "w16-all", "w06-auto")
EXPECTED_QUALITY_PROMPT_IDS = ("smoke_arithmetic", "smoke_unicode", "smoke_json_arithmetic", "smoke_python_function",
                              "smoke_three_key_retrieval", "long_1k", "long_4k", "long_16k", "long_30k7")
RUN_ID_RE = re.compile(r"^20261009-47-(?:w10-all|w04-all|w06-all|w16-all|w06-auto)-(?:quality-on|performance-off)$")
RUN41_REL = Path("bench/hetero/20261009-41-hetero41-direct-quality")
RUN41_TEMPLATES = {
    "launch": (RUN41_REL / "launch-owned.ps1", "6967ea74e87447d66b153b218a9a336d9041628b14e52ec5a046309bcd06a679"),
    "watch": (RUN41_REL / "startup-watch.ps1", "e976655be126686081d2c17823f35cf356d4311f91a57f840bed0cfc09f020d7"),
    "quality": (RUN41_REL / "run-quality-after-ready.ps1", "dcb68b86b81c0d2a7db92cc0ec7343d72123c8b48e2a2f76d6a383b32932a078"),
    "cleanup": (RUN41_REL / "cleanup-owned-model.ps1", "6ee4aff158730b04b3def8b9feafa62d6658bb6a964ddd9edced79f8c65663ef"),
    "jsonl_reader": (Path("tools/hetero_jsonl_tail.ps1"), "52e15e5fbe3f798efeeb91d0ad09d46fc0ac3ccd737ef7073e63f46b82984e1d"),
}
RUN41_ID = "20261009-41-hetero41-direct-quality"
RUN41_ID_REPLACEMENT_COUNTS = {"launch": 7, "watch": 1, "quality": 4, "cleanup": 1}
PWsh7 = Path(r"C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe")
PWSH7_SHA256 = "362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139"


class SweepError(ValueError):
    pass


class LifecycleTimeout(SweepError):
    def __init__(self, owner: dict[str, Any], stdout: bytes, stderr: bytes, process: subprocess.Popen):
        super().__init__(f"owned PowerShell controller timed out with PID {owner.get('pid')}; process/evidence preserved")
        self.owner, self.stdout, self.stderr, self.process = owner, stdout, stderr, process


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_windows_argv(argv: Iterable[str]) -> str:
    """Return a Windows command-line rendering of argv, without shell parsing."""
    values = list(argv)
    if any(not isinstance(value, str) or "\0" in value for value in values):
        raise SweepError("argv must contain NUL-free strings")
    return subprocess.list2cmdline(values)


def validate_create_time(value: str) -> str:
    """Validate and preserve a literal full-precision UTC CreateTime string."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SweepError("CreateTime must remain an ISO UTC string ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SweepError("invalid CreateTime string") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise SweepError("CreateTime must be UTC")
    # Compare the original text later; do not normalize or drop fractional digits.
    return value


def owned_cleanup_order(snapshot: list[dict[str, Any]],
                        owners: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure PID-reuse-safe cleanup plan; caller must re-snapshot before stopping."""
    by_pid = {row.get("pid"): row for row in snapshot if type(row.get("pid")) is int}
    order = []
    for role in ("engine", "server", "launcher"):
        owner = owners.get(role)
        if not isinstance(owner, dict) or type(owner.get("pid")) is not int:
            raise SweepError(f"missing owned process identity for {role}")
        row = by_pid.get(owner["pid"])
        if row is None:
            order.append({"role": role, "pid": owner["pid"], "status": "already_absent"})
            continue
        if (row.get("exe") != owner.get("exe") or
                validate_create_time(row.get("create_utc")) != validate_create_time(owner.get("create_utc")) or
                row.get("parent_pid") != owner.get("parent_pid") or
                not isinstance(owner.get("command_contains"), str) or
                owner["command_contains"] not in str(row.get("command_line", ""))):
            raise SweepError(f"PID {owner['pid']} no longer matches the owned {role} identity")
        order.append({"role": role, "pid": owner["pid"], "status": "exact_match_stop_child_first"})
    return order


def _reject_json_constant(value: str) -> None:
    raise SweepError(f"non-finite JSON constant: {value}")


def _save_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


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
    """Read one bounded complete newline row; preserve any incomplete tail bytes."""
    if max_record_bytes < 1 or max_tail_bytes < max_record_bytes:
        raise SweepError("invalid JSONL read bounds")
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size == 0:
            raise SweepError("JSONL file is empty")
        take = min(size, max_tail_bytes + 1)
        offset = size - take
        stream.seek(offset)
        buffer = stream.read(take)
        if len(buffer) != take:
            raise SweepError("short read while reading JSONL tail")

    start = 0
    if offset:
        boundary = buffer.find(b"\n")
        if boundary < 0:
            raise SweepError("no newline boundary within bounded tail")
        start = boundary + 1
    last_lf = buffer.rfind(b"\n")
    if last_lf < start:
        fragment = buffer[start:]
        if fragment and partial_evidence_dir is not None:
            fragment_hash = sha256_bytes(fragment)
            fragment_path = partial_evidence_dir / f"{path.name}.partial.{fragment_hash}.bin"
            _save_exclusive(fragment_path, fragment)
        raise SweepError("no complete newline-terminated row; partial tail preserved")

    trailing = buffer[last_lf + 1:] if last_lf < len(buffer) - 1 else b""
    previous_lf = buffer.rfind(b"\n", start, last_lf)
    row_start = previous_lf + 1 if previous_lf >= start else start
    row_bytes = buffer[row_start:last_lf]
    if row_bytes.endswith(b"\r"):
        row_bytes = row_bytes[:-1]
    if not row_bytes:
        raise SweepError("last complete JSONL row is empty")
    if len(row_bytes) > max_record_bytes:
        raise SweepError("last complete JSONL row exceeds size bound")

    partial_path = None
    if trailing and partial_evidence_dir is not None:
        partial_hash = sha256_bytes(trailing)
        partial_path = partial_evidence_dir / f"{path.name}.partial.{partial_hash}.bin"
        _save_exclusive(partial_path, trailing)

    try:
        text = row_bytes.decode("utf-8", errors="strict")
        row = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, SweepError) as exc:
        raise SweepError(f"last complete row is not exactly one finite JSON value: {exc}") from exc
    if not isinstance(row, dict):
        raise SweepError("JSONL row must be an object")

    observed = row.get(timestamp_field) if timestamp_field else None
    age_seconds = None
    if timestamp_field:
        if not isinstance(observed, str):
            raise SweepError("timestamp is missing or not a string")
        try:
            stamp = dt.datetime.fromisoformat(observed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SweepError("timestamp is not ISO format") from exc
        if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
            raise SweepError("timestamp must be UTC")
        now = now_utc or dt.datetime.now(dt.timezone.utc)
        age_seconds = (now - stamp).total_seconds()
        if age_seconds < -2 or (max_age_seconds is not None and age_seconds > max_age_seconds):
            raise SweepError("last complete JSONL row is future-dated or stale")

    owner = row.get(owner_field) if owner_field else None
    if owner_field and (type(owner) is not int or
                        (expected_owner_pid is not None and owner != expected_owner_pid)):
        raise SweepError("JSONL owner is missing, malformed, or mismatched")
    record_type = row.get(record_type_field) if record_type_field else None
    if record_type_field and not isinstance(record_type, str):
        raise SweepError("JSONL record type is missing or malformed")
    if expected_record_type is not None and record_type != expected_record_type:
        raise SweepError("JSONL record type mismatch")

    return {
        "record": row,
        "raw_record": text,
        "record_sha256": sha256_bytes(row_bytes),
        "record_start_offset": offset + row_start,
        "record_end_offset": offset + last_lf,
        "file_length_bytes": size,
        "observed_at_utc": observed,
        "age_seconds": age_seconds,
        "owner_pid": owner,
        "record_type": record_type,
        "trailing_partial_bytes": len(trailing),
        "trailing_partial_sha256": sha256_bytes(trailing) if trailing else None,
        "trailing_partial_path": str(partial_path) if partial_path else None,
    }


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _has_bang_loop(text: str) -> bool:
    compact = "".join(text.split())
    return (len(compact) >= 8 and set(compact) == {"!"}) or compact.endswith("!" * 32)


def validate_performance_request(run_dir: Path, index: int, max_tokens: int,
                                 *, allow_capped: bool = True,
                                 expected_prompt_sha256: str | None = None,
                                 expected_prompt_bytes: int | None = None) -> dict[str, Any]:
    """Fail closed on malformed SSE, missing tokens/cache data, nonfinite timing, or degenerate output."""
    stem = f"formal-{index:04d}"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("max_tokens") != max_tokens or metadata.get("repeats") != 3 or metadata.get("warmup") != 1:
        raise SweepError("metadata does not match planned cap/repeats/warmup")
    if expected_prompt_sha256 is not None and metadata.get("prompt_sha256") != expected_prompt_sha256:
        raise SweepError("prompt SHA differs from the immutable scenario recipe")
    if expected_prompt_bytes is not None and metadata.get("prompt_bytes") != expected_prompt_bytes:
        raise SweepError("prompt byte count differs from the immutable scenario recipe")
    warmups = json.loads((run_dir / "warmups.json").read_text(encoding="utf-8"))
    if not isinstance(warmups, list) or len(warmups) != 1:
        raise SweepError("scenario must retain exactly one warmup record")
    records = json.loads((run_dir / "requests.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 3 or index >= len(records):
        raise SweepError("formal request record missing")
    record = records[index]
    if not isinstance(record, dict) or _contains_nonfinite(record):
        raise SweepError("formal request JSON is malformed or nonfinite")
    if record.get("status") not in ("completed", "truncated_by_length"):
        raise SweepError("formal request status is not complete or cleanly length-capped")
    if record.get("finish_reason") not in ("stop", "length"):
        raise SweepError("formal request finish_reason is missing/invalid")
    if (record["status"] == "completed") != (record["finish_reason"] == "stop"):
        raise SweepError("formal request status/finish_reason disagree")
    if record["finish_reason"] == "length" and (not allow_capped or record.get("usage", {}).get("completion_tokens") != max_tokens):
        raise SweepError("length-capped output is not eligible for this timing recipe")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        raise SweepError("usage missing")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if any(type(x) is not int or x < 0 for x in (prompt_tokens, completion_tokens, total_tokens)):
        raise SweepError("actual token counts missing or invalid")
    if completion_tokens <= 1 or completion_tokens > max_tokens or total_tokens != prompt_tokens + completion_tokens:
        raise SweepError("single-token, over-cap, or inconsistent token counts")
    timings = record.get("timings")
    if not isinstance(timings, dict) or timings.get("cache_n") != 0:
        raise SweepError("performance record is missing explicit cache_n=0")
    for key in ("client_e2e_s", "first_generated_s", "first_visible_content_s"):
        value = record.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise SweepError(f"invalid finite positive timing: {key}")
    for value in timings.values():
        if isinstance(value, (float, int)) and _contains_nonfinite(value):
            raise SweepError("nonfinite server timing value")

    events_path = run_dir / f"{stem}.events.json"
    sse_path = run_dir / f"{stem}.sse.txt"
    text_path = run_dir / f"{stem}.text.txt"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(events, list) or _contains_nonfinite(events):
        raise SweepError("SSE event JSON missing/malformed/nonfinite")
    raw = sse_path.read_text(encoding="utf-8")
    data_values = [line[5:].lstrip() for line in raw.splitlines() if line.startswith("data:")]
    if not data_values or data_values.count("[DONE]") != 1 or data_values[-1] != "[DONE]":
        raise SweepError("SSE does not end with exactly one data: [DONE]")
    try:
        raw_events = [json.loads(value, parse_constant=_reject_json_constant) for value in data_values[:-1]]
    except (json.JSONDecodeError, SweepError) as exc:
        raise SweepError(f"SSE data event malformed or nonfinite: {exc}") from exc
    if raw_events != events:
        raise SweepError("raw SSE data events differ from the saved event artifact")
    sse_usage = [event.get("usage") for event in raw_events if isinstance(event, dict) and isinstance(event.get("usage"), dict)]
    if not sse_usage:
        raise SweepError("raw SSE stream is missing its usage/count event")
    final_usage = sse_usage[-1]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if final_usage.get(key) != usage.get(key):
            raise SweepError(f"raw SSE usage count differs from requests.json: {key}")
    event_contents: list[str] = []
    sse_finish = None
    for event in events:
        choices = event.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                event_contents.append(delta["content"])
            if choice.get("finish_reason") is not None:
                sse_finish = choice["finish_reason"]
    if sse_finish != record["finish_reason"]:
        raise SweepError("SSE finish_reason differs from request metadata")
    output = text_path.read_bytes()
    if not output or len(event_contents) == 0 or b"".join(x.encode("utf-8") for x in event_contents) != output:
        raise SweepError("SSE-visible output and stored UTF-8 bytes differ or are empty")
    if _has_bang_loop(output.decode("utf-8")):
        raise SweepError("performance output contains the quality bang-loop degeneracy")
    if record.get("content_sha256") != sha256_bytes(output) or record.get("content_chars") != len(output.decode("utf-8")):
        raise SweepError("stored output hash/character count mismatch")
    return {
        "status": "timing_eligible_quality_not_assessed",
        "formal_index": index,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": record["finish_reason"],
        "client_e2e_s": record["client_e2e_s"],
        "first_generated_s": record["first_generated_s"],
        "cache_n": timings["cache_n"],
        "content_sha256": record["content_sha256"],
    }


def build_plan(matrix_path: Path) -> dict[str, Any]:
    requested_matrix = Path(matrix_path)
    if requested_matrix.is_symlink():
        raise SweepError("matrix file may not be a symlink")
    matrix_path = requested_matrix.resolve(strict=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "prepared_for_root_review_not_launched":
        raise SweepError("matrix is not a prepared plan")
    actual_arms = {(x.get("arm"), x.get("workers"), x.get("affinity")) for x in matrix.get("arms", [])}
    if actual_arms != EXPECTED_ARMS:
        raise SweepError("worker-pool arm set does not exactly match the reviewed five-arm matrix")
    if matrix.get("arm_execution_order") != list(ARM_EXECUTION_ORDER):
        raise SweepError("arm execution order must place w10-all baseline first for both phases")
    scenarios = matrix.get("performance_phase", {}).get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 6:
        raise SweepError("performance recipe must contain the three decode and three context scenarios")
    decode = [x for x in scenarios if str(x.get("id", "")).startswith("decode-cap-")]
    if len(decode) != 3 or {x.get("max_tokens") for x in decode} != {256, 512, 1024}:
        raise SweepError("decode recipe must use the fixed 256/512/1024 caps")
    decode_prompts = set()
    for scenario in decode:
        prompt_path = Path(scenario.get("prompt_file", "")).resolve(strict=True)
        if (not prompt_path.is_file() or sha256_file(prompt_path) != scenario.get("prompt_sha256") or
                prompt_path.stat().st_size != scenario.get("prompt_bytes")):
            raise SweepError("sustained decode prompt does not match the immutable recipe hash/size")
        decode_prompts.add((str(prompt_path), scenario["prompt_sha256"], scenario["prompt_bytes"]))
    if len(decode_prompts) != 1 or any(s.get("prompt_id") != "sustained_decode" for s in decode):
        raise SweepError("all decode caps must use the same fixed sustained-output prompt")
    context = [x for x in scenarios if str(x.get("id", "")).startswith("context-")]
    if len(context) != 3 or {x.get("max_tokens") for x in context} != {128}:
        raise SweepError("context recipe must contain the three fixed 128-token caps")
    context_ids = {"context-4k": ("long_4k", 4109), "context-16k": ("long_16k", 16396),
                    "context-32k": ("long_30k7", 30712)}
    manifest_binding = matrix.get("quality_prompt_manifest")
    implementation_binding = matrix.get("tokenizer_implementation_snapshot")
    if not isinstance(manifest_binding, dict) or not isinstance(implementation_binding, dict):
        raise SweepError("matrix lacks frozen quality manifest/tokenizer snapshot bindings")
    manifest_path = Path(manifest_binding.get("path", "")).resolve(strict=True)
    expected_manifest_path = (matrix_path.parent / "quality-manifest" / "manifest.json").resolve(strict=True)
    if manifest_path != expected_manifest_path or sha256_file(manifest_path) != manifest_binding.get("sha256"):
        raise SweepError("quality manifest path/hash differs from the exact frozen matrix binding")
    implementation_path = Path(implementation_binding.get("path", "")).resolve(strict=True)
    if (implementation_path != Path(manifest_path.parent.parent / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py").resolve(strict=True) or
            sha256_file(implementation_path) != implementation_binding.get("sha256") or
            implementation_path.stat().st_size != implementation_binding.get("bytes")):
        raise SweepError("tokenizer implementation snapshot version/path/size/SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    implementation = manifest.get("tokenizer", {}).get("implementation_source", {})
    if (Path(implementation.get("path", "")).resolve(strict=True) != implementation_path or
            implementation.get("sha256") != implementation_binding.get("sha256") or
            implementation.get("bytes") != implementation_binding.get("bytes")):
        raise SweepError("manifest tokenizer source identity differs from the approved snapshot")
    prompts = {item["id"]: item for item in manifest.get("prompts", [])}
    if tuple(item.get("id") for item in manifest.get("prompts", [])) != EXPECTED_QUALITY_PROMPT_IDS:
        raise SweepError("frozen manifest prompt IDs/order differ from the original nine-prompt suite")
    for prompt in manifest["prompts"]:
        prompt_path = (manifest_path.parent / prompt["file"]).resolve(strict=True)
        if manifest_path.parent.resolve(strict=True) not in prompt_path.parents:
            raise SweepError(f"prompt file escaped the frozen manifest directory: {prompt['id']}")
        if prompt_path.stat().st_size != prompt.get("bytes") or sha256_file(prompt_path) != prompt.get("sha256"):
            raise SweepError(f"frozen prompt bytes/hash mismatch: {prompt['id']}")
    for scenario in context:
        expected_id, expected_tokens = context_ids.get(scenario.get("id"), (None, None))
        prompt = prompts.get(scenario.get("prompt_id"))
        if scenario.get("prompt_id") != expected_id or scenario.get("prompt_tokens") != expected_tokens or not prompt:
            raise SweepError("context scenario ID/prompt/token count differs from the fixed long-prompt recipe")
        prompt_path = (manifest_path.parent / prompt["file"]).resolve(strict=True)
        if prompt_path.stat().st_size != prompt["bytes"] or sha256_file(prompt_path) != prompt["sha256"]:
            raise SweepError(f"context prompt asset changed: {scenario['id']}")
    plan_steps = []
    arms_by_id = {arm["arm"]: arm for arm in matrix["arms"]}
    for phase in ("quality_on", "performance_off"):
        for arm_id in ARM_EXECUTION_ORDER:
            arm = arms_by_id[arm_id]
            row = arm[phase]
            run_id = row["run_id"]
            if not RUN_ID_RE.fullmatch(run_id):
                raise SweepError("RunId is outside the fixed matrix namespace")
            expected_leaf = (matrix_path.parent / "arms" / arm["arm"] /
                             ("quality-on" if phase == "quality_on" else "performance-off"))
            leaf = Path(row["path"])
            if leaf.resolve(strict=True) != expected_leaf.resolve(strict=True):
                raise SweepError(f"phase directory escaped the exact matrix output tree: {run_id}")
            current = expected_leaf
            while current != matrix_path.parent:
                if current.is_symlink():
                    raise SweepError(f"phase output path traverses a symlink: {current}")
                current = current.parent
            config_doc = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
            identity_doc = json.loads((leaf / "identity.json").read_text(encoding="utf-8"))
            args = config_doc.get("args", [])
            try:
                workers = args[args.index("--pool-workers") + 1]
                affinity = args[args.index("--pool-affinity") + 1]
            except (ValueError, IndexError) as exc:
                raise SweepError(f"pool args missing in {run_id}") from exc
            if workers != str(arm["workers"]) or affinity != arm["affinity"]:
                raise SweepError(f"pool args disagree with matrix in {run_id}")
            if config_doc.get("hetero_capture_token_ids") is not (phase == "quality_on"):
                raise SweepError(f"capture mode disagrees with matrix in {run_id}")
            if identity_doc.get("config", {}).get("capture_actual_engine_token_ids") is not (phase == "quality_on"):
                raise SweepError(f"identity capture mode disagrees with matrix in {run_id}")
            for name, expected in (("config.json", row["config_sha256"]),
                                   ("identity.json", row["identity_sha256"]),
                                   ("provenance.json", row["provenance_sha256"])):
                if sha256_file(leaf / name) != expected:
                    raise SweepError(f"prepared {name} hash mismatch in {run_id}")
            plan_steps.append({
                "run_id": run_id,
                "arm": arm["arm"],
                "worker_count": arm["workers"],
                "affinity": arm["affinity"],
                "phase": phase,
                "matrix_root": str(matrix_path.parent),
                "token_capture": phase == "quality_on",
                "config": str(leaf / "config.json"),
                "identity": str(leaf / "identity.json"),
                "provenance": str(leaf / "provenance.json"),
                "quality_prompt_manifest_path": str(manifest_path),
                "quality_prompt_manifest_sha256": manifest_binding["sha256"],
                "tokenizer_implementation_path": str(implementation_path),
                "tokenizer_implementation_sha256": implementation_binding["sha256"],
                "launch_template": "pinned run41 launch-owned.ps1 (parameterized in memory)",
                "quality_template": "pinned run41 quality controller (quality-on) or guarded hetero_bench loop (performance-off)",
            })
    return {
        "schema_version": 1,
        "status": "plan_only_no_hardware_or_process_access",
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "steps": plan_steps,
        "arm_execution_order": list(ARM_EXECUTION_ORDER),
        "performance_scenarios": enrich_performance_scenarios(scenarios, manifest_path),
        "execution_enabled": False,
        "production_adapter_installed": True,
        "requires_explicit_execute_authorization": True,
    }


def validate_root_authorization(matrix_path: Path, authorization_path: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve(strict=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "root_reviewed_authorized":
        raise SweepError("authorization is not root reviewed")
    if authorization.get("matrix_sha256") != sha256_file(matrix_path):
        raise SweepError("authorization does not bind this exact matrix hash")
    arms = {x["arm"] for x in matrix["arms"]}
    if set(authorization.get("arms", [])) != arms:
        raise SweepError("authorization arm set differs from matrix")
    nonce = authorization.get("nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise SweepError("authorization nonce must be 32 lowercase hexadecimal digits")
    repo = Path(__file__).resolve().parents[1]
    output_root = authorization.get("output_root")
    if not isinstance(output_root, str):
        raise SweepError("authorization must bind the matrix output root")
    requested_output = Path(output_root)
    if requested_output.is_symlink():
        raise SweepError("authorized output_root may not be a symlink")
    resolved_output = requested_output.resolve(strict=True)
    if resolved_output != matrix_path.parent.resolve(strict=True) or not resolved_output.is_dir():
        raise SweepError("authorized output_root must be the exact prepared matrix directory")
    expected_adapter_sha = sha256_file(Path(__file__).resolve())
    if authorization.get("adapter_sha256") != expected_adapter_sha:
        raise SweepError("authorization does not bind this exact adapter source SHA")
    expected_templates = template_source_hashes(repo)
    if authorization.get("template_sha256s") != expected_templates:
        raise SweepError("authorization does not bind the reviewed run41 template/JSONL-reader sources")
    authorization["_resolved_output_root"] = str(resolved_output)
    return authorization


def execute_controlled_coordinator(plan: dict[str, Any], authorization: dict[str, Any], backend: Any) -> list[dict[str, Any]]:
    """Run one reviewed lifecycle backend serially; clean only a prior successful phase."""
    if not callable(getattr(backend, "run_phase", None)) or not callable(getattr(backend, "cleanup_previous", None)):
        raise SweepError("controlled coordinator requires launch/watch/task and owned-cleanup methods")
    if not isinstance(authorization, dict) or not authorization.get("_resolved_output_root"):
        raise SweepError("controlled coordinator requires validated root authorization")
    if plan.get("matrix_sha256") != authorization.get("matrix_sha256"):
        raise SweepError("authorization does not bind this exact immutable plan hash")
    results = []
    previous = None
    for step in plan["steps"]:
        if step["arm"] not in authorization["arms"]:
            raise SweepError("step arm is not authorized")
        if previous is not None:
            cleanup = backend.cleanup_previous(previous)
            if cleanup.get("status") != "terminal_owned_cleanup_pass":
                raise SweepError("stop-first: prior exact-owned model/sampler cleanup failed")
        result = backend.run_phase(step)
        output_path = result.get("path") if isinstance(result, dict) else None
        if not isinstance(output_path, str):
            raise SweepError("lifecycle backend did not return an evidence path")
        resolved = Path(output_path).resolve(strict=True)
        if Path(authorization["_resolved_output_root"]) not in resolved.parents:
            raise SweepError("lifecycle evidence escaped the authorized matrix root")
        if result.get("status") != "pass":
            raise SweepError("stop-first: lifecycle phase failed; retain partial owner/evidence for review")
        results.append(result)
        previous = result
    # Deliberately leave the last successful service/sampler owned and live for root review.
    return results


def template_source_hashes(repo: Path) -> dict[str, str]:
    return {name: sha256_file(repo / rel) for name, (rel, _) in RUN41_TEMPLATES.items()}


def enrich_performance_scenarios(scenarios: list[dict[str, Any]], manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = {item["id"]: item for item in manifest.get("prompts", [])}
    enriched = []
    for source in scenarios:
        item = dict(source)
        if item.get("id", "").startswith("context-"):
            prompt = prompts[item["prompt_id"]]
            item.update(prompt_file=str((manifest_path.parent / prompt["file"]).resolve(strict=True)),
                        prompt_sha256=prompt["sha256"], prompt_bytes=prompt["bytes"])
        enriched.append(item)
    return enriched


def pinned_template_hashes() -> dict[str, str]:
    return {name: expected for name, (_, expected) in RUN41_TEMPLATES.items()}


def render_run41_template(template_name: str, step: dict[str, Any], repo: Path) -> str:
    if template_name not in ("launch", "watch", "quality", "cleanup"):
        raise SweepError("unknown run41 template")
    rel, expected_sha = RUN41_TEMPLATES[template_name]
    source_path = repo / rel
    source = source_path.read_text(encoding="utf-8-sig")
    if sha256_file(source_path) != expected_sha:
        raise SweepError(f"pinned run41 {template_name} source SHA changed")
    run_id = step.get("run_id")
    run_dir = Path(step.get("config", "")).resolve(strict=True).parent
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise SweepError("template RunId is not allowlisted")
    expected_run = (Path(step.get("matrix_root", "")) / "arms" /
                    step["arm"] / ("quality-on" if step["phase"] == "quality_on" else "performance-off")).resolve(strict=True)
    if run_dir != expected_run:
        raise SweepError("template config path does not bind the prepared arm directory")
    rendered, count = source.replace(RUN41_ID, run_id), source.count(RUN41_ID)
    if count != RUN41_ID_REPLACEMENT_COUNTS[template_name] or RUN41_ID in rendered:
        raise SweepError("run41 template RunId substitution count differs from the pinned source contract")
    if template_name == "launch":
        old = "$run = Join-Path $repo 'bench\\hetero\\" + run_id + "'"
        if rendered.count(old) != 1: raise SweepError("run41 launch run-directory anchor not unique")
        rendered = rendered.replace(old, "$run = '" + str(run_dir) + "'")
    if template_name == "watch":
        old = "$runId='" + run_id + "'; $run=Join-Path $repo ('bench\\hetero\\'+$runId)"
        if rendered.count(old) != 1: raise SweepError("run41 watcher run-directory anchor not unique")
        rendered = rendered.replace(old, "$runId='" + run_id + "'; $run='" + str(run_dir) + "'")
    if template_name == "quality":
        old = "$runDir = Join-Path $repo 'bench\\hetero\\" + run_id + "'"
        if rendered.count(old) != 1: raise SweepError("run41 quality run-directory anchor not unique")
        rendered = rendered.replace(old, "$runDir = '" + str(run_dir) + "'")
    if template_name == "cleanup":
        old = "$repo='C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero';$run=Join-Path $repo 'bench\\hetero\\" + run_id + "';"
        if rendered.count(old) != 1: raise SweepError("run41 cleanup run-directory anchor not unique")
        rendered = rendered.replace(old, "$repo='C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero';$run='" + str(run_dir) + "';")
    if template_name == "quality" and step["phase"] == "performance_off":
        old = "$identity.config.capture_actual_engine_token_ids -ne $true"
        if rendered.count(old) != 1:
            raise SweepError("run41 capture-mode guard anchor changed")
        rendered = rendered.replace(old, "$identity.config.capture_actual_engine_token_ids -ne $false")
    if template_name in ("launch", "quality"):
        manifest_path = Path(step.get("quality_prompt_manifest_path", "")).resolve(strict=True)
        if sha256_file(manifest_path) != step.get("quality_prompt_manifest_sha256"):
            raise SweepError("parameterized phase is not bound to the frozen quality manifest SHA")
        old_launch_manifest = "$manifestPath = Join-Path $repo 'bench\\hetero\\20261009-06-quality-prompts\\manifest.json'"
        old_quality_manifest = "$manifestPath = 'C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero\\bench\\hetero\\20261009-06-quality-prompts\\manifest.json'"
        old = old_launch_manifest if template_name == "launch" else old_quality_manifest
        if rendered.count(old) != 1:
            raise SweepError(f"run41 {template_name} manifest-path source anchor not unique")
        rendered = rendered.replace(old, "$manifestPath = '" + str(manifest_path) + "'")
    return rendered


def render_performance_controller(step: dict[str, Any], repo: Path) -> str:
    source = render_run41_template("quality", step, repo)
    start_marker = "$readyBindingPath = Join-Path $runDir 'quality\\ready-binding.json'"
    if source.count(start_marker) != 1:
        raise SweepError("run41 quality-controller performance replacement anchor changed")
    capture_guard = "$identity.config.capture_actual_engine_token_ids -ne $false"
    if source.count(capture_guard) != 1:
        raise SweepError("run41 capture-mode assertion anchor changed")
    tail = r'''$phaseRunId = $contract.provenance.run_id
$performanceDir = Join-Path $runDir 'performance'
if (-not (Test-Path -LiteralPath $performanceDir -PathType Container)) { throw 'performance output directory missing' }
$bindingPath = Join-Path $performanceDir 'ready-binding.json'
if (Test-Path -LiteralPath $bindingPath) { throw 'performance ready binding exists; refusing overwrite' }
$binding = [ordered]@{run_id=$phaseRunId;checked_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$LauncherPid;
    server_pid=$ready.server.ProcessId;engine_pid=$ready.engine.ProcessId;sampler_actual_child_pid=(Get-Content -LiteralPath (Join-Path $runDir 'sampler-process.json') -Raw | ConvertFrom-Json).actual_child_pid;
    engine_sha256=(Get-Sha256 $identity.engine.path);server_file_sha256=(Get-Sha256 $serverScript);mode='performance-off'}
Write-ExclusiveJson $bindingPath $binding
$progressPath = Join-Path $performanceDir 'performance-progress.jsonl'
if (Test-Path -LiteralPath $progressPath) { throw 'performance progress exists; refusing overwrite' }
$progress = [IO.File]::Open($progressPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
function Write-Progress($Event) { $line=(ConvertTo-Json -InputObject ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');event=$Event}) -Compress)+"`n";$bytes=[Text.UTF8Encoding]::new($false).GetBytes($line);$progress.Write($bytes,0,$bytes.Length);$progress.Flush($true) }
$scenarioResults=@();$scenarios=@($env:RUN47_PERF_SCENARIOS_JSON | ConvertFrom-Json -DateKind String)
if ($scenarios.Count -ne 6) { throw 'fixed six-scenario performance recipe is missing' }
try {
    $null=Get-LatestResourceGate 'performance start';$null=Get-ReadyBindings 'performance start' 0
    Write-Progress @{event='performance_start';scenario_count=$scenarios.Count}
    foreach ($scenario in $scenarios) {
        if ($scenario.id -notmatch '^(decode-cap-(256|512|1024)|context-(4k|16k|32k))$') { throw 'performance scenario ID not allowlisted' }
        $promptPath=[IO.Path]::GetFullPath([string]$scenario.prompt_file)
        if (-not (Test-Path -LiteralPath $promptPath -PathType Leaf) -or (Get-Sha256 $promptPath) -ne $scenario.prompt_sha256 -or
            (Get-Item -LiteralPath $promptPath).Length -ne $scenario.prompt_bytes) { throw "$($scenario.id): prompt bytes/hash differ from matrix recipe" }
        $cap=[int]$scenario.max_tokens
        if (($scenario.id -like 'decode-cap-*' -and $cap -notin @(256,512,1024)) -or
            ($scenario.id -like 'context-*' -and $cap -ne 128)) { throw "$($scenario.id): output cap differs from fixed recipe" }
        $null=Get-ReadyBindings "before $($scenario.id)" 0;$null=Get-LatestResourceGate "before $($scenario.id)"
        $requestRunId=$phaseRunId+'-'+$scenario.id
        $outputPath=Join-Path $performanceDir $requestRunId
        $stdoutPath=Join-Path $performanceDir ($scenario.id+'.stdout.log');$stderrPath=Join-Path $performanceDir ($scenario.id+'.stderr.log')
        foreach($path in @($outputPath,$stdoutPath,$stderrPath)){if(Test-Path -LiteralPath $path){throw "$($scenario.id): existing output; refusing overwrite"}}
        $runnerArgs=@((Join-Path $repo 'tools\hetero_bench.py'),'--prompt-file',$promptPath,'--output',$performanceDir,
            '--run-id',$requestRunId,'--repeats','3','--warmup','1','--max-tokens',[string]$cap,'--concurrency','1',
            '--seed','42','--reasoning-effort','none','--identity-json',$identityPath,'--base-url',"http://127.0.0.1:$port",
            '--allow-capped-performance')
        $runner=Start-Process -FilePath $python -ArgumentList $runnerArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $runnerLauncher=Get-CimInstance Win32_Process -Filter "ProcessId = $($runner.Id)" -ErrorAction SilentlyContinue
        if (-not $runnerLauncher) { throw "$($scenario.id): benchmark launcher identity missing" }
        $actual=$null
        if ($runnerLauncher.ExecutablePath -eq 'C:\Python314\python.exe' -and $runnerLauncher.CommandLine.Contains('tools\hetero_bench.py')) {$actual=$runnerLauncher}
        for($i=0;$i -lt 50 -and -not $actual;$i++){$actual=Get-CimInstance Win32_Process|Where-Object{$_.ParentProcessId -eq $runner.Id -and $_.ExecutablePath -eq 'C:\Python314\python.exe' -and $_.CommandLine.Contains('tools\hetero_bench.py') -and $_.CommandLine.Contains($requestRunId)}|Select-Object -First 1;if(-not $actual){Start-Sleep -Milliseconds 100}}
        if(-not $actual){throw "$($scenario.id): benchmark actual C Python child identity missing"}
        $launcherCreated=$runnerLauncher.CreationDate.ToUniversalTime().ToString('o');$actualCreated=$actual.CreationDate.ToUniversalTime().ToString('o')
        Write-Progress @{scenario=$scenario.id;event='runner_started';launcher_pid=$runner.Id;launcher_exe=$runnerLauncher.ExecutablePath;launcher_create_utc=$launcherCreated;
            actual_child_pid=$actual.ProcessId;actual_child_exe=$actual.ExecutablePath;actual_child_parent_pid=$actual.ParentProcessId;actual_child_create_utc=$actualCreated;argv=$runnerArgs}
        try {
            while ($true) {
                $runner.Refresh();$live=Get-CimInstance Win32_Process -Filter "ProcessId = $($actual.ProcessId)" -ErrorAction SilentlyContinue
                if ($runner.HasExited -and -not $live) { break }
                if ($runner.HasExited -and $live) { throw "$($scenario.id): actual benchmark child outlived launcher" }
                $null=Get-LatestResourceGate "during $($scenario.id)";$null=Get-ReadyBindings "during $($scenario.id)" 1
                Start-Sleep -Seconds 1;$runner.Refresh()
            }
        } catch {
            Stop-OwnedRunner $runner.Id $launcherCreated $runnerLauncher.ExecutablePath $actual.ProcessId $actualCreated $requestRunId
            throw
        }
        $runnerFinalChild=Get-CimInstance Win32_Process -Filter "ProcessId = $($actual.ProcessId)" -ErrorAction SilentlyContinue
        if($runnerFinalChild){throw "$($scenario.id): actual benchmark child remained after exit"}
        if($runner.ExitCode -ne 0 -or -not(Test-Path -LiteralPath $outputPath -PathType Container)){throw "$($scenario.id): benchmark failed; preserve partial output"}
        $null=Get-LatestResourceGate "after $($scenario.id)";$null=Get-ReadyBindings "after $($scenario.id)" 0
        $scenarioResults+=,[ordered]@{id=$scenario.id;run_id=$requestRunId;status='client_exit_zero_output_preserved';exit_code=[int]$runner.ExitCode;
            max_tokens=$cap;prompt_sha256=$scenario.prompt_sha256;output_path=$outputPath;launcher_pid=$runner.Id;actual_child_pid=$actual.ProcessId;
            launcher_create_utc=$launcherCreated;actual_child_create_utc=$actualCreated}
        Write-Progress @{scenario=$scenario.id;event='runner_complete';result=$scenarioResults[-1]}
    }
    $summary=[ordered]@{schema_version=1;run_id=$phaseRunId;status='six_performance_clients_completed_pending_offline_guard';completed_utc=[DateTime]::UtcNow.ToString('o');scenarios=$scenarioResults}
    Write-ExclusiveJson (Join-Path $performanceDir 'performance-stage.json') $summary
    Write-Progress @{event='performance_complete';status=$summary.status}
    $summary|ConvertTo-Json -Depth 10 -Compress
} catch {
    Write-Progress @{event='failed';error_type=$_.Exception.GetType().Name;error=$_.Exception.Message;completed_scenarios=$scenarioResults.Count}
    $failurePath=Join-Path $performanceDir 'performance-failure.json'
    if(-not(Test-Path -LiteralPath $failurePath)){Write-ExclusiveJson $failurePath ([ordered]@{schema_version=1;run_id=$phaseRunId;status='failed_stopped';utc=[DateTime]::UtcNow.ToString('o');error_type=$_.Exception.GetType().Name;error=$_.Exception.Message;completed_scenarios=$scenarioResults})}
    throw
} finally { $progress.Dispose() }
'''
    prefix = source[:source.index(start_marker)]
    rendered = prefix + tail
    return rendered


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SweepError("PowerShell lifecycle returned no JSON result object")


def validate_powershell_syntax(script_text: str) -> None:
    """Use the pinned PowerShell parser only; this never evaluates the script."""
    if not PWsh7.is_file() or sha256_file(PWsh7) != PWSH7_SHA256:
        raise SweepError("pinned PowerShell 7 executable identity mismatch")
    command = "$s=[Console]::In.ReadToEnd();$t=$null;$e=$null;[void][System.Management.Automation.Language.Parser]::ParseInput($s,[ref]$t,[ref]$e);if($e.Count){$e|ForEach-Object{[Console]::Error.WriteLine($_.Message)};exit 2};Write-Output syntax_ok"
    completed = subprocess.run([str(PWsh7), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                               input=script_text, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    if completed.returncode != 0 or "syntax_ok" not in completed.stdout:
        raise SweepError("parameterized run41 PowerShell text failed parse-only syntax validation")


def _query_pwsh_process_identity(pid: int) -> dict[str, Any]:
    command = (f'$p=Get-CimInstance Win32_Process -Filter "ProcessId = {int(pid)}" -ErrorAction SilentlyContinue;'
               'if(-not $p){exit 3};[ordered]@{pid=[int]$p.ProcessId;parent_pid=[int]$p.ParentProcessId;'
               'exe=$p.ExecutablePath;create_utc=$p.CreationDate.ToUniversalTime().ToString(\'o\');'
               'command_line=$p.CommandLine}|ConvertTo-Json -Compress')
    completed = subprocess.run([str(PWsh7), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                               cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=False, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10)
    if completed.returncode != 0:
        raise SweepError(f"unable to snapshot owned PowerShell PID {pid} through CIM")
    try:
        owner = json.loads(completed.stdout.decode("utf-8", "strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SweepError(f"invalid CIM identity JSON for owned PowerShell PID {pid}") from exc
    if (not isinstance(owner, dict) or owner.get("pid") != pid or owner.get("parent_pid") != os.getpid() or
            str(owner.get("exe", "")).casefold() != str(PWsh7).casefold() or
            not isinstance(owner.get("command_line"), str) or "-Command" not in owner["command_line"] or
            "ReadToEnd" not in owner["command_line"]):
        raise SweepError(f"owned PowerShell PID {pid} failed executable/parent/command identity verification")
    validate_create_time(owner.get("create_utc"))
    return owner


def _close_unfed_owned_process(proc: subprocess.Popen) -> None:
    """Close only the exact Popen-owned process while it is still blocked before script input."""
    if proc.stdin and not proc.stdin.closed:
        proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _invoke_pwsh_text(script_text: str, action: str, *, env: dict[str, str], cwd: Path,
                      on_started: Callable[[dict[str, Any], subprocess.Popen], None],
                      timeout_seconds: float = 3600) -> tuple[int, bytes, bytes, dict[str, Any]]:
    if not PWsh7.is_file() or sha256_file(PWsh7) != PWSH7_SHA256:
        raise SweepError("pinned PowerShell 7 executable identity mismatch")
    suffix = {
        "validate": "",
        "launch": "-Start",
        "watch": "-Monitor -TimeoutSeconds 1800",
        "check-ready": "-CheckReady -RootReadyConfirmed -LauncherPid ([int]$env:RUN47_LAUNCHER_PID) -SamplerPid ([int]$env:RUN47_SAMPLER_PID)",
        "quality": "-Run -RootReadyConfirmed -LauncherPid ([int]$env:RUN47_LAUNCHER_PID) -SamplerPid ([int]$env:RUN47_SAMPLER_PID)",
        "performance": "-Run -RootReadyConfirmed -LauncherPid ([int]$env:RUN47_LAUNCHER_PID) -SamplerPid ([int]$env:RUN47_SAMPLER_PID)",
        "cleanup": "-Execute",
    }.get(action)
    if suffix is None:
        raise SweepError("unknown lifecycle script action")
    wrapper = "$source=[Console]::In.ReadToEnd();$block=[scriptblock]::Create($source);& $block " + suffix
    argv = [str(PWsh7), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrapper]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                            creationflags=flags)
    try:
        owner = _query_pwsh_process_identity(proc.pid)
        owner.update({"argv": argv, "argv_windows": serialize_windows_argv(argv),
                      "controller_streams": "stdout/stderr pipes captured before script input; exclusive evidence paths supplied by caller"})
        on_started(owner, proc)
    except Exception:
        _close_unfed_owned_process(proc)
        raise
    try:
        out, err = proc.communicate(script_text.encode("utf-8"), timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        # Preserve the exact controller PID for external owner review. Do not kill by PID here.
        raise LifecycleTimeout(owner, exc.output or b"", exc.stderr or b"", proc) from exc
    return int(proc.returncode), out, err, owner


class Run41LifecycleAdapter:
    """Shared per-recipe adapter over pinned run41 launch/watch/quality/cleanup code."""
    def __init__(self, matrix_path: Path, authorization: dict[str, Any], *, repo: Path | None = None):
        self.matrix_path = matrix_path.resolve(strict=True)
        self.authorization = authorization
        self.repo = (repo or Path(__file__).resolve().parents[1]).resolve(strict=True)
        self.live_result: dict[str, Any] | None = None
        self._owned_controller_handles: dict[int, subprocess.Popen] = {}
        self.execution_state: dict[str, Any] = {
            "execution_attempted": False,
            "lifecycle_controller_started": False,
            "model_launch_reported": False,
            "stage": "authorization_validated_preflight",
            "run_id": None,
            "run_dir": None,
            "controller_dir": None,
            "controller_process": None,
            "owner_receipt_paths": [],
            "live_owner_state": "none",
            "last_cleanup_receipt": None,
        }
        if self.authorization.get("_resolved_output_root") != str(self.matrix_path.parent.resolve(strict=True)):
            raise SweepError("adapter output root differs from the authorized matrix root")
        if template_source_hashes(self.repo) != pinned_template_hashes():
            raise SweepError("run41 lifecycle template set is not the reviewed exact source snapshot")

    def _controller_dir(self, step: dict[str, Any]) -> Path:
        run_dir = Path(step["config"]).resolve(strict=True).parent
        folder = run_dir / "controller"
        folder.mkdir(exist_ok=False)
        return folder

    def _call(self, step: dict[str, Any], template: str, action: str, controller_dir: Path,
              *, launcher_pid: int | None = None, sampler_pid: int | None = None,
              scenarios: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        script_text = render_performance_controller(step, self.repo) if action == "performance" else render_run41_template(template, step, self.repo)
        stamp = {"action": action, "run_id": step["run_id"], "template": template,
                 "rendered_script_sha256": sha256_bytes(script_text.encode("utf-8")),
                 "template_source_sha256": sha256_file(self.repo / RUN41_TEMPLATES[template][0]),
                 "adapter_sha256": sha256_file(Path(__file__).resolve()),
                 "matrix_sha256": sha256_file(self.matrix_path), "authorization_nonce": self.authorization["nonce"],
                 "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                 "argv": [str(PWsh7), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "<stdin-scriptblock>"],
                 "argv_windows": serialize_windows_argv([str(PWsh7), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "<stdin-scriptblock>"])}
        _save_exclusive(controller_dir / f"{action}-invocation.json", (json.dumps(stamp, indent=2) + "\n").encode())
        env = os.environ.copy()
        env["RUN47_LAUNCHER_PID"] = str(launcher_pid or 0)
        env["RUN47_SAMPLER_PID"] = str(sampler_pid or 0)
        if scenarios is not None:
            env["RUN47_PERF_SCENARIOS_JSON"] = json.dumps(scenarios, separators=(",", ":"))
        process_receipt_path = controller_dir / f"{action}-process.json"

        def on_started(owner: dict[str, Any], proc: subprocess.Popen) -> None:
            process_receipt = {"schema_version": 1, "run_id": step["run_id"], "action": action,
                               "status": "owned_controller_started_before_script_input", "owner": owner,
                               "rendered_script_sha256": stamp["rendered_script_sha256"],
                               "template_source_sha256": stamp["template_source_sha256"],
                               "matrix_sha256": stamp["matrix_sha256"], "authorization_nonce": stamp["authorization_nonce"],
                               "stdout_evidence_path": str(controller_dir / f"{action}.stdout.log"),
                               "stderr_evidence_path": str(controller_dir / f"{action}.stderr.log")}
            _save_exclusive(process_receipt_path, (json.dumps(process_receipt, indent=2) + "\n").encode())
            self._owned_controller_handles[proc.pid] = proc
            owner_state = self.execution_state.get("live_owner_state", "none")
            if action == "launch":
                owner_state = "launch_controller_started_owner_state_unverified"
            elif action == "cleanup":
                owner_state = "exact_owned_cleanup_controller_running"
            self.execution_state.update({"lifecycle_controller_started": True, "execution_attempted": True,
                                         "stage": f"{action}_controller_running", "controller_process": owner,
                                         "controller_process_receipt": str(process_receipt_path),
                                         "live_owner_state": owner_state})
        try:
            code, stdout_bytes, stderr_bytes, owner = _invoke_pwsh_text(script_text, action, env=env, cwd=self.repo,
                                                                        on_started=on_started)
        except LifecycleTimeout as exc:
            _save_exclusive(controller_dir / f"{action}.stdout.log", exc.stdout)
            _save_exclusive(controller_dir / f"{action}.stderr.log", exc.stderr)
            _save_exclusive(controller_dir / f"{action}-timeout.json", (json.dumps({"action": action,
                "status": "controller_timeout_process_preserved_for_exact_review", "controller_pid": exc.owner["pid"],
                "owner": exc.owner,
                "run_id": step["run_id"], "rendered_script_sha256": stamp["rendered_script_sha256"],
                "stdout_partial_sha256": sha256_bytes(exc.stdout), "stderr_partial_sha256": sha256_bytes(exc.stderr)}, indent=2) + "\n").encode())
            raise
        self._owned_controller_handles.pop(int(owner["pid"]), None)
        _save_exclusive(controller_dir / f"{action}.stdout.log", stdout_bytes)
        _save_exclusive(controller_dir / f"{action}.stderr.log", stderr_bytes)
        if len(stdout_bytes) > 4 * 1024 * 1024 or len(stderr_bytes) > 4 * 1024 * 1024:
            raise SweepError(f"run41 {action} output exceeded bound; complete raw logs retained in {controller_dir}")
        stdout = stdout_bytes.decode("utf-8", "replace")
        stderr = stderr_bytes.decode("utf-8", "replace")
        result = {"exit_code": code, "controller_pid": owner["pid"], "controller_owner": owner, "stdout": stdout, "stderr": stderr}
        _save_exclusive(controller_dir / f"{action}-exit.json", (json.dumps({"action": action, "exit_code": code,
            "controller_pid": owner["pid"], "owner": owner, "stdout_sha256": sha256_bytes(stdout_bytes),
            "stderr_sha256": sha256_bytes(stderr_bytes)}, indent=2) + "\n").encode())
        if code != 0:
            raise SweepError(f"run41 {action} phase failed with exit {code}; logs retained in {controller_dir}")
        result["json"] = _last_json_object(stdout)
        return result

    def run_phase(self, step: dict[str, Any]) -> dict[str, Any]:
        if self.live_result is not None:
            raise SweepError("previous successful phase must be exactly cleaned before next launch")
        config_path = Path(step["config"]).resolve(strict=True)
        identity_path = Path(step["identity"]).resolve(strict=True)
        provenance_path = Path(step["provenance"]).resolve(strict=True)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.execution_state.update({"stage": "phase_preflight", "run_id": step["run_id"],
                                     "run_dir": str(config_path.parent), "controller_dir": None,
                                     "model_launch_reported": False})
        if (sha256_file(config_path) != next(x for x in json.loads(self.matrix_path.read_text(encoding="utf-8"))["arms"]
                                               if x["arm"] == step["arm"])[step["phase"]]["config_sha256"] or
                config.get("exe") != identity.get("engine", {}).get("path") or
                provenance.get("run_id") != step["run_id"] or provenance.get("engine_sha256") != identity.get("engine", {}).get("sha256") or
                provenance.get("shared_controls", {}).get("quality_prompt_manifest") != step.get("quality_prompt_manifest_path") or
                provenance.get("shared_controls", {}).get("quality_prompt_manifest_sha256") != step.get("quality_prompt_manifest_sha256") or
                identity.get("quality_prompt_manifest", {}).get("path") != step.get("quality_prompt_manifest_path") or
                identity.get("quality_prompt_manifest", {}).get("sha256") != step.get("quality_prompt_manifest_sha256") or
                identity.get("tokenizer_implementation", {}).get("path") != step.get("tokenizer_implementation_path") or
                identity.get("tokenizer_implementation", {}).get("sha256") != step.get("tokenizer_implementation_sha256") or
                config.get("hetero_capture_token_ids") is not step["token_capture"]):
            raise SweepError("phase config/identity/provenance changed from the immutable matrix recipe")
        controller_dir = self._controller_dir(step)
        if controller_dir.parent != config_path.parent:
            raise SweepError("controller evidence directory escaped its run leaf")
        self.execution_state["controller_dir"] = str(controller_dir)
        self.execution_state["owner_receipt_paths"] = [
            str(config_path.parent / name) for name in ("process.json", "sampler-process.json", "resource/startup-watch-result.json")
        ]
        self.execution_state["stage"] = "launch_controller_starting"
        launch = self._call(step, "launch", "launch", controller_dir)
        launch_obj = launch["json"]
        if launch_obj.get("status") != "launched_not_yet_ready" or launch_obj.get("start_performed") is not True:
            raise SweepError("run41 launch template did not return its expected owned-launch state")
        self.execution_state.update({"model_launch_reported": True, "live_owner_state": "owned_model_and_sampler_startup_pending",
                                     "launcher_pid": launch_obj.get("launcher_pid"), "sampler_pid": launch_obj.get("sampler_pid"),
                                     "stage": "startup_watch"})
        watch = self._call(step, "watch", "watch", controller_dir)
        watch_obj = watch["json"]
        if watch_obj.get("status") != "startup_ready":
            raise SweepError("run41 startup watcher did not prove startup_ready")
        self.execution_state["live_owner_state"] = "owned_service_startup_ready"
        launcher_pid = int(launch_obj["launcher_pid"])
        sampler_pid = int(launch_obj["sampler_pid"])
        process_receipt = json.loads((config_path.parent / "process.json").read_text(encoding="utf-8"))
        sampler_receipt = json.loads((config_path.parent / "sampler-process.json").read_text(encoding="utf-8"))
        if (process_receipt.get("run_id") != step["run_id"] or process_receipt.get("launcher_pid") != launcher_pid or
                process_receipt.get("engine_sha256") != identity["engine"]["sha256"] or
                process_receipt.get("config_sha256") != sha256_file(config_path) or
                process_receipt.get("server_file_sha256") != provenance.get("server_file_sha256") or
                sampler_receipt.get("sampler_launcher_pid") != sampler_pid or
                sampler_receipt.get("actual_child_pid") != watch_obj.get("sampler_child_pid") or
                sampler_receipt.get("actual_child_create_utc") != watch_obj.get("sampler_child_create_utc") or
                watch_obj.get("launcher_pid") != launcher_pid or watch_obj.get("engine_exe") != identity["engine"]["path"]):
            raise SweepError("launch/watch/sampler receipts do not bind exact config/binary/bridge/PID identities")
        check = self._call(step, "quality", "check-ready", controller_dir, launcher_pid=launcher_pid, sampler_pid=sampler_pid)
        if (check["json"].get("status") != "live_quality_preflight_pass" or
                check["json"].get("chat_requests_sent") != 0 or check["json"].get("model_requests") != 0 or
                check["json"].get("manifest_has_prompt_paths") is not True or check["json"].get("identity_bound") is not True or
                float(check["json"].get("physical_gib", 0)) < 12 or float(check["json"].get("commit_gib", 0)) < 4):
            raise SweepError("run41 live CheckReady did not pass zero-request readiness binding")
        self.execution_state["live_owner_state"] = "owned_service_ready_task_pending"
        self.execution_state["stage"] = "quality_or_performance_task"
        if step["phase"] == "quality_on":
            task_guard_results = []
            task = self._call(step, "quality", "quality", controller_dir, launcher_pid=launcher_pid, sampler_pid=sampler_pid)
            if task["json"].get("status") != "nine_prompt_matrix_complete" or task["json"].get("formal_requests") != 27:
                raise SweepError("strict nine-prompt H4 quality stage did not complete")
        else:
            scenarios = self._performance_scenarios()
            task = self._call(step, "quality", "performance", controller_dir, launcher_pid=launcher_pid,
                              sampler_pid=sampler_pid, scenarios=scenarios)
            stage_path = config_path.parent / "performance" / "performance-stage.json"
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            task_guard_results = []
            for scenario in scenarios:
                scenario_dir = config_path.parent / "performance" / (step["run_id"] + "-" + scenario["id"])
                for formal in range(3):
                    task_guard_results.append(validate_performance_request(scenario_dir, formal, int(scenario["max_tokens"]), allow_capped=True,
                        expected_prompt_sha256=scenario.get("prompt_sha256"), expected_prompt_bytes=scenario.get("prompt_bytes")))
            if task["json"].get("status") != "six_performance_clients_completed_pending_offline_guard" or stage.get("status") != task["json"].get("status"):
                raise SweepError("performance controller output/status did not pass the offline strict validator")
        result_path = controller_dir / "phase-result.json"
        summary = {"schema_version": 1, "status": "pass", "run_id": step["run_id"], "phase": step["phase"],
                   "arm": step["arm"], "worker_count": step["worker_count"], "affinity": step["affinity"],
                   "token_capture": step["token_capture"], "launch": launch_obj, "startup": watch_obj,
                   "readiness_check": check["json"], "task": task["json"], "task_offline_guard_status":
                       ("quality_controller_strict_pass" if step["phase"] == "quality_on" else "timing_only_offline_guard_pass_quality_not_assessed"),
                   "performance_formal_guards": task_guard_results, "source_template_hashes": template_source_hashes(self.repo)}
        _save_exclusive(result_path, (json.dumps(summary, indent=2) + "\n").encode())
        self.live_result = {"step": step, "path": str(result_path), "status": "pass", "launcher_pid": launcher_pid,
                            "sampler_pid": sampler_pid, "run_dir": str(config_path.parent)}
        self.execution_state.update({"stage": "phase_complete", "live_owner_state": "owned_service_live_for_root_review",
                                     "phase_result_path": str(result_path)})
        return self.live_result

    def write_failure_receipt(self, exc: BaseException) -> str | None:
        controller_dir = self.execution_state.get("controller_dir")
        if not controller_dir:
            return None
        target = Path(controller_dir) / "coordinator-failure.json"
        if target.exists():
            return str(target)
        body = {"schema_version": 1, "status": "failed_preserved_after_execution_attempt" if self.execution_state.get("execution_attempted") else "phase_preflight_failed_no_lifecycle_start",
                "utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "error_type": type(exc).__name__, "error": str(exc), "execution_state": self.execution_state,
                "model_launch_reported": bool(self.execution_state.get("model_launch_reported")),
                "live_state_path": str(controller_dir), "owner_receipts": self.execution_state.get("owner_receipt_paths", [])}
        _save_exclusive(target, (json.dumps(body, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        return str(target)

    def _performance_scenarios(self) -> list[dict[str, Any]]:
        matrix = json.loads(self.matrix_path.read_text(encoding="utf-8"))
        return enrich_performance_scenarios(matrix["performance_phase"]["scenarios"],
                                            Path(matrix["quality_prompt_manifest"]["path"]))

    def cleanup_previous(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.live_result is None or result.get("path") != self.live_result.get("path"):
            raise SweepError("cleanup target is not this adapter's exact last successful phase")
        step = self.live_result["step"]
        controller_dir = Path(self.live_result["path"]).parent
        cleanup = self._call(step, "cleanup", "cleanup", controller_dir)
        raw_receipt = cleanup["json"]
        if raw_receipt.get("status") != "terminal_owned_server_and_sampler_released":
            raise SweepError("exact owned server/sampler cleanup is not terminal")
        self.live_result = None
        self.execution_state["last_cleanup_receipt"] = raw_receipt
        self.execution_state["live_owner_state"] = "terminal_owned_server_and_sampler_released"
        return {"status": "terminal_owned_cleanup_pass", "raw_receipt": raw_receipt,
                "run_id": step["run_id"], "controller_dir": str(controller_dir)}


def invoke_host_fake_cli(fake_cli: Path, argv: list[str], fixture_root: Path) -> dict[str, Any]:
    root = fixture_root.resolve(strict=True)
    cli = fake_cli.resolve(strict=True)
    if root not in cli.parents or cli.suffix.casefold() != ".py":
        raise SweepError("fake CLI must reside inside the isolated fixture root")
    completed = subprocess.run([sys.executable, str(cli), *argv], cwd=str(root),
                               shell=False, capture_output=True, text=True, timeout=10)
    if completed.returncode:
        raise SweepError("host fake CLI exited nonzero")
    return json.loads(completed.stdout)


def launch_model() -> None:
    raise SweepError("production model launching is disabled in this prepared harness")


def load_model_payload() -> None:
    raise SweepError("payload loading is disabled in this prepared harness")


def initialize_core() -> None:
    raise SweepError("OpenVINO/Core initialization is disabled in this prepared harness")


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--adapter", choices=("run41",), help="explicit reviewed shared lifecycle adapter")
    args = parser.parse_args(argv)
    adapter = None
    try:
        plan = build_plan(Path(args.matrix))
        if args.execute:
            if not args.authorization:
                raise SweepError("execution requires a separate root authorization receipt")
            if args.adapter != "run41":
                raise SweepError("execution requires explicit --adapter run41")
            authorization = validate_root_authorization(Path(args.matrix), Path(args.authorization))
            adapter = Run41LifecycleAdapter(Path(args.matrix), authorization)
            results = execute_controlled_coordinator(plan, authorization, adapter)
            print(json.dumps({"status": "matrix_phases_complete_last_service_left_for_root_review",
                              "phase_count": len(results), "results": results}, indent=2, ensure_ascii=False))
            return 0
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        execution_state = adapter.execution_state if adapter is not None else {"execution_attempted": False,
                                                                               "lifecycle_controller_started": False,
                                                                               "stage": "authorization_or_adapter_preflight",
                                                                               "live_owner_state": "none"}
        failure_path = None
        receipt_error = None
        if adapter is not None and execution_state.get("lifecycle_controller_started"):
            try:
                failure_path = adapter.write_failure_receipt(exc)
            except Exception as write_exc:
                receipt_error = f"{type(write_exc).__name__}: {write_exc}"
        status = ("failed_preserved_after_execution_attempt"
                  if execution_state.get("lifecycle_controller_started") else "rejected_no_execution")
        print(json.dumps({"status": status, "error": f"{type(exc).__name__}: {exc}",
                          "execution_attempted": bool(execution_state.get("execution_attempted")),
                          "model_launch_reported": bool(execution_state.get("model_launch_reported")),
                          "execution_state": execution_state, "live_state_path": execution_state.get("controller_dir"),
                          "failure_receipt": failure_path, "failure_receipt_error": receipt_error}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
