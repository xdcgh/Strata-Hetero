"""Plan and validate isolated two-conversation KV parking experiments.

The default command is read-only and plan-only. This module never imports a
model/runtime adapter at import time; an adapter can be loaded only after a
contract-bound root authorization has been checked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterable, Protocol


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "bench/hetero/20261009-49-kv-storage-preparation"
DEFAULT_CONTRACT = PREPARATION / "execution-contract.json"
V2_CONTRACT = PREPARATION / "execution-contract-v2.json"
RAM_FLOOR = 12 * 1024**3
COMMIT_FLOOR = 4 * 1024**3
STATE_HASH_KEYS = ("gdn", "ple", "tail", "dead", "pooled", "pooled_full", "kv", "mtp", "stale")
STATE_METADATA_KEYS = ("L_tokens", "ple_prev")
TURN_IDS = ("warmup", "formal1", "formal2", "formal3")
SESSION_RESULT_RE = re.compile(r"^(SAVED|RESTORED) ([0-9]+) ([0-9]+) ([0-9]+(?:\.[0-9]+)?)$")
SESSION_PROGRESS_RE = re.compile(r"^SESSION ([0-9]+) ([0-9]+)$")
SESSION_WAIT_RE = re.compile(r"^SWAIT ([A-Za-z0-9_-]+) ([0-9]+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAM_PARK_RE = re.compile(r"^strata serve: conversation cache: parked ([0-9]+) tokens in ([0-9]+(?:\.[0-9]+)?) ms; parked=([0-9]+) bytes=([0-9]+) evictions=([0-9]+) snapshot_bytes=([0-9]+) reused_kv_bytes=([0-9]+)$")
RAM_RESTORE_RE = re.compile(r"^strata serve: conversation cache: restored ([0-9]+) tokens \((live|checkpoint)\) in ([0-9]+(?:\.[0-9]+)?) ms; parked=([0-9]+) bytes=([0-9]+)$")


class ParkingError(ValueError):
    pass


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str):
    raise ParkingError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParkingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="strict")
        return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParkingError(f"invalid strict UTF-8 JSON: {type(exc).__name__}") from None


def strict_json_file(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return strict_json_loads(raw), sha256_bytes(raw)


def _no_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ParkingError("non-finite value is forbidden")
    if isinstance(value, dict):
        for item in value.values(): _no_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _no_nonfinite(item)


def canonical_json_bytes(value: Any) -> bytes:
    _no_nonfinite(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_json_object(raw: bytes | str) -> dict[str, Any]:
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise ParkingError("assistant completion must be one JSON object")
    return value


def _hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ParkingError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ParkingError(f"{label} must be an integer >= {minimum}")
    return value


def _option_values(args: list[str], option: str) -> list[str | None]:
    values = []
    for i, value in enumerate(args):
        if value == option:
            values.append(args[i + 1] if i + 1 < len(args) else None)
    return values


def _one_option(args: list[str], option: str, expected: str | None) -> bool:
    return _option_values(args, option) == [expected]


def validate_config(config: dict[str, Any], config_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or not isinstance(config.get("args"), list):
        raise ParkingError(f"{config_name} config must contain an argv array")
    args = config["args"]
    cache = contract["configs"][config_name]["conversation_cache_mib"]
    expected_cache = 4096 if config_name == "ram-pair" else 0
    if config_name == "disk-pair":
        values = _option_values(args, "--conversation-cache-mib")
        if values != ["0"]: raise ParkingError("disk-pair must disable RAM conversation parking")
    elif config_name == "ram-pair":
        if _option_values(args, "--conversation-cache-mib") != ["4096"]:
            raise ParkingError("ram-pair must use the fixed 4096 MiB cache")
    else:
        if _option_values(args, "--conversation-cache-mib") != ["0"]:
            raise ParkingError(f"{config_name} must disable RAM conversation parking")
    if cache != expected_cache:
        raise ParkingError(f"{config_name} cache metadata differs from the contract")
    required = {
        "--max-context": "32768", "--kv": "int8", "--prompt-cache": "3",
        "--conversation-cache-slots": "2", "--conversation-cache-min-free-mib": "12288",
        "--session-min-free-mib": "4096", "--resident-budget-gib": "20",
        "--spec": "2", "--spec-min-p": "0", "--mtp-max-t": "1", "--suffix-draft": "0",
        "--pcie-frac": "0", "--ple-io": "direct",
    }
    for option, value in required.items():
        if not _one_option(args, option, value):
            raise ParkingError(f"{config_name} has an unexpected or duplicate {option}")
    if args.count("--no-kv-grow") != 1:
        raise ParkingError(f"{config_name} must disable KV growth exactly once")
    if config.get("parallel") != 1 or config.get("fit_max_tokens") is not False:
        raise ParkingError(f"{config_name} must remain single-request and disable cap rewriting")
    if config.get("sampling") != {"temperature": 0, "top_p": 1, "seed": 42}:
        raise ParkingError(f"{config_name} sampling controls differ")
    if config.get("hetero_capture_token_ids") is not True:
        raise ParkingError(f"{config_name} must capture actual emitted token ids")
    if config.get("hetero_transport") != "private stdio StrataEngine; no HTTP listener":
        raise ParkingError(f"{config_name} transport is not the reviewed private stdio mode")
    if config.get("host") != "127.0.0.1" or config.get("open_browser") is not False:
        raise ParkingError(f"{config_name} host/browser controls differ")
    if config_name == "disk-pair":
        expected_dir = contract["snapshot_paths"]["directory"]
        if config.get("slot_save_path") != expected_dir or not _one_option(args, "--conversation-cache-mib", "0"):
            raise ParkingError("disk-pair must bind the dedicated snapshot directory and RAM cache off")
    elif config.get("slot_save_path") is not None:
        raise ParkingError(f"{config_name} must not enable disk slot persistence")
    return {"name": config_name, "cache_mib": cache, "prompt_cache": 3,
            "max_context": 32768, "transport": config["hetero_transport"],
            "sampling": dict(config["sampling"]), "config_sha256_checked": True}


def load_contract(path: Path = DEFAULT_CONTRACT) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    contract, digest = strict_json_file(path)
    if not isinstance(contract, dict) or contract.get("schema_version") not in (1, 2):
        raise ParkingError("unsupported KV parking contract schema")
    if contract.get("schema_version") == 1 and contract.get("status") != "source_inventory_and_runtime_plan_only":
        raise ParkingError("unexpected preparation contract status")
    if contract.get("schema_version") == 2:
        if (contract.get("execution_contract_version") != 2 or
                contract.get("status") not in ("execution_contract_v2_not_authorized", "execution_contract_v2_authorized")):
            raise ParkingError("unexpected v2 preparation contract state")
        binding = contract.get("stdio_protocol_binding")
        if not isinstance(binding, dict) or binding.get("stop_token_ids") != [248046, 248044]:
            raise ParkingError("v2 contract must bind the pinned stop-token pair")
        if binding.get("stop_tokens") != {"<|im_end|>": 248046, "<|endoftext|>": 248044}:
            raise ParkingError("v2 stop-token text/ID mapping differs from the pinned tokenizer")
        for key in ("bridge_sha256", "tokenizer_source_sha256", "tokenizer_config_sha256", "vocab_sha256", "token_type_sha256", "merges_sha256"):
            _hex_digest(binding.get(key), f"stdio_protocol_binding.{key}")
        if binding.get("engine_source_sha") != "ffeb748d190ddeedff23587ce848596d4db0c752":
            raise ParkingError("v2 contract is not bound to the reviewed engine source")
    if contract.get("snapshot_paths", {}).get("create_now") is not False or contract.get("snapshots_written") is not False:
        raise ParkingError("the preparation contract must not authorize snapshot creation")
    if contract.get("context", {}).get("actual_tokenizer_calibration_status") != "NOT_RUN":
        raise ParkingError("tokenizer calibration state changed; contract must be reviewed again")
    source_bindings_path = Path(contract.get("source_bindings_path", path.parent / "source-bindings.json"))
    if not source_bindings_path.is_absolute(): source_bindings_path = path.parent / source_bindings_path
    source_bindings_raw = source_bindings_path.read_bytes()
    if sha256_bytes(source_bindings_raw) != contract.get("source_bindings_sha256"):
        raise ParkingError("preparation source-binding receipt changed")
    source_bindings = strict_json_loads(source_bindings_raw)
    if source_bindings.get("source") is None or not isinstance(source_bindings.get("source"), list):
        raise ParkingError("preparation source-binding receipt is malformed")
    configs = {}
    for name, spec in contract.get("configs", {}).items():
        config_path = Path(spec["path"])
        raw = config_path.read_bytes()
        actual = sha256_bytes(raw)
        if actual != spec["sha256"]:
            raise ParkingError(f"{name} config SHA differs from reviewed contract")
        config = strict_json_loads(raw)
        configs[name] = validate_config(config, name, contract)
        configs[name]["path"] = str(config_path)
        configs[name]["sha256"] = actual
    expected_names = {"reference-A", "reference-B", "ram-pair", "disk-pair"}
    if set(configs) != expected_names:
        raise ParkingError("contract must bind exactly the four reviewed config arms")
    if contract.get("schema_version") == 2:
        if contract.get("status") == "execution_contract_v2_not_authorized" and (
                contract.get("execution_authorized") is not False or contract.get("runner_execution_permitted") is not False):
            raise ParkingError("new v2 preparation contract must remain unauthorized until root review")
        if contract.get("runner_implemented") is not True or contract.get("runtime_validation_status") != "NOT_RUN":
            raise ParkingError("v2 source adapter must be bound while runtime validation remains NOT_RUN")
        source = contract.get("runner_source_binding")
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ParkingError("v2 contract must bind the concrete stdio runner source")
        _hex_digest(source.get("sha256"), "runner_source_binding.sha256")
        source_path = Path(source["path"])
        if sha256_bytes(source_path.read_bytes()) != source["sha256"]:
            raise ParkingError("v2 stdio runner source differs from its binding")
        binary = contract.get("runtime_binary_binding")
        if not isinstance(binary, dict) or binary.get("sha256") != "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb":
            raise ParkingError("v2 contract must pin the reviewed 6bbc runtime binary")
        if binary.get("source_sha") != "ffeb748d190ddeedff23587ce848596d4db0c752":
            raise ParkingError("v2 contract binary/source binding differs from the reviewed 6bbc/ffeb pair")
        runtime_root = Path(contract.get("runtime_output_directory", ""))
        tokenizer_dir = Path(contract["stdio_protocol_binding"].get("tokenizer_directory", ""))
        if not runtime_root.is_absolute() or not tokenizer_dir.is_absolute():
            raise ParkingError("v2 runtime and tokenizer paths must be absolute")
    return contract, digest, configs


def build_plan(contract: dict[str, Any], contract_sha256: str, configs: dict[str, dict[str, Any]],
               request_manifest_sha256: str | None = None,
               reference_manifest_sha256: str | None = None,
               contract_path: Path | None = None) -> dict[str, Any]:
    phases = []
    for conversation in ("A", "B"):
        phases.append({"phase": "uninterrupted_reference", "conversation": conversation,
                       "config": f"reference-{conversation}", "requests": [f"{conversation}-{x}" for x in TURN_IDS],
                       "comparison_reference": "independent_same_conversation_uninterrupted_history"})
    phases.append({"phase": "ram-hot-parking", "config": "ram-pair",
                   "sequence": list(contract["phases"][1]["sequence"]),
                   "snapshot_evidence_type": "RamParkingEvidence",
                   "require_both_conversation_references": True})
    phases.append({"phase": "disk-parked-cold", "config": "disk-pair",
                   "sequence": list(contract["phases"][2]["sequence"]),
                   "snapshot_evidence_type": "DiskSnapshotEvidence",
                   "snapshot_directory": contract["snapshot_paths"]["directory"],
                   "require_both_conversation_references": True,
                   "OS_file_cache_state": "uncontrolled; no cold-SSD claim"})
    blockers = []
    if contract.get("execution_authorized") is not True: blockers.append("contract_execution_authorized_false")
    if contract.get("runtime_binary_binding") is None: blockers.append("runtime_binary_not_bound")
    if contract.get("runner_implemented") is not True or not isinstance(contract.get("runner_source_binding"), dict):
        blockers.append("execution_adapter_not_reviewed_or_bound")
    if contract.get("runtime_validation_status") == "NOT_RUN": blockers.append("execution_adapter_runtime_validation_not_run")
    if contract.get("context", {}).get("actual_tokenizer_calibration_status") == "NOT_RUN": blockers.append("prompt_tokenizer_calibration_not_run")
    if request_manifest_sha256 is None: blockers.append("exact_prompt_token_cap_manifest_not_supplied")
    if reference_manifest_sha256 is None: blockers.append("independent_A_B_reference_manifest_not_supplied_or_not_yet_captured")
    return {"schema_version": contract.get("schema_version", 1),
            "execution_contract_version": contract.get("execution_contract_version", 1),
            "run_id": contract.get("run_id"), "contract_path": str(contract_path.resolve()) if contract_path else None,
            "status": "plan_only",
            "contract_sha256": contract_sha256, "config_bindings": configs, "phases": phases,
            "request_manifest_sha256": request_manifest_sha256,
            "reference_manifest_sha256": reference_manifest_sha256,
            "blockers": blockers, "execution_authorized": False,
            "model_or_adapter_imported": False, "snapshot_directory_created": False,
            "snapshot_written": False, "api_or_ov_used": False, "gpu_used": False,
            "deadlines": {"total_arm_seconds": contract["resource_budget"]["total_arm_deadline_seconds"],
                          "session_wait_cap_seconds": 3600,
                          "resource_sampling_interval_seconds": contract["resource_budget"]["global_interval_seconds"],
                          "resource_owner_sample_max_gap_seconds": 1.5},
            "all_storage_tiers_complete": False,
            "claim_limit": "preparation plan only; hot RAM parking, disk restore, prefetch, and cold-storage performance are unmeasured"}


def _validate_token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ParkingError(f"{label} must be a nonempty list of nonnegative token IDs")
    return value


def validate_request_manifest(manifest: dict[str, Any], contract: dict[str, Any], contract_sha256: str) -> dict[str, dict[str, dict[str, Any]]]:
    version = contract.get("schema_version")
    if manifest.get("schema_version") != version or manifest.get("contract_sha256") != contract_sha256:
        raise ParkingError("request manifest is not bound to this exact contract")
    if version == 2:
        binding = contract["stdio_protocol_binding"]
        if manifest.get("stdio_protocol_binding") != binding:
            raise ParkingError("request manifest protocol/tokenizer/stop bindings differ from the v2 contract")
    conversations = manifest.get("conversations")
    if not isinstance(conversations, dict) or set(conversations) != {"A", "B"}:
        raise ParkingError("request manifest must bind exactly conversations A and B")
    normalized = {}
    expected_cap = contract["context"]["max_generated_per_turn"]
    sampling = {"temperature": 0, "top_p": 1, "seed": 42}
    for conversation in ("A", "B"):
        item = conversations[conversation]
        if not isinstance(item, dict) or item.get("prompt_cache") != 3:
            raise ParkingError(f"{conversation} request manifest must bind prompt-cache=3 semantics")
        turns = item.get("turns")
        if not isinstance(turns, list) or len(turns) != len(TURN_IDS):
            raise ParkingError(f"{conversation} must have one warmup and exactly three formal turns")
        by_id = {}
        for expected_id, turn in zip(TURN_IDS, turns):
            if not isinstance(turn, dict) or turn.get("turn_id") != expected_id:
                raise ParkingError(f"{conversation} turn order must be {TURN_IDS}")
            request_id = f"{conversation}-{expected_id}"
            prompt = turn.get("prompt_utf8")
            if not isinstance(prompt, str): raise ParkingError(f"{request_id} exact prompt text is missing")
            try: prompt_bytes = prompt.encode("utf-8", errors="strict")
            except UnicodeEncodeError: raise ParkingError(f"{request_id} prompt has invalid Unicode") from None
            digest = _hex_digest(turn.get("prompt_sha256"), f"{request_id} prompt hash")
            if sha256_bytes(prompt_bytes) != digest: raise ParkingError(f"{request_id} prompt bytes/hash mismatch")
            ids = _validate_token_ids(turn.get("prompt_token_ids"), f"{request_id} prompt_token_ids")
            cap = _exact_int(turn.get("max_tokens"), f"{request_id} max_tokens", 1)
            if cap != expected_cap: raise ParkingError(f"{request_id} generation cap differs from the fixed contract")
            if turn.get("sampling") != sampling: raise ParkingError(f"{request_id} sampling differs from the fixed contract")
            if len(ids) + cap > contract["context"]["limit_tokens"]:
                raise ParkingError(f"{request_id} exact prompt tokens plus cap exceed context")
            if expected_id == "warmup":
                target = contract["context"]["base_prompt_target_tokens"]
                tolerance = contract["context"]["base_prompt_tolerance_tokens"]
                if abs(len(ids) - target) > tolerance:
                    raise ParkingError(f"{request_id} has not passed exact-token-count calibration")
            by_id[expected_id] = {"request_id": request_id, "conversation": conversation,
                "turn_id": expected_id, "prompt_utf8": prompt, "prompt_sha256": digest,
                "prompt_token_ids": ids, "max_tokens": cap, "sampling": sampling,
                "prompt_token_count": len(ids), "prompt_cache": 3,
                "stop_token_ids": list(contract["stdio_protocol_binding"]["stop_token_ids"]) if version == 2 else None}
        normalized[conversation] = by_id
    return normalized


def validate_reference_manifest(references: dict[str, Any], requests: dict[str, dict[str, dict[str, Any]]],
                                contract: dict[str, Any], contract_sha256: str,
                                request_manifest_sha256: str) -> dict[str, dict[str, dict[str, Any]]]:
    version = contract.get("schema_version")
    if (references.get("schema_version") != version or references.get("contract_sha256") != contract_sha256 or
            references.get("request_manifest_sha256") != request_manifest_sha256):
        raise ParkingError("reference manifest is not bound to this contract and request manifest")
    if version == 2 and references.get("stdio_protocol_binding") != contract.get("stdio_protocol_binding"):
        raise ParkingError("reference manifest protocol binding differs from the v2 contract")
    conversations = references.get("conversations")
    if not isinstance(conversations, dict) or set(conversations) != {"A", "B"}:
        raise ParkingError("reference manifest must contain independent A and B histories")
    result = {}
    for conv in ("A", "B"):
        turns = conversations[conv]
        if not isinstance(turns, list) or len(turns) != len(TURN_IDS):
            raise ParkingError(f"{conv} reference must contain warmup plus three formals")
        by_id = {}
        for expected_id, ref in zip(TURN_IDS, turns):
            req = requests[conv][expected_id]
            if not isinstance(ref, dict) or ref.get("request_id") != req["request_id"]:
                raise ParkingError(f"{conv} reference turn order/identity mismatch")
            if (ref.get("prompt_sha256") != req["prompt_sha256"] or
                    ref.get("prompt_token_ids") != req["prompt_token_ids"] or
                    ref.get("max_tokens") != req["max_tokens"] or ref.get("sampling") != req["sampling"] or
                    ref.get("prompt_cache_entries") != req["prompt_cache"] or
                    (contract.get("schema_version") == 2 and ref.get("stop_token_ids") != req["stop_token_ids"])):
                raise ParkingError(f"{conv} reference request binding mismatch for {expected_id}")
            _validate_completion(ref, contract["canonical_outputs"][conv], req, reference=True)
            by_id[expected_id] = ref
        result[conv] = by_id
    return result


def _validate_state_fingerprint(value: Any, label: str) -> dict[str, Any] | None:
    if value is None: return None
    if (not isinstance(value, dict) or value.get("algorithm") != "fnv1a64" or
            value.get("phase") != "post_gen" or value.get("cryptographic") is not False):
        raise ParkingError(f"{label} state fingerprint must be typed post-GEN noncryptographic FNV-1a-64")
    domains = value.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(STATE_HASH_KEYS):
        raise ParkingError(f"{label} FNV domains differ from the pinned 64-bit source hashes")
    for key, item in domains.items():
        if not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{16}", item) is None:
            raise ParkingError(f"{label} state fingerprint {key} must be a 64-bit hex hash")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != set(STATE_METADATA_KEYS):
        raise ParkingError(f"{label} non-hash state metadata domains differ from the pinned source")
    if type(metadata["L_tokens"]) is not int or metadata["L_tokens"] < 0:
        raise ParkingError(f"{label} L_tokens metadata must be an exact nonnegative integer")
    prev = metadata["ple_prev"]
    if not isinstance(prev, list) or len(prev) != 2 or any(type(x) is not int for x in prev):
        raise ParkingError(f"{label} ple_prev metadata must preserve its two integer components")
    return value


def _validate_completion(record: dict[str, Any], expected_object: dict[str, Any], request: dict[str, Any], *, reference: bool) -> dict[str, Any]:
    label = f"{request['request_id']} {'reference' if reference else 'actual'}"
    for field in ("request_id", "prompt_sha256", "prompt_token_ids", "max_tokens", "sampling"):
        want = request[field]
        if record.get(field) != want: raise ParkingError(f"{label} changed bound {field}")
    output = strict_json_object(record.get("raw_output_utf8", ""))
    if output != expected_object or canonical_json_bytes(output) != canonical_json_bytes(expected_object):
        raise ParkingError(f"{label} JSON object differs from its own conversation's canonical expected object")
    emitted = _validate_token_ids(record.get("emitted_token_ids"), f"{label} emitted_token_ids")
    if len(emitted) > request["max_tokens"]: raise ParkingError(f"{label} emitted more than its exact token cap")
    if record.get("prompt_cache_entries") != request["prompt_cache"]:
        raise ParkingError(f"{label} prompt-cache configuration changed")
    if not isinstance(record.get("finish_reason"), str) or not record["finish_reason"]:
        raise ParkingError(f"{label} finish_reason missing")
    if request.get("stop_token_ids") is not None and record.get("stop_token_ids") != request["stop_token_ids"]:
        raise ParkingError(f"{label} stop IDs differ from the v2 request binding")
    if any(k in record for k in ("checkpoint_tail_sha256", "checkpoint_tail_reread_sha256", "state_hashes")):
        raise ParkingError(f"{label} contains an invented checkpoint-tail digest or legacy untyped state hash")
    fingerprint = _validate_state_fingerprint(record.get("state_fingerprint"), label)
    return {"object": output, "emitted_token_ids": emitted, "finish_reason": record["finish_reason"],
            "state_fingerprint": fingerprint}


def validate_arm_completions(actual: dict[str, Any], requests: dict[str, dict[str, dict[str, Any]]],
                             references: dict[str, dict[str, dict[str, Any]]],
                             contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(actual, dict) or set(actual) != {"A", "B"}:
        raise ParkingError("arm completion evidence must independently include A and B")
    checked = {"A": [], "B": []}; fnv_matches = 0; metadata_matches = 0; fingerprint_missing = 0
    for conv in ("A", "B"):
        records = actual[conv]
        if not isinstance(records, list) or len(records) != len(TURN_IDS):
            raise ParkingError(f"{conv} must return its warmup and three formal results")
        for turn_id, record in zip(TURN_IDS, records):
            request = requests[conv][turn_id]
            ref = references[conv][turn_id]
            parsed = _validate_completion(record, contract["canonical_outputs"][conv], request, reference=False)
            ref_parsed = _validate_completion(ref, contract["canonical_outputs"][conv], request, reference=True)
            if parsed["emitted_token_ids"] != ref_parsed["emitted_token_ids"]:
                raise ParkingError(f"{conv}/{turn_id} actual emitted token IDs differ from its own uninterrupted reference")
            if parsed["finish_reason"] != ref_parsed["finish_reason"]:
                raise ParkingError(f"{conv}/{turn_id} finish_reason differs from its own reference")
            if parsed["state_fingerprint"] is not None and ref_parsed["state_fingerprint"] is not None:
                if parsed["state_fingerprint"]["domains"] != ref_parsed["state_fingerprint"]["domains"]:
                    raise ParkingError(f"{conv}/{turn_id} post-GEN FNV hashes differ from its own reference")
                fnv_matches += 1
                if parsed["state_fingerprint"]["metadata"] != ref_parsed["state_fingerprint"]["metadata"]:
                    raise ParkingError(f"{conv}/{turn_id} post-GEN metadata differs from its own reference")
                metadata_matches += 1
            else:
                fingerprint_missing += 1
            checked[conv].append({"turn_id": turn_id, "quality": "pass"})
    return {"status": "pass", "checked": checked, "post_gen_fnv_matches": fnv_matches,
            "post_gen_metadata_matches": metadata_matches, "post_gen_fingerprint_missing": fingerprint_missing,
            "reference_scope": "A and B are each compared only to their own uninterrupted history"}


def validate_ram_evidence(evidence: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict) or evidence.get("tier") != "ram-hot":
        raise ParkingError("RAM evidence must use the ram-hot typed schema")
    expected_sequence = contract["phases"][1]["sequence"]
    if evidence.get("request_sequence") != expected_sequence:
        raise ParkingError("RAM pair request order differs from the contract")
    if evidence.get("fallback_or_skip") != []:
        raise ParkingError("RAM pair contains a skip, fallback, or admission event")
    cache_cap = contract["resource_budget"]["ram_cache_cap_bytes"]
    snapshots = evidence.get("events")
    if not isinstance(snapshots, list) or not snapshots:
        raise ParkingError("RAM parking event evidence is missing")
    seen = {"A": {"park": 0, "restore": 0}, "B": {"park": 0, "restore": 0}}
    max_bytes = 0
    for i, event in enumerate(snapshots):
        if not isinstance(event, dict) or event.get("tier") != "ram-hot":
            raise ParkingError("RAM event type/tier mismatch")
        conv = event.get("conversation")
        if conv not in seen: raise ParkingError("RAM event has unknown conversation")
        action = event.get("action")
        if action not in ("park", "restore"):
            raise ParkingError("RAM event must be typed park/restore with its matching engine log line")
        current = _exact_int(event.get("cache_bytes"), f"RAM event {i} cache_bytes")
        if current > cache_cap: raise ParkingError("RAM cache exceeded its fixed byte cap")
        max_bytes = max(max_bytes, current)
        if event.get("admitted") is not True or event.get("fallback") is not False:
            raise ParkingError("RAM event was skipped or fell back to prompt reread")
        line = event.get("log_line")
        if not isinstance(line, str): raise ParkingError("RAM park/restore must retain its actual engine log line")
        if action == "park":
            match = RAM_PARK_RE.fullmatch(line)
            if not match: raise ParkingError("RAM park log did not match the source's success record")
            tokens, elapsed, parked_slots, logged_cache, evictions, snapshot_bytes, reused_bytes = match.groups()
            for name, value in (("parked tokens", tokens), ("parked slots", parked_slots),
                                ("parked cache bytes", logged_cache), ("evictions", evictions),
                                ("snapshot bytes", snapshot_bytes), ("reused KV bytes", reused_bytes)):
                _exact_int(int(value), name)
            if not math.isfinite(float(elapsed)) or float(elapsed) < 0: raise ParkingError("invalid RAM park elapsed time")
            if int(logged_cache) != current or int(snapshot_bytes) != event.get("snapshot_bytes"):
                raise ParkingError("RAM event byte accounting differs from the actual park log")
            if int(tokens) < contract["context"]["base_prompt_target_tokens"] - contract["context"]["base_prompt_tolerance_tokens"]:
                raise ParkingError("RAM parked snapshot is shorter than the calibrated base prompt")
        else:
            match = RAM_RESTORE_RE.fullmatch(line)
            if not match: raise ParkingError("RAM restore log did not match the source's success record")
            tokens, source, elapsed, parked_slots, logged_cache = match.groups()
            _exact_int(int(tokens), "restored prefix tokens", 1); _exact_int(int(parked_slots), "parked slots")
            if not math.isfinite(float(elapsed)) or float(elapsed) < 0: raise ParkingError("invalid RAM restore elapsed time")
            if int(logged_cache) != current: raise ParkingError("RAM restore byte accounting differs from its actual log")
            expected_tokens = _exact_int(event.get("reused_prefix_tokens"), "RAM reused_prefix_tokens", 1)
            if int(tokens) != expected_tokens:
                raise ParkingError("RAM reused prefix count differs from the actual restore log")
            minimum = contract["context"]["base_prompt_target_tokens"] - contract["context"]["base_prompt_tolerance_tokens"]
            if expected_tokens < minimum: raise ParkingError("RAM restore did not reuse the calibrated base prompt prefix")
        seen[conv][action] += 1
    for conv in ("A", "B"):
        if seen[conv]["park"] < 3 or seen[conv]["restore"] < 3:
            raise ParkingError(f"RAM evidence lacks three actual park/restore events for {conv}")
    retained = evidence.get("repark_retained_bytes")
    if not isinstance(retained, dict) or set(retained) != {"A", "B"}:
        raise ParkingError("RAM re-park retained-byte accounting must separately include A and B")
    for conv, value in retained.items(): _exact_int(value, f"RAM retained bytes {conv}")
    return {"tier": "ram-hot", "status": "pass", "max_cache_bytes": max_bytes,
            "event_counts": seen, "repark_retained_bytes": retained}


def parse_session_line(line: str, max_wait_seconds: int = 3600) -> dict[str, Any]:
    if not isinstance(line, str) or "\r" in line or "\n" in line or not line:
        raise ParkingError("session protocol line must be one nonempty line")
    match = SESSION_PROGRESS_RE.fullmatch(line)
    if match:
        done, total = map(int, match.groups())
        if total <= 0 or done > total: raise ParkingError("invalid SESSION progress counts")
        return {"kind": "progress", "done": done, "total": total}
    match = SESSION_WAIT_RE.fullmatch(line)
    if match:
        phase, seconds = match.group(1), int(match.group(2))
        if seconds < 1 or seconds > max_wait_seconds: raise ParkingError("SWAIT duration outside the bounded allowance")
        return {"kind": "wait", "phase": phase, "seconds": seconds}
    match = SESSION_RESULT_RE.fullmatch(line)
    if match:
        verb, tokens_s, bytes_s, millis_s = match.groups()
        millis = float(millis_s)
        if not math.isfinite(millis) or millis < 0: raise ParkingError("invalid session elapsed time")
        tokens, count = int(tokens_s), int(bytes_s)
        if tokens <= 0 or count <= 0: raise ParkingError("session result counts must be positive")
        action = "save" if verb == "SAVED" else "restore"
        return {"kind": "result", "action": action, "tokens": tokens, "bytes": count, "milliseconds": millis}
    if line.startswith(("SERR ", "FATAL", "ERR ")):
        raise ParkingError("session operation refusal/fatal result cannot pass")
    raise ParkingError("unknown or malformed session protocol line")


def validate_disk_evidence(evidence: dict[str, Any], contract: dict[str, Any],
                           expected_seed_tokens: dict[str, int]) -> dict[str, Any]:
    if not isinstance(evidence, dict) or evidence.get("tier") != "disk-parked-cold":
        raise ParkingError("disk evidence must use the disk-parked-cold typed schema")
    directory = Path(contract["snapshot_paths"]["directory"])
    expected_files = {"A": "A-seed.bin", "B": "B-seed.bin"}
    seeds = evidence.get("seed_files")
    if not isinstance(seeds, list) or len(seeds) != 2:
        raise ParkingError("disk evidence must contain exactly two initial seed files")
    by_conv = {}
    total = 0
    for seed in seeds:
        conv = seed.get("conversation") if isinstance(seed, dict) else None
        if conv not in expected_files or conv in by_conv: raise ParkingError("duplicate/unknown disk seed conversation")
        path = Path(seed.get("path", ""))
        if not path.is_absolute() or path.name != expected_files[conv] or path.parent != directory:
            raise ParkingError("disk seed path differs from its exclusive contract path")
        if seed.get("destination_absent_before_save") is not True or seed.get("created_exclusive") is not True:
            raise ParkingError("disk SAVE destination was not confirmed absent/exclusive")
        size = _exact_int(seed.get("bytes"), f"disk seed {conv} bytes", 1)
        digest = _hex_digest(seed.get("sha256"), f"disk seed {conv} sha256")
        if seed.get("readback_sha256") != digest or seed.get("after_sha256") != digest:
            raise ParkingError("disk seed readback or immutability hash changed")
        if seed.get("before_file_id") != seed.get("after_file_id") or seed.get("before_mtime_ns") != seed.get("after_mtime_ns"):
            raise ParkingError("disk seed identity changed after SAVE")
        if seed.get("regular_single_link_no_reparse") is not True:
            raise ParkingError("disk seed file type/link validation is incomplete")
        if size > contract["snapshot_paths"]["max_file_bytes"]: raise ParkingError("disk seed exceeds per-file size cap")
        total += size; by_conv[conv] = seed
    if set(by_conv) != {"A", "B"} or total > contract["snapshot_paths"]["max_total_file_bytes"]:
        raise ParkingError("disk seed set/aggregate byte cap invalid")
    if set(expected_seed_tokens) != {"A", "B"}:
        raise ParkingError("independent A/B seed token bindings are required")
    for conv in ("A", "B"):
        _exact_int(expected_seed_tokens[conv], f"expected {conv} seed tokens", 1)
        if "checkpoint_tail_sha256" in by_conv[conv]:
            raise ParkingError("disk evidence must not claim a cryptographic checkpoint-tail digest")
    if evidence.get("save_order") != ["A", "B"]:
        raise ParkingError("disk seed files must be saved once in A then B order")
    expected_restores = ["A", "B", "A", "B", "A", "B"]
    if evidence.get("restore_order") != expected_restores:
        raise ParkingError("disk seed restore order differs from the contract")
    results = evidence.get("operations")
    if not isinstance(results, list) or len(results) != 8:
        raise ParkingError("disk trace must include two SAVE and six RESTORE operation records")
    parsed = []
    restore_seen = []
    expected_operations = [("save", "A"), ("save", "B")] + [("restore", conv) for conv in expected_restores]
    actual_operations = [(x.get("action"), x.get("conversation")) if isinstance(x, dict) else (None, None) for x in results]
    if actual_operations != expected_operations:
        raise ParkingError("SAVE/RESTORE operation order differs from the reviewed two-seed sequence")
    for operation in results:
        if not isinstance(operation, dict) or operation.get("tier") != "disk-parked-cold":
            raise ParkingError("disk operation tier/type mismatch")
        if any(key in operation for key in ("checkpoint_tail_sha256", "checkpoint_tail_reread_sha256")):
            raise ParkingError("disk evidence must not claim a pre-restore checkpoint-tail digest")
        action = operation.get("action"); conv = operation.get("conversation")
        if action not in ("save", "restore") or conv not in by_conv:
            raise ParkingError("invalid disk operation identity")
        expected_verb = "SAVED" if action == "save" else "RESTORED"
        lines = operation.get("protocol_lines")
        if not isinstance(lines, list) or not lines:
            raise ParkingError("disk operation lacks SESSION/SWAIT/result lines")
        last_by_total = {}; result = None; progress_count = 0; waits = []
        per_phase_wait = min(3600, 60 + (by_conv[conv]["bytes"] >> 22),
                             contract["resource_budget"]["total_arm_deadline_seconds"])
        for line in lines:
            row = parse_session_line(line, per_phase_wait)
            if row["kind"] == "progress":
                key = row["total"]
                if row["done"] < last_by_total.get(key, -1): raise ParkingError("SESSION progress moved backwards")
                last_by_total[key] = row["done"]; progress_count += 1
            elif row["kind"] == "wait": waits.append(row)
            else:
                if result is not None or row["action"] != action: raise ParkingError("wrong/duplicate final SAVED or RESTORED line")
                result = row
        if result is None: raise ParkingError("disk operation has no final SAVED/RESTORED result")
        if any(last_by_total[total] != total for total in last_by_total):
            raise ParkingError("SESSION progress did not reach the complete file size")
        seed = by_conv[conv]
        if operation.get("seed_sha256") != seed["sha256"]:
            raise ParkingError("disk operation is not bound to the immutable seed hash")
        if action == "save":
            if result["tokens"] != expected_seed_tokens.get(conv): raise ParkingError("SAVED token count differs from the captured seed")
            if result["bytes"] != by_conv[conv]["bytes"]: raise ParkingError("SAVED byte count differs from the immutable seed")
        else:
            if result["tokens"] != expected_seed_tokens.get(conv) or result["bytes"] != by_conv[conv]["bytes"]:
                raise ParkingError("RESTORED token/byte count differs from the immutable seed")
            if (operation.get("seed_sha256_before_restore") != seed["sha256"] or
                    operation.get("seed_bytes_before_restore") != seed["bytes"]):
                raise ParkingError("full seed SHA-256 and byte count were not rechecked before this restore")
            if (operation.get("seed_sha256_after_restore") != seed["sha256"] or
                    operation.get("seed_bytes_after_restore") != seed["bytes"]):
                raise ParkingError("immutable seed changed during restore")
            if "checkpoint_tail_reread_sha256" in operation:
                raise ParkingError("disk operation must not claim a pre-restore checkpoint-tail digest")
            restore_seen.append(conv)
        if progress_count < 1: raise ParkingError("disk operation omitted SESSION progress evidence")
        parsed.append({"action": action, "conversation": conv, "tokens": result["tokens"],
                       "bytes": result["bytes"], "milliseconds": result["milliseconds"],
                       "session_progress_lines": progress_count, "wait_phases": waits})
    if [x["conversation"] for x in parsed if x["action"] == "save"] != ["A", "B"] or restore_seen != expected_restores:
        raise ParkingError("disk operation list does not preserve the two independent seed histories")
    return {"tier": "disk-parked-cold", "status": "pass", "total_seed_bytes": total,
            "save_restore_operations": parsed, "snapshot_integrity": "full-file-sha256-and-per-restore-recheck",
            "continuation_quality": "actual-token-id-and-output-parity-validated-separately",
            "os_file_cache_state": "uncontrolled"}


def disk_space_gate(free_bytes: Any, contract: dict[str, Any], additional_bytes: int = 0) -> dict[str, Any]:
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ParkingError("additional disk bytes must be a nonnegative exact integer")
    floor = contract["resource_budget"]["initial_disk_free_required_bytes"]
    ok = type(free_bytes) is int and free_bytes >= floor + additional_bytes
    return {"status": "pass" if ok else "fail_closed", "free_bytes": free_bytes,
            "required_free_bytes": floor + additional_bytes}


def sample_snapshot_disk(snapshot_dir: Path, contract: dict[str, Any], additional_bytes: int = 0) -> dict[str, Any]:
    if not snapshot_dir.is_absolute(): raise ParkingError("snapshot directory must be an absolute path")
    usage = shutil.disk_usage(snapshot_dir if snapshot_dir.exists() else snapshot_dir.parent)
    return disk_space_gate(usage.free, contract, additional_bytes)


def resource_gate(sample: dict[str, Any], errors: list[Any] | None = None,
                  physical_floor: int = RAM_FLOOR, commit_floor: int = COMMIT_FLOOR,
                  additional_physical_bytes: int = 0) -> dict[str, Any]:
    errors = errors or []
    physical = sample.get("physical_available_bytes") if isinstance(sample, dict) else None
    commit = sample.get("commit_available_bytes") if isinstance(sample, dict) else None
    sources = sample.get("sources", {}) if isinstance(sample, dict) else {}
    source = sample.get("source", "") if isinstance(sample, dict) else ""
    psapi = "PSAPI GetPerformanceInfo" in str(source) or sources.get("system_commit") == "PSAPI GetPerformanceInfo"
    ok = (not errors and type(physical) is int and type(commit) is int and psapi and
          physical >= physical_floor + additional_physical_bytes and commit >= commit_floor)
    return {"status": "pass" if ok else "fail_closed", "physical_available_bytes": physical,
            "commit_available_bytes": commit, "physical_floor_bytes": physical_floor,
            "additional_physical_bytes": additional_physical_bytes, "commit_floor_bytes": commit_floor,
            "PSAPI_commit_source_verified": psapi, "errors": errors}


def sample_windows_resources() -> tuple[dict[str, Any], list[Any]]:
    # Intentionally deferred: default import/plan mode does not load host telemetry or device tooling.
    from tools.hetero_resources import memory_snapshot
    return memory_snapshot()


def validate_owner_snapshot(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("role", "pid", "create_time_utc", "executable", "parent_pid", "argv")
    if not isinstance(expected, list) or not expected or not isinstance(observed, list):
        raise ParkingError("owned process tree identity is unavailable")
    exp_by_pid = {}
    for row in expected:
        if not isinstance(row, dict) or any(key not in row for key in required): raise ParkingError("incomplete expected process identity")
        if type(row["pid"]) is not int or row["pid"] <= 0 or type(row["parent_pid"]) is not int:
            raise ParkingError("PID/parent PID must be exact integers")
        if not isinstance(row["create_time_utc"], str) or not row["create_time_utc"].endswith("Z"):
            raise ParkingError("process CreateTime must be full UTC text")
        if not isinstance(row["executable"], str) or not Path(row["executable"]).is_absolute():
            raise ParkingError("process executable path must be absolute")
        if not isinstance(row["argv"], list) or not row["argv"] or any(not isinstance(x, str) for x in row["argv"]):
            raise ParkingError("process argv must preserve the exact ordered argument list")
        if row["pid"] in exp_by_pid: raise ParkingError("duplicate PID in expected owner tree")
        exp_by_pid[row["pid"]] = row
    obs_by_pid = {}
    for row in observed:
        if not isinstance(row, dict) or type(row.get("pid")) is not int: raise ParkingError("observed process identity lacks exact PID")
        if row["pid"] in obs_by_pid: raise ParkingError("duplicate PID in observed owner tree")
        obs_by_pid[row["pid"]] = row
    if set(exp_by_pid) != set(obs_by_pid): raise ParkingError("observed process tree contains missing or unowned PIDs")
    for pid, want in exp_by_pid.items():
        got = obs_by_pid[pid]
        for field in required:
            if field == "executable":
                equal = os.path.normcase(os.path.normpath(got.get(field, ""))) == os.path.normcase(os.path.normpath(want[field]))
            else: equal = got.get(field) == want[field]
            if not equal: raise ParkingError(f"owned PID identity changed: pid={pid}, field={field}")
    roots = [row for row in expected if row["parent_pid"] not in exp_by_pid]
    if len(roots) != 1: raise ParkingError("owner tree must have one exact launcher root")
    return sorted(expected, key=lambda row: _owner_depth(row, exp_by_pid), reverse=True)


def _owner_depth(row: dict[str, Any], by_pid: dict[int, dict[str, Any]]) -> int:
    depth = 0; parent = row["parent_pid"]; seen = {row["pid"]}
    while parent in by_pid:
        if parent in seen: raise ParkingError("cycle in owned process tree")
        seen.add(parent); depth += 1; parent = by_pid[parent]["parent_pid"]
    return depth


def cleanup_exact_owner_tree(expected: list[dict[str, Any]],
                             observe: Callable[[], list[dict[str, Any]]],
                             stop_exact: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    """Stop only exact previously-owned identities, child first; re-snapshot before every stop."""
    validate_owner_snapshot(expected, expected)
    by_pid = {row["pid"]: row for row in expected}
    order = sorted(expected, key=lambda row: _owner_depth(row, by_pid), reverse=True)
    actions = []
    initial = observe()
    if any(row.get("pid") not in by_pid for row in initial):
        raise ParkingError("cleanup snapshot includes a process outside the exact owned tree")
    for identity in order:
        current = observe()
        current_by_pid = {row.get("pid"): row for row in current}
        if any(pid not in by_pid for pid in current_by_pid):
            raise ParkingError("cleanup snapshot grew to include an unowned process")
        candidate = current_by_pid.get(identity["pid"])
        if candidate is None:
            actions.append({"pid": identity["pid"], "action": "already_exited"}); continue
        validate_owner_snapshot([identity], [candidate])
        result = stop_exact(identity)
        actions.append({"pid": identity["pid"], "action": "stopped_exact_identity", "result": result})
    remaining = observe()
    if remaining: raise ParkingError("owned process tree or unknown child remains after exact cleanup")
    return actions


class ExecutionAdapter(Protocol):
    """Live-adapter interface; the pinned private-stdio implementation is in tools.hetero_kv_stdio."""
    api_version: int
    def execute_plan(self, plan: dict[str, Any], requests: dict[str, Any],
                     checkpoint: Callable[[str, list[dict[str, Any]], int], dict[str, Any]]) -> dict[str, Any]: ...
    def owned_process_trees(self) -> list[list[dict[str, Any]]]: ...
    def owner_snapshot(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
    def stop_exact(self, identity: dict[str, Any]) -> Any: ...


def checkpoint(adapter: ExecutionAdapter, expected_owners: list[dict[str, Any]],
               contract: dict[str, Any], label: str, additional_physical_bytes: int = 0,
               snapshot_dir: Path | None = None, additional_disk_bytes: int = 0) -> dict[str, Any]:
    sample, errors = sample_windows_resources()
    gate = resource_gate(sample, errors, additional_physical_bytes=additional_physical_bytes)
    owner_order = validate_owner_snapshot(expected_owners, adapter.owner_snapshot(expected_owners))
    disk_gate = sample_snapshot_disk(snapshot_dir, contract, additional_disk_bytes) if snapshot_dir else None
    if gate["status"] != "pass" or (disk_gate is not None and disk_gate["status"] != "pass"):
        # Fail closed: no new adapter operation is allowed after a failed/unknown floor.
        cleanup_exact_owner_tree(expected_owners, lambda: adapter.owner_snapshot(expected_owners), adapter.stop_exact)
        raise ParkingError(f"resource/disk floor failed at {label}; exact owned tree was stopped")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return {"label": label, "resource": gate, "owner_tree": owner_order,
            "disk": disk_gate,
            "observed_at_utc": timestamp}


def prelaunch_checkpoint(contract: dict[str, Any], additional_physical_bytes: int = 0,
                         snapshot_dir: Path | None = None, additional_disk_bytes: int = 0) -> dict[str, Any]:
    """Fresh 12/4 gate before an adapter starts a new owned process tree."""
    sample, errors = sample_windows_resources()
    gate = resource_gate(sample, errors, additional_physical_bytes=additional_physical_bytes)
    disk_gate = sample_snapshot_disk(snapshot_dir, contract, additional_disk_bytes) if snapshot_dir else None
    if gate["status"] != "pass" or (disk_gate is not None and disk_gate["status"] != "pass"):
        raise ParkingError("prelaunch resource/disk floor failed or is unknown")
    return {"label": "prelaunch", "resource": gate, "disk": disk_gate,
            "owner_tree": [], "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")}


def validate_checkpoint_history(samples: list[dict[str, Any]], max_gap_seconds: float = 1.5) -> dict[str, Any]:
    if not isinstance(samples, list) or not samples:
        raise ParkingError("execution adapter did not provide periodic resource/owner checkpoints")
    parsed = []
    for row in samples:
        if not isinstance(row, dict) or row.get("resource", {}).get("status") != "pass":
            raise ParkingError("an execution resource checkpoint failed or is unknown")
        owners = row.get("owner_tree")
        if not isinstance(owners, list) or (not owners and row.get("label") != "prelaunch"):
            raise ParkingError("resource checkpoint lacks the exact owned process tree")
        disk = row.get("disk")
        if disk is not None and disk.get("status") != "pass":
            raise ParkingError("disk-space checkpoint failed or is unknown")
        text = row.get("observed_at_utc")
        if not isinstance(text, str) or not text.endswith("Z"):
            raise ParkingError("resource checkpoint timestamp must be UTC")
        try: parsed.append(dt.datetime.fromisoformat(text[:-1] + "+00:00"))
        except ValueError: raise ParkingError("invalid resource checkpoint timestamp") from None
    gaps = [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]
    if any(gap <= 0 or gap > max_gap_seconds for gap in gaps):
        raise ParkingError("resource/owner sampling cadence exceeded its bounded interval")
    return {"sample_count": len(samples), "max_observed_gap_seconds": max(gaps, default=0.0),
            "cadence_gate_seconds": max_gap_seconds}


def execute_authorized_adapter(adapter: ExecutionAdapter, plan: dict[str, Any],
                               request_manifest: dict[str, Any], request_bindings: dict[str, Any],
                               contract: dict[str, Any], contract_sha256: str,
                               request_manifest_sha256: str) -> dict[str, Any]:
    """Run a separately reviewed adapter and validate its typed A/B, RAM, and disk evidence."""
    if contract.get("schema_version") != 2 or contract.get("execution_contract_version") != 2:
        raise ParkingError("live KV execution is forbidden for v1 contracts; use the root-reviewed v2 schema")

    def on_checkpoint(label: str, owners: list[dict[str, Any]], additional_bytes: int = 0,
                      additional_disk_bytes: int = 0):
        disk_phase = any(word in label.casefold() for word in ("disk", "save", "restore", "snapshot"))
        snapshot_dir = Path(contract["snapshot_paths"]["directory"]) if disk_phase else None
        if not owners:
            return prelaunch_checkpoint(contract, additional_bytes, snapshot_dir, additional_disk_bytes)
        return checkpoint(adapter, owners, contract, label, additional_bytes,
                          snapshot_dir, additional_disk_bytes)

    bundle = None
    failure = None
    try:
        bundle = adapter.execute_plan(plan, request_manifest, on_checkpoint)
    except BaseException as exc:
        failure = exc
    finally:
        # A phase that returns with an owned tree still alive is closed using exact identities only.
        try:
            trees = adapter.owned_process_trees()
            for owners in reversed(trees):
                if owners:
                    cleanup_exact_owner_tree(owners, lambda e=owners: adapter.owner_snapshot(e), adapter.stop_exact)
                    if adapter.owner_snapshot(owners): raise ParkingError("adapter-owned process tree remains after cleanup")
        except BaseException as cleanup_exc:
            if failure is not None:
                raise ParkingError(f"adapter failed ({type(failure).__name__}); exact cleanup failed ({type(cleanup_exc).__name__})") from failure
            raise
    if failure is not None: raise ParkingError(f"execution adapter failed: {type(failure).__name__}") from failure
    if not isinstance(bundle, dict): raise ParkingError("execution adapter returned no typed result bundle")
    elapsed = bundle.get("arm_elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed > contract["resource_budget"]["total_arm_deadline_seconds"]:
        raise ParkingError("execution arm exceeded or did not prove the total bounded deadline")
    refs_doc = bundle.get("references")
    refs = validate_reference_manifest(refs_doc, request_bindings, contract, contract_sha256,
                                       request_manifest_sha256)
    ram = bundle.get("ram_pair", {})
    disk = bundle.get("disk_pair", {})
    ram_quality = validate_arm_completions(ram.get("completions"), request_bindings, refs, contract)
    disk_quality = validate_arm_completions(disk.get("completions"), request_bindings, refs, contract)
    ram_evidence = validate_ram_evidence(ram.get("evidence"), contract)
    disk_evidence = validate_disk_evidence(disk.get("evidence"), contract,
        disk.get("expected_seed_tokens", {}))
    checks = validate_checkpoint_history(bundle.get("resource_checkpoints", []))
    return {"schema_version": 1, "status": "validated", "run_id": contract["run_id"],
            "reference_A_B_independent": True, "ram_quality": ram_quality,
            "disk_quality": disk_quality, "ram_evidence": ram_evidence,
            "disk_evidence": disk_evidence, "resource_owner_sampling": checks,
            "all_storage_tiers_complete": False,
            "claim_limit": "validated only the contract's hot-RAM and parked-disk arms; prefetch/cold-storage performance remain separate"}


def verify_execution_authorization(contract: dict[str, Any], contract_sha256: str,
                                   authorization: dict[str, Any], adapter_path: Path,
                                   request_manifest_sha256: str | None = None) -> str:
    if contract.get("schema_version") != 2 or contract.get("execution_contract_version") != 2:
        raise ParkingError("live KV execution is forbidden for v1 contracts")
    if contract.get("execution_authorized") is not True or contract.get("runner_execution_permitted") is not True:
        raise ParkingError("execution is disabled by the prepared contract")
    if contract.get("runtime_binary_binding") is None or contract.get("runner_implemented") is not True:
        raise ParkingError("execution requires a separately reviewed binary and adapter binding")
    adapter_sha = sha256_bytes(adapter_path.read_bytes())
    runner = contract.get("runner_source_binding")
    if (not isinstance(runner, dict) or Path(runner.get("path", "")).resolve() != adapter_path.resolve() or
            runner.get("sha256") != adapter_sha):
        raise ParkingError("contract runner-source binding differs from the selected adapter")
    if (authorization.get("status") != "approved" or authorization.get("run_id") != contract.get("run_id") or
            authorization.get("contract_sha256") != contract_sha256 or
            authorization.get("adapter_sha256") != adapter_sha or authorization.get("adapter_reviewed") is not True):
        raise ParkingError("root authorization does not bind this exact contract and reviewed adapter")
    if (not isinstance(request_manifest_sha256, str) or
            authorization.get("request_manifest_sha256") != request_manifest_sha256):
        raise ParkingError("root authorization does not bind the exact v2 pretokenized request manifest")
    return adapter_sha


def import_reviewed_adapter(path: Path, expected_sha256: str):
    # Called only after verify_execution_authorization succeeds.
    if sha256_bytes(path.read_bytes()) != expected_sha256: raise ParkingError("adapter changed after authorization")
    spec = importlib.util.spec_from_file_location("reviewed_kv_execution_adapter", path)
    if spec is None or spec.loader is None: raise ParkingError("cannot load reviewed execution adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_adapter", None)
    if not callable(factory): raise ParkingError("reviewed adapter must provide create_adapter()")
    adapter = factory()
    if getattr(adapter, "api_version", None) != 1: raise ParkingError("execution adapter API version mismatch")
    return adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args(argv)
    contract, contract_sha, configs = load_contract(args.contract)
    request_sha = reference_sha = None
    requests = refs = None
    if args.requests:
        req, request_sha = strict_json_file(args.requests)
        requests = validate_request_manifest(req, contract, contract_sha)
    if args.references:
        if requests is None: parser.error("--references requires --requests")
        ref, reference_sha = strict_json_file(args.references)
        refs = validate_reference_manifest(ref, requests, contract, contract_sha, request_sha)
    plan = build_plan(contract, contract_sha, configs, request_sha, reference_sha, args.contract)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False)); return 0
    if not args.adapter or not args.authorization:
        parser.error("execution requires a root-bound --adapter and --authorization receipt")
    if requests is None:
        parser.error("execution requires an exact pretokenized --requests manifest")
    authorization, _ = strict_json_file(args.authorization)
    adapter_sha = verify_execution_authorization(contract, contract_sha, authorization, args.adapter, request_sha)
    adapter = import_reviewed_adapter(args.adapter, adapter_sha)
    result = execute_authorized_adapter(adapter, plan, strict_json_loads(args.requests.read_bytes()),
                                        requests, contract, contract_sha, request_sha)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"KV parking harness: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
