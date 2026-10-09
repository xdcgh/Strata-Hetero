#!/usr/bin/env python3
"""Offline A/B comparison of saved Strata quality artifacts; never calls model endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import hetero_quality as quality

MAX_METADATA_BYTES = 8 << 20
MAX_TEXT_BYTES = 2 << 20
MAX_SSE_BYTES = 16 << 20
MAX_SMALL_IDENTITY_BYTES = 128 << 20


class EvidenceError(ValueError):
    pass


LEGACY_WRITER_COMMIT = "1cd62b0089ab5777c318d9e28e7fefbe92410d83"
LEGACY_WRITER_BLOB = "bc26b6996a3f96cc5e0bb284ddbbaf307509f780"
LEGACY_WRITER_SOURCE_SHA256 = "130a32c2fc1c458e3aba0dfd6754d6ef22c96dd7f7931c377c1bbd0e8110213f"
LEGACY_WRITER_API = 'Path.write_text(content, encoding="utf-8") with newline=None'
LEGACY_TEXT_NEWLINE_CONVERSION = "windows_textio_newline_translation"
LEGACY_MODEL_HASH_SCOPE = "logical UTF8 before legacy text writer newline conversion"
NEW_TEXT_ENCODING = "utf-8 bytes without newline translation"
NEW_CONTENT_HASH_SCOPE = "exact stored model-output bytes"


def _json_load(path: Path, limit: int = MAX_METADATA_BYTES) -> Any:
    if path.is_symlink():
        raise EvidenceError(f"symlink rejected: {path.name}")
    size = path.stat().st_size
    if size > limit:
        raise EvidenceError(f"JSON exceeds size limit: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, max_bytes: int = MAX_SMALL_IDENTITY_BYTES) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"not a regular file: {path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise EvidenceError(f"file exceeds hash limit: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest(), size


def resolve_run(spec: str, base: Path, manifest_date: str | None = None) -> Path:
    candidate = Path(spec)
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    if spec.isdigit():
        run_number = int(spec)
        matches = [p for p in base.iterdir() if p.is_dir() and
                   re.match(rf"^\d{{8}}-0*{run_number}(?:-|$)", p.name)]
        if len(matches) != 1:
            raise EvidenceError(f"run selector {spec!r} matched {len(matches)} run-number directories")
        return matches[0].resolve()
    matches = [p for p in base.iterdir() if p.is_dir() and spec.casefold() in p.name.casefold()]
    if len(matches) != 1:
        raise EvidenceError(f"run selector {spec!r} matched {len(matches)} directories")
    return matches[0].resolve()


def _arg_map(args: Any) -> dict[str, Any]:
    if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
        raise EvidenceError("config args is not a string list")
    result: dict[str, Any] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if not key.startswith("--"):
            raise EvidenceError("config args contains a non-option token")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            value: Any = args[i + 1]
            i += 2
        else:
            value = True
            i += 1
        if key == "--native-dense-gguf":
            if value is True:
                raise EvidenceError("--native-dense-gguf requires a path")
            result.setdefault(key, []).append(value)
        else:
            if key in result:
                raise EvidenceError(f"repeated non-list option in config args: {key}")
            result[key] = value
    return result


def _normal_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return os.path.normcase(os.path.normpath(value))


def _public_path(value: Any) -> Any:
    if not isinstance(value, str): return value
    home = os.environ.get("USERPROFILE") or str(Path.home())
    if home:
        try:
            if os.path.normcase(os.path.normpath(value)).startswith(os.path.normcase(os.path.normpath(home)) + os.sep):
                return "%USERPROFILE%" + value[len(home):].replace("/", "\\")
        except Exception:
            pass
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_file_tier_log_line(line: str) -> str:
    """Parse only the mode token after the file-tier prefix; ignore override explanations."""
    match = re.search(r"the file tier reads\s+(unbuffered|through the file cache)(?=\s|\()", line, re.IGNORECASE)
    if not match: return "unknown"
    return "unbuffered" if match.group(1).casefold() == "unbuffered" else "buffered"


def verify_tokenizer_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        return {"status": "missing", "errors": ["manifest tokenizer object missing"], "identity_sha256": None}
    errors: list[str] = []
    records = []
    implementation = tokenizer.get("implementation_source")
    if not isinstance(implementation, dict):
        errors.append("tokenizer implementation hash missing")
    else:
        records.append(("implementation_source", implementation))
    assets = tokenizer.get("assets")
    if not isinstance(assets, list) or not assets:
        errors.append("tokenizer asset hashes missing")
    else:
        records.extend((f"asset_{i}", item) for i, item in enumerate(assets))
    observed = []
    for label, record in records:
        path_text, expected_hash, expected_size = record.get("path"), record.get("sha256"), record.get("bytes")
        if not isinstance(path_text, str) or not isinstance(expected_hash, str) or type(expected_size) is not int:
            errors.append(f"{label} identity malformed")
            continue
        try:
            digest, size = sha256_file(Path(path_text))
        except Exception as exc:
            errors.append(f"{label} unreadable: {type(exc).__name__}")
            continue
        match = digest == expected_hash and size == expected_size
        observed.append({"label": label, "sha256": digest, "bytes": size, "matches_manifest": match})
        if not match:
            errors.append(f"{label} hash/size mismatch")
    return {"status": "pass" if not errors and len(observed) == len(records) else "failed",
            "identity_sha256": _canonical_hash(tokenizer), "artifacts": observed, "errors": errors}


def _verified_file_hash(path_text: Any, recorded_hash: Any, label: str,
                        max_bytes: int = MAX_SMALL_IDENTITY_BYTES) -> dict[str, Any]:
    if not isinstance(path_text, str) or not isinstance(recorded_hash, str):
        return {"status": "unknown", "error": f"{label} path/hash missing"}
    try:
        digest, size = sha256_file(Path(path_text), max_bytes=max_bytes)
    except Exception as exc:
        return {"status": "unknown", "error": f"{label} hash failed: {type(exc).__name__}"}
    return {"status": "pass" if digest.casefold() == recorded_hash.casefold() else "failed",
            "path": path_text, "sha256": digest, "recorded_sha256": recorded_hash,
            "bytes": size, "matches_record": digest.casefold() == recorded_hash.casefold()}


def _provenance_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} path missing")
    relative = Path(value)
    win_relative = PureWindowsPath(value)
    if (relative.is_absolute() or relative.drive or win_relative.is_absolute() or win_relative.drive or
            ".." in relative.parts or ".." in win_relative.parts):
        raise EvidenceError(f"{label} path must be a relative path within the run")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise EvidenceError(f"{label} symlink rejected")
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise EvidenceError(f"{label} path escapes the run or is not a regular file")
    return resolved


def _checked_receipt(root: Path, name: Any, recorded_hash: Any, label: str) -> tuple[Path, str, dict[str, Any]]:
    path = _provenance_child(root, name, label)
    digest, _ = sha256_file(path)
    if not isinstance(recorded_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", recorded_hash):
        raise EvidenceError(f"{label} recorded SHA-256 missing/invalid")
    if digest.casefold() != recorded_hash.casefold():
        raise EvidenceError(f"{label} SHA-256 does not match provenance")
    value = _json_load(path)
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} JSON must be an object")
    return path, digest, value


def _verify_admission_evidence(root: Path) -> dict[str, Any]:
    provenance_path = root / "provenance.json"
    if provenance_path.is_symlink():
        return {"status": "unknown", "errors": ["provenance symlink rejected"], "mode": "provenance"}
    provenance = None
    provenance_digest = None
    if provenance_path.is_file():
        try:
            provenance = _json_load(provenance_path)
            provenance_digest, _ = sha256_file(provenance_path)
        except Exception as exc:
            return {"status": "unknown", "errors": [f"provenance unreadable: {type(exc).__name__}"],
                    "mode": "provenance"}
        if not isinstance(provenance, dict):
            return {"status": "unknown", "errors": ["provenance JSON must be an object"], "mode": "provenance"}
    has_launch_field = isinstance(provenance, dict) and "launch_admission" in provenance
    launch = provenance.get("launch_admission") if has_launch_field else None
    if not has_launch_field:
        # Legacy runs without an explicit checked launch receipt continue to use admission.json.
        try:
            admission_path = _provenance_child(root, "admission.json", "legacy admission")
            digest, _ = sha256_file(admission_path)
            admission = _json_load(admission_path)
            ok = admission.get("pass") is True and admission.get("status") == "pass"
            return {"status": "pass" if ok else "unknown", "mode": "legacy_admission",
                    "provenance_path": str(provenance_path) if provenance_path.is_file() else None,
                    "provenance_sha256": provenance_digest,
                    "selected_receipt": "admission.json", "selected_receipt_path": str(admission_path),
                    "selected_receipt_sha256": digest, "sha256_scope": "exact stored receipt file bytes",
                    "admission_status": admission.get("status"), "admission_pass": admission.get("pass"),
                    "errors": [] if ok else ["legacy admission receipt does not prove pass"]}
        except Exception as exc:
            return {"status": "unknown", "mode": "legacy_admission",
                    "errors": [f"legacy admission proof missing/invalid: {type(exc).__name__}: {exc}"]}
    if not isinstance(launch, dict):
        return {"status": "unknown", "mode": "launch_admission_provenance",
                "errors": ["launch_admission provenance must be an object"]}
    if launch.get("status") != "pass":
        return {"status": "unknown", "mode": "launch_admission_provenance",
                "errors": ["launch_admission provenance status is not pass"]}
    try:
        admission_path, admission_hash, admission = _checked_receipt(
            root, launch.get("receipt"), launch.get("receipt_sha256"), "launch admission")
        process_path, process_hash, process = _checked_receipt(
            root, launch.get("launch_process_receipt"), launch.get("process_receipt_sha256"),
            "launch process receipt")
        admission_ok = admission.get("pass") is True and admission.get("status") == "pass"
        process_pid = process.get("launcher_pid")
        process_run = process.get("run")
        process_ok = (type(process_pid) is int and process_pid > 0 and
                      isinstance(process_run, str) and Path(process_run).resolve() == root.resolve() and
                      isinstance(process.get("config"), str) and
                      Path(process["config"]).resolve() == (root / "config.json").resolve())
        if not admission_ok or not process_ok:
            raise EvidenceError("selected admission or launch process receipt does not prove a valid pass/launch")
        return {"status": "pass", "mode": "launch_admission_provenance",
                "provenance_path": str(provenance_path), "provenance_sha256": provenance_digest,
                "selected_receipt": launch["receipt"], "selected_receipt_path": str(admission_path),
                "selected_receipt_sha256": admission_hash, "recorded_receipt_sha256": launch["receipt_sha256"],
                "launch_process_receipt": launch["launch_process_receipt"],
                "launch_process_receipt_path": str(process_path), "launch_process_receipt_sha256": process_hash,
                "recorded_process_receipt_sha256": launch["process_receipt_sha256"],
                "sha256_scope": "each digest covers exact stored JSON file bytes",
                "admission_status": admission.get("status"), "admission_pass": admission.get("pass"),
                "process_receipt_structurally_valid": process_ok, "launcher_pid": process_pid,
                "errors": []}
    except Exception as exc:
        return {"status": "unknown", "mode": "launch_admission_provenance",
                "errors": [f"launch admission provenance proof invalid: {type(exc).__name__}: {exc}"]}


def _load_run_controls(root: Path) -> dict[str, Any]:
    try:
        config = _json_load(root / "config.json")
        identity = _json_load(root / "identity.json")
    except Exception as exc:
        return {"root": str(root), "status": "unknown", "errors": [f"control files unavailable: {type(exc).__name__}"]}
    args = _arg_map(config.get("args"))
    env = config.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) for k in env):
        env = {}
    engine = identity.get("engine", {}) if isinstance(identity.get("engine"), dict) else {}
    model = identity.get("model", {}) if isinstance(identity.get("model"), dict) else {}
    server = identity.get("server_python", {}) if isinstance(identity.get("server_python"), dict) else {}
    identity_cfg = identity.get("config", {}) if isinstance(identity.get("config"), dict) else {}
    engine_path = engine.get("path")
    config_exe = config.get("exe")
    exe_path_match = _normal_path(engine_path) == _normal_path(config_exe)
    binary_hash = _verified_file_hash(config_exe, engine.get("sha256"), "engine_binary")
    raw_config_hash = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()

    log_path = Path(config.get("log", "")) if isinstance(config.get("log"), str) else None
    file_modes: list[dict[str, Any]] = []
    r4 = False
    resident_ram_lines: list[str] = []
    mtp_load_line = None
    ple_observation: dict[str, Any] = {"configured_args": {k: v for k, v in args.items() if k.startswith("--ple-")},
                                      "observed_placement": "not_logged", "locked_bytes": None}
    if log_path is not None and log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if "the file tier reads" in line:
                        mode = classify_file_tier_log_line(line)
                        file_modes.append({"mode": mode, "evidence": line.strip()[:500]})
                    if "R4 hit path ON" in line:
                        r4 = True
                    if "resident RAM mode:" in line:
                        resident_ram_lines.append(line.strip()[:500])
                    if "mtp: draft layer loaded" in line:
                        mtp_load_line = line.strip()[:500]
                    if "PLE table locked in RAM" in line:
                        match = re.search(r"(\d+) locked table bytes", line)
                        ple_observation = {"configured_args": ple_observation["configured_args"],
                                           "observed_placement": "locked_in_ram",
                                           "locked_bytes": int(match.group(1)) if match else None,
                                           "evidence": line.strip()[:500]}
                    elif "PLE table" in line and ple_observation["observed_placement"] == "not_logged":
                        ple_observation["observed_placement"] = "file_or_other_placement_observed"
                        ple_observation["evidence"] = line.strip()[:500]
        except Exception:
            pass
    effective_mode = file_modes[-1]["mode"] if file_modes else "unknown"
    server_log = root / "logs" / "server.stdout.log"
    ready_seen = False
    if server_log.is_file():
        try:
            with server_log.open("r", encoding="utf-8", errors="replace") as stream:
                ready_seen = any(line.startswith("ready:") for line in stream)
        except Exception:
            pass
    expected_env_override = env.get("STRATA_UNBUFFERED_LOAD")
    env_snapshot = {str(k): v for k, v in env.items()}
    # Never serialize all environment values; retain only names and the one authorized value.
    safe_env = {k: v for k, v in env_snapshot.items() if k == "STRATA_UNBUFFERED_LOAD"}
    core_args = {k: v for k, v in args.items() if not k.startswith("--ple-")}
    strict_env = {str(k): hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                  for k, v in env.items() if k != "STRATA_UNBUFFERED_LOAD"}
    config_controls = {"model_name": config.get("model_name"), "tokenizer_path": config.get("tokenizer"),
                       "parallel": config.get("parallel"), "fit_max_tokens": config.get("fit_max_tokens"),
                       "sampling": config.get("sampling"), "hetero_capture_token_ids": config.get("hetero_capture_token_ids"),
                       "engine_silence_s": config.get("engine_silence_s"), "core_args": core_args,
                       "strict_env_value_sha256": strict_env}
    integrity_path = model.get("integrity_manifest")
    integrity_sha = None
    if isinstance(integrity_path, str):
        p = (root / integrity_path).resolve()
        if p.is_file():
            try: integrity_sha = sha256_file(p)[0]
            except Exception: pass
    profile_path = args.get("--expert-profile")
    profile_hash = None
    if isinstance(profile_path, str):
        try: profile_hash = sha256_file(Path(profile_path))[0]
        except Exception: pass
    core_controls = {"config": config_controls,
                     "model": {"id": model.get("id"), "native_path": model.get("native_path"),
                               "integrity_manifest_sha256": integrity_sha},
                     "engine_version": engine.get("version"),
                     "expert_profile_sha256": profile_hash,
                     "server_python": {"checkout_sha": server.get("checkout_sha"),
                                       "file_sha256": server.get("server_file_sha256"),
                                       "quality_capture_opt_in": server.get("quality_capture_opt_in")},
                     "manifest_tokenizer_identity": None}
    errors = []
    required_args = ("--pack", "--native", "--expert-profile", "--expert-cache", "--prefill", "--spec",
                     "--spec-min-p", "--mtp", "--max-context", "--kv", "--resident-budget-gib",
                     "--vram-reserve-mib", "--pool-workers", "--prompt-cache", "--adapt-swaps", "--pcie-frac")
    missing = [k for k in required_args if k not in args]
    if missing: errors.append("missing engine controls: " + ",".join(missing))
    if not isinstance(model.get("id"), str) or not model.get("id"):
        errors.append("model identity ID missing")
    if not isinstance(model.get("native_path"), str) or not model.get("native_path"):
        errors.append("native model identity path missing")
    if not isinstance(model.get("integrity_manifest"), str) or not model.get("integrity_manifest"):
        errors.append("model integrity manifest reference missing")
    if not isinstance(engine.get("version"), str) or not engine.get("version"):
        errors.append("engine version missing")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(identity.get("source", {}).get("sha", ""))):
        errors.append("engine source identity hash missing/invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(server.get("checkout_sha", ""))):
        errors.append("server Python checkout identity missing/invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(server.get("server_file_sha256", ""))):
        errors.append("server Python file identity missing/invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(identity_cfg.get("sha256", ""))):
        errors.append("config identity hash missing/invalid")
    if str(args.get("--resident-budget-gib")) != "20": errors.append("resident budget is not deterministic 20 GiB")
    if identity_cfg.get("resident_budget_gib") != 20: errors.append("identity does not prove resident budget 20 GiB")
    if identity_cfg.get("deterministic_placement") is not True: errors.append("identity does not prove deterministic placement")
    if identity_cfg.get("capture_actual_engine_token_ids") is not True or config.get("hetero_capture_token_ids") is not True:
        errors.append("actual engine token-ID capture is not enabled in saved identity/config")
    if binary_hash.get("status") != "pass": errors.append("engine binary hash does not match identity")
    if not exe_path_match: errors.append("config executable path differs from identity path")
    if integrity_sha is None: errors.append("model integrity-manifest evidence missing")
    if profile_hash is None: errors.append("expert profile file hash could not be verified")
    if effective_mode == "unknown": errors.append("effective file tier mode is unproven in engine log")
    if not r4: errors.append("R4 resident GPU path is unproven in engine log")
    if not resident_ram_lines: errors.append("resident RAM mode is unproven in engine log")
    resident_match = re.search(r"resident RAM mode:\s*([0-9]+(?:\.[0-9]+)?) GiB of experts", resident_ram_lines[-1]) if resident_ram_lines else None
    resident_actual_gib = float(resident_match.group(1)) if resident_match else None
    if resident_actual_gib is None or abs(resident_actual_gib - 20.0) > 0.02:
        errors.append("run log does not prove the 20 GiB resident complement")
    if mtp_load_line is None: errors.append("MTP draft layer load is unproven in engine log")
    if not ready_seen: errors.append("server ready marker missing")
    admission_evidence = _verify_admission_evidence(root)
    admission_ok = admission_evidence.get("status") == "pass"
    if not admission_ok: errors.append("admission receipt does not prove pass")
    return {"root": str(root), "status": "pass" if not errors else "incomplete",
            "errors": errors, "engine_path": config_exe, "engine_version": engine.get("version"),
            "engine_sha256_recorded": engine.get("sha256"), "engine_sha256_actual": binary_hash.get("sha256"),
            "engine_sha256_status": binary_hash.get("status"), "engine_source_sha": identity.get("source", {}).get("sha"),
            "identity_config_sha256": identity_cfg.get("sha256"), "config_file_sha256": raw_config_hash,
            "normalized_controls": core_controls, "normalized_controls_sha256": _canonical_hash(core_controls),
            "run_specific_config": {"cwd": _public_path(config.get("cwd")), "log": _public_path(config.get("log")),
                                    "host": config.get("host"), "port": config.get("port"),
                                    "open_browser": config.get("open_browser")},
            "model_identity": {"id": model.get("id"), "native_path": model.get("native_path"),
                               "integrity_manifest": integrity_path, "integrity_manifest_sha256": integrity_sha},
            "tokenizer_path": config.get("tokenizer"), "server_python": core_controls["server_python"],
            "strict_env_keys": sorted(k for k in env_snapshot if k != "STRATA_UNBUFFERED_LOAD"),
            "strict_env_value_sha256": strict_env,
            "allowed_env_override": {"STRATA_UNBUFFERED_LOAD": expected_env_override},
            "ple_observation": ple_observation, "effective_file_tier_mode": effective_mode,
            "file_tier_log_evidence": file_modes[-1]["evidence"] if file_modes else None,
            "r4_resident_gpu_path_proven": r4, "resident_ram_log_evidence": resident_ram_lines[-1] if resident_ram_lines else None,
            "resident_ram_actual_gib": resident_actual_gib, "mtp_load_proven": mtp_load_line is not None,
            "mtp_load_log_evidence": mtp_load_line,
            "server_ready_proven": ready_seen, "admission_pass_proven": admission_ok,
            "admission_evidence": admission_evidence,
            "config_sha256_actual": raw_config_hash}


def extract_final_sse(path: Path, max_bytes: int = MAX_SSE_BYTES) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file() or path.is_symlink():
        return {"valid": False, "errors": ["SSE file missing or unsafe"]}
    if path.stat().st_size > max_bytes:
        return {"valid": False, "errors": ["SSE exceeds size limit"]}
    contents: list[str] = []
    diag_records: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    final_records: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    data_index = 0
    done = False
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
            for line in stream:
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    errors.append("invalid SSE JSON data record")
                    data_index += 1
                    continue
                if not isinstance(obj, dict):
                    errors.append("SSE data record is not an object")
                    data_index += 1
                    continue
                choices = obj.get("choices")
                choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
                if isinstance(choices, list) and len(choices) != 1:
                    errors.append("SSE record does not contain exactly one choice")
                if choice:
                    delta = choice.get("delta", {})
                    if isinstance(delta, dict) and "content" in delta:
                        chunk = delta.get("content")
                        if isinstance(chunk, str): contents.append(chunk)
                        elif chunk is not None: errors.append("SSE visible content chunk is not text")
                    if choice.get("finish_reason") is not None:
                        final_records.append((data_index, obj, choice))
                diagnostics = obj.get("strata_diagnostics")
                if diagnostics is not None:
                    if isinstance(diagnostics, dict) and choice is not None:
                        diag_records.append((data_index, obj, choice))
                    else:
                        errors.append("final diagnostics record malformed")
                data_index += 1
    except Exception as exc:
        return {"valid": False, "errors": errors + [f"SSE read failed: {type(exc).__name__}"]}
    if len(diag_records) != 1:
        errors.append(f"expected exactly one final diagnostics record, found {len(diag_records)}")
    if len(final_records) != 1:
        errors.append(f"expected exactly one final choice record, found {len(final_records)}")
    if len(diag_records) == 1 and len(final_records) == 1:
        diag_idx, obj, choice = diag_records[0]
        if final_records[0][0] != diag_idx:
            errors.append("diagnostics record is not attached to the final choice")
        if done is False:
            errors.append("SSE done marker missing")
        if not isinstance(obj.get("usage"), dict):
            errors.append("final SSE usage missing")
        diagnostics = obj["strata_diagnostics"]
        token_ids = diagnostics.get("actual_generated_token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
            errors.append("actual_generated_token_ids missing or not typed nonnegative integers")
            token_ids = None
        source = diagnostics.get("source")
        if not isinstance(source, str) or "engine.generate" not in source:
            errors.append("token-ID diagnostic source does not prove engine.generate")
        if diagnostics.get("include_stop") is not True:
            errors.append("token-ID diagnostics does not include emitted stop IDs")
        finish = choice.get("finish_reason")
        if finish != "stop":
            errors.append(f"final finish reason is not stop: {finish}")
        return {"valid": not errors, "errors": errors, "actual_generated_token_ids": token_ids,
                "actual_generated_token_ids_sha256": _canonical_hash(token_ids) if token_ids is not None else None,
                "token_count": len(token_ids) if token_ids is not None else None,
                "finish_reason": finish, "include_stop": diagnostics.get("include_stop"),
                "source": source, "usage": obj.get("usage"), "visible_text_from_sse": "".join(contents),
                "sse_done_marker": done}
    return {"valid": False, "errors": errors}


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def load_artifact_writer_receipt(run_root: Path) -> dict[str, Any]:
    path = run_root / "artifact_writer_provenance.json"
    if not path.is_file():
        return {"status": "missing", "files": {}, "errors": ["artifact writer provenance receipt missing"]}
    try:
        receipt = _json_load(path)
    except Exception as exc:
        return {"status": "invalid", "files": {}, "errors": [f"writer receipt unreadable: {type(exc).__name__}"]}
    errors = []
    source_ok = (receipt.get("schema_version") == 1 and
                 receipt.get("status") == "verified_each_formal_against_raw_sse_and_legacy_writer" and
                 receipt.get("client_os_name") == "nt" and
                 receipt.get("writer_git_commit") == LEGACY_WRITER_COMMIT and
                 receipt.get("writer_git_blob_sha1") == LEGACY_WRITER_BLOB and
                 receipt.get("writer_source_bytes_sha256") == LEGACY_WRITER_SOURCE_SHA256 and
                 receipt.get("writer_api") == LEGACY_WRITER_API and
                 receipt.get("model_content_hash_scope") == LEGACY_MODEL_HASH_SCOPE)
    if not source_ok: errors.append("receipt does not identify the reviewed legacy Windows text writer")
    rows = receipt.get("files")
    if not isinstance(rows, list):
        rows = []
        errors.append("receipt files list missing")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            errors.append("receipt file row malformed")
            continue
        rel = Path(row["relative_path"]).as_posix()
        if rel in by_path: errors.append(f"duplicate receipt row: {rel}")
        by_path[rel] = row
    return {"status": "pass" if not errors else "invalid", "files": by_path, "errors": errors}


def verify_text_artifact(run_root: Path, text_path: Path, prompt_id: str, repeat: int,
                         metadata_sha256: str | None, model_text: str, raw_artifact: bytes,
                         receipt: dict[str, Any] | None, artifact_writer: dict[str, Any] | None,
                         request_artifact_encoding: Any) -> dict[str, Any]:
    model_bytes = model_text.encode("utf-8")
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    artifact_sha = hashlib.sha256(raw_artifact).hexdigest()
    try:
        artifact_text = raw_artifact.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        artifact_text = None
    if raw_artifact == model_bytes:
        metadata_writer_ok = (isinstance(artifact_writer, dict) and
                              artifact_writer.get("client_os_name") in {"nt", "posix"} and
                              artifact_writer.get("text_encoding") == NEW_TEXT_ENCODING and
                              artifact_writer.get("content_sha256_scope") == NEW_CONTENT_HASH_SCOPE and
                              request_artifact_encoding == NEW_TEXT_ENCODING)
        if metadata_writer_ok:
            return {"status": "verified_metadata_byte_exact", "model_text_sha256": model_sha,
                    "artifact_text_sha256": artifact_sha, "conversion": "none"}
        rel = text_path.relative_to(run_root).as_posix()
        row = (receipt or {}).get("files", {}).get(rel)
        if (receipt or {}).get("status") == "pass" and isinstance(row, dict):
            expected = {"prompt_id": prompt_id, "repeat_index": repeat,
                        "metadata_sha256": metadata_sha256, "model_text_sha256": model_sha,
                        "artifact_sha256": artifact_sha, "conversion": "exact_utf8"}
            mismatches = [key for key, value in expected.items() if row.get(key) != value]
            if not mismatches:
                return {"status": "verified_legacy_receipt_exact_utf8", "model_text_sha256": model_sha,
                        "artifact_text_sha256": artifact_sha, "conversion": "exact_utf8",
                        "receipt_source": {"git_commit": LEGACY_WRITER_COMMIT,
                                           "git_blob_sha1": LEGACY_WRITER_BLOB}}
        return {"status": "unproven", "model_text_sha256": model_sha,
                "artifact_text_sha256": artifact_sha, "conversion": None,
                "errors": ["byte-exact artifact lacks a valid new writer flag or matching reviewed receipt"]}

    expected_windows_bytes = model_bytes.replace(b"\n", b"\r\n")
    rel = text_path.relative_to(run_root).as_posix()
    row = (receipt or {}).get("files", {}).get(rel)
    if (receipt or {}).get("status") == "pass" and isinstance(row, dict):
        expected = {"prompt_id": prompt_id, "repeat_index": repeat,
                    "metadata_sha256": metadata_sha256, "model_text_sha256": model_sha,
                    "artifact_sha256": artifact_sha,
                    "conversion": "windows_textio_lf_to_crlf"}
        mismatches = [key for key, value in expected.items() if row.get(key) != value]
        if not mismatches and raw_artifact == expected_windows_bytes:
            return {"status": "verified_legacy_conversion", "model_text_sha256": model_sha,
                    "artifact_text_sha256": artifact_sha,
                    "conversion": "windows_textio_lf_to_crlf",
                    "receipt_source": {"git_commit": LEGACY_WRITER_COMMIT,
                                       "git_blob_sha1": LEGACY_WRITER_BLOB,
                                       "source_bytes_sha256": LEGACY_WRITER_SOURCE_SHA256}}
        return {"status": "unproven", "model_text_sha256": model_sha,
                "artifact_text_sha256": artifact_sha,
                "conversion": None, "errors": ["receipt row mismatch: " + ",".join(mismatches or ["artifact bytes do not match Windows newline conversion"])]}
    return {"status": "unproven", "model_text_sha256": model_sha,
            "artifact_text_sha256": artifact_sha, "conversion": None,
            "errors": ["raw artifact differs from SSE model text and no matching reviewed writer receipt exists"]}


def evaluate_formal(run_root: Path, prompt: dict[str, Any], request: dict[str, Any], index: int,
                    metadata: dict[str, Any], writer_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    req_dir = Path(metadata["_request_dir"])
    stem = f"formal-{index:04d}"
    sse_path, text_path = req_dir / f"{stem}.sse.txt", req_dir / f"{stem}.text.txt"
    errors: list[str] = []
    if request.get("status") != "completed": errors.append(f"request status is {request.get('status')!r}")
    if request.get("error") is not None: errors.append("requests.json contains an error")
    if not sse_path.is_file() or not text_path.is_file():
        errors.append("formal SSE or text artifact missing")
        return {"repeat": index, "status": "incomplete", "errors": errors,
                "text_sha256": None, "token_ids_sha256": None, "token_ids": None, "quality": None}
    if text_path.stat().st_size > MAX_TEXT_BYTES:
        errors.append("formal text exceeds size limit")
        return {"repeat": index, "status": "incomplete", "errors": errors,
                "text_sha256": None, "token_ids_sha256": None, "token_ids": None, "quality": None}
    raw_text = text_path.read_bytes()
    try:
        artifact_text = raw_text.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        artifact_text = None
        errors.append("formal artifact is not valid UTF-8")
    sse = extract_final_sse(sse_path)
    errors.extend(sse.get("errors", []))
    model_text = sse.get("visible_text_from_sse")
    if not isinstance(model_text, str):
        model_text = ""
        errors.append("SSE visible model text missing")
    metadata_hash = metadata.get("_metadata_raw_sha256")
    artifact_proof = verify_text_artifact(run_root, text_path, str(prompt.get("id")), index,
                                          metadata_hash, model_text, raw_text, writer_receipt,
                                          metadata.get("artifact_writer"), request.get("output_artifact_encoding"))
    if artifact_text is None:
        errors.append("formal artifact text decode failed")
    if artifact_proof.get("status") == "unproven":
        errors.extend(artifact_proof.get("errors", ["formal artifact provenance unproven"]))
    req_usage = request.get("usage")
    sse_usage = sse.get("usage")
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(req_usage, dict) or not isinstance(sse_usage, dict):
        errors.append("usage missing in requests.json or final SSE")
    else:
        for key in usage_fields:
            if not _nonnegative_int(req_usage.get(key)) or not _nonnegative_int(sse_usage.get(key)):
                errors.append(f"usage {key} is missing or not a nonnegative integer")
            elif req_usage[key] != sse_usage[key]:
                errors.append(f"requests/SSE usage mismatch for {key}")
        if all(_nonnegative_int(req_usage.get(k)) for k in usage_fields):
            if req_usage["total_tokens"] != req_usage["prompt_tokens"] + req_usage["completion_tokens"]:
                errors.append("request usage total does not equal prompt+completion")
    if request.get("finish_reason") != sse.get("finish_reason"):
        errors.append("requests/SSE finish-reason mismatch")
    if sse.get("finish_reason") != "stop" or sse.get("include_stop") is not True:
        errors.append("stop label is not comparable")
    if req_usage and _nonnegative_int(req_usage.get("completion_tokens")) and sse.get("token_count") is not None:
        if req_usage["completion_tokens"] != sse["token_count"]:
            errors.append("actual engine token count differs from completion usage")
    content_hash = request.get("content_sha256")
    if not isinstance(content_hash, str) or content_hash.casefold() != artifact_proof.get("model_text_sha256", "").casefold():
        errors.append("requests content hash differs from SSE model-text bytes")

    finish = sse.get("finish_reason")
    run_status = str(request.get("status", "missing"))
    if finish in {"length", "max_tokens", "truncated"}:
        run_status = "truncated_by_length"
        errors.append("formal output was length-truncated")
    max_tokens = metadata.get("max_tokens")
    if finish != "stop" and type(max_tokens) is int and req_usage and req_usage.get("completion_tokens") == max_tokens:
        run_status = "truncated_by_length"
        errors.append("formal output reached configured token cap without stop")
    quality_result = quality.evaluate_text(prompt, model_text, run_status=run_status)
    quality_result.pop("detail", None)  # do not duplicate response fragments in the comparison receipt
    ids = sse.get("actual_generated_token_ids")
    quality_result["actual_emitted_token_ids"] = None  # raw IDs remain in the source SSE; receipt stores hash/count
    quality_result["actual_emitted_token_ids_sha256"] = sse.get("actual_generated_token_ids_sha256")
    quality_result["model_text_from_sse_sha256"] = artifact_proof.get("model_text_sha256")
    quality_result["artifact_text_sha256"] = artifact_proof.get("artifact_text_sha256")
    accepted = not errors and quality_result.get("quality_status") == "pass"
    return {"repeat": index, "status": "pass" if accepted else "incomplete" if errors else "quality_failed",
            "errors": errors, "text_sha256": artifact_proof.get("artifact_text_sha256"),
            "artifact_text_sha256": artifact_proof.get("artifact_text_sha256"),
            "model_text_sha256": artifact_proof.get("model_text_sha256"),
            "artifact_conversion": {k:v for k,v in artifact_proof.items() if k not in ("model_text_sha256", "artifact_text_sha256")},
            "token_ids_sha256": sse.get("actual_generated_token_ids_sha256"),
            "token_ids": ids, "token_count": sse.get("token_count"),
            "finish_reason": finish, "include_stop": sse.get("include_stop"),
            "usage": {key: req_usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                     if isinstance(req_usage, dict) else None,
            "request": {key: request.get(key) for key in ("status", "finish_reason", "request_start_monotonic_ns",
                                                           "request_end_monotonic_ns", "batch_index")},
            "quality": quality_result}


def _discover_requests(run_root: Path, prompts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    found: dict[str, Any] = {}
    errors: list[str] = []
    requests_root = run_root / "requests"
    if not requests_root.is_dir():
        return {}, ["requests directory missing"]
    metadata_files = sorted(requests_root.rglob("metadata.json"))
    if len(metadata_files) > 1000:
        return {}, ["too many request metadata records"]
    for meta_path in metadata_files:
        try:
            metadata = _json_load(meta_path)
        except Exception as exc:
            errors.append(f"metadata unreadable: {meta_path.parent.name} ({type(exc).__name__})")
            continue
        prompt_id = quality.normalize_prompt_id(str(metadata.get("prompt_file", "")))
        if prompt_id not in prompts:
            errors.append(f"request refers to unknown prompt: {prompt_id or 'empty'}")
            continue
        if prompt_id in found:
            errors.append(f"duplicate request directory for prompt {prompt_id}")
            found[prompt_id] = {"metadata": metadata, "request_dir": meta_path.parent, "duplicate": True}
            continue
        metadata["_request_dir"] = str(meta_path.parent)
        found[prompt_id] = {"metadata": metadata, "request_dir": meta_path.parent, "duplicate": False}
    return found, errors


def _load_request_entries(request_dir: Path, repeats: int) -> tuple[list[dict[str, Any]], list[str]]:
    path = request_dir / "requests.json"
    if not path.is_file(): return [], ["requests.json missing"]
    try: entries = _json_load(path)
    except Exception as exc: return [], [f"requests.json unreadable: {type(exc).__name__}"]
    if not isinstance(entries, list): return [], ["requests.json is not a list"]
    errors = []
    if len(entries) != repeats: errors.append(f"expected {repeats} formal request records, found {len(entries)}")
    if any(not isinstance(entry, dict) for entry in entries): errors.append("requests.json contains non-object row")
    return [x for x in entries if isinstance(x, dict)], errors


def _prompt_identity_check(manifest_path: Path, prompt: dict[str, Any]) -> dict[str, Any]:
    rel = prompt.get("file")
    expected = prompt.get("sha256")
    if not isinstance(rel, str) or not isinstance(expected, str):
        return {"status": "failed", "error": "manifest prompt file/hash missing"}
    source = (manifest_path.parent / rel).resolve()
    try:
        digest, size = sha256_file(source, MAX_TEXT_BYTES)
    except Exception as exc:
        return {"status": "failed", "error": f"prompt source hash failed: {type(exc).__name__}"}
    return {"status": "pass" if digest == expected else "failed", "path": str(source),
            "sha256": digest, "expected_sha256": expected, "bytes": size}


def _resources_for_run(run_root: Path, windows: list[tuple[float, float]] | None) -> dict[str, Any]:
    path = run_root / "resource" / "samples.jsonl"
    if not path.is_file(): return {"status": "unknown", "errors": ["resource samples missing"]}
    ranges = windows or []
    found = []
    unknown_physical = unknown_commit = error_samples = 0
    commit_scopes: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 1 << 20:
                    error_samples += 1
                    continue
                try: sample = json.loads(line)
                except json.JSONDecodeError:
                    error_samples += 1
                    continue
                if sample.get("record_type") != "resource_sample": continue
                mono = sample.get("monotonic_seconds")
                if not isinstance(mono, (int, float)) or not math.isfinite(float(mono)): continue
                if not any(lo <= mono <= hi for lo, hi in ranges): continue
                memory = sample.get("memory") or {}
                physical = memory.get("physical_available_bytes")
                commit = memory.get("commit_available_bytes")
                if type(physical) is not int: unknown_physical += 1
                if type(commit) is not int: unknown_commit += 1
                sources = memory.get("sources") or {}
                commit_scopes.add(str(sources.get("system_commit") or memory.get("source") or "unknown"))
                found.append((float(mono), memory, sample))
    except Exception as exc:
        return {"status": "unknown", "errors": [f"resource stream failed: {type(exc).__name__}"]}
    physical_rows = [(r[1].get("physical_available_bytes"), r[0]) for r in found if type(r[1].get("physical_available_bytes")) is int]
    commit_rows = [(r[1].get("commit_available_bytes"), r[0]) for r in found if type(r[1].get("commit_available_bytes")) is int]
    def minimum(rows):
        if not rows: return None
        value, mono = min(rows, key=lambda x:x[0])
        return {"bytes": value, "gib": round(value / 1024**3, 3), "monotonic_seconds": mono}
    disk_start: dict[str, dict[str, int]] = {}
    disk_end: dict[str, dict[str, int]] = {}
    if found:
        for device, counters in (found[0][2].get("disk_io") or {}).items():
            if isinstance(counters, dict): disk_start[device] = counters
        for device, counters in (found[-1][2].get("disk_io") or {}).items():
            if isinstance(counters, dict): disk_end[device] = counters
    disk_delta = {}
    for device in sorted(set(disk_start) | set(disk_end)):
        a, b = disk_start.get(device, {}), disk_end.get(device, {})
        disk_delta[device] = {}
        for counter in ("read_bytes", "write_bytes", "read_count", "write_count"):
            x, y = a.get(counter), b.get(counter)
            disk_delta[device][counter] = y - x if type(x) is int and type(y) is int and y >= x else None
    return {"status": "pass" if found else "unknown", "window_sample_count": len(found),
            "physical_available_min": minimum(physical_rows), "physical_unknown_samples": unknown_physical,
            "commit_available_min_observed": minimum(commit_rows), "commit_unknown_samples": unknown_commit,
            "commit_scope_values": sorted(commit_scopes),
            "commit_systemwide_proven": bool(commit_scopes) and all(x == "PSAPI GetPerformanceInfo" for x in commit_scopes),
            "malformed_or_oversize_samples": error_samples,
            "disk_io_delta": {"scope": "systemwide per-device counters; not attributable only to this model", "devices": disk_delta}}


def _resource_windows(run_data: dict[str, Any]) -> list[tuple[float, float]]:
    intervals = []
    for prompt in run_data.get("prompts", {}).values():
        for record in prompt.get("formals", []):
            request = record.get("request") or {}
            start = request.get("request_start_monotonic_ns")
            end = request.get("request_end_monotonic_ns")
            if type(start) is int and type(end) is int and end >= start:
                intervals.append((start / 1e9, end / 1e9))
    return intervals


def _buffer_override_error(baseline: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    same_run = _normal_path(baseline.get("root")) == _normal_path(candidate.get("root"))
    if same_run:
        return None  # self-comparison validates a run's evidence; it is not the buffered A/B candidate arm
    if candidate.get("allowed_env_override", {}).get("STRATA_UNBUFFERED_LOAD") != "0":
        return "candidate does not record the approved STRATA_UNBUFFERED_LOAD=0 override"
    return None


def analyze_run(run_root: Path, manifest_path: Path, manifest_prompts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    discovered, discovery_errors = _discover_requests(run_root, manifest_prompts)
    writer_receipt = load_artifact_writer_receipt(run_root)
    prompt_results: dict[str, Any] = {}
    for prompt_id, prompt in manifest_prompts.items():
        source_check = _prompt_identity_check(manifest_path, prompt)
        discovered_prompt = discovered.get(prompt_id)
        if not discovered_prompt:
            prompt_results[prompt_id] = {"status": "incomplete", "prompt_source": source_check,
                                         "errors": ["prompt request directory missing"], "formals": []}
            continue
        metadata = discovered_prompt["metadata"]
        request_dir = discovered_prompt["request_dir"]
        metadata_path = request_dir / "metadata.json"
        try:
            metadata["_metadata_raw_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        except Exception:
            metadata["_metadata_raw_sha256"] = None
        errors: list[str] = []
        if discovered_prompt["duplicate"]: errors.append("duplicate prompt request directory")
        prompt_file = metadata.get("prompt_file")
        prompt_id_from_metadata = quality.normalize_prompt_id(str(prompt_file or ""))
        if prompt_id_from_metadata != prompt_id: errors.append("metadata prompt identity mismatch")
        if metadata.get("prompt_sha256") != prompt.get("sha256"): errors.append("metadata prompt hash differs from manifest")
        if metadata.get("prompt_bytes") != source_check.get("bytes"): errors.append("metadata prompt byte count differs from manifest file")
        copied_prompt_path = request_dir / "prompt.txt"
        copied_hash = None
        if copied_prompt_path.is_file():
            try:
                copied_hash, copied_size = sha256_file(copied_prompt_path, MAX_TEXT_BYTES)
                if copied_hash != source_check.get("sha256") or copied_size != source_check.get("bytes"):
                    errors.append("copied request prompt differs from manifest source bytes")
            except Exception as exc: errors.append(f"copied prompt hash failed: {type(exc).__name__}")
        else: errors.append("copied request prompt missing")
        repeats = metadata.get("repeats")
        if type(repeats) is not int or repeats < 1:
            errors.append("metadata repeats missing/invalid")
            repeats = 0
        requests, req_errors = _load_request_entries(request_dir, repeats)
        errors.extend(req_errors)
        formals = []
        for index in range(repeats):
            if index >= len(requests):
                formals.append({"repeat": index, "status": "incomplete", "errors": ["request record missing"],
                                "text_sha256": None, "token_ids_sha256": None, "quality": None})
                continue
            request = requests[index]
            batch_index = request.get("batch_index")
            if batch_index is not None and batch_index != index:
                errors.append(f"requests.json order mismatch at repeat {index}")
            formal = evaluate_formal(run_root, prompt, request, index, metadata, writer_receipt)
            formal["request"] = request
            formal["token_ids"] = formal.pop("token_ids", None)
            formals.append(formal)
        formal_statuses = {f.get("status") for f in formals}
        prompt_status = "pass" if not errors and formals and formal_statuses == {"pass"} and source_check.get("status") == "pass" else (
            "incomplete" if "incomplete" in formal_statuses or source_check.get("status") != "pass" else "failed")
        prompt_results[prompt_id] = {"status": prompt_status, "metadata": {k: metadata.get(k) for k in
            ("prompt_file", "prompt_sha256", "prompt_bytes", "repeats", "warmup", "max_tokens", "concurrency", "seed", "reasoning_effort")},
            "prompt_source": source_check, "copied_prompt_sha256": copied_hash,
            "errors": errors, "formals": formals}
    overall_errors = discovery_errors[:]
    missing = sorted(set(manifest_prompts) - set(discovered))
    if missing: overall_errors.append("manifest prompts missing: " + ",".join(missing))
    extra = sorted(set(discovered) - set(manifest_prompts))
    if extra: overall_errors.append("unexpected prompt ids: " + ",".join(extra))
    return {"root": str(run_root), "prompts": prompt_results, "discovery_errors": discovery_errors,
            "missing_prompt_ids": missing, "unexpected_prompt_ids": extra,
            "resource_window": _resources_for_run(run_root, _resource_windows({"prompts": prompt_results}))}


def compare_runs(baseline_root: Path, candidate_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, prompts = quality.load_manifest(manifest_path)
    if len(prompts) != 9:
        raise EvidenceError(f"manifest must contain nine quality prompts, found {len(prompts)}")
    manifest_sha, _ = sha256_file(manifest_path)
    tokenizer_check = verify_tokenizer_manifest(manifest)
    baseline = analyze_run(baseline_root, manifest_path, prompts)
    candidate = analyze_run(candidate_root, manifest_path, prompts)
    control_differences: dict[str, Any] = {}
    allowed_differences: dict[str, Any] = {}
    unexpected_differences: list[str] = []
    control_errors: list[str] = []
    base_control = _load_run_controls(baseline_root)
    cand_control = _load_run_controls(candidate_root)
    if not isinstance(base_control.get("normalized_controls"), dict): base_control["normalized_controls"] = {}
    if not isinstance(cand_control.get("normalized_controls"), dict): cand_control["normalized_controls"] = {}
    if base_control.get("status") != "pass" or cand_control.get("status") != "pass":
        control_errors.extend([f"baseline control: {e}" for e in base_control.get("errors", [])])
        control_errors.extend([f"candidate control: {e}" for e in cand_control.get("errors", [])])
    for key in ("normalized_controls",):
        if base_control.get(key) != cand_control.get(key):
            control_differences[key] = {"baseline_sha256": base_control.get("normalized_controls_sha256"),
                                        "candidate_sha256": cand_control.get("normalized_controls_sha256")}
            unexpected_differences.append("normalized shared controls differ")
    if base_control.get("model_identity") != cand_control.get("model_identity"):
        control_differences["model_identity"] = {"baseline": base_control.get("model_identity"),
                                                  "candidate": cand_control.get("model_identity")}
        unexpected_differences.append("model identity differs")
    if base_control.get("server_python") != cand_control.get("server_python"):
        control_differences["server_python"] = {"baseline": base_control.get("server_python"),
                                                 "candidate": cand_control.get("server_python")}
        unexpected_differences.append("server Python bridge identity differs")
    allowed_differences = {
        "binary": {"baseline_path": base_control.get("engine_path"), "candidate_path": cand_control.get("engine_path"),
                   "baseline_sha256": base_control.get("engine_sha256_actual"), "candidate_sha256": cand_control.get("engine_sha256_actual"),
                   "baseline_source_sha": base_control.get("engine_source_sha"), "candidate_source_sha": cand_control.get("engine_source_sha")},
        "raw_config_hash": {"baseline": base_control.get("config_file_sha256"), "candidate": cand_control.get("config_file_sha256")},
        "recorded_config_identity_hash": {"baseline": base_control.get("identity_config_sha256"),
                                           "candidate": cand_control.get("identity_config_sha256")},
        "ple": {"baseline": base_control.get("ple_observation"), "candidate": cand_control.get("ple_observation")},
        "run_specific_config": {"baseline": base_control.get("run_specific_config"),
                                "candidate": cand_control.get("run_specific_config")},
        "file_io_override": {"baseline": base_control.get("allowed_env_override"),
                             "candidate": cand_control.get("allowed_env_override")},
    }
    base_env = base_control.get("strict_env_value_sha256", {})
    cand_env = cand_control.get("strict_env_value_sha256", {})
    env_diffs = {k: {"baseline_value_sha256": base_env.get(k), "candidate_value_sha256": cand_env.get(k)}
                 for k in sorted(set(base_env) | set(cand_env)) if base_env.get(k) != cand_env.get(k)}
    if env_diffs:
        control_differences["strict_environment_value_differences"] = env_diffs
        unexpected_differences.extend(f"unapproved environment difference: {k}" for k in env_diffs)
    if base_control.get("effective_file_tier_mode") != "buffered" or cand_control.get("effective_file_tier_mode") != "buffered":
        control_errors.append("buffered file-tier mode not proved by run logs")
    override_error = _buffer_override_error(base_control, cand_control)
    if override_error: control_errors.append(override_error)
    manifest_tokenizer_hash = tokenizer_check.get("identity_sha256")
    base_control["normalized_controls"]["manifest_tokenizer_identity"] = manifest_tokenizer_hash
    cand_control["normalized_controls"]["manifest_tokenizer_identity"] = manifest_tokenizer_hash
    base_control["normalized_controls_sha256"] = _canonical_hash(base_control["normalized_controls"])
    cand_control["normalized_controls_sha256"] = _canonical_hash(cand_control["normalized_controls"])
    if base_control["normalized_controls_sha256"] != cand_control["normalized_controls_sha256"]:
        control_differences["normalized_controls_after_tokenizer_check"] = {
            "baseline_sha256": base_control["normalized_controls_sha256"],
            "candidate_sha256": cand_control["normalized_controls_sha256"]}
        unexpected_differences.append("normalized shared controls differ")
    if tokenizer_check.get("status") != "pass": control_errors.append("manifest tokenizer artifact identity not verified")

    prompt_comparisons: dict[str, Any] = {}
    all_token_equal = True
    all_text_equal = True
    all_artifact_equal = True
    all_quality_pass = True
    any_incomplete = bool(baseline["missing_prompt_ids"] or candidate["missing_prompt_ids"] or control_errors)
    expected_pairs = len(prompts) * 3
    observed_pairs = 0
    required_pairs_complete = True
    prompt_pair_counts: dict[str, dict[str, int]] = {}
    for prompt_id in prompts:
        b = baseline["prompts"].get(prompt_id)
        c = candidate["prompts"].get(prompt_id)
        if b is None or c is None:
            any_incomplete = True
            required_pairs_complete = False
            prompt_pair_counts[prompt_id] = {"expected_pairs": 3, "observed_pairs": 0}
            prompt_comparisons[prompt_id] = {"status": "incomplete", "expected_pairs": 3,
                                             "observed_pairs": 0, "errors": ["prompt side missing"]}
            continue
        b_formals, c_formals = b.get("formals", []), c.get("formals", [])
        prompt_required_complete = len(b_formals) >= 3 and len(c_formals) >= 3
        prompt_observed_pairs = sum(1 for i in range(3) if i < len(b_formals) and i < len(c_formals))
        prompt_pair_counts[prompt_id] = {"expected_pairs": 3, "observed_pairs": prompt_observed_pairs}
        observed_pairs += prompt_observed_pairs
        if not prompt_required_complete:
            required_pairs_complete = False
            any_incomplete = True
        metadata_fields = ("prompt_file", "prompt_sha256", "prompt_bytes", "repeats", "warmup",
                           "max_tokens", "concurrency", "seed", "reasoning_effort")
        base_meta, cand_meta = b.get("metadata") or {}, c.get("metadata") or {}
        metadata_observed = all(k in base_meta for k in metadata_fields) and all(k in cand_meta for k in metadata_fields)
        metadata_matches = all(base_meta.get(k) == cand_meta.get(k) for k in metadata_fields) if metadata_observed else None
        if metadata_observed and not metadata_matches:
            control_differences.setdefault("per_prompt_metadata", {})[prompt_id] = {
                "baseline": {k: base_meta.get(k) for k in metadata_fields},
                "candidate": {k: cand_meta.get(k) for k in metadata_fields}}
            unexpected_differences.append(f"request controls differ for {prompt_id}")
        repeats = max(len(b_formals), len(c_formals))
        pairs = []
        for i in range(repeats):
            bf = b_formals[i] if i < len(b_formals) else None
            cf = c_formals[i] if i < len(c_formals) else None
            if bf is None or cf is None:
                any_incomplete = True
                pairs.append({"repeat": i, "status": "incomplete", "errors": ["formal artifact missing on one side"]})
                continue
            if bf.get("status") != "pass" or cf.get("status") != "pass": any_incomplete = True
            token_equal = (bf.get("token_ids") is not None and cf.get("token_ids") is not None and
                           bf.get("token_ids") == cf.get("token_ids") and bf.get("include_stop") is True and cf.get("include_stop") is True)
            text_equal = bf.get("model_text_sha256") is not None and bf.get("model_text_sha256") == cf.get("model_text_sha256")
            artifact_equal = (bf.get("artifact_text_sha256") is not None and
                              bf.get("artifact_text_sha256") == cf.get("artifact_text_sha256"))
            quality_equal = ((bf.get("quality") or {}).get("quality_status") ==
                             (cf.get("quality") or {}).get("quality_status"))
            if not token_equal: all_token_equal = False
            if not text_equal: all_text_equal = False
            if not artifact_equal: all_artifact_equal = False
            if ((bf.get("quality") or {}).get("quality_status") != "pass" or
                    (cf.get("quality") or {}).get("quality_status") != "pass"):
                all_quality_pass = False
            pair_errors = []
            if bf.get("status") != "pass" or cf.get("status") != "pass": pair_errors.append("formal evidence incomplete/invalid")
            if metadata_matches is False: pair_errors.append("per-prompt request controls differ")
            if not token_equal: pair_errors.append("actual generated token-ID sequences/stop tags differ or are unproven")
            if not text_equal: pair_errors.append("raw visible text byte hashes differ or are unproven")
            pairs.append({"repeat": i, "status": "pass" if not pair_errors else "incomplete" if "formal evidence incomplete/invalid" in pair_errors else "mismatch",
                          "token_ids_equal": token_equal, "baseline_token_count": bf.get("token_count"),
                          "candidate_token_count": cf.get("token_count"),
                          "baseline_token_ids_sha256": bf.get("token_ids_sha256"),
                          "candidate_token_ids_sha256": cf.get("token_ids_sha256"),
                          "baseline_stop": {"finish_reason": bf.get("finish_reason"), "include_stop": bf.get("include_stop")},
                          "candidate_stop": {"finish_reason": cf.get("finish_reason"), "include_stop": cf.get("include_stop")},
                          "model_text_from_sse_equal": text_equal,
                          "baseline_model_text_sha256": bf.get("model_text_sha256"),
                          "candidate_model_text_sha256": cf.get("model_text_sha256"),
                          "raw_artifact_bytes_equal": artifact_equal,
                          "baseline_artifact_sha256": bf.get("artifact_text_sha256"),
                          "candidate_artifact_sha256": cf.get("artifact_text_sha256"),
                          "baseline_artifact_conversion": bf.get("artifact_conversion"),
                          "candidate_artifact_conversion": cf.get("artifact_conversion"),
                          "quality_status_equal": quality_equal,
                          "request_controls": "matched" if metadata_matches is True else "different" if metadata_matches is False else "not_observed",
                          "baseline_quality": (bf.get("quality") or {}).get("quality_status"),
                          "candidate_quality": (cf.get("quality") or {}).get("quality_status"),
                          "errors": pair_errors})
        prompt_comparisons[prompt_id] = {"request_controls": "matched" if metadata_matches is True else "different" if metadata_matches is False else "not_observed",
                                         "expected_pairs": 3, "observed_pairs": prompt_observed_pairs,
                                         "status": "pass" if pairs and all(p.get("status") == "pass" for p in pairs)
                                         else "incomplete" if any(p.get("status") == "incomplete" for p in pairs)
                                         else "mismatch", "formals": pairs}
    pairs_complete = required_pairs_complete and observed_pairs == expected_pairs
    if not pairs_complete:
        any_incomplete = True
        all_token_equal = False
        all_text_equal = False
        all_artifact_equal = False
        all_quality_pass = False
    controls_match = not control_errors and not unexpected_differences and base_control.get("normalized_controls_sha256") == cand_control.get("normalized_controls_sha256")
    if not controls_match: any_incomplete = True
    accepted = (not any_incomplete and controls_match and all_token_equal and all_text_equal and all_quality_pass and
                all(v.get("status") == "pass" for v in prompt_comparisons.values()))
    status = "accepted" if accepted else "incomplete" if any_incomplete else "not_accepted"
    for run_data in (baseline, candidate):
        for prompt_data in run_data.get("prompts", {}).values():
            for formal in prompt_data.get("formals", []):
                formal.pop("token_ids", None)
    return {"schema_version": 1, "tool": "hetero-compare-quality", "offline_only": True,
            "status": status, "accepted": accepted, "manifest": {"path": str(manifest_path), "sha256": manifest_sha,
            "prompt_ids": sorted(prompts)}, "tokenizer_identity": tokenizer_check,
            "controls": {"match": controls_match, "errors": control_errors,
                         "unexpected_differences": unexpected_differences, "differences": control_differences,
                         "baseline": base_control, "candidate": cand_control,
                         "allowed_differences": allowed_differences},
            "runs": {"baseline": baseline, "candidate": candidate},
            "prompt_comparisons": prompt_comparisons,
            "overall": {"all_actual_token_ids_equal": all_token_equal,
                        "all_model_text_from_sse_equal": all_text_equal,
                        "all_artifact_byte_hashes_equal": all_artifact_equal,
                        "all_quality_pass": all_quality_pass,
                        "expected_pairs": expected_pairs, "observed_pairs": observed_pairs,
                        "required_pairs_complete": pairs_complete,
                        "required_pairs_by_prompt": prompt_pair_counts}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file(): parser.error("manifest must exist")
    base = manifest_path.resolve().parent.parent
    output = Path(args.output)
    if output.exists():
        print("output already exists", file=sys.stderr)
        return 2
    if not output.parent.is_dir(): parser.error("output parent directory must exist")
    try:
        manifest_header = _json_load(manifest_path)
        created_utc = manifest_header.get("created_utc") if isinstance(manifest_header, dict) else None
        manifest_date = created_utc[:10] if isinstance(created_utc, str) and len(created_utc) >= 10 else None
        baseline = resolve_run(args.baseline_run, base, manifest_date)
        candidate = resolve_run(args.candidate_run, base, manifest_date)
        result = compare_runs(baseline, candidate, manifest_path.resolve())
    except Exception as exc:
        result = {"schema_version": 1, "tool": "hetero-compare-quality", "offline_only": True,
                  "status": "incomplete", "accepted": False,
                  "errors": [f"comparison failed: {type(exc).__name__}: {str(exc)[:300]}"]}
    if args.validate_only:
        print(json.dumps({"status": result["status"], "accepted": result["accepted"],
                          "output_written": False, "offline_only": True}, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] in {"accepted", "not_accepted", "incomplete"} else 1
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        print("output already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"output write failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "accepted": result["accepted"],
                      "output": str(output)}, ensure_ascii=False))
    return 0 if result["accepted"] else 3 if result["status"] == "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
