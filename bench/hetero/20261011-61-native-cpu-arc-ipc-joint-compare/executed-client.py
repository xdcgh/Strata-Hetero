#!/usr/bin/env python3
"""Plan or explicitly run a framed CPU-versus-Arc FFN boundary comparison.

The default path reads the small compare contract and pinned run42 IPC helper sources.
It does not read model/activation payloads, import NumPy/OpenVINO, launch services, or access a device.
Execution is separately gated by a root authorization receipt and explicit stop flags.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONTRACT_PATH = ROOT / "bench/hetero/20261011-53-native-cpu-arc-ipc-compare/compare-contract.json"
RUN42_PATH = ROOT / "bench/hetero/20261009-42-native-q4k-q5_1-ipc/executed-client.py"
RESULT_PREFIX = struct.Struct("<IQ")
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _load_run42():
    spec = importlib.util.spec_from_file_location("native_ipc_run42_helpers", RUN42_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed run42 IPC ownership/resource helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN42 = _load_run42()


class CompareError(RuntimeError):
    pass


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_sha(path: Path) -> str:
    data, _ = RUN42._sha_file_stable(path)
    return sha(data)


def load_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def plan_record(contract: dict[str, Any]) -> dict[str, Any]:
    backends = [{"id": b["id"], "pool_workers": b["pool_workers"], "pool_affinity": b["pool_affinity"],
                 "optional": b["optional"], "binary_sha256": b["exe_sha256"]}
                for b in contract["backends"]]
    return {
        "status": "plan_only", "run_id": contract["run_id"], "execution_mode": contract.get("execution_mode", "full"),
        "execution_authorized": False,
        "rows": [x["rows"] for x in contract["row_cases"]], "warmup": contract["warmup"],
        "formal_repeats": contract["formal_repeats"], "backends": backends,
        "arms": contract["arm_plan"], "arc": {"device": "GPU.0", "precision": "f32", "execution_mode": "ACCURACY"},
        "timing_contract": contract["timing_contract"], "quality_tolerances": contract["quality_tolerances"],
        "quality_policy": contract["quality_policy"],
        "resource_gate": contract["resource_gate"], "cpu_blob_hash_pinned": True,
        "model_payloads_read": False, "native_f32_payloads_read": False, "native_cpu_service_started": False,
        "arc_service_started": False, "openvino_imported": False, "core_created": False, "device_accessed": False,
        "inference_called": False, "large_f32_outputs_written": False,
        "raw_output_files_planned": contract["output"]["outputs_write_large_f32"],
        "planned_output_format": contract["output"]["output_format"],
        "output_format": contract["output"]["output_format"],
        "output_root": contract["output"]["private_output_root"], "receipt_dir": contract["output"]["receipt_dir"],
        "run42_fresh_arc_remeasure_required": True,
    }


def _process_with_utc(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    row["create_time_utc"] = dt.datetime.fromtimestamp(row["create_time"], dt.timezone.utc).isoformat()
    return row


def init_payload(nonce: str, rows: int, identity_sha256: str) -> bytes:
    if len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
        raise CompareError("nonce must be 32 lowercase hexadecimal characters")
    if type(rows) is not int or rows not in ROWS:
        raise CompareError("rows are outside the frozen nine-shape matrix")
    if len(identity_sha256) != 64 or any(c not in "0123456789abcdef" for c in identity_sha256):
        raise CompareError("identity SHA-256 must be lowercase 64-character hex")
    return json.dumps({"schema": "strata-xpu-init-v1", "nonce": nonce, "rows": rows,
                       "identity_sha256": identity_sha256}, separators=(",", ":")).encode("ascii")


def parse_result_payload(kind: int, request_id: int, frame) -> tuple[bytes, int]:
    if frame.kind != (kind | 0x8000) or frame.request_id != request_id or len(frame.payload) < RESULT_PREFIX.size:
        raise CompareError("IPC response kind/id/prefix mismatch")
    status, worker_ns = RESULT_PREFIX.unpack_from(frame.payload)
    body = bytes(frame.payload[RESULT_PREFIX.size:])
    if status != 0:
        try:
            message = json.loads(body.decode("utf-8")).get("message", "service request failed")
        except Exception:
            message = "invalid error response"
        raise CompareError(f"IPC kind={kind} id={request_id}: {message}")
    return body, worker_ns


def validate_cpu_ready(ready: dict[str, Any], contract: dict[str, Any], backend: dict[str, Any],
                       nonce: str, rows: int) -> None:
    pins = contract["source_pins"]
    cpu_pool = ready.get("pool")
    checks = {
        "schema": ready.get("schema") == "strata-xpu-ready-v1", "status": ready.get("status") == "ready",
        "nonce": ready.get("nonce") == nonce, "identity": ready.get("identity_sha256") == contract["arc"]["identity_sha256"],
        "rows": ready.get("rows") == rows, "shape": (ready.get("hidden"), ready.get("intermediate")) == (2560, 640),
        "operator": ready.get("operator") == "native-q4k-q5_1",
        "signature": ready.get("operator_signature") == "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
        "blob_verified": ready.get("native_blob_loaded_and_hash_verified") is True,
        "blob_bytes": ready.get("blob_bytes") == 3072000,
        "blob_sha": ready.get("blob_sha256") == contract["arc"]["blob_sha256"],
        "identity_hash": ready.get("identity_bytes_sha256") == contract["arc"]["identity_sha256"],
        "identity_size": type(ready.get("identity_bytes")) is int and 0 <= ready["identity_bytes"] <= 1048576,
        "native_source": ready.get("native_source_sha256") == pins["cpu_kernel_source_sha256"],
        "service_source": ready.get("service_source_sha256") == pins["cpu_service_source_sha256"],
        "service_cmake": ready.get("service_cmake_sha256") == pins["cpu_harness_cmake_sha256"],
        "pool_cpp": ready.get("pool_cpp_sha256") == pins["cpu_pool_cpp_sha256"],
        "pool_hpp": ready.get("pool_hpp_sha256") == pins["cpu_pool_hpp_sha256"],
        "requested_workers": isinstance(cpu_pool, dict) and cpu_pool.get("requested_workers") == backend["pool_workers"],
        "actual_workers": isinstance(cpu_pool, dict) and cpu_pool.get("actual_workers") == backend["pool_workers"],
        "affinity": isinstance(cpu_pool, dict) and cpu_pool.get("affinity") == backend["pool_affinity"],
        "host_pin": isinstance(cpu_pool, dict) and cpu_pool.get("host_pinned") == (backend["pool_affinity"] != "none"),
    }
    if not all(checks.values()):
        raise CompareError(f"CPU INIT READY binding mismatch: {[k for k, ok in checks.items() if not ok]}")
    reports = cpu_pool.get("worker_affinity_readback")
    if not isinstance(reports, list) or len(reports) != backend["pool_workers"]:
        raise CompareError("CPU READY pool affinity readback count mismatch")
    required_types = {"worker": int, "requested_core": int, "ready": bool, "pin_requested": bool,
        "pin_applied": bool, "mask_observed": bool, "mask_matches_request": bool,
        "observed_group": int, "observed_mask": int, "startup_processor": int}
    if any(not isinstance(x, dict) or any(type(x.get(key)) is not kind for key, kind in required_types.items()) or
           x.get("ready") is not True or x.get("pin_requested") != (backend["pool_affinity"] != "none")
           for x in reports):
        raise CompareError("CPU READY pool affinity worker readback is incomplete")
    if backend["pool_affinity"] != "none" and any(
        x["requested_core"] < 0 or not (x["pin_applied"] and x["mask_observed"] and x["mask_matches_request"])
        for x in reports):
        raise CompareError("CPU READY pool pin readback did not match every requested core")
    if backend["pool_affinity"] == "none" and any(x["pin_applied"] for x in reports):
        raise CompareError("CPU READY reports an unexpected pinned worker for the none-affinity backend")


def _serialize_input(np, x) -> bytes:
    return np.asarray(x, dtype="<f4", order="C").tobytes(order="C")


def _safe_component(value: str) -> str:
    result = "".join(c if c.isalnum() or c in "-+_" else "_" for c in value)
    if not result or result in {".", ".."}: raise CompareError("unsafe output path component")
    return result


def _worker_stderr_path(logs: Path, arm_id: str, role: str) -> Path:
    return logs / f"{_safe_component(arm_id)}-{_safe_component(role)}.stderr.log"


def _write_raw_response(path: Path, body: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb", buffering=0) as stream:
            view = memoryview(body)
            while view:
                written = stream.write(view)
                if not written: raise OSError(f"no progress while preserving raw response: {path}")
                view = view[written:]
            stream.flush(); os.fsync(stream.fileno())
        return {"path": str(path), "bytes": len(body), "sha256": sha(body), "exclusive": True, "complete": True}
    except Exception as exc:
        # Preserve/hash any partial response file and report the storage failure without retrying.
        if path.is_file():
            partial = path.read_bytes()
            count, digest = len(partial), sha(partial)
        else:
            count, digest = 0, sha(b"")
        return {"path": str(path), "bytes": count, "sha256": digest, "exclusive": True,
                "complete": False, "write_error": f"{type(exc).__name__}: {exc}"}


def _timed_infer(worker, request_kind: int, request_id: int, rows: int, x, expected_input_bytes: bytes, expected_input_sha: str,
                 reference_bytes: bytes, reference, backend_id: str, arm: str, monitor,
                 np, worker_api, *, cpu_gold_sha: str | None = None, barrier=None,
                 output_base: Path | None = None, output_writer=None, post_transfer_callback=None) -> dict[str, Any]:
    # Gate and input integrity checks stay outside the measured interval.
    try:
        monitor.require()
        preflight_bytes = expected_input_bytes
        if len(preflight_bytes) != rows * 2560 * 4 or sha(preflight_bytes) != expected_input_sha:
            raise CompareError(f"pre-timing input serialization/hash mismatch for rows={rows}")
        monitor.require()
        if barrier is not None: barrier.wait(timeout=30)
    except BaseException:
        if barrier is not None: barrier.abort()
        raise
    start_ns = time.perf_counter_ns()
    peer_transfer_error = None
    try:
        raw_input = _serialize_input(np, x)
        frame = worker.request(request_kind, request_id, raw_input, timeout=120.0)
        body, worker_ns = parse_result_payload(request_kind, request_id, frame)
        expected_bytes = rows * 2560 * 4
        shape_ok = len(body) == expected_bytes
        owned = np.frombuffer(body, dtype="<f4").reshape(rows, 2560).copy(order="C") if shape_ok else None
        end_ns = time.perf_counter_ns()
        # Reuse the barrier for a second phase: both timestamps and owned copies exist
        # before either peer starts output IO, integrity checks, or numerical quality work.
        if barrier is not None:
            try: barrier.wait(timeout=30)
            except threading.BrokenBarrierError as exc: peer_transfer_error = f"{type(exc).__name__}: {exc}"
    except BaseException:
        if barrier is not None: barrier.abort()
        raise

    if post_transfer_callback is not None: post_transfer_callback()
    if output_base is not None:
        extension = ".f32" if shape_ok else ".response.bin"
        output_path = output_base.with_suffix(extension)
        output_record = (output_writer or _write_raw_response)(output_path, body)
    else:
        output_path = None; output_record = None
    output_storage_pass = output_record is None or output_record.get("complete", True) is True
    post_gate_error = None
    try: monitor.require()
    except Exception as exc: post_gate_error = f"{type(exc).__name__}: {exc}"
    # SHA, finiteness, and numerical checks are outside both clocks.
    input_hash_pass = sha(raw_input) == expected_input_sha
    finite_owned = bool(owned is not None and owned.flags.c_contiguous and owned.flags.owndata and np.isfinite(owned).all())
    output_sha = sha(body)
    result: dict[str, Any] = {
        "arm": arm, "backend": backend_id, "rows": rows, "request_id": request_id,
        "input_sha256": expected_input_sha, "output_sha256": output_sha,
        "input_bytes": len(raw_input), "output_bytes": len(body),
        "caller_boundary_rtt_ns": end_ns - start_ns, "worker_wall_ns": worker_ns,
        "rtt_minus_worker_ns_descriptive": end_ns - start_ns - worker_ns,
        "output_shape_ok": shape_ok, "output_path": str(output_path) if output_path is not None else None,
        "raw_output_storage": output_record,
        "output_storage_complete": output_storage_pass,
        "response_format": "f32le" if shape_ok else "raw_response",
        "paired_transfer_complete": peer_transfer_error is None,
        "paired_transfer_error": peer_transfer_error,
        "owned_output_copy": bool(owned is not None and owned.flags.c_contiguous and owned.flags.owndata), "finite": finite_owned,
        "timed_input_hash_match": input_hash_pass, "resource_gate_pass_after_transfer": post_gate_error is None,
        "resource_gate_error_after_transfer": post_gate_error,
    }
    if cpu_gold_sha is not None:
        bitwise = output_sha == cpu_gold_sha and body == reference_bytes
        result.update({"gold_sha256": cpu_gold_sha, "bitwise_gold_match": bitwise,
                       "quality_metric": "exact F32LE byte match to frozen native CPU output"})
        result["comparison_pass"] = bitwise and finite_owned and input_hash_pass and output_storage_pass and post_gate_error is None and peer_transfer_error is None
    else:
        if post_gate_error is not None:
            quality = {"shape_ok": shape_ok, "finite": finite_owned, "pass": False,
                       "skipped": "resource monitor gate failed after transfer", "gate_error": post_gate_error}
        elif peer_transfer_error is not None:
            quality = {"shape_ok": shape_ok, "finite": finite_owned, "pass": False,
                       "skipped": "peer transfer failed before shared end barrier"}
        else:
            try:
                quality = worker_api.quality_report(owned, reference, worker_api.DEFAULT_TOLERANCES) if shape_ok else {
                    "shape_ok": False, "finite": False, "pass": False, "output_bytes": len(body), "expected_bytes": expected_bytes}
            except Exception as exc:
                quality = {"shape_ok": shape_ok, "finite": finite_owned, "pass": False,
                           "quality_error": f"{type(exc).__name__}: {exc}"}
        result["numerical_quality_vs_native_cpu"] = quality
        result["quality_metric"] = "run42 DEFAULT_TOLERANCES: max_abs 0.01, relative_rmse 0.001, row_norm_relative 0.001"
        result["comparison_pass"] = bool(quality.get("pass")) and finite_owned and input_hash_pass and output_storage_pass and post_gate_error is None and peer_transfer_error is None
    if barrier is not None:
        result["paired_start_perf_counter_ns"] = start_ns
        result["paired_end_perf_counter_ns"] = end_ns
    return result


class DirectCpuTree:
    """Use run42's owned-tree monitor for CPU, where exe and service are the same process."""
    def __init__(self, worker, psutil, executable: Path, argv: list[str]):
        self.worker = worker
        self.inner = RUN42.OwnedServiceTree(worker, psutil, executable, executable, executable, argv[1:])
        self.root_identity = _process_with_utc(self.inner.root_record)
        got = self.root_identity["command_line"]
        expected0 = os.path.normcase(os.path.realpath(str(executable)))
        if not got or os.path.normcase(os.path.realpath(got[0])) != expected0 or got[1:] != argv[1:]:
            raise CompareError(f"CPU service full argv mismatch: {got}")

    def snapshot(self):
        state = self.inner.snapshot()
        state["actual_service_process"] = self.root_identity
        state["service_tree_topology"] = "owned direct Popen child; launcher and service are the same process"
        return state

    def cleanup_descendants(self):
        return self.inner.cleanup_descendants()


class WorkerGroup:
    def __init__(self, handles): self.handles = list(handles)
    @property
    def pid(self): return [h["worker"].pid for h in self.handles]
    @property
    def returncode(self): return [h["worker"].returncode for h in self.handles]
    def close(self):
        errors = []
        for handle in self.handles:
            try: handle["worker"].close()
            except Exception as exc: errors.append(f"{handle['role']}: {type(exc).__name__}: {exc}")
        if errors: raise CompareError("owned worker group cleanup failed: " + "; ".join(errors))


class TreeGroup:
    def __init__(self, handles): self.handles = list(handles)
    def snapshot(self):
        return {h["role"]: h["tree"].snapshot() for h in self.handles}
    def cleanup_descendants(self):
        rows = {h["role"]: h["tree"].cleanup_descendants() for h in self.handles}
        return {"members": rows, "terminal": all(x.get("terminal") is True for x in rows.values())}


def _update_receipt(stream, record: dict[str, Any]) -> None:
    RUN42.update_receipt(stream, record)


def _raw_output_status(value) -> tuple[bool, bool]:
    if isinstance(value, dict):
        stored = value.get("raw_output_storage")
        here = isinstance(stored, dict) and isinstance(stored.get("path"), str)
        children = [_raw_output_status(v) for v in value.values() if v is not stored]
        return (here or any(x[0] for x in children),
                (here and value.get("response_format") == "f32le") or any(x[1] for x in children))
    if isinstance(value, list):
        children = [_raw_output_status(v) for v in value]
        return any(x[0] for x in children), any(x[1] for x in children)
    return False, False


def _persist_progress(receipt, record):
    raw, f32 = _raw_output_status(record.get("arms", []))
    record["raw_output_files_written"] = raw
    record["large_f32_outputs_written"] = f32
    _update_receipt(receipt, record)


def _authorization(args, contract, contract_path: Path):
    if not (args.execute and args.root_start_confirmed and args.model_matrix_stopped_confirmed):
        raise CompareError("execution requires explicit root start and model-matrix-stopped confirmations")
    path = args.root_authorization
    if path is None or not path.is_absolute() or not path.is_file():
        raise CompareError("an absolute existing root authorization JSON is required")
    auth = json.loads(path.read_text(encoding="utf-8-sig"))
    source_freeze_path = Path(contract.get("source_freeze_path", contract_path.parent / "source-freeze.json"))
    if not source_freeze_path.is_absolute() or not source_freeze_path.is_file():
        raise CompareError("an absolute existing source-freeze receipt is required")
    expected = {"schema": "strata-native-ipc-compare-root-authorization-v1", "run_id": contract["run_id"],
        "execution_authorized": True, "model_matrix_stopped": True, "root_review_complete": True,
        "client_sha256": _stable_sha(Path(__file__)), "contract_sha256": _stable_sha(contract_path),
        "source_freeze_sha256": _stable_sha(source_freeze_path)}
    execution_mode = contract.get("execution_mode", "full")
    if execution_mode not in {"full", "paired_only"}:
        raise CompareError(f"unsupported comparison execution mode: {execution_mode}")
    if bool(args.paired_only) != (execution_mode == "paired_only"):
        raise CompareError("--paired-only must exactly match the selected contract execution mode")
    for key, value in expected.items():
        if auth.get(key) != value: raise CompareError(f"root authorization binding mismatch: {key}")
    if execution_mode == "paired_only":
        paired = contract.get("paired_only")
        if not isinstance(paired, dict): raise CompareError("paired-only contract baseline binding is missing")
        paired_expected = {"execution_mode": "paired_only",
            "baseline_run_id": paired["baseline_run_id"],
            "baseline_contract_sha256": paired["baseline_contract_sha256"],
            "baseline_receipt_sha256": paired["baseline_receipt_sha256"],
            "baseline_client_sha256": paired["baseline_client_sha256"],
            "root_solo_review_sha256": paired["root_solo_review_sha256"],
            "root_arc_numerical_review_sha256": paired["root_arc_numerical_review_sha256"]}
        for key, value in paired_expected.items():
            if auth.get(key) != value: raise CompareError(f"paired-only root authorization binding mismatch: {key}")
    if args.include_cpu16_none and not auth.get("cpu16_none_authorized", False):
        raise CompareError("optional CPU pool w16/none needs separate root authorization")
    return {"path": str(path), "sha256": _stable_sha(path), "authorization": auth}


def _check_source_pins(contract):
    pins = contract["source_pins"]
    mapping = (("run42_client_path", "run42_client_sha256"), ("protocol_path", "protocol_sha256"),
        ("arc_service_path", "arc_service_sha256"), ("worker_api_path", "worker_api_sha256"),
        ("activation_path", "activation_sha256"), ("resource_path", "resource_sha256"),
        ("verify_path", "verify_sha256"))
    actual = {}
    for path_key, digest_key in mapping:
        digest = _stable_sha(Path(pins[path_key]))
        if digest != pins[digest_key]: raise CompareError(f"pinned source changed: {pins[path_key]}")
        actual[digest_key] = digest
    for path_key, digest_key in (("cpu_service_source_path", "cpu_service_source_sha256"),
        ("cpu_kernel_source_path", "cpu_kernel_source_sha256"), ("cpu_harness_cmake_path", "cpu_harness_cmake_sha256"),
        ("cpu_pool_cpp_path", "cpu_pool_cpp_sha256"), ("cpu_pool_hpp_path", "cpu_pool_hpp_sha256")):
        digest = _stable_sha(Path(pins[path_key]))
        if digest != pins[digest_key]: raise CompareError(f"pinned CPU source changed: {pins[path_key]}")
        actual[digest_key] = digest
    if "cpu_service_fixture_path" in pins or "cpu_service_fixture_sha256" in pins:
        if not ("cpu_service_fixture_path" in pins and "cpu_service_fixture_sha256" in pins):
            raise CompareError("CPU service fixture source pin is incomplete")
        digest = _stable_sha(Path(pins["cpu_service_fixture_path"]))
        if digest != pins["cpu_service_fixture_sha256"]:
            raise CompareError(f"pinned source changed: {pins['cpu_service_fixture_path']}")
        actual["cpu_service_fixture_sha256"] = digest
    if contract.get("contract_version") == 2:
        old_recipe = Path(contract["supersedes_contract_path"])
        old_sha = _stable_sha(old_recipe)
        if old_sha != contract["supersedes_contract_sha256"]:
            raise CompareError("preserved compare-contract-v1 recipe changed")
        actual["compare_contract_v1_sha256"] = old_sha
    return actual


def _verify_cpu_binary(contract, backend):
    path = Path(backend["exe"]); digest = _stable_sha(path)
    if digest != backend["exe_sha256"]: raise CompareError("CPU service executable SHA differs from contract")
    build_path = Path(backend["build_receipt_path"])
    if _stable_sha(build_path) != backend["build_receipt_sha256"]: raise CompareError("CPU service build receipt changed")
    receipt = json.loads(build_path.read_text(encoding="utf-8-sig"))
    expected_build_run_id = backend.get("build_run_id", "native-cpu-service-20261011-52")
    if receipt.get("status") != "success" or receipt.get("run_id") != expected_build_run_id:
        raise CompareError(f"CPU service build receipt is not a successful {expected_build_run_id} build")
    b = receipt.get("binaries", {}).get("hetero_native_cpu_service", {})
    if b.get("sha256") != digest or os.path.normcase(os.path.realpath(b.get("path", ""))) != os.path.normcase(os.path.realpath(path)):
        raise CompareError("CPU build receipt does not bind this executable")
    pins = contract["source_pins"]
    source_map = {"src/hetero/native_cpu_service.cpp": pins["cpu_service_source_sha256"],
        "src/hetero/native_expert_bench.cpp": pins["cpu_kernel_source_sha256"],
        "tools/hetero_native_cpu/CMakeLists.txt": pins["cpu_harness_cmake_sha256"],
        "src/kernels/cpu/pool.cpp": pins["cpu_pool_cpp_sha256"],
        "include/strata/kernels/cpu/pool.hpp": pins["cpu_pool_hpp_sha256"]}
    before = receipt.get("source_sha256_before", {})
    if "cpu_service_fixture_sha256" in pins:
        source_map["src/hetero/native_cpu_service_fixture.cpp"] = pins["cpu_service_fixture_sha256"]
    if any(before.get(rel) != digest for rel, digest in source_map.items()):
        raise CompareError(f"{expected_build_run_id} build receipt source freeze differs from compare contract")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "build_receipt_sha256": backend["build_receipt_sha256"]}


def _comparison_basis(contract):
    backend_keys = ("id", "kind", "exe", "exe_sha256", "build_receipt_path", "build_receipt_sha256",
                    "pool_workers", "pool_affinity", "optional", "build_run_id")
    return ({key: contract[key] for key in (
        "contract_version", "source_pins", "arc", "artifacts", "row_cases", "warmup", "formal_repeats",
        "quality_tolerances", "quality_policy", "timing_contract", "resource_gate", "concurrency")}
        | {"backends": [{key: backend.get(key) for key in backend_keys} for backend in contract["backends"]]})


def _validate_baseline_trial(trial, mode, row_case, output_root, expected_label, tolerances):
    if not isinstance(trial, dict) or trial.get("label") != expected_label or trial.get("comparison_pass") is not True:
        raise CompareError(f"paired-only baseline {mode} trial {expected_label} did not pass")
    for key in ("output_shape_ok", "finite", "timed_input_hash_match", "resource_gate_pass_after_transfer",
                "paired_transfer_complete", "owned_output_copy", "output_storage_complete"):
        if trial.get(key) is not True:
            raise CompareError(f"paired-only baseline {mode} {expected_label} failed {key}")
    if trial.get("response_format") != "f32le":
        raise CompareError(f"paired-only baseline {mode} {expected_label} is not F32LE")
    if trial.get("rows") != row_case["rows"] or trial.get("input_sha256") != row_case["input_sha256"]:
        raise CompareError(f"paired-only baseline {mode} {expected_label} input binding changed")
    if trial.get("input_bytes") != row_case["input_bytes"] or trial.get("output_bytes") != row_case["rows"] * 2560 * 4:
        raise CompareError(f"paired-only baseline {mode} {expected_label} tensor size changed")
    if mode == "cpu":
        if trial.get("bitwise_gold_match") is not True or trial.get("output_sha256") != row_case["native_cpu_output_sha256"]:
            raise CompareError(f"paired-only CPU baseline {expected_label} did not bitwise match native gold")
    else:
        quality = trial.get("numerical_quality_vs_native_cpu")
        if (not isinstance(quality, dict) or quality.get("pass") is not True or
            quality.get("shape_ok") is not True or quality.get("finite") is not True):
            raise CompareError(f"paired-only Arc baseline {expected_label} failed numerical quality")
        if quality.get("tolerances") != tolerances:
            raise CompareError(f"paired-only Arc baseline {expected_label} tolerance contract changed")
    storage = trial.get("raw_output_storage")
    if not isinstance(storage, dict) or storage.get("exclusive") is not True or storage.get("complete") is not True:
        raise CompareError(f"paired-only baseline {mode} {expected_label} lacks complete exclusive output storage")
    output_path = Path(storage.get("path", ""))
    if not output_path.is_absolute() or not output_path.is_file():
        raise CompareError(f"paired-only baseline {mode} {expected_label} output is missing")
    try:
        if os.path.normcase(os.path.commonpath([os.path.realpath(output_path), os.path.realpath(output_root)])) != os.path.normcase(os.path.realpath(output_root)):
            raise CompareError(f"paired-only baseline {mode} {expected_label} output escaped its private root")
    except ValueError as exc:
        raise CompareError(f"paired-only baseline {mode} {expected_label} output root mismatch") from exc
    if output_path.stat().st_size != trial["output_bytes"] or storage.get("bytes") != trial["output_bytes"]:
        raise CompareError(f"paired-only baseline {mode} {expected_label} output byte count changed")
    if storage.get("sha256") != trial.get("output_sha256") or _stable_sha(output_path) != trial.get("output_sha256"):
        raise CompareError(f"paired-only baseline {mode} {expected_label} output hash changed")


def _verify_paired_baseline(contract):
    paired = contract.get("paired_only")
    if contract.get("execution_mode") != "paired_only" or not isinstance(paired, dict):
        raise CompareError("paired-only baseline requested for a non-paired contract")
    baseline_contract_path = Path(paired["baseline_contract_path"])
    baseline_receipt_path = Path(paired["baseline_receipt_path"])
    review_path = Path(paired["root_solo_review_path"])
    arc_review_path = Path(paired["root_arc_numerical_review_path"])
    baseline_client_path = Path(paired["baseline_executed_client_path"])
    for path in (baseline_contract_path, baseline_receipt_path, review_path, arc_review_path, baseline_client_path):
        if not path.is_absolute() or not path.is_file():
            raise CompareError(f"paired-only baseline file is absent or not absolute: {path}")
    hashes = {"baseline_contract_sha256": _stable_sha(baseline_contract_path),
              "baseline_receipt_sha256": _stable_sha(baseline_receipt_path),
              "baseline_client_sha256": _stable_sha(baseline_client_path),
              "root_solo_review_sha256": _stable_sha(review_path),
              "root_arc_numerical_review_sha256": _stable_sha(arc_review_path)}
    for key, value in hashes.items():
        if paired.get(key) != value:
            raise CompareError(f"paired-only baseline changed: {key}")
    baseline_contract = json.loads(baseline_contract_path.read_text(encoding="utf-8-sig"))
    if baseline_contract.get("run_id") != paired.get("baseline_run_id"):
        raise CompareError("paired-only baseline contract run id changed")
    if _comparison_basis(contract) != _comparison_basis(baseline_contract):
        raise CompareError("paired-only baseline source/timing/tolerance/input/binary/worker contract differs")
    baseline = json.loads(baseline_receipt_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    arc_review = json.loads(arc_review_path.read_text(encoding="utf-8-sig"))
    if baseline.get("run_id") != paired["baseline_run_id"] or baseline.get("status") != "failed":
        raise CompareError("paired-only baseline receipt is not the reviewed partial run60")
    baseline_auth = baseline.get("root_authorization", {}).get("authorization", {})
    baseline_freeze_path = Path(baseline_contract.get("source_freeze_path", baseline_contract_path.parent / "source-freeze.json"))
    if (baseline_auth.get("client_sha256") != hashes["baseline_client_sha256"] or
        baseline_auth.get("contract_sha256") != hashes["baseline_contract_sha256"] or
        not baseline_freeze_path.is_file() or
        baseline_auth.get("source_freeze_sha256") != _stable_sha(baseline_freeze_path)):
        raise CompareError("paired-only baseline execution authorization does not bind its frozen client/contract/source")
    failure = baseline.get("failure", {})
    if (failure.get("type") != "FileExistsError" or failure.get("no_restart") is not True or
        "cpu-pool-w10-all.stderr.log" not in failure.get("message", "")):
        raise CompareError("paired-only baseline did not stop at the reviewed solo-to-joint log collision")
    if baseline.get("raw_output_files_written") is not True or baseline.get("large_f32_outputs_written") is not True:
        raise CompareError("paired-only baseline raw F32 responses were not retained")
    if (baseline.get("resource_gate_all_samples_passed") is not True or
        type(baseline.get("resource_gate_samples")) is not int or baseline["resource_gate_samples"] < 1):
        raise CompareError("paired-only baseline resource observer has failed or missing samples")
    if baseline.get("source_pins") != _check_source_pins(contract):
        raise CompareError("paired-only baseline source pin receipt differs from the current contract")
    if review.get("status") != "root_raw_hash_review_pass_solo_only" or review.get("receipt_sha256") != hashes["baseline_receipt_sha256"]:
        raise CompareError("root solo-only raw-response review does not bind the baseline receipt")
    if (review.get("checked_response_files") != 162 or review.get("concurrent_unrun") is not True or
        review.get("modelgain_unrun") is not True):
        raise CompareError("root solo-only review did not check all 162 responses or confirm joint arms were unrun")
    if (arc_review.get("status") != "root_independent_arc_numerical_replay_pass" or
        arc_review.get("outputs_checked") != 54 or arc_review.get("formal_outputs") != 45 or
        arc_review.get("warmups") != 9 or arc_review.get("tolerances") != contract["quality_tolerances"]):
        raise CompareError("root independent Arc numerical review is incomplete or uses different tolerances")
    row_cases = {item["rows"]: item for item in contract["row_cases"]}
    expected_cpu_ids = {b["id"] + "-solo" for b in contract["backends"]}
    expected_ids = expected_cpu_ids | {"arc-gpu0-f32-accuracy-solo"}
    arms = baseline.get("arms", [])
    by_id = {arm.get("id"): arm for arm in arms}
    if len(by_id) != len(arms) or set(by_id) != expected_ids | {"cpu-pool-w10-all+arc-gpu0-concurrent"}:
        raise CompareError("paired-only baseline is missing one or more complete solo arms")
    joint_id = "cpu-pool-w10-all+arc-gpu0-concurrent"
    interrupted = by_id.get(joint_id)
    if (interrupted is None or interrupted.get("status") != "running" or interrupted.get("rows") != [] or
        interrupted.get("process_start") is not None or interrupted.get("cleanup") is not None):
        raise CompareError("paired-only baseline partial joint arm differs from the reviewed pre-start failure")
    reviewed = {entry.get("arm"): entry for entry in review.get("arms", [])}
    if set(reviewed) != expected_ids:
        raise CompareError("root solo review arm list differs from the completed baseline arms")
    arc_points = {(point.get("rows"), point.get("label")): point for point in arc_review.get("points", [])}
    if len(arc_points) != 54:
        raise CompareError("root independent Arc numerical review has duplicate or missing output points")
    for arm_id in expected_ids:
        arm = by_id[arm_id]
        if arm.get("status") != "complete" or not arm.get("process_start"):
            raise CompareError(f"paired-only baseline arm did not complete: {arm_id}")
        cleanup = arm.get("cleanup")
        if not isinstance(cleanup, dict) or not cleanup:
            raise CompareError(f"paired-only baseline cleanup receipt missing: {arm_id}")
        for row in cleanup.values():
            if row.get("terminal") is not True or row.get("remaining_owned_descendants") != []:
                raise CompareError(f"paired-only baseline owned process tree did not terminate: {arm_id}")
        rows = arm.get("rows", [])
        if [item.get("rows") for item in rows] != list(ROWS) or len(rows) != len(ROWS):
            raise CompareError(f"paired-only baseline row matrix is incomplete: {arm_id}")
        review_rows = reviewed[arm_id].get("rows", [])
        if [item.get("rows") for item in review_rows] != list(ROWS):
            raise CompareError(f"root solo review row matrix is incomplete: {arm_id}")
        mode = "arc" if arm_id == "arc-gpu0-f32-accuracy-solo" else "cpu"
        for row in rows:
            expected_trials = ["warmup"] + [f"formal-{i:02d}" for i in range(1, contract["formal_repeats"] + 1)]
            trials = [row.get("warmup"), *row.get("formals", [])]
            if [trial.get("label") if isinstance(trial, dict) else None for trial in trials] != expected_trials:
                raise CompareError(f"paired-only baseline trial labels are incomplete: {arm_id}")
            case = row_cases[row["rows"]]
            for trial in trials:
                _validate_baseline_trial(trial, mode, case, Path(baseline_contract["output"]["private_output_root"]),
                                         trial["label"], contract["quality_tolerances"])
                if mode == "arc":
                    point = arc_points.get((row["rows"], trial["label"]))
                    quality = point.get("quality") if isinstance(point, dict) else None
                    if (not isinstance(quality, dict) or quality != trial.get("numerical_quality_vs_native_cpu") or
                        quality.get("pass") is not True or quality.get("shape_ok") is not True or quality.get("finite") is not True or
                        quality.get("tolerances") != contract["quality_tolerances"] or
                        point.get("sha256") != trial.get("output_sha256")):
                        raise CompareError(f"root independent Arc review differs from baseline output: rows={row['rows']} {trial['label']}")
        reviewed_rows = [
            {"rows": row["rows"], "rtt_ms": round(statistics.median([x["caller_boundary_rtt_ns"] for x in row["formals"]]) / 1e6, 4),
             "worker_ms": round(statistics.median([x["worker_wall_ns"] for x in row["formals"]]) / 1e6, 4)} for row in rows]
        if reviewed[arm_id].get("rows") != reviewed_rows:
            raise CompareError(f"root solo review statistics differ from receipt: {arm_id}")
    return {"contract": baseline_contract, "receipt": baseline, "review": review, "arc_review": arc_review,
        **hashes, "run_id": paired["baseline_run_id"]}


def _start_cpu(backend, contract, monitor, logs, psutil, arm_id):
    argv = [backend["exe"], "--serve", "--blob", contract["arc"]["blob_path"], "--blob-sha256",
        contract["arc"]["blob_sha256"], "--identity", contract["arc"]["identity_path"], "--identity-sha256",
        contract["arc"]["identity_sha256"], "--pool-workers", str(backend["pool_workers"]),
        "--pool-affinity", backend["pool_affinity"]]
    t0 = time.perf_counter_ns()
    worker = RUN42.transport.OwnedPipeWorker(argv, _worker_stderr_path(logs, arm_id, backend["id"]),
        request_timeout=contract["request_timeout_seconds"], shutdown_timeout=5.0).start()
    tree = None
    try:
        tree = DirectCpuTree(worker, psutil, Path(backend["exe"]), argv)
        state = tree.snapshot(); monitor.attach(worker, tree); monitor.require()
        return {"role": backend["id"], "worker": worker, "tree": tree, "argv": argv,
            "launch_perf_counter_ns": t0,
            "process_startup_wall_ns": time.perf_counter_ns() - t0,
            "launcher_identity": state["launcher"], "actual_service_process": state["actual_service_process"],
            "service_tree_topology": "direct-child"}
    except BaseException:
        if tree is not None:
            try: tree.cleanup_descendants()
            except Exception: pass
        try: worker.close()
        except Exception: pass
        if tree is not None:
            cleanup = tree.cleanup_descendants()
            if cleanup.get("terminal") is not True: raise CompareError("CPU startup cleanup did not reach terminal state")
        raise


def _start_arc(contract, monitor, logs, psutil, arm_id, existing_handles=()):
    arc = contract["arc"]; argv = list(arc["service_argv_template"]); t0 = time.perf_counter_ns()
    worker = RUN42.transport.OwnedPipeWorker(argv, _worker_stderr_path(logs, arm_id, "arc-gpu0-f32-accuracy"),
        request_timeout=contract["request_timeout_seconds"], shutdown_timeout=5.0).start()
    tree = None
    try:
        tree = RUN42.OwnedServiceTree(worker, psutil, Path(arc["launcher"]), Path(arc["actual_child"]),
            Path(contract["source_pins"]["arc_service_path"]), argv[1:])
        handle = {"role": "arc-gpu0-f32-accuracy", "worker": worker, "tree": tree, "argv": argv,
            "launch_perf_counter_ns": t0,
            "process_startup_wall_ns": time.perf_counter_ns() - t0,
            "launcher_identity": None, "actual_service_process": None,
            "service_tree_topology": "venv Python launcher with verified C:\\Python314 child"}
        handles = [*existing_handles, handle]
        if len(handles) > 1:
            monitor.attach(WorkerGroup(handles), TreeGroup(handles))
        else:
            monitor.attach(worker, tree)
        state = tree.await_actual_child(timeout=20.0); monitor.require()
        handle["launcher_identity"] = _process_with_utc(state["launcher"])
        handle["actual_service_process"] = _process_with_utc(state["actual_service_child"])
        return handle
    except BaseException:
        if tree is not None:
            try: tree.cleanup_descendants()
            except Exception: pass
        try: worker.close()
        except Exception: pass
        if tree is not None:
            cleanup = tree.cleanup_descendants()
            if cleanup.get("terminal") is not True: raise CompareError("Arc startup cleanup did not reach terminal state")
        raise


def _init(handle, rows, contract, arc_contract, identity, backend, nonce, monitor, protocol, mode, np, records):
    payload = init_payload(nonce, rows, contract["arc"]["identity_sha256"])
    monitor.require(); request_id = handle.setdefault("request_id", 0) + 1; handle["request_id"] = request_id
    start = time.perf_counter_ns()
    frame = handle["worker"].request(protocol.INIT_KIND, request_id, payload,
        timeout=contract["request_timeout_seconds"])
    body, worker_ns = parse_result_payload(protocol.INIT_KIND, request_id, frame)
    client_ns = time.perf_counter_ns() - start
    ready = json.loads(body.decode("utf-8"))
    if mode == "cpu": validate_cpu_ready(ready, contract, backend, nonce, rows)
    else: RUN42.validate_ready(ready, arc_contract, identity, nonce, rows)
    monitor.require()
    result = {"rows": rows, "nonce": nonce, "request_id": request_id, "client_rtt_ns": client_ns,
        "worker_wall_ns": worker_ns, "shape_compile_ns": ready.get("compile_ns"), "ready": ready}
    if not handle.get("first_init_completed"):
        result["launch_to_first_init_ready_ns"] = time.perf_counter_ns() - handle["launch_perf_counter_ns"]
        result["launch_to_first_init_scope"] = "includes service/blob/pool or Arc runtime/weight startup and first shape INIT; excluded from inference latency"
        handle["first_init_completed"] = True
    records.append(result)
    return result


def _trial(handle, mode, backend, arm, rows, input_record, gold_record, contract,
           monitor, protocol, np, worker_api, label, barrier=None, output_writer=None,
           post_transfer_callback=None):
    request_id = handle.setdefault("request_id", 0) + 1; handle["request_id"] = request_id
    backend_id = backend["id"] if mode == "cpu" else "arc-gpu0-f32-accuracy"
    base = Path(contract["output"]["private_output_root"]) / _safe_component(arm) / _safe_component(backend_id) / f"rows-{rows:03d}" / _safe_component(label)
    result = _timed_infer(handle["worker"], protocol.INFER_KIND, request_id, rows,
        input_record["array"], input_record["bytes"], input_record["sha256"], gold_record["bytes"], gold_record["array"],
        backend_id, arm, monitor, np, worker_api,
        cpu_gold_sha=gold_record["sha256"] if mode == "cpu" else None, barrier=barrier,
        output_base=base, output_writer=output_writer, post_transfer_callback=post_transfer_callback)
    result["label"] = label
    return result


def _require_trial_gate(result, mode, label):
    if (result.get("timed_input_hash_match") is not True or result.get("output_shape_ok") is not True or
        result.get("finite") is not True or result.get("owned_output_copy") is not True or
        result.get("output_storage_complete") is not True or result.get("resource_gate_pass_after_transfer") is not True or
        result.get("paired_transfer_complete") is not True):
        raise CompareError(f"inference response integrity failed for rows={result.get('rows')} label={label}: {result}")
    if result.get("comparison_pass") is not True:
        raise CompareError(f"inference comparison failed for rows={result.get('rows')} label={label}: {result}")


def _run_single(handle, mode, backend, arm, contract, arc_contract, identity, inputs, gold,
                monitor, protocol, np, worker_api, rows_result=None, on_update=None):
    if rows_result is None: rows_result = []
    for item in contract["row_cases"]:
        rows = item["rows"]; nonce = secrets.token_hex(16); init_result = []
        _init(handle, rows, contract, arc_contract, identity, backend, nonce, monitor, protocol, mode, np, init_result)
        one = {"rows": rows, "init": init_result[0], "warmup": None, "formals": []}
        rows_result.append(one)
        if on_update is not None: on_update()
        one["warmup"] = _trial(handle, mode, backend, arm, rows, inputs[rows], gold[rows], contract,
                               monitor, protocol, np, worker_api, "warmup")
        if on_update is not None: on_update()
        _require_trial_gate(one["warmup"], mode, "warmup")
        for rep in range(1, contract["formal_repeats"] + 1):
            monitor.require()
            one["formals"].append(_trial(handle, mode, backend, arm, rows, inputs[rows], gold[rows], contract,
                monitor, protocol, np, worker_api, f"formal-{rep:02d}"))
            if on_update is not None: on_update()
            _require_trial_gate(one["formals"][-1], mode, f"formal-{rep:02d}")
        one["status"] = "complete"
        if on_update is not None: on_update()
    return rows_result


def _run_paired(cpu_handle, arc_handle, cpu_backend, arm, contract, arc_contract, identity,
                inputs, gold, monitor, protocol, np, worker_api, rows_result=None, on_update=None):
    from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
    if rows_result is None: rows_result = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-arc-pair") as executor:
        for item in contract["row_cases"]:
            rows = item["rows"]; nonce = secrets.token_hex(16); cpu_init = []; arc_init = []
            _init(cpu_handle, rows, contract, arc_contract, identity, cpu_backend, nonce, monitor, protocol, "cpu", np, cpu_init)
            _init(arc_handle, rows, contract, arc_contract, identity, cpu_backend, nonce, monitor, protocol, "arc", np, arc_init)
            paired = {"rows": rows, "nonce": nonce, "cpu_init": cpu_init[0], "arc_init": arc_init[0],
                      "warmup": None, "formals": []}
            rows_result.append(paired)
            if on_update is not None: on_update()
            labels = ["warmup"] + [f"formal-{i:02d}" for i in range(1, contract["formal_repeats"] + 1)]
            for label in labels:
                monitor.require(); barrier = threading.Barrier(2)
                cpu_future = executor.submit(_trial, cpu_handle, "cpu", cpu_backend, arm, rows,
                    inputs[rows], gold[rows], contract, monitor, protocol, np, worker_api, label, barrier)
                arc_future = executor.submit(_trial, arc_handle, "arc", cpu_backend, arm, rows,
                    inputs[rows], gold[rows], contract, monitor, protocol, np, worker_api, label, barrier)
                done, pending = wait((cpu_future, arc_future), return_when=FIRST_EXCEPTION)
                failures = []
                for side, future in (("cpu", cpu_future), ("arc", arc_future)):
                    if future in done and future.exception() is not None:
                        failures.append(f"{side}: {type(future.exception()).__name__}: {future.exception()}")
                if failures:
                    barrier.abort()
                    for future, handle in ((cpu_future, cpu_handle), (arc_future, arc_handle)):
                        if future in pending:
                            try: handle["worker"].close()
                            except Exception as exc: failures.append(f"{handle['role']} peer cleanup: {type(exc).__name__}: {exc}")
                pair = {"label": label, "cpu": None, "arc": None}
                for side, future in (("cpu", cpu_future), ("arc", arc_future)):
                    try: pair[side] = future.result()
                    except BaseException as exc:
                        message = f"{side}: {type(exc).__name__}: {exc}"
                        if message not in failures: failures.append(message)
                if pair["cpu"] is not None and pair["arc"] is not None:
                    pair["start_skew_ns"] = abs(pair["cpu"]["paired_start_perf_counter_ns"] - pair["arc"]["paired_start_perf_counter_ns"])
                    pair["end_skew_ns"] = abs(pair["cpu"]["paired_end_perf_counter_ns"] - pair["arc"]["paired_end_perf_counter_ns"])
                if failures: pair["transfer_errors"] = failures
                if label == "warmup": paired["warmup"] = pair
                else: paired["formals"].append(pair)
                if on_update is not None: on_update()
                if failures: raise CompareError("paired transfer failed: " + "; ".join(failures))
                _require_trial_gate(pair["cpu"], "cpu", label)
                _require_trial_gate(pair["arc"], "arc", label)
                monitor.require()
            paired["status"] = "complete"
            if on_update is not None: on_update()
    return rows_result


def _graceful_close(handles, monitor):
    errors = []
    for handle in handles:
        proc = handle["worker"]._proc
        if proc is not None and proc.poll() is None and proc.stdin is not None:
            try: proc.stdin.close()
            except (OSError, ValueError): pass
    for handle in handles:
        proc = handle["worker"]._proc
        if proc is not None and proc.poll() is None:
            try: proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired: pass
    closer = WorkerGroup(handles) if len(handles) > 1 else handles[0]["worker"]
    try: closer.close()
    except Exception as exc: errors.append(f"worker close: {type(exc).__name__}: {exc}")
    cleanup = {}
    for handle in handles:
        try:
            result = handle["tree"].cleanup_descendants()
            cleanup[handle["role"]] = result
            if result.get("terminal") is not True: errors.append(f"{handle['role']} tree did not reach terminal state")
        except Exception as exc: errors.append(f"{handle['role']} tree cleanup: {type(exc).__name__}: {exc}")
        if handle["worker"].returncode is None: errors.append(f"{handle['role']} launcher remains alive")
    monitor.attach(None, None)
    if errors: raise CompareError("owned phase cleanup incomplete: " + "; ".join(errors))
    return cleanup


def _flatten_solo_rows(arm):
    return {row["rows"]: row for row in arm.get("rows", [])}


def _contention_deltas(arms, baseline_arms=None, baseline_run_id=None, joint_run_id=None):
    """Describe latency deltas only; separately executed baselines are labeled cross-run."""
    by_id = {a["id"]: a for a in arms}
    baseline_by_id = {a["id"]: a for a in (baseline_arms if baseline_arms is not None else arms)}
    cross_run = baseline_arms is not None
    deltas = []
    for backend in ("cpu-pool-w10-all", "cpu-pool-w16-none"):
        cpu_solo = baseline_by_id.get(backend + "-solo")
        arc_solo = baseline_by_id.get("arc-gpu0-f32-accuracy-solo")
        concurrent = by_id.get(backend + "+arc-gpu0-concurrent")
        if (cpu_solo is None or arc_solo is None or concurrent is None or
            cpu_solo.get("status") != "complete" or arc_solo.get("status") != "complete" or
            concurrent.get("status") != "complete"):
            continue
        cpu_rows = _flatten_solo_rows(cpu_solo)
        arc_rows = _flatten_solo_rows(arc_solo)
        for joint in concurrent["rows"]:
            cpu_row = cpu_rows.get(joint["rows"]); arc_row = arc_rows.get(joint["rows"])
            if cpu_row is None or arc_row is None: continue
            solo_trials = {
                "cpu": {"warmup": cpu_row["warmup"], **{x["label"]: x for x in cpu_row["formals"]}},
                "arc": {"warmup": arc_row["warmup"], **{x["label"]: x for x in arc_row["formals"]}},
            }
            joint_trials = [joint["warmup"], *joint["formals"]]
            for pair in joint_trials:
                for side in ("cpu", "arc"):
                    baseline_trial = solo_trials[side].get(pair["label"])
                    if baseline_trial is not None:
                        baseline = baseline_trial["caller_boundary_rtt_ns"]
                        during = pair[side]["caller_boundary_rtt_ns"]
                        deltas.append({"backend": backend, "rows": joint["rows"], "label": pair["label"],
                            "service": side, "solo_ns": baseline, "concurrent_ns": during,
                            "delta_ns": during - baseline,
                            "baseline_run_id": baseline_run_id or joint_run_id,
                            "joint_run_id": joint_run_id,
                            "comparison_mode": "cross-run baseline; not interleaved" if cross_run else "same-run sequential baseline",
                            "interpretation": "latency contention only; no bandwidth inference"})
    return deltas


def execute(args, contract, authorization):
    import numpy as np
    import psutil
    from tools import hetero_xpu_service as arc_service, hetero_xpu_worker as worker_api
    from tools.hetero_resources import memory_snapshot

    execution_mode = contract.get("execution_mode", "full")
    if execution_mode not in {"full", "paired_only"} or bool(args.paired_only) != (execution_mode == "paired_only"):
        raise CompareError("execution selector does not match the authorized comparison contract")
    if contract.get("warmup") != 1 or contract.get("formal_repeats") != 5 or [r.get("rows") for r in contract.get("row_cases", [])] != list(ROWS):
        raise CompareError("comparison contract differs from the reviewed one-warmup/five-formal nine-shape matrix")
    if contract.get("quality_tolerances") != worker_api.DEFAULT_TOLERANCES:
        raise CompareError("comparison tolerances differ from the pinned run42 worker defaults")

    output_root = Path(contract["output"]["private_output_root"])
    receipt_dir = Path(contract["output"]["receipt_dir"])
    resource_path = Path(contract["output"]["resource_log"])
    receipt_path = receipt_dir / "compare-receipt.json"
    if output_root.exists() or receipt_dir.exists() or resource_path.exists():
        raise FileExistsError(f"refusing an existing {contract['run_id']} output/receipt/resource path")
    gate = contract["resource_gate"]
    if not output_root.parent.is_dir() or shutil.disk_usage(output_root.parent).free < gate["minimum_free_disk_bytes"]:
        raise CompareError("private E output parent absent or free-space floor not met")
    initial, errors = memory_snapshot()
    if (errors or type(initial.get("physical_available_bytes")) is not int or
        type(initial.get("commit_available_bytes")) is not int or
        initial["physical_available_bytes"] < gate["minimum_physical_bytes"] or
        initial["commit_available_bytes"] < gate["minimum_commit_bytes"]):
        raise CompareError("fresh PSAPI 12/4 resource gate failed or unknown")
    source_pins = _check_source_pins(contract)
    selected = [b for b in contract["backends"] if not b["optional"] or args.include_cpu16_none]
    if execution_mode == "paired_only":
        expected_joint = set(contract.get("paired_only", {}).get("joint_backend_ids", []))
        if {backend["id"] for backend in selected} != expected_joint:
            raise CompareError("paired-only selected backends differ from the authorized joint-arm set")
    cpu_binaries = {b["id"]: _verify_cpu_binary(contract, b) for b in selected}
    paired_baseline = _verify_paired_baseline(contract) if execution_mode == "paired_only" else None

    output_root.mkdir(parents=False, exist_ok=False)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    resource_path.parent.mkdir(parents=False, exist_ok=False)
    logs = receipt_dir / "logs"; logs.mkdir(exist_ok=False)
    record = {"schema_version": 1, "status": "running", "run_id": contract["run_id"], "execution_mode": execution_mode,
        "started_utc": utc(),
        "root_authorization": authorization, "resource_gate_initial": initial, "source_pins": source_pins,
        "controller_identity": _process_with_utc(RUN42._process_record(psutil.Process(os.getpid()))),
        "cpu_binaries": cpu_binaries, "arc_identity_sha256": contract["arc"]["identity_sha256"],
        "native_blob_sha256": contract["arc"]["blob_sha256"], "verified_inputs": None,
        "verified_native_cpu_gold": None, "asset_stats": None, "weights_header": None,
        "arms": [], "model_route_enabled": False,
        "timing_contract": contract["timing_contract"], "quality_tolerances": contract["quality_tolerances"],
        "raw_output_files_written": False, "large_f32_outputs_written": False}
    if paired_baseline is not None:
        record["paired_only_baseline"] = {key: paired_baseline[key] for key in (
            "run_id", "baseline_contract_sha256", "baseline_receipt_sha256", "baseline_client_sha256",
            "root_solo_review_sha256", "root_arc_numerical_review_sha256")}
        record["paired_only_baseline"]["comparison_scope"] = "cross-run solo baseline; not interleaved with run61 joint arms"
    monitor = RUN42.ResourceMonitor(resource_path, psutil,
        minimum_physical=gate["minimum_physical_bytes"], minimum_commit=gate["minimum_commit_bytes"])
    handles = []
    with receipt_path.open("x+", encoding="utf-8", newline="\n") as receipt:
        _update_receipt(receipt, record)
        try:
            monitor.start(); monitor.require()
            # Artifact hashing/loading runs only after authorization and under the live observer.
            arc_contract_path = Path(contract["artifacts"]["run42_contract_path"])
            if _stable_sha(arc_contract_path) != contract["artifacts"]["run42_contract_sha256"]:
                raise CompareError("frozen run42 artifact contract changed")
            arc_contract = json.loads(arc_contract_path.read_text(encoding="utf-8-sig"))
            identity, header, inputs, gold, asset_stats = RUN42.load_artifacts(arc_contract, np, worker_api, arc_service)
            if (header.get("hidden"), header.get("intermediate")) != (2560, 640):
                raise CompareError("Arc NPZ matrix geometry differs from the native CPU operator")
            if sorted(inputs) != list(ROWS) or sorted(gold) != list(ROWS):
                raise CompareError("verified input/gold set is not the frozen nine-shape matrix")
            monitor.require()
            record.update({"verified_inputs": {str(k): v["stat"] for k, v in inputs.items()},
                "verified_native_cpu_gold": {str(k): v["stat"] for k, v in gold.items()},
                "asset_stats": asset_stats, "weights_header": header})
            _update_receipt(receipt, record)
            if execution_mode != "paired_only":
                # Full runs measure their solo baseline arms in this same run.
                for backend in selected:
                    arm = {"id": backend["id"] + "-solo", "kind": "cpu_solo", "backend": backend["id"],
                           "rows": [], "status": "running"}
                    record["arms"].append(arm); _update_receipt(receipt, record)
                    handle = _start_cpu(backend, contract, monitor, logs, psutil, arm["id"]); handles = [handle]
                    monitor.attach(handle["worker"], handle["tree"])
                    arm["process_start"] = {k: handle[k] for k in ("argv", "process_startup_wall_ns", "launcher_identity",
                                                                     "actual_service_process", "service_tree_topology")}
                    try:
                        _run_single(handle, "cpu", backend, arm["id"], contract, arc_contract,
                            identity, inputs, gold, monitor, arc_service, np, worker_api, arm["rows"],
                            lambda: _persist_progress(receipt, record))
                        arm["status"] = "complete"
                    finally:
                        arm["cleanup"] = _graceful_close(handles, monitor); handles = []
                    _update_receipt(receipt, record); monitor.require()

                arc_arm = {"id": "arc-gpu0-f32-accuracy-solo", "kind": "arc_solo", "rows": [], "status": "running"}
                record["arms"].append(arc_arm); _update_receipt(receipt, record)
                arc_handle = _start_arc(contract, monitor, logs, psutil, arc_arm["id"]); handles = [arc_handle]
                arc_arm["process_start"] = {k: arc_handle[k] for k in ("argv", "process_startup_wall_ns", "launcher_identity",
                                                                        "actual_service_process", "service_tree_topology")}
                try:
                    _run_single(arc_handle, "arc", None, arc_arm["id"], contract, arc_contract,
                        identity, inputs, gold, monitor, arc_service, np, worker_api, arc_arm["rows"],
                        lambda: _persist_progress(receipt, record))
                    arc_arm["status"] = "complete"
                finally:
                    arc_arm["cleanup"] = _graceful_close(handles, monitor); handles = []
                _update_receipt(receipt, record); monitor.require()

            # Both processes remain attached to one ResourceMonitor through the synchronized pairs.
            for backend in selected:
                arm = {"id": backend["id"] + "+arc-gpu0-concurrent", "kind": "synchronized_cpu_arc_concurrent",
                       "backend": backend["id"], "rows": [], "status": "running"}
                record["arms"].append(arm); _update_receipt(receipt, record)
                cpu_handle = _start_cpu(backend, contract, monitor, logs, psutil, arm["id"])
                handles = [cpu_handle]; monitor.attach(cpu_handle["worker"], cpu_handle["tree"])
                try:
                    arc_handle = _start_arc(contract, monitor, logs, psutil, arm["id"], existing_handles=handles)
                except BaseException:
                    arm["cleanup"] = _graceful_close(handles, monitor); handles = []
                    raise
                handles = [cpu_handle, arc_handle]
                arm["process_start"] = [{k: h[k] for k in ("role", "argv", "process_startup_wall_ns", "launcher_identity",
                                                              "actual_service_process", "service_tree_topology")} for h in handles]
                try:
                    _run_paired(cpu_handle, arc_handle, backend, arm["id"], contract, arc_contract,
                        identity, inputs, gold, monitor, arc_service, np, worker_api, arm["rows"],
                        lambda: _persist_progress(receipt, record))
                    arm["status"] = "complete"
                finally:
                    arm["cleanup"] = _graceful_close(handles, monitor); handles = []
                _update_receipt(receipt, record); monitor.require()
            record["solo_vs_concurrent_latency_deltas"] = _contention_deltas(record["arms"],
                baseline_arms=paired_baseline["receipt"]["arms"] if paired_baseline else None,
                baseline_run_id=paired_baseline["run_id"] if paired_baseline else None,
                joint_run_id=contract["run_id"])
            record["contention_interpretation"] = (
                "cross-run solo baseline deltas; solo and joint arms were not interleaved; latency contention only, no memory-bandwidth claim"
                if paired_baseline else
                "same-run sequential solo baseline deltas; paired latency only, no memory-bandwidth claim without a measured hardware counter")
            source_after = _check_source_pins(contract)
            record["source_sha256_after"] = source_after
            if source_after != source_pins: raise CompareError("pinned code sources changed during comparison")
            record["source_unchanged_during_run"] = True
            record["status"] = "complete"
        except BaseException as exc:
            record["status"] = "failed"
            record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1200], "failed_utc": utc(),
                                 "no_restart": True}
            if handles:
                try: record["failure_cleanup"] = _graceful_close(handles, monitor)
                except Exception as cleanup_exc:
                    record["failure_cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                handles = []
        finally:
            try: monitor.close()
            except Exception as exc:
                record["resource_monitor_close_error"] = f"{type(exc).__name__}: {exc}"; record["status"] = "failed"
            try:
                samples = [json.loads(line) for line in resource_path.read_text(encoding="utf-8").splitlines() if line]
                samples = [row for row in samples if row.get("record_type") == "resource_sample"]
                record["resource_gate_samples"] = len(samples)
                record["resource_gate_all_samples_passed"] = bool(samples) and all(row.get("gate_ok") is True for row in samples)
                record["physical_available_min_bytes"] = min((row["physical_available_bytes"] for row in samples), default=None)
                record["commit_available_min_bytes"] = min((row["commit_available_bytes"] for row in samples), default=None)
                if not record["resource_gate_all_samples_passed"]: record["status"] = "failed"
            except Exception as exc:
                record["resource_gate_summary_error"] = f"{type(exc).__name__}: {exc}"; record["status"] = "failed"
            record["finished_utc"] = utc(); _update_receipt(receipt, record)
    return 0 if record["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH,
        help="comparison contract JSON (defaults to the preserved run53 recipe)")
    parser.add_argument("--execute", action="store_true", help="run only after explicit root review and authorization")
    parser.add_argument("--paired-only", action="store_true",
        help="run only concurrent arms against a separately reviewed solo baseline")
    parser.add_argument("--root-start-confirmed", action="store_true")
    parser.add_argument("--model-matrix-stopped-confirmed", action="store_true")
    parser.add_argument("--root-authorization", type=Path)
    parser.add_argument("--include-cpu16-none", action="store_true", help="add distinct optional w16/none arms")
    args = parser.parse_args(argv)
    if not args.contract.is_absolute():
        args.contract = args.contract.resolve()
    contract = load_contract(args.contract)
    if not args.execute:
        print(json.dumps(plan_record(contract), indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    authorization = _authorization(args, contract, args.contract)
    return execute(args, contract, authorization)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"hetero_native_ipc_compare: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
