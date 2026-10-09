#!/usr/bin/env python3
"""Plan or explicitly run one owned native-Q4_K/Q5_1 IPC worker on GPU.0."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools import hetero_xpu_transport as transport

RUN = Path(__file__).resolve().parent
CONTRACT = RUN / "execution-contract.json"
RESULT_PREFIX = struct.Struct("<IQ")
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
REQUEST_TIMEOUT = 120.0
RESOURCE_INTERVAL = 1.0


class BenchError(RuntimeError):
    pass


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise BenchError(f"receipt path must be absolute with an existing parent: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def update_receipt(stream, record: dict[str, Any]) -> None:
    stream.seek(0); json.dump(record, stream, indent=2, ensure_ascii=True, allow_nan=False)
    stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())


def make_plan(c: dict[str, Any]) -> dict[str, Any]:
    return {"status": "plan_only", "run_id": c["run_id"], "execution_authorized": c["execution_authorized"],
            "rows": [item["rows"] for item in c["row_cases"]], "operator": c["operator"],
            "device": c["device"], "precision": c["precision"], "warmup": c["warmup"],
            "formal_repeats": c["formal_repeats"], "output_root": c["output_root"],
            "kernel_execution": False, "core_created": False, "child_started": False,
            "payload_read": False, "argv": c["service_argv_template"]}


def _sha_file_stable(path: Path, expected_bytes: int | None = None,
                     expected_sha: str | None = None) -> tuple[bytes, dict[str, Any]]:
    from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged

    before = file_stat(path)
    if expected_bytes is not None and before["size_bytes"] != expected_bytes:
        raise BenchError(f"size mismatch for {path}: {before['size_bytes']} != {expected_bytes}")
    digest = hashlib.sha256(); chunks = []
    with path.open("rb", buffering=0) as stream:
        if not fd_matches_path(os.fstat(stream.fileno()), before):
            raise BenchError(f"path/fd identity mismatch before read: {path}")
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block); chunks.append(block)
        final_fd = os.fstat(stream.fileno())
    after = file_stat(path)
    if not stat_unchanged(before, after) or not fd_matches_path(final_fd, before):
        raise BenchError(f"file identity changed during read: {path}")
    actual_sha = digest.hexdigest()
    if expected_sha is not None and actual_sha != expected_sha:
        raise BenchError(f"SHA-256 mismatch for {path}: {actual_sha} != {expected_sha}")
    return b"".join(chunks), {**before, "path": str(path), "sha256": actual_sha}


def _process_record(proc) -> dict[str, Any]:
    return {"pid": int(proc.pid), "parent_pid": int(proc.ppid()), "create_time": float(proc.create_time()),
            "executable": os.path.normcase(os.path.realpath(proc.exe())), "command_line": list(proc.cmdline())}


class OwnedServiceTree:
    """Observe and reap only descendants of this run's OwnedPipeWorker launcher."""

    def __init__(self, worker, psutil, wrapper_path: Path, child_path: Path,
                 service_script: Path, expected_args: list[str]):
        self.worker, self.psutil = worker, psutil
        self.wrapper_path = os.path.normcase(os.path.realpath(wrapper_path))
        self.child_path = os.path.normcase(os.path.realpath(child_path))
        self.service_script = os.path.normcase(os.path.realpath(service_script))
        self.expected_args = expected_args
        self.records: dict[tuple[int, float], dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.service_child: dict[str, Any] | None = None
        root = psutil.Process(worker.pid)
        self.root_record = _process_record(root)
        if self.root_record["executable"] != self.wrapper_path:
            raise BenchError(f"OwnedPipeWorker launcher executable mismatch: {self.root_record}")
        self.records[(self.root_record["pid"], self.root_record["create_time"])] = self.root_record

    def _is_actual_child(self, row: dict[str, Any]) -> bool:
        if row["executable"] != self.child_path or row["parent_pid"] != self.worker.pid:
            return False
        args = row["command_line"]
        if not args:
            return False
        argv0 = os.path.normcase(os.path.realpath(args[0]))
        if argv0 not in {self.child_path, self.wrapper_path} or args[1:] != self.expected_args:
            return False
        scripts = [os.path.normcase(os.path.realpath(x)) for x in args[1:]
                   if os.path.isabs(x) and x.lower().endswith(".py")]
        return self.service_script in scripts

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            try:
                root = self.psutil.Process(self.worker.pid)
                current_root = _process_record(root)
                if not self._same(self.root_record, current_root):
                    raise BenchError("worker launcher PID identity changed")
                processes = [root] + root.children(recursive=True)
            except self.psutil.NoSuchProcess:
                processes = []
            for proc in processes:
                try:
                    row = _process_record(proc)
                except self.psutil.NoSuchProcess:
                    continue
                key = (row["pid"], row["create_time"])
                old = self.records.get(key)
                if old is not None and not self._same(old, row):
                    raise BenchError(f"owned process identity changed for PID {row['pid']}")
                self.records[key] = row
            children = [r for r in self.records.values() if r["pid"] != self.worker.pid]
            live_service = []
            for row in children:
                try:
                    current = _process_record(self.psutil.Process(row["pid"]))
                except self.psutil.NoSuchProcess:
                    continue
                if self._same(row, current) and self._is_actual_child(row):
                    live_service.append(row)
            if len(live_service) > 1:
                raise BenchError("more than one matching native service interpreter child is alive")
            if live_service:
                self.service_child = live_service[0]
            return {"launcher": self.root_record, "descendants": children,
                    "actual_service_child": self.service_child, "live_service_children": len(live_service)}

    @staticmethod
    def _same(a, b) -> bool:
        return (a["pid"], a["create_time"], a["executable"], a["command_line"]) == (
            b["pid"], b["create_time"], b["executable"], b["command_line"])

    @staticmethod
    def _same_process_identity(a, b) -> bool:
        # Parent PID can change to the OS reaper after launcher exit; create-time/exe/argv still bind the process.
        return (a["pid"], a["create_time"], a["executable"], a["command_line"]) == (
            b["pid"], b["create_time"], b["executable"], b["command_line"])

    def await_actual_child(self, timeout: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.snapshot()
            if state["actual_service_child"] is not None:
                return state
            if self.worker.returncode is not None:
                raise BenchError(f"service launcher exited before its C Python child was identified: {self.worker.returncode}")
            time.sleep(0.1)
        raise BenchError("timed out locating the owned C:\\Python314 service child")

    def cleanup_descendants(self) -> dict[str, Any]:
        # Refresh while the launcher is available so a just-created child is bound before teardown.
        try:
            self.snapshot()
        except self.psutil.NoSuchProcess:
            pass
        results = []
        with self.lock:
            known = list(self.records.values())
        for record in reversed(known):
            if record["pid"] == self.worker.pid:
                continue  # OwnedPipeWorker.close() exclusively reaps its Popen root.
            outcome = {"identity": record, "action": "already_exited"}
            try:
                proc = self.psutil.Process(record["pid"])
                current = _process_record(proc)
                if not self._same_process_identity(record, current):
                    outcome["action"] = "identity_changed_pid_not_touched"
                else:
                    proc.terminate(); outcome["action"] = "terminate_exact_descendant"
                    try:
                        proc.wait(timeout=3.0)
                    except self.psutil.TimeoutExpired:
                        current = _process_record(self.psutil.Process(record["pid"]))
                        if self._same_process_identity(record, current):
                            self.psutil.Process(record["pid"]).kill(); outcome["action"] = "kill_exact_descendant_after_timeout"
                            self.psutil.Process(record["pid"]).wait(timeout=3.0)
                        else:
                            outcome["action"] = "identity_changed_after_timeout_pid_not_touched"
                    outcome["exit_code"] = proc.wait(timeout=0.1)
            except self.psutil.NoSuchProcess:
                outcome["action"] = "exited_during_cleanup"
            except Exception as exc:
                outcome["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            results.append(outcome)
        remaining = []
        for record in known:
            if record["pid"] == self.worker.pid:
                continue
            try:
                current = _process_record(self.psutil.Process(record["pid"]))
                if self._same_process_identity(record, current):
                    remaining.append(record)
            except self.psutil.NoSuchProcess:
                pass
        return {"descendant_cleanup": results, "remaining_owned_descendants": remaining,
                "terminal": not remaining and self.worker.returncode is not None}


class ResourceMonitor:
    def __init__(self, path: Path, psutil, *, minimum_physical: int, minimum_commit: int):
        self.path, self.psutil = path, psutil
        self.minimum_physical, self.minimum_commit = minimum_physical, minimum_commit
        self.stop_event = threading.Event(); self.first_sample = threading.Event()
        self.failed: dict[str, Any] | None = None; self.startup_error: str | None = None
        self.tree: OwnedServiceTree | None = None; self.worker = None
        self.lock = threading.RLock(); self.last: dict[str, Any] | None = None
        self.stream = path.open("x", encoding="utf-8", buffering=1)
        self.thread = threading.Thread(target=self._loop, name="native-ipc-resource-watch", daemon=True)

    def start(self, timeout: float = 15.0):
        self.thread.start()
        if not self.first_sample.wait(timeout):
            with self.lock: self.startup_error = f"first resource sample timed out after {timeout}s"
            self.stop_event.set(); self.thread.join(timeout=2.0)
            raise BenchError(self.startup_error)
        with self.lock:
            if self.startup_error is not None:
                raise BenchError(f"resource monitor startup failed: {self.startup_error}")
    def attach(self, worker, tree):
        with self.lock: self.worker, self.tree = worker, tree

    def _stop_owned_after_gate(self, row, worker, tree):
        if tree is not None:
            try:
                row["owned_descendant_gate_cleanup"] = tree.cleanup_descendants()
            except Exception as exc:
                row["owned_descendant_gate_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if worker is not None:
            try:
                worker.close()
                row["owned_worker_gate_cleanup"] = {"launcher_pid": worker.pid, "returncode": worker.returncode}
            except Exception as exc:
                row["owned_worker_gate_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        cleanup = {k: v for k, v in row.items() if "cleanup" in k}
        self.stream.write(json.dumps({"record_type": "gate_cleanup", "observed_at_utc": utc(),
                                      "cleanup": cleanup}, ensure_ascii=True, allow_nan=False) + "\n")
        self.stream.flush()

    def _loop(self):
        from tools.hetero_resources import memory_snapshot
        while not self.stop_event.is_set():
            try:
                mem, errors = memory_snapshot()
            except BaseException as exc:
                with self.lock: self.startup_error = f"{type(exc).__name__}: {exc}"
                self.first_sample.set(); self.stop_event.set(); break
            physical, commit = mem.get("physical_available_bytes"), mem.get("commit_available_bytes")
            with self.lock: tree = self.tree
            try:
                tree_info = tree.snapshot() if tree is not None else None
            except Exception as exc:
                tree_info = {"error": f"{type(exc).__name__}: {exc}"}
                errors = list(errors) + [{"resource": "owned_process_tree", "message": str(exc)}]
            gate_ok = (not errors and type(physical) is int and type(commit) is int and
                       physical >= self.minimum_physical and commit >= self.minimum_commit)
            row = {"record_type": "resource_sample", "observed_at_utc": utc(), "owner_pid": os.getpid(),
                   "physical_available_bytes": physical, "commit_available_bytes": commit,
                   "memory_source": mem.get("source"), "errors": errors, "gate_ok": gate_ok,
                   "owned_service_tree": tree_info}
            self.stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"); self.stream.flush()
            with self.lock: self.last = row
            self.first_sample.set()
            if not gate_ok:
                with self.lock: self.failed = row; worker, tree = self.worker, self.tree
                self._stop_owned_after_gate(row, worker, tree)
                self.stop_event.set(); break
            self.stop_event.wait(1.0)

    def require(self):
        with self.lock: failure, row, startup_error = self.failed, self.last, self.startup_error
        if startup_error is not None: raise BenchError(f"resource monitor failed: {startup_error}")
        if failure is not None: raise BenchError(f"12/4 GiB resource gate failed: {failure}")
        if row is None or not row["gate_ok"]: raise BenchError("resource gate is absent, stale, or failed")
        observed = dt.datetime.fromisoformat(row["observed_at_utc"])
        if abs((dt.datetime.now(dt.timezone.utc) - observed).total_seconds()) > 5:
            raise BenchError("resource sample is stale")
        return row

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=5.0)
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        if self.thread.is_alive(): raise BenchError("resource sampler thread failed to stop")


def read_verified(path: Path, expected_bytes: int, expected_sha: str):
    from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged
    before = file_stat(path)
    if before["size_bytes"] != expected_bytes: raise BenchError(f"size mismatch: {path}")
    with path.open("rb", buffering=0) as stream:
        if not fd_matches_path(os.fstat(stream.fileno()), before): raise BenchError(f"file ID mismatch: {path}")
        raw = stream.read(expected_bytes + 1); last = os.fstat(stream.fileno())
    after = file_stat(path)
    if len(raw) != expected_bytes or not stat_unchanged(before, after) or not fd_matches_path(last, before):
        raise BenchError(f"file changed while reading: {path}")
    if sha(raw) != expected_sha: raise BenchError(f"SHA-256 mismatch: {path}")
    return raw, before


def load_artifacts(c, np, worker_api, service):
    input_receipt_path = Path(c["inputs_receipt"])
    if sha(input_receipt_path.read_bytes()) != c["inputs_receipt_sha256"]:
        raise BenchError("frozen run10 inputs receipt changed")
    summary_path = Path(c["native_cpu_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if sha(summary_path.read_bytes()) != c["native_cpu_summary_sha256"] or summary.get("status") != c["native_cpu_summary_status"]:
        raise BenchError("frozen native CPU run summary changed or has an unexpected status")
    identity_path = Path(c["identity_path"])
    identity_raw, identity_stat = read_verified(identity_path, identity_path.stat().st_size, c["identity_sha256"])
    identity = worker_api._safe_identity(json.loads(identity_raw.decode("utf-8")))
    if identity["weights_sha256"] != c["identity_weights_sha256"]:
        raise BenchError("identity decoded-weight hashes changed")
    extraction_path = Path(c["extraction_receipt"])
    if sha(extraction_path.read_bytes()) != c["extraction_receipt_sha256"]:
        raise BenchError("expert extraction receipt changed")
    binding_path = Path(c["native_blob_binding_receipt"])
    binding_receipt = json.loads(binding_path.read_text(encoding="utf-8"))
    if sha(binding_path.read_bytes()) != c["native_blob_binding_receipt_sha256"] or binding_receipt["status"] != "pass":
        raise BenchError("native blob binding receipt changed or is not passing")
    source = identity["source"]
    if (source["quantized_source_types"] != {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"} or
        source["raw_selected_payload_sha256"] != c["native_blob_binding"]["raw_selected_payload_sha256"]):
        raise BenchError("native operator/source identity mismatch")
    blob_path = Path(c["blob_path"])
    from tools.hetero_verify_existing import file_stat
    blob_stat = file_stat(blob_path)
    expected_blob = binding_receipt["blob_stat_after"]
    if (blob_stat["size_bytes"], blob_stat["mtime_ns"], blob_stat["ctime_ns"], blob_stat["file_id"]) != (
        c["blob_bytes"], expected_blob["mtime_ns"], expected_blob["ctime_ns"], expected_blob["file_id"]):
        raise BenchError("native blob path identity differs from its scoped binding receipt")
    header = worker_api.validate_npz_metadata(Path(c["weights_npz_path"]), identity)
    if (header["hidden"], header["intermediate"]) != (2560, 640): raise BenchError("NPZ decoded matrix shape mismatch")
    inputs, native = {}, {}
    for item in c["row_cases"]:
        rows = item["rows"]
        native_receipt_path = Path(item["native_cpu_receipt_path"])
        native_receipt = json.loads(native_receipt_path.read_text(encoding="utf-8-sig"))
        if (sha(native_receipt_path.read_bytes()) != item["native_cpu_receipt_sha256"] or
            native_receipt.get("status") != "success" or native_receipt.get("execution", {}).get("native_kernel_called") is not True or
            native_receipt.get("output", {}).get("sha256") != item["native_cpu_output_sha256"] or
            item.get("native_cpu_process_exit") != 0):
            raise BenchError(f"native CPU reference receipt mismatch for rows={rows}")
        xraw, xstat = read_verified(Path(item["input_path"]), item["input_bytes"], item["input_sha256"])
        nraw, nstat = read_verified(Path(item["native_cpu_output_path"]), item["native_cpu_output_bytes"], item["native_cpu_output_sha256"])
        x = np.frombuffer(xraw, dtype="<f4").reshape(rows, 2560)
        reference = np.frombuffer(nraw, dtype="<f4").reshape(rows, 2560).copy(order="C")
        if not np.isfinite(x).all() or not np.isfinite(reference).all(): raise BenchError(f"non-finite row {rows} fixture")
        inputs[rows] = {"bytes": xraw, "array": x, "stat": xstat, "sha256": sha(xraw)}
        native[rows] = {"bytes": nraw, "array": reference, "stat": nstat,
                        "sha256": item["native_cpu_output_sha256"], "receipt": item["native_cpu_receipt_path"]}
    return identity, header, inputs, native, {"identity": identity_stat, "blob": blob_stat}


def decode_result(frame, kind: int, request_id: int):
    if frame.kind != (kind | transport.RESPONSE_BIT) or frame.request_id != request_id or len(frame.payload) < RESULT_PREFIX.size:
        raise BenchError("IPC response identity/prefix mismatch")
    status, worker_ns = RESULT_PREFIX.unpack_from(frame.payload)
    body = bytes(frame.payload[RESULT_PREFIX.size:])
    if status != 0:
        try: message = json.loads(body.decode("utf-8")).get("message", "service request failed")
        except Exception: message = "invalid error response"
        raise BenchError(f"IPC kind={kind} id={request_id}: {message}")
    return body, worker_ns


def _f32_property(value: Any) -> bool:
    return str(value).strip().casefold() in {"f32", "float32", "<type: 'float32'>", '<type: "float32">'}


def validate_ready(ready: dict[str, Any], c: dict[str, Any], identity: dict[str, Any], nonce: str, rows: int) -> None:
    bind = ready.get("native_blob_binding", {})
    if (ready.get("schema") != "strata-xpu-ready-v1" or ready.get("status") != "ready" or
        ready.get("nonce") != nonce or ready.get("identity_sha256") != c["identity_sha256"] or
        ready.get("rows") != rows or ready.get("operator") != c["operator"] or
        ready.get("operator_signature") != c["operator_signature"] or ready.get("weights_ready") is not True or
        ready.get("weights_loaded_and_hash_verified") is not True or ready.get("weights_sha256") != identity["weights_sha256"]):
        raise BenchError("INIT ready/operator/weights identity metadata mismatch")
    if (bind.get("blob_sha256") != c["blob_sha256"] or bind.get("segment_sha256_actual") != identity["source"]["raw_selected_payload_sha256"] or
        bind.get("down_offset_bytes") != 1843200 or bind.get("q5_1_minimums_sha256_le_f32") != c["native_blob_binding"]["minimums_sha256_le_f32"]):
        raise BenchError("INIT native raw-blob/minimum binding mismatch")
    if (ready.get("device") != "GPU.0" or ready.get("execution_devices") != ["GPU.0"] or
        "intel" not in ready.get("device_full_name", "").casefold() or "arc" not in ready.get("device_full_name", "").casefold()):
        raise BenchError("INIT device identity/fallback mismatch")
    if (ready.get("precision_requested") != "f32" or not _f32_property(ready.get("reported_precision")) or
        ready.get("precision_policy", {}).get("precision_policy_matches") is not True or
        ready.get("compile_property_policy", {}).get("submitted") != {"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}):
        raise BenchError("INIT F32/ACCURACY policy mismatch")
    build = ready.get("native_operator_build", {})
    if build.get("operator") != c["operator_signature"] or build.get("compiled") is not False or build.get("inference_run") is not False:
        raise BenchError("INIT reports an unexpected native operator/fallback state")


def _save_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    with path.open("xb", buffering=0) as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    stored, stat = _sha_file_stable(path, len(raw), sha(raw))
    return {"path": str(path), "bytes": len(stored), "sha256": sha(stored), "readback_verified": True, "stat": stat}


def execute(c: dict[str, Any]) -> int:
    import numpy as np
    import psutil
    from tools import hetero_xpu_service as service, hetero_xpu_worker as worker_api
    from tools.hetero_resources import memory_snapshot

    if Path(sys.executable).resolve() != Path(c["runner_interpreter"]).resolve():
        raise BenchError(f"runner must use reviewed package interpreter {c['runner_interpreter']}; got {sys.executable}")
    for path_key, digest_key in (("protocol_source", "protocol_source_sha256"),
                                 ("service_source", "service_source_sha256"),
                                 ("activation_source", "activation_source_sha256"),
                                 ("native_graph_source", "native_graph_source_sha256"),
                                 ("runner_path", "runner_source_sha256"),
                                 ("worker_api_source", "worker_api_source_sha256"),
                                 ("verify_source", "verify_source_sha256"),
                                 ("resource_source", "resource_source_sha256")):
        if sha(Path(c[path_key]).read_bytes()) != c[digest_key]:
            raise BenchError(f"reviewed source changed: {c[path_key]}")
    runroot, outroot = RUN, Path(c["output_root"])
    if outroot.exists(): raise FileExistsError(f"refusing existing E output root: {outroot}")
    if any((runroot / p).exists() for p in ("logs", "resource", "receipt.json")):
        raise FileExistsError("run42 receipt/log/resource path already exists")
    gate = c["resource_gate"]
    initial, errors = memory_snapshot()
    if errors or type(initial.get("physical_available_bytes")) is not int or type(initial.get("commit_available_bytes")) is not int or initial["physical_available_bytes"] < gate["minimum_physical_bytes"] or initial["commit_available_bytes"] < gate["minimum_commit_bytes"]:
        raise BenchError("fresh 12/4 GiB memory gate failed or unknown")
    outroot.parent.mkdir(parents=True, exist_ok=True)
    if __import__("shutil").disk_usage(outroot.parent).free < c["resource_gate"]["minimum_free_disk_bytes"]:
        raise BenchError("E: free-space floor not met")
    logs, resource_dir = runroot / "logs", runroot / "resource"
    logs.mkdir(exist_ok=False); resource_dir.mkdir(exist_ok=False); outroot.mkdir(exist_ok=False)
    record = {"schema_version": 1, "status": "running", "run_id": c["run_id"], "started_utc": utc(),
              "contract": str(CONTRACT), "operator": c["operator"], "rows": [], "resource_gate_initial": initial,
              "timing_contract": c["timing_contract"], "quality_tolerances": c["quality_tolerances"],
              "model_route_claim": False}
    receipt_path = runroot / "receipt.json"
    with receipt_path.open("x+", encoding="utf-8", newline="\n") as receipt:
        update_receipt(receipt, record)
        worker = monitor = tree = None; outputs_complete = False
        request_id = 0; last_output_failure = None
        try:
            monitor = ResourceMonitor(resource_dir / "samples.jsonl", psutil,
                minimum_physical=gate["minimum_physical_bytes"], minimum_commit=gate["minimum_commit_bytes"])
            monitor.start(); monitor.require()
            identity, header, inputs, native, asset_stats = load_artifacts(c, np, worker_api, service)
            monitor.require()
            record.update({"identity_sha256": c["identity_sha256"], "identity_weights_sha256": identity["weights_sha256"],
                           "weights_header": header, "verified_inputs": {str(k): v["stat"] for k,v in inputs.items()},
                           "verified_native_cpu_references": {str(k): v["stat"] for k,v in native.items()},
                           "native_blob_stat": asset_stats["blob"]})
            update_receipt(receipt, record)
            service_argv = list(c["service_argv_template"])
            worker = transport.OwnedPipeWorker(service_argv, logs / "service.stderr.log",
                request_timeout=REQUEST_TIMEOUT, shutdown_timeout=5.0).start()
            service_path = str(Path(c["service_source"]).resolve())
            expected_args = service_argv[1:]
            tree = OwnedServiceTree(worker, psutil, Path(c["interpreter"]["service_launcher"]),
                                    Path(c["interpreter"]["actual_child_interpreter"]), Path(service_path), expected_args)
            monitor.attach(worker, tree)
            tree_state = tree.await_actual_child()
            monitor.require()
            record["worker_start"] = {"argv": service_argv, "launcher_pid": worker.pid,
                "launcher_identity": tree_state["launcher"], "actual_service_child": tree_state["actual_service_child"]}
            update_receipt(receipt, record)

            def rpc(kind, payload):
                nonlocal request_id
                monitor.require(); request_id += 1; started = time.perf_counter_ns()
                wire_start = time.perf_counter_ns()
                frame = worker.request(kind, request_id, payload, timeout=REQUEST_TIMEOUT)
                body, worker_ns = decode_result(frame, kind, request_id)
                wire_end = time.perf_counter_ns()
                return body, worker_ns, started, wire_start, wire_end, request_id

            for item in c["row_cases"]:
                rows = item["rows"]; x = inputs[rows]["array"]; reference = native[rows]["array"]
                nonce = secrets.token_hex(16)
                init = json.dumps({"schema": service.SERVICE_INIT_SCHEMA, "nonce": nonce, "rows": rows,
                    "identity_sha256": c["identity_sha256"]}, separators=(",", ":")).encode("utf-8")
                init_body, init_worker_ns, init_start, init_wire_start, init_wire_end, init_id = rpc(service.INIT_KIND, init)
                ready = json.loads(init_body); validate_ready(ready, c, identity, nonce, rows)
                shape = {"rows": rows, "input_sha256": inputs[rows]["sha256"], "native_cpu_reference_sha256": native[rows]["sha256"],
                    "nonce": nonce, "init": {"request_id": init_id, "ready": ready, "client_rtt_ns": time.perf_counter_ns()-init_start,
                    "wire_rtt_ns": init_wire_end-init_wire_start, "worker_wall_ns": init_worker_ns}, "warmup": None, "formals": [], "status": "running"}
                record["rows"].append(shape); update_receipt(receipt, record)

                def infer_once(formal_label):
                    nonlocal request_id, last_output_failure
                    t0 = time.perf_counter_ns()
                    raw_input = np.asarray(x, dtype="<f4", order="C").tobytes(order="C")
                    encode_end = time.perf_counter_ns(); monitor.require(); request_id += 1; request_id_local = request_id
                    frame = worker.request(service.INFER_KIND, request_id_local, raw_input, timeout=REQUEST_TIMEOUT)
                    body, worker_ns = decode_result(frame, service.INFER_KIND, request_id_local)
                    received = time.perf_counter_ns()
                    expected_bytes = rows * 2560 * 4
                    if len(raw_input) != expected_bytes or sha(raw_input) != inputs[rows]["sha256"]:
                        raise BenchError(f"serialized X mismatch for rows={rows}")
                    if len(body) != expected_bytes:
                        raise BenchError(f"native operator output size mismatch for rows={rows}")
                    owned = np.frombuffer(body, dtype="<f4").reshape(rows, 2560).copy(order="C")
                    rtt_end = time.perf_counter_ns()
                    if not owned.flags.c_contiguous or not owned.flags.owndata or not np.isfinite(owned).all():
                        bad = outroot / f"failure-rows-{rows:03d}-{formal_label}.f32"
                        last_output_failure = _save_bytes(bad, body)
                        raise BenchError(f"native operator output is non-finite or not owned for rows={rows}; preserved {bad}")
                    quality = worker_api.quality_report(owned, reference, worker_api.DEFAULT_TOLERANCES)
                    out_sha = sha(body)
                    result = {"label": formal_label, "request_id": request_id_local, "input_sha256": sha(raw_input),
                        "output_sha256": out_sha, "input_bytes": len(raw_input), "output_bytes": len(body),
                        "client_rtt_ns_after_output_copy": rtt_end-t0, "input_encode_ns": encode_end-t0,
                        "wire_rtt_ns_before_output_copy": received-encode_end, "output_owned_copy_ns": rtt_end-received,
                        "worker_wall_ns": worker_ns, "rtt_minus_worker_ns_descriptive": rtt_end-t0-worker_ns,
                        "numerical_quality_vs_native_cpu": quality}
                    if not quality.get("finite") or not quality.get("shape_ok") or not quality.get("pass"):
                        failed_path=outroot/f"failure-rows-{rows:03d}-{formal_label}.f32"
                        last_output_failure=_save_bytes(failed_path,body)
                        result["preserved_failure_output"]=last_output_failure
                    return result, owned, body

                warm, _, _ = infer_once("warmup")
                shape["warmup"] = warm; update_receipt(receipt, record)
                final_bytes = None
                for formal in range(1, int(c["formal_repeats"])+1):
                    monitor.require()
                    result, owned, raw_out = infer_once(f"formal-{formal:02d}")
                    shape["formals"].append(result); update_receipt(receipt, record)
                    if result["numerical_quality_vs_native_cpu"]["pass"] is not True:
                        shape["status"]="failed_numerical_gate"; update_receipt(receipt,record)
                        raise BenchError(f"native CPU numerical gate failed: rows={rows}, formal={formal}")
                    final_bytes = raw_out
                if final_bytes is None: raise BenchError(f"no output was produced for rows={rows}")
                path=outroot/f"output-{rows:03d}.f32"; shape["stored_output"]=_save_bytes(path,final_bytes)
                shape["status"]="complete"; update_receipt(receipt,record)
            outputs_complete=True; record["status"]="complete"
        except BaseException as exc:
            record["status"]="failed"; record["failure"]={"type":type(exc).__name__,"message":str(exc)[:1000],"failed_utc":utc()}
            if last_output_failure is not None: record["failure_output"] = last_output_failure
        finally:
            if monitor is not None:
                try: monitor.close()
                except Exception as exc: record["resource_monitor_close_error"]=f"{type(exc).__name__}: {exc}"; record["status"]="failed"
            if worker is not None:
                try:
                    proc=worker._proc
                    if not worker.in_flight and proc is not None and proc.stdin is not None and proc.poll() is None:
                        proc.stdin.close()
                        try: proc.wait(timeout=8)
                        except subprocess.TimeoutExpired: pass
                    worker.close()
                    record["worker_launcher_reaped"]={"pid":worker.pid,"returncode":worker.returncode}
                except Exception as exc: record["worker_close_error"]=f"{type(exc).__name__}: {exc}"; record["status"]="failed"
            if tree is not None:
                try:
                    record["process_tree_cleanup"]=tree.cleanup_descendants()
                    if not record["process_tree_cleanup"]["terminal"]: record["status"]="failed"
                except Exception as exc: record["process_tree_cleanup_error"]=f"{type(exc).__name__}: {exc}"; record["status"]="failed"
            record["finished_utc"]=utc(); record["outputs_complete"]=outputs_complete
            update_receipt(receipt,record); receipt.close()
    return 0 if record["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root-start-confirmed", action="store_true")
    parser.add_argument("--cuda-build-stopped-confirmed", action="store_true")
    args = parser.parse_args(argv)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if not args.run:
        print(json.dumps(make_plan(contract), indent=2))
        return 0
    if not (args.root_start_confirmed and args.cuda_build_stopped_confirmed and contract["execution_authorized"] is True):
        parser.error("run42 needs root start authorization, CUDA-build-stopped confirmation, and execution_authorized=true")
    return execute(contract)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"native IPC: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
