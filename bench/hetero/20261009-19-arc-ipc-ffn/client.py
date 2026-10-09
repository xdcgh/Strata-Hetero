#!/usr/bin/env python3
"""Explicit post-root-signal IPC benchmark client for one real expert FFN on Intel Arc."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
HIDDEN = 2560
DEVICE = "GPU.0"
PRECISION = "f32"
WARMUPS = 1
REPEATS = 5
REQUEST_TIMEOUT = 300.0
RAM_MIN = 12.0
COMMIT_MIN = 4.0
OUTPUT_ROOT = Path(r"E:\Strata-Hetero-data\experts\20261009-19-arc-ipc-ffn")
INPUT_RECEIPT = ROOT / "bench" / "hetero" / "20261009-10-native-cpu" / "inputs-receipt.json"
NATIVE_RUN = ROOT / "bench" / "hetero" / "20261009-10-native-cpu" / "runs" / "native-cpu-01"
WEIGHTS = Path(r"E:\Strata-Hetero-data\experts\20261009-01-real-expert-01\weights.npz")
IDENTITY = Path(r"E:\Strata-Hetero-data\experts\20261009-01-real-expert-01\identity.json")
WEIGHT_EXTRACTION = ROOT / "bench" / "hetero" / "20261009-01-real-expert" / "extraction-01.json"
SAMPLER_PYTHON = Path(r"E:\Strata-Hetero-data\venv\Scripts\python.exe")
SERVICE_PYTHON = Path(r"E:\Strata-Hetero-data\venv-intel\Scripts\python.exe")
STOP = RUN / "resource" / "STOP"
RECEIPT = RUN / "receipt.json"
CSV_PATH = RUN / "formal-results.csv"

from tools import hetero_xpu_service as service  # noqa: E402; module import does not import OpenVINO or create Core
from tools import hetero_xpu_transport as transport  # noqa: E402
from tools import hetero_xpu_worker as worker_api  # noqa: E402

RESULT_PREFIX = struct.Struct("<IQ")


class BenchError(RuntimeError):
    pass


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def is_reported_f32(value: object) -> bool:
    """Accept only known OpenVINO float32 spellings and the service f32 label."""
    return str(value).strip().casefold() in {"f32", "float32", "<type: 'float32'>", '<type: "float32">'}


def own_output_and_stop_rtt(raw: bytes, rows: int, start_ns: int) -> tuple[np.ndarray, int]:
    owned = np.frombuffer(raw, dtype="<f4").reshape(rows, HIDDEN).copy(order="C")
    return owned, time.perf_counter_ns() - start_ns


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_update(stream, record: dict) -> None:
    stream.seek(0)
    json.dump(record, stream, indent=2, ensure_ascii=True, allow_nan=False)
    stream.write("\n")
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def memory_gate() -> dict:
    if os.name != "nt":
        raise BenchError("real IPC device run is Windows-only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    mem = MEMORYSTATUSEX(); mem.dwLength = ctypes.sizeof(mem)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise BenchError("physical-memory query unknown")
    perf = PERFORMANCE_INFORMATION(); perf.cb = ctypes.sizeof(perf)
    psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), ctypes.c_ulong]
    psapi.GetPerformanceInfo.restype = ctypes.c_int
    if not psapi.GetPerformanceInfo(ctypes.byref(perf), ctypes.sizeof(perf)):
        raise BenchError("PSAPI system-commit query unknown")
    commit = (int(perf.CommitLimit) - int(perf.CommitTotal)) * int(perf.PageSize)
    ram = int(mem.ullAvailPhys)
    return {"at_utc": utc(), "physical_bytes": ram, "physical_gib": ram / 1024**3,
            "commit_bytes": commit, "commit_gib": commit / 1024**3,
            "status": "pass" if ram >= RAM_MIN * 1024**3 and commit >= COMMIT_MIN * 1024**3 else "blocked"}


def read_latest_sample(path: Path, observer_pid: int, owner_pid: int) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise BenchError("resource observer has not produced a sample")
    row = json.loads(lines[-1])
    if row.get("record_type") != "resource_sample" or row.get("owner_pid") != owner_pid:
        raise BenchError("resource observer record/owner is unknown")
    if not row.get("ram_gate") or row["ram_gate"].get("status") != "pass" or row["ram_gate"].get("available_gib", 0) < RAM_MIN:
        raise BenchError("physical RAM gate failed or is unknown")
    if not row.get("commit_gate") or row["commit_gate"].get("status") != "pass" or row["commit_gate"].get("available_gib", 0) < COMMIT_MIN:
        raise BenchError("system commit gate failed or is unknown")
    if row.get("alerts") or row.get("errors"):
        raise BenchError("resource observer recorded alerts/errors")
    return row


def decode_response(frame, kind: int, request_id: int) -> tuple[dict | None, int, bytes]:
    if frame.kind != (kind | transport.RESPONSE_BIT) or frame.request_id != request_id or len(frame.payload) < RESULT_PREFIX.size:
        raise BenchError("IPC response identity/prefix mismatch")
    status, worker_ns = RESULT_PREFIX.unpack_from(frame.payload)
    body = bytes(frame.payload[RESULT_PREFIX.size:])
    if status != 0:
        try:
            detail = json.loads(body.decode("utf-8")).get("message", "service request failed")
        except Exception:
            detail = "service returned an invalid error frame"
        raise BenchError(f"IPC kind={kind} id={request_id}: {detail}")
    return None, worker_ns, body


def load_artifacts(contract: dict):
    input_receipt_path = Path(contract["inputs"]["receipt"])
    if sha(input_receipt_path.read_bytes()) != contract["inputs"]["receipt_sha256"]:
        raise BenchError("native CPU input receipt SHA mismatch")
    input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
    identity_raw = IDENTITY.read_bytes()
    identity = worker_api._safe_identity(json.loads(identity_raw.decode("utf-8")))
    identity_sha = sha(identity_raw)
    if identity_sha != contract["weights"]["identity_sha256"]:
        raise BenchError("weights identity SHA differs from contract")
    extraction_sha = sha(WEIGHT_EXTRACTION.read_bytes())
    if extraction_sha != contract["weights"]["extraction_receipt_sha256"]:
        raise BenchError("real-expert extraction receipt SHA differs from contract")
    header = worker_api.validate_npz_metadata(WEIGHTS, identity)
    weights = worker_api.load_verified_weights(WEIGHTS, identity, header)
    inputs = {}; native_outputs = {}
    for item in input_receipt["inputs"]:
        rows = int(item["rows"]); path = Path(item["path"]); raw = path.read_bytes()
        if len(raw) != item["bytes"] or sha(raw) != item["sha256"] or len(raw) != rows * HIDDEN * 4:
            raise BenchError(f"input rows={rows} size/SHA/shape differs from inputs receipt")
        x = np.frombuffer(raw, dtype="<f4").reshape(rows, HIDDEN)
        if not np.isfinite(x).all():
            raise BenchError(f"input rows={rows} contains non-finite values")
        nr = json.loads((NATIVE_RUN / f"rows-{rows:03d}" / "native-receipt.json").read_text(encoding="utf-8"))
        if nr.get("status") != "success" or nr.get("execution", {}).get("native_kernel_called") is not True:
            raise BenchError(f"native CPU receipt for rows={rows} is not a successful native kernel run")
        native_path = Path(nr["output"]["path"]); native_raw = native_path.read_bytes()
        if len(native_raw) != nr["output"]["bytes"] or sha(native_raw) != nr["output"]["sha256"] or len(native_raw) != rows * HIDDEN * 4:
            raise BenchError(f"native CPU output rows={rows} size/SHA/shape differs from its receipt")
        native = np.frombuffer(native_raw, dtype="<f4").reshape(rows, HIDDEN).copy()
        if not np.isfinite(native).all():
            raise BenchError(f"native CPU output rows={rows} contains non-finite values")
        inputs[rows] = {"array": x, "bytes": raw, "sha256": sha(raw), "path": str(path)}
        native_outputs[rows] = {"array": native, "sha256": sha(native_raw), "path": str(native_path),
                                "receipt": str(NATIVE_RUN / f"rows-{rows:03d}" / "native-receipt.json"),
                                "prior_f32_relative_rmse": nr.get("comparison", {}).get("quality", {}).get("relative_rmse")}
    if tuple(sorted(inputs)) != ROWS or tuple(sorted(native_outputs)) != ROWS:
        raise BenchError("expected exactly nine prepared input/native output row sizes")
    return identity, identity_sha, header, weights, inputs, native_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="required explicit start signal; otherwise no device/child action")
    parser.add_argument("--root-start-confirmed", action="store_true")
    parser.add_argument("--cuda-build-stopped-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.run or not args.root_start_confirmed or not args.cuda_build_stopped_confirmed:
        parser.error("wait for root and CUDA-build-stop signals, then pass --run --root-start-confirmed --cuda-build-stopped-confirmed")
    contract_path = RUN / "execution-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("execution_authorized") is not False or contract.get("run_id") != RUN.name:
        raise BenchError("execution contract is not the prepared H19 contract")
    if RUN.exists() is False or RUN.parent.resolve() != (ROOT / "bench" / "hetero").resolve():
        raise BenchError("run directory is not the explicit H19 target")
    out_dir = OUTPUT_ROOT
    if out_dir.exists():
        raise FileExistsError(f"exclusive E: output directory already exists: {out_dir}")
    for p in (RECEIPT, CSV_PATH, RUN / "resource", RUN / "logs"):
        if p.exists():
            raise FileExistsError(f"exclusive run output already exists: {p}")

    run_record: dict = {"schema_version": 1, "status": "running", "run_id": RUN.name,
                        "started_utc": utc(), "contract": str(contract_path), "shapes": [],
                        "device_requested": DEVICE, "precision_requested": PRECISION,
                        "native_cpu_comparison_note": "Native quantized CPU uses Q4_K/Q5_1 with Q8 activation arithmetic; compare descriptively under unchanged F32 tolerances, not as the F32 acceptance reference.",
                        "timing_note": "Client RTT includes F32LE input serialization, IPC, response receipt and client-owned contiguous NumPy F32 output copy. The separately recorded wire RTT ends before that NumPy copy. worker_wall_ns is service-local duration; subtraction is descriptive only.",
                        "model_route_claim": False}
    receipt = RECEIPT.open("x+", encoding="utf-8", newline="\n")
    worker = None; sampler = None; sampler_stdout = None; sampler_stderr = None; rows_complete = False
    request_id = 0
    csv_file = None
    csv_path = None
    try:
        write_update(receipt, run_record)
        identity, identity_sha, header, weights, inputs, native_outputs = load_artifacts(contract)
        if header["hidden"] != HIDDEN or header["intermediate"] != 640 or identity["layer"] != 28 or identity["expert"] != 288:
            raise BenchError("real expert identity/shape differs from reviewed layer28/expert288 contract")
        before_gates = memory_gate()
        if before_gates["status"] != "pass":
            raise BenchError("fresh physical RAM/system commit gate failed before worker startup")
        if shutil.disk_usage(out_dir.parent).free < 12 * 1024**3 + sum(rows * HIDDEN * 4 for rows in ROWS):
            raise BenchError("E: output capacity gate failed")
        out_dir.mkdir(parents=False, exist_ok=False)
        (RUN / "logs").mkdir(exist_ok=False); (RUN / "resource").mkdir(exist_ok=False)
        run_record.update({"identity_sha256": identity_sha, "weights_header_info": header,
                           "weights_payload_hashes_verified": identity["weights_sha256"],
                           "input_receipt": contract["inputs"]["receipt"],
                           "inputs_verified": {str(rows): {"sha256": inputs[rows]["sha256"], "bytes": len(inputs[rows]["bytes"])} for rows in ROWS},
                           "native_cpu_outputs_verified": {str(rows): native_outputs[rows]["sha256"] for rows in ROWS},
                           "system_gates_before_worker": before_gates})
        write_update(receipt, run_record)
        csv_path = CSV_PATH.open("x", encoding="utf-8", newline="")
        csv_file = csv.writer(csv_path)
        csv_file.writerow(["rows","formal_index","request_id","input_sha256","output_sha256","input_bytes","output_bytes","rtt_ns_after_owned_output_copy","wire_rtt_ns_before_numpy_output_copy","worker_wall_ns","rtt_minus_worker_ns_descriptive","f32_reference_pass","f32_max_abs","f32_relative_rmse","f32_row_norm_relative","native_cpu_informational_pass","native_cpu_relative_rmse"])
        csv_path.flush(); os.fsync(csv_path.fileno())

        resource_path = RUN / "resource" / "samples.jsonl"
        sampler_stdout = (RUN / "logs" / "resource.stdout.log").open("xb", buffering=0)
        sampler_stderr = (RUN / "logs" / "resource.stderr.log").open("xb", buffering=0)
        sampler_argv = [str(SAMPLER_PYTHON), "-u", "-B", str(ROOT / "tools" / "hetero_resources.py"),
                        "--output", str(resource_path), "--pid", str(os.getpid()), "--interval", "1",
                        "--max-seconds", "3600", "--minimum-commit-gib", "4", "--stop-file", str(STOP)]
        sampler = subprocess.Popen(sampler_argv, shell=False, stdin=subprocess.DEVNULL,
                                   stdout=sampler_stdout, stderr=sampler_stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        run_record["resource_observer"] = {"pid": sampler.pid, "argv": sampler_argv, "output": str(resource_path)}
        write_update(receipt, run_record)

        def gate() -> dict:
            if sampler is None or sampler.poll() is not None:
                raise BenchError("owned resource observer exited")
            deadline = time.monotonic() + 8.0
            while not resource_path.is_file() or resource_path.stat().st_size == 0:
                if time.monotonic() >= deadline: raise BenchError("resource observer produced no sample")
                time.sleep(0.1)
            lines = resource_path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[-1])
            if row.get("record_type") != "resource_sample" or row.get("owner_pid") != os.getpid():
                raise BenchError("resource observer owner/record is unknown")
            sample_time = datetime.fromisoformat(row["observed_at_utc"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - sample_time.astimezone(timezone.utc)).total_seconds()
            if age < -2 or age > 5: raise BenchError("resource sample is stale/unknown")
            ram, commit = row.get("ram_gate", {}), row.get("commit_gate", {})
            if ram.get("status") != "pass" or ram.get("available_gib", 0) < 12:
                raise BenchError("physical RAM gate below 12 GiB or unknown")
            if commit.get("status") != "pass" or commit.get("available_gib", 0) < 4:
                raise BenchError("PSAPI system commit gate below 4 GiB or unknown")
            if row.get("alerts") or row.get("errors"): raise BenchError("resource observer reported alerts/errors")
            return row

        first_sample = gate()
        run_record["resource_gate_initial"] = first_sample
        write_update(receipt, run_record)
        worker_start_ns = time.perf_counter_ns()
        service_argv = [str(SERVICE_PYTHON), "-u", "-B", str(ROOT / "tools" / "hetero_xpu_service.py"),
                        "--serve", "--weights", str(WEIGHTS), "--identity", str(IDENTITY),
                        "--device", DEVICE, "--precision", PRECISION]
        worker = transport.OwnedPipeWorker(service_argv, RUN / "logs" / "arc-worker.stderr.log",
                                            request_timeout=REQUEST_TIMEOUT, shutdown_timeout=5.0).start()
        run_record["worker"] = {"pid": worker.pid, "argv": service_argv,
                                "spawn_wall_ns": time.perf_counter_ns() - worker_start_ns}
        write_update(receipt, run_record)

        def rpc(kind: int, make_payload) -> dict:
            nonlocal request_id
            request_id += 1
            t0 = time.perf_counter_ns()
            payload = make_payload()
            frame = worker.request(kind, request_id, payload, timeout=REQUEST_TIMEOUT)
            if frame.kind != (kind | transport.RESPONSE_BIT) or frame.request_id != request_id:
                raise BenchError("IPC response kind/request ID mismatch")
            if len(frame.payload) < RESULT_PREFIX.size:
                raise BenchError("IPC response missing status/worker time prefix")
            status, worker_ns = RESULT_PREFIX.unpack_from(frame.payload)
            body = bytes(frame.payload[RESULT_PREFIX.size:])
            rtt_ns = time.perf_counter_ns() - t0
            if status != 0:
                try: message = json.loads(body.decode("utf-8")).get("message", "service error")
                except Exception: message = "malformed service error payload"
                raise BenchError(f"IPC kind {kind} request {request_id} failed: {message}")
            return {"request_id": request_id, "worker_wall_ns": worker_ns, "wire_rtt_ns_before_numpy_output_copy": rtt_ns,
                    "rtt_ns": rtt_ns, "rtt_start_ns": t0,
                    "payload": payload if kind == service.INFER_KIND else None, "body": body}

        for rows in ROWS:
            sample = gate()
            x = inputs[rows]["array"]
            reference = worker_api.numpy_ffn(x, weights)
            if reference.shape != (rows,HIDDEN) or not np.isfinite(reference).all():
                raise BenchError(f"NumPy F32 reference shape/finite check failed for rows={rows}")
            nonce = secrets.token_hex(16)
            init_id_before = request_id + 1
            init = rpc(service.INIT_KIND, lambda: json.dumps({"schema":service.SERVICE_INIT_SCHEMA,"nonce":nonce,
                            "rows":rows,"identity_sha256":identity_sha},separators=(",",":"),sort_keys=True).encode("utf-8"))
            ready = json.loads(init["body"].decode("utf-8"))
            dev_name = str(ready.get("device_full_name", ""))
            policy = ready.get("precision_policy", {})
            submitted = (ready.get("compile_property_policy") or {}).get("submitted", {})
            if (ready.get("status") != "ready" or ready.get("nonce") != nonce or
                    ready.get("identity_sha256") != identity_sha or ready.get("rows") != rows or
                    ready.get("device") != DEVICE or ready.get("execution_devices") != [DEVICE] or
                    "intel" not in dev_name.casefold() or "arc" not in dev_name.casefold() or
                    ready.get("precision_requested") != PRECISION or
                    not is_reported_f32(ready.get("reported_precision")) or
                    policy.get("precision_policy_matches") is not True or
                    ready.get("weights_sha256") != identity["weights_sha256"] or
                    submitted != {"INFERENCE_PRECISION_HINT":"f32","EXECUTION_MODE_HINT":"ACCURACY"} or
                    ready.get("cached_graph_reused") is not False):
                raise BenchError(f"INIT ready/device/precision/identity contract failed for rows={rows}")
            shape_record = {"rows":rows,"shape":[rows,HIDDEN],"input_path":inputs[rows]["path"],
                            "input_sha256":inputs[rows]["sha256"],"native_cpu_output_path":native_outputs[rows]["path"],
                            "native_cpu_output_sha256":native_outputs[rows]["sha256"],"resource_before_init":sample,
                            "init":{"request_id":init_id_before,"nonce":nonce,"client_rtt_ns":init["rtt_ns"],
                                    "worker_wall_ns":init["worker_wall_ns"],"reported_compile_ns":ready.get("compile_ns"),
                                    "ready":ready},"warmup":None,"formals":[],"status":"running"}
            run_record["shapes"].append(shape_record)
            write_update(receipt,run_record)
            def infer_once() -> dict:
                result = rpc(service.INFER_KIND,
                             lambda: np.asarray(x,dtype="<f4",order="C").tobytes(order="C"))
                raw_input = result.pop("payload")
                output = result.pop("body")
                expected_bytes = rows * HIDDEN * 4
                array, rtt_ns = own_output_and_stop_rtt(output, rows, result.pop("rtt_start_ns"))
                result["rtt_ns"] = rtt_ns
                if len(raw_input) != expected_bytes or sha(raw_input) != inputs[rows]["sha256"]:
                    raise BenchError(f"serialized F32LE input differs from verified X for rows={rows}")
                if len(output) != expected_bytes:
                    raise BenchError(f"Arc output byte count differs from expected shape for rows={rows}")
                if not np.isfinite(array).all():
                    raise BenchError(f"Arc output contains NaN/Inf for rows={rows}")
                return {"request_id":result["request_id"],"rtt_ns":result["rtt_ns"],
                        "wire_rtt_ns_before_numpy_output_copy":result["wire_rtt_ns_before_numpy_output_copy"],
                        "worker_wall_ns":result["worker_wall_ns"],"rtt_minus_worker_ns_descriptive":result["rtt_ns"]-result["worker_wall_ns"],
                        "input_sha256":sha(raw_input),"input_bytes":len(raw_input),
                        "output_sha256":sha(output),"output_bytes":len(output),"output_bytes_owned":output,
                        "output_array":array}

            gate()
            warm = infer_once()
            shape_record["warmup"]={k:v for k,v in warm.items() if k not in ("output_bytes_owned","output_array")}
            last_output=None
            for formal in range(1,REPEATS+1):
                gate()
                measured=infer_once()
                f32_quality=worker_api.quality_report(measured["output_array"], reference,
                                                      worker_api.DEFAULT_TOLERANCES)
                native_quality=worker_api.quality_report(measured["output_array"],
                    native_outputs[rows]["array"], worker_api.DEFAULT_TOLERANCES)
                formal_record={k:v for k,v in measured.items() if k not in ("output_bytes_owned","output_array")}
                formal_record.update({"formal_index":formal,"input_shape":[rows,HIDDEN],"output_shape":list(measured["output_array"].shape),
                                     "execution_devices":ready["execution_devices"],"reported_precision":ready["reported_precision"],
                                     "precision_policy":ready["precision_policy"],"f32_reference_quality":f32_quality,
                                     "native_cpu_informational_quality":native_quality,
                                     "native_cpu_comparison_scope":"different native Q4_K/Q5_1 + Q8 activation arithmetic; report only, do not widen F32 reference tolerance"})
                shape_record["formals"].append(formal_record)
                last_output=measured["output_bytes_owned"]
                csv_file.writerow([rows,formal,measured["request_id"],measured["input_sha256"],measured["output_sha256"],
                                   measured["input_bytes"],measured["output_bytes"],measured["rtt_ns"],measured["wire_rtt_ns_before_numpy_output_copy"],measured["worker_wall_ns"],
                                   measured["rtt_minus_worker_ns_descriptive"],f32_quality["pass"],f32_quality.get("max_abs_error"),
                                   f32_quality.get("relative_rmse"),f32_quality.get("max_row_norm_relative_error"),
                                   native_quality["pass"],native_quality.get("relative_rmse")])
                csv_path.flush(); os.fsync(csv_path.fileno())
                write_update(receipt,run_record)
                if not f32_quality.get("finite") or not f32_quality.get("shape_ok") or not f32_quality.get("pass"):
                    raise BenchError(f"strict NumPy F32 reference quality failed for rows={rows}, formal={formal}")
                gate()
            if last_output is None:
                raise BenchError(f"no final output exists for rows={rows}")
            output_path=OUTPUT_ROOT/f"output-{rows:03d}.f32"
            with output_path.open("xb",buffering=0) as f:
                f.write(last_output); f.flush(); os.fsync(f.fileno())
            stored=output_path.read_bytes()
            readback_sha=sha(stored)
            if len(stored)!=rows*HIDDEN*4 or readback_sha!=shape_record["formals"][-1]["output_sha256"]:
                raise BenchError(f"stored F32 output/readback SHA mismatch for rows={rows}")
            shape_record["stored_last_formal_f32_output"]={"path":str(output_path),"bytes":len(stored),"sha256":readback_sha,
                                                           "readback_sha256":readback_sha,"readback_verified":True}
            rtts=[x["rtt_ns"] for x in shape_record["formals"]]
            workers=[x["worker_wall_ns"] for x in shape_record["formals"]]
            shape_record["timing_summary"]={"rtt_ns_mean":statistics.mean(rtts),"rtt_ns_median":statistics.median(rtts),
                                             "rtt_ns_spread_max_minus_min":max(rtts)-min(rtts),
                                             "worker_wall_ns_mean":statistics.mean(workers),"worker_wall_ns_median":statistics.median(workers),
                                             "worker_wall_ns_spread_max_minus_min":max(workers)-min(workers),
                                             "compile_ns_excluded":ready["compile_ns"]}
            shape_record["status"]="complete"
            write_update(receipt,run_record)
        rows_complete=True
        run_record["status"]="complete"
    except BaseException as exc:
        run_record["status"]="failed"
        run_record["failure"]={"type":type(exc).__name__,"message":str(exc)[:600],"failed_utc":utc()}
    finally:
        if worker is not None:
            try:
                worker_pid=worker.pid
                worker.close()
                run_record["worker_closed"]={"pid":worker_pid,"returncode":worker.returncode,"reaped":True}
            except BaseException as exc:
                run_record["worker_close_error"]={"type":type(exc).__name__,"message":str(exc)[:300]}
                run_record["status"]="failed"
        if sampler is not None:
            try:
                if sampler.poll() is None:
                    if STOP.exists():
                        pass
                    else:
                        with STOP.open("xb") as f: f.write(b"H19 owned sampler stop\n"); f.flush(); os.fsync(f.fileno())
                    sampler.wait(timeout=10)
                run_record["resource_observer_closed"]={"pid":sampler.pid,"returncode":sampler.poll(),"stop_file":str(STOP)}
            except BaseException as exc:
                try:
                    if sampler.poll() is None: sampler.terminate(); sampler.wait(timeout=5)
                except BaseException: pass
                run_record["resource_observer_close_error"]={"type":type(exc).__name__,"message":str(exc)[:300]}
                run_record["status"]="failed"
        if csv_path is not None:
            csv_path.flush(); os.fsync(csv_path.fileno()); csv_path.close()
        run_record["finished_utc"]=utc()
        run_record["all_shapes_complete"]=rows_complete
        write_update(receipt,run_record)
        receipt.close()
        for stream in (sampler_stdout,sampler_stderr):
            if stream is not None: stream.close()
    print(json.dumps({"status":run_record["status"],"receipt":str(RECEIPT),"rows_complete":rows_complete,
                      "worker_pid":run_record.get("worker",{}).get("pid"),"service_argv":service_argv if 'service_argv' in locals() else None},indent=2))
    return 0 if run_record["status"]=="complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
