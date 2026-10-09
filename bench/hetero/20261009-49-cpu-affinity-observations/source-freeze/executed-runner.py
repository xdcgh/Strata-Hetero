#!/usr/bin/env python3
"""Plan, prepare approved row inputs, or run the isolated native CPU pool sweep."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT = RUN / "sweep-execution-contract-v2.json"
ROWS = (1, 16, 256)
WORKERS = (4,)
AFFINITIES = ("none", "all", "auto", "p-cores")
PHYS_MIN = 12 * 1024**3
COMMIT_MIN = 4 * 1024**3
TIMEOUT_SECONDS = 90
CHUNK = 1024 * 1024


class SweepError(RuntimeError):
    pass


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value: Any) -> str:
    if not path.is_absolute() or not path.parent.is_dir():
        raise SweepError(f"exclusive receipt requires absolute path/existing parent: {path}")
    raw = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return sha(raw)


def write_stream_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb", buffering=0) as stream:
        view = memoryview(raw)
        while view:
            n = stream.write(view)
            if not n:
                raise OSError(f"short write made no progress: {path}")
            view = view[n:]
        stream.flush(); os.fsync(stream.fileno())


def make_plan(c: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    by_affinity = c["affinity_candidates"]
    for rows in c["default_gate_rows"]:
        case_id = f"default-r{rows:03d}"
        cases.append({"case_id": case_id, "output_path": str(Path(c["output_root"]) / case_id / "output.f32"),
                      "receipt_path": str(Path(c["output_root"]) / case_id / "native-receipt.json"),
                      "phase": "default_reference_gate",
                      "status": "planned", "rows": rows, "workers": None, "affinity": None,
                      "host_participates": False, "effective_compute_participants": 1,
                      "reference_policy": "frozen_native_output_and_same_candidate_default"})
    strict_overflow = 0
    planned_pool = 0
    for affinity in AFFINITIES:
        for workers in WORKERS:
            candidate = by_affinity.get(affinity, {})
            worker_candidates = candidate.get("worker_candidates", []) if affinity != "none" else []
            overflow = max(0, workers - len(worker_candidates)) if affinity != "none" else 0
            for rows in ROWS:
                excluded = affinity == "p-cores" and overflow > 0
                case_id = f"pool-{affinity}-w{workers:02d}-r{rows:03d}"
                cases.append({"case_id": case_id,
                              "output_path": str(Path(c["output_root"]) / case_id / "output.f32"),
                              "receipt_path": str(Path(c["output_root"]) / case_id / "native-receipt.json"),
                              "phase": "pool_matrix",
                              "status": "not_run_strict_candidate_overflow" if excluded else "planned",
                              "rows": rows, "workers": workers, "affinity": affinity,
                              "host_participates": True, "effective_compute_participants": workers + 1,
                              "pin_expected": affinity != "none",
                              "candidate_worker_count": len(worker_candidates) if affinity != "none" else None,
                              "candidate_cpu_set_ids": [x["cpu_set_id"] for x in worker_candidates],
                              "candidate_native_cpu_ids": [x["native_cpu_id"] for x in worker_candidates],
                              "planned_host_cpu_set_id": candidate.get("host_core_candidate", {}).get("cpu_set_id") if affinity != "none" else None,
                              "planned_host_native_cpu_id": candidate.get("host_core_candidate", {}).get("native_cpu_id") if affinity != "none" else None,
                              "planned_unpinned_overflow": overflow,
                              "worker_pin_success_evidence": False,
                              "reason": f"{workers} workers exceed {len(worker_candidates)} p-cores candidates" if excluded else None,
                              "reference_policy": "frozen_native_and_same_binary_default" if rows in (6, 10) else "frozen_native_output"})
                if excluded: strict_overflow += 1
                else: planned_pool += 1
    return {"schema_version": 1, "run_id": c["run_id"], "status": "plan_only",
            "execution_authorized": c["execution_authorized"],
            "input_preparation_authorized": c["input_preparation_authorized"],
            "scope": c["scope"], "rows": ROWS, "workers": WORKERS, "affinities": AFFINITIES,
            "default_reference_gates": len(c["default_gate_rows"]), "planned_pool_cases": planned_pool,
            "strict_pcore_overflow_not_run": strict_overflow,
            "planned_case_count_including_gates": len(c["default_gate_rows"]) + planned_pool,
            "plan_entry_count_including_exclusions": len(cases),
            "all_auto_all_worker16_overflow_cases": sum(1 for x in cases if x.get("affinity") in ("all", "auto") and x.get("workers") == 16 and x["status"] == "planned"),
            "cases": cases, "payload_read": False, "kernel_execution": False,
            "affinity_mutation": False, "output_root_created": False,
            "planned_output_root": c["output_root"], "planned_data_root": c["data_root"],
            "cpu_type_labels": "unknown; numeric CPU Set IDs, group/logical indices, and EfficiencyClass only"}


def generate_input_bytes(rows: int, numpy_module=None) -> tuple[bytes, dict[str, Any]]:
    if rows not in (6, 10):
        raise ValueError("only the two missing frozen-input rows may be generated")
    if numpy_module is None:
        import numpy as numpy_module
    seed = 42 + rows
    values = numpy_module.random.default_rng(seed).standard_normal((rows, 2560), dtype=numpy_module.float32)
    values *= numpy_module.float32(0.25)
    values = numpy_module.ascontiguousarray(values, dtype="<f4")
    raw = values.tobytes(order="C")
    if len(raw) != rows * 2560 * 4 or not bool(numpy_module.isfinite(values).all()):
        raise SweepError(f"generated input validation failed for rows={rows}")
    return raw, {"seed": seed, "numpy_version": numpy_module.__version__,
                 "formula": "default_rng(42+rows).standard_normal((rows,2560),dtype=float32)*float32(0.25)",
                 "encoding": "contiguous little-endian float32", "shape": [rows, 2560]}


def minimum_free_disk_bytes(c: dict[str, Any]) -> int:
    return int(c["resource_gate"]["minimum_free_disk_bytes"])


def file_sha(path: Path, expected_bytes: int | None = None) -> tuple[str, dict[str, Any]]:
    from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged
    before = file_stat(path)
    if expected_bytes is not None and before["size_bytes"] != expected_bytes:
        raise SweepError(f"size mismatch: {path}")
    digest = hashlib.sha256(); count = 0
    with path.open("rb", buffering=0) as stream:
        if not fd_matches_path(os.fstat(stream.fileno()), before):
            raise SweepError(f"path/handle identity differs before read: {path}")
        while block := stream.read(CHUNK):
            digest.update(block); count += len(block)
        fd_after = os.fstat(stream.fileno())
    after = file_stat(path)
    if count != before["size_bytes"] or not stat_unchanged(before, after) or not fd_matches_path(fd_after, before):
        raise SweepError(f"file identity changed while reading: {path}")
    return digest.hexdigest(), {**before, "path": str(path)}


def checked_file(spec: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    path = Path(spec["path"])
    actual_sha, identity = file_sha(path, int(spec["bytes"]))
    if spec.get("sha256") and actual_sha != spec["sha256"]:
        raise SweepError(f"SHA mismatch: {path}")
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or sha(raw) != actual_sha:
        raise SweepError(f"file changed between stable hash and readback: {path}")
    return raw, {**identity, "sha256": actual_sha}


def sample_resource(label: str, samples, *, enforce: bool = True) -> dict[str, Any]:
    from tools.hetero_resources import memory_snapshot
    memory, errors = memory_snapshot()
    row = {"record_type": "resource_sample", "label": label, "observed_at_utc": utc(),
           "owner_pid": os.getpid(), "physical_available_bytes": memory.get("physical_available_bytes"),
           "commit_available_bytes": memory.get("commit_available_bytes"), "source": memory.get("source"), "errors": errors}
    row["gate_ok"] = (not errors and type(row["physical_available_bytes"]) is int and
                      type(row["commit_available_bytes"]) is int and
                      row["physical_available_bytes"] >= PHYS_MIN and row["commit_available_bytes"] >= COMMIT_MIN)
    samples.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"); samples.flush()
    if enforce and not row["gate_ok"]:
        raise SweepError(f"12/4 GiB resource gate failed: {row}")
    return row


def cpu_identity(proc, psutil) -> dict[str, Any]:
    return {"pid": int(proc.pid), "parent_pid": int(proc.ppid()), "create_time": float(proc.create_time()),
            "executable": os.path.normcase(os.path.realpath(proc.exe())), "command_line": list(proc.cmdline())}


def same_process(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a[k] == b[k] for k in ("pid", "parent_pid", "create_time", "executable", "command_line"))


def finite_f32le(raw: bytes, expected_count: int) -> bool:
    if len(raw) != expected_count * 4 or len(raw) % 4:
        return False
    return all(math.isfinite(value) for (value,) in __import__("struct").iter_unpack("<f", raw))


def validate_direct_process(proc, psutil, expected_exe: Path, argv: list[str]) -> dict[str, Any]:
    observed = cpu_identity(psutil.Process(proc.pid), psutil)
    if observed["executable"] != os.path.normcase(os.path.realpath(expected_exe)):
        raise SweepError(f"owned C++ executable identity mismatch: {observed}")
    if observed["parent_pid"] != os.getpid():
        raise SweepError(f"owned C++ process has an unexpected parent: {observed}")
    actual = observed["command_line"]
    if not actual or os.path.normcase(os.path.realpath(actual[0])) != observed["executable"] or actual[1:] != argv:
        raise SweepError(f"owned C++ command line mismatch: {observed}")
    return observed


def current_identity(path: Path) -> dict[str, Any]:
    from tools.hetero_verify_existing import file_stat
    return {**file_stat(path), "path": str(path)}


def wait_owned_child(proc, psutil, identity: dict[str, Any], sample_fn,
                     *, deadline_seconds: float = TIMEOUT_SECONDS, poll_seconds: float = 1.0) -> dict[str, Any]:
    """Bound a direct Popen child, rechecking its PID/create-time/exe/argv/parent every poll."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_seconds)
            return {"stdout": stdout or b"", "stderr": stderr or b"", "timed_out": False,
                    "resource_gate_failure": None, "identity_failure": None}
        except subprocess.TimeoutExpired:
            timed_out = time.monotonic() >= deadline
            current = None
            if not timed_out:
                try:
                    current = cpu_identity(psutil.Process(proc.pid), psutil)
                except psutil.NoSuchProcess:
                    current = None
                if current is None or not same_process(identity, current):
                    try:
                        if proc.poll() is None: proc.kill()
                    except OSError:
                        pass
                    stdout, stderr = proc.communicate(timeout=10)
                    return {"stdout": stdout or b"", "stderr": stderr or b"", "timed_out": False,
                            "resource_gate_failure": None, "identity_failure": current or "owned process disappeared"}
                try:
                    gate = sample_fn()
                except BaseException as exc:
                    gate = {"gate_ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if not gate.get("gate_ok"):
                    try:
                        if proc.poll() is None: proc.kill()
                    except OSError:
                        pass
                    stdout, stderr = proc.communicate(timeout=10)
                    return {"stdout": stdout or b"", "stderr": stderr or b"", "timed_out": False,
                            "resource_gate_failure": gate, "identity_failure": None}
                continue
            try:
                if proc.poll() is None: proc.kill()
            except OSError:
                pass
            stdout, stderr = proc.communicate(timeout=10)
            return {"stdout": stdout or b"", "stderr": stderr or b"", "timed_out": True,
                    "resource_gate_failure": None, "identity_failure": None}


def assert_identity_equal(expected: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    fields = ("size_bytes", "mtime_ns", "ctime_ns", "file_id")
    if any(expected.get(k) != actual.get(k) for k in fields):
        raise SweepError(f"{label} identity changed: expected={expected}, actual={actual}")


def run_case(case: dict[str, Any], c: dict[str, Any], inputs: dict[int, bytes], input_sha: dict[int, str],
             references: dict[int, bytes], exe_identity: dict[str, Any], samples) -> dict[str, Any]:
    import psutil
    if case["status"] != "planned":
        raise SweepError("attempted to execute an excluded/unplanned case")
    root = Path(c["output_root"])
    case_dir = root / case["case_id"]
    case_dir.mkdir(parents=False, exist_ok=False)
    rows = int(case["rows"]); phase = case["phase"]
    input_spec = next(x for x in c["input_specs"] if x["rows"] == rows)
    output = case_dir / "output.f32"; native_receipt_path = case_dir / "native-receipt.json"
    argv = ["--run", "--blob", c["blob_path"], "--input", input_spec["path"],
            "--rows", str(rows), "--hidden", "2560", "--intermediate", "640",
            "--gu-type", "12", "--down-type", "7", "--output", str(output),
            "--receipt-json", str(native_receipt_path), "--warmup", "1", "--repeat", "5"]
    if phase == "pool_matrix":
        argv += ["--pool-workers", str(case["workers"]), "--pool-affinity", case["affinity"]]
    attempt = {"schema_version": 1, "case": case, "argv": [str(Path(c["binary_path"])), *argv],
               "binary_identity_expected": exe_identity, "input_sha256_expected": input_sha[rows],
               "started_utc": utc(), "inherited_strata_environment_names_removed": True,
               "affinity_mutation_scope": "owned C++ child only; no parent process affinity calls"}
    write_new(case_dir / "attempt.json", attempt)
    before = sample_resource(f"{case['case_id']}:prelaunch", samples)
    if not before["gate_ok"]:
        write_new(case_dir / "failure.json", {"status": "failed_prelaunch_gate", "resource": before})
        raise SweepError(f"resource gate failed before {case['case_id']}")
    assert_identity_equal(exe_identity, current_identity(Path(c["binary_path"])), "candidate executable")
    env = os.environ.copy(); stripped = sorted(k for k in env if k.startswith("STRATA_"))
    for key in stripped: env.pop(key, None)
    proc = None; process_identity = None; stdout = stderr = b""; watch = None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.Popen([str(Path(c["binary_path"])), *argv], cwd=case_dir,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, creationflags=flags)
        process_identity = validate_direct_process(proc, psutil, Path(c["binary_path"]), argv)
        write_new(case_dir / "process-start.json", {"started_utc": utc(), "identity": process_identity,
                    "argv": [str(Path(c["binary_path"])), *argv], "popen_owned_handle": True,
                    "strata_environment_names_removed": stripped})
        watch = wait_owned_child(proc, psutil, process_identity,
                    lambda: sample_resource(f"{case['case_id']}:running", samples, enforce=False))
        stdout, stderr = watch["stdout"], watch["stderr"]
        write_stream_exclusive(case_dir / "stdout.log", stdout or b"")
        write_stream_exclusive(case_dir / "stderr.log", stderr or b"")
        process_result = {"identity": process_identity, "exit_code": proc.returncode,
                          "finished_utc": utc(), "timed_out": watch["timed_out"],
                          "resource_gate_failure": watch["resource_gate_failure"],
                          "identity_failure": watch["identity_failure"],
                          "full_argv": [str(Path(c["binary_path"])), *argv]}
        write_new(case_dir / "process.json", process_result)
        after = sample_resource(f"{case['case_id']}:finished", samples, enforce=False)
        if proc.returncode != 0 or not after["gate_ok"] or watch["timed_out"] or watch["resource_gate_failure"] or watch["identity_failure"]:
            failure = {"status": "failed_process_or_postflight_gate", "process": process_result,
                       "resource_after": after, "partial_output_retained": output.exists(),
                       "stdout_log": str(case_dir / "stdout.log"), "stderr_log": str(case_dir / "stderr.log")}
            write_new(case_dir / "failure.json", failure)
            raise SweepError(f"native harness failed for {case['case_id']}: {failure}")
        native = load(native_receipt_path)
        output_sha, output_stat = file_sha(output, rows * 2560 * 4)
        output_bytes = output.read_bytes()
        if sha(output_bytes) != output_sha or len(output_bytes) != rows * 2560 * 4:
            raise SweepError(f"output changed during readback: {output}")
        expected_reference = references.get(rows)
        if expected_reference is None:
            if phase != "default_reference_gate":
                raise SweepError(f"same-binary default reference missing for rows={rows}")
            references[rows] = output_bytes; expected_reference = output_bytes
        matches_reference = output_bytes == expected_reference
        receipt_ok = validate_native_receipt(native, case, c, input_sha[rows], output_sha, len(output_bytes))
        finite = finite_f32le(output_bytes, rows * 2560)
        result = {"schema_version": 1, "status": "pass" if matches_reference and receipt_ok and finite else "failed_parity_or_receipt",
                  "case": case, "process": process_result, "resource_before": before,
                  "resource_after": after, "output": {"path": str(output), "bytes": len(output_bytes),
                  "sha256": output_sha, "stat": output_stat, "finite": finite},
                  "matches_required_native_reference_bitwise": matches_reference,
                  "native_receipt_validated": receipt_ok, "pool_report": native.get("pool"),
                  "worker_pin_success_claimed": False}
        write_new(case_dir / "case-result.json", result)
        if result["status"] != "pass":
            write_new(case_dir / "failure.json", {"status": result["status"], "result": result})
            raise SweepError(f"native output/reference parity failed for {case['case_id']}")
        return result
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            try:
                # Popen owns the Windows process handle; never retarget termination by a bare PID.
                proc.kill(); stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                pass
        if not (case_dir / "stdout.log").exists():
            try: write_stream_exclusive(case_dir / "stdout.log", stdout or b"")
            except Exception: pass
        if not (case_dir / "stderr.log").exists():
            try: write_stream_exclusive(case_dir / "stderr.log", stderr or b"")
            except Exception: pass
        if not (case_dir / "failure.json").exists():
            write_new(case_dir / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "process_identity": process_identity,
                        "resource_gate_failure": watch.get("resource_gate_failure") if watch else None,
                        "identity_failure": watch.get("identity_failure") if watch else None,
                        "timed_out": watch.get("timed_out") if watch else None,
                        "full_argv": [str(Path(c["binary_path"])), *argv],
                        "exit_code": proc.poll() if proc is not None else None,
                        "partial_output_retained": output.exists(), "failed_utc": utc()})
        raise


def validate_native_receipt(native: dict[str, Any], case: dict[str, Any], c: dict[str, Any],
                            input_digest: str, output_digest: str, output_bytes: int) -> bool:
    parameters = native.get("parameters", {})
    expected_params = {"rows": case["rows"], "hidden": 2560, "intermediate": 640,
                       "gu_type": 12, "down_type": 7, "warmup": 1, "repeat": 5}
    source = native.get("source", {}); blob = native.get("blob", {}); inp = native.get("input", {})
    out = native.get("output", {}); execute = native.get("execution", {})
    controls = native.get("reference_controls", {})
    valid = (native.get("status") == "success" and parameters == expected_params and
             source.get("native_harness_source_sha256") == c["binary_source_sha256"] and
             blob.get("sha256") == c["blob_sha256"] and inp.get("sha256") == input_digest and
             out.get("sha256") == output_digest and out.get("bytes") == output_bytes and
             execute.get("native_kernel_called") is True and execute.get("gpu_executed") is False and
             execute.get("model_cli_executed") is False and
             all(controls.get(k) == v for k, v in c["reference_controls_expected"].items()))
    if case["phase"] == "default_reference_gate":
        return valid and execute.get("engine_pool_called") is False and native.get("pool") is None
    pool = native.get("pool", {})
    affinity = case["affinity"]
    expected_pin = affinity != "none"
    candidate = c["affinity_candidates"].get(affinity, {})
    worker_candidates = candidate.get("worker_candidates", []) if expected_pin else []
    if expected_pin and len(worker_candidates) < int(case["workers"]):
        raise SweepError("pool case exceeds the reviewed worker candidate list")
    observations = pool.get("worker_affinity_observations")
    observations_ok = isinstance(observations, list) and len(observations) == int(case["workers"])
    if observations_ok:
        for index, row in enumerate(observations):
            if (row.get("worker") != index or row.get("ready") is not True or
                    row.get("pin_requested") is not expected_pin or
                    row.get("pin_applied") is not expected_pin or row.get("mask_observed") is not True):
                observations_ok = False; break
            if expected_pin:
                core = worker_candidates[index]["native_cpu_id"]
                if (row.get("requested_core") != core or row.get("mask_matches_request") is not True or
                        row.get("observed_group") != core // 64 or row.get("observed_mask") != 1 << (core % 64)):
                    observations_ok = False; break
    return (valid and execute.get("engine_pool_called") is True and
            pool.get("requested_background_workers") == case["workers"] and
            pool.get("actual_background_workers") == case["workers"] and
            pool.get("effective_compute_participants") == case["workers"] + 1 and
            pool.get("host_works") is True and pool.get("affinity") == affinity and
            pool.get("pin") is expected_pin and
            pool.get("host_pin_applied") is expected_pin and
            pool.get("worker_pin_success_available") is True and observations_ok)


def prepare_inputs(c: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from tools.hetero_resources import memory_snapshot
    if Path(sys.executable).resolve() != Path(c["runner_interpreter"]).resolve():
        raise SweepError(f"input preparation must use {c['runner_interpreter']}")
    data_root = Path(c["data_root"]); inputs_dir = Path(c["inputs_root"])
    prep_receipt = Path(c["preparation_receipt"])
    if data_root.exists() or inputs_dir.exists() or prep_receipt.exists():
        raise FileExistsError(f"refusing existing prepared-input path: {data_root}")
    memory, errors = memory_snapshot()
    if (errors or type(memory.get("physical_available_bytes")) is not int or
            type(memory.get("commit_available_bytes")) is not int or
            memory["physical_available_bytes"] < PHYS_MIN or memory["commit_available_bytes"] < COMMIT_MIN):
        raise SweepError("fresh 12/4 GiB prepare gate failed or unknown")
    if shutil.disk_usage(data_root.parent).free < minimum_free_disk_bytes(c):
        raise SweepError("E: free-space reserve is below the reviewed floor")
    data_root.mkdir(parents=False, exist_ok=False); inputs_dir.mkdir(parents=False, exist_ok=False)
    entries = []
    for rows in (6, 10):
        raw, recipe = generate_input_bytes(rows, np)
        path = inputs_dir / f"input-{rows:03d}.f32"
        write_stream_exclusive(path, raw)
        readback_sha, readback_stat = file_sha(path, len(raw))
        readback = path.read_bytes()
        values = np.frombuffer(readback, dtype="<f4")
        if (readback_sha != sha(raw) or readback != raw or len(readback) != rows * 2560 * 4 or
                not bool(np.isfinite(values).all())):
            raise SweepError(f"written input readback validation failed: rows={rows}")
        entries.append({"rows": rows, "path": str(path), "bytes": len(raw), "sha256": sha(raw),
                        "readback_identity": readback_stat, "readback_sha256": readback_sha,
                        "readback_finite": True, "recipe": recipe})
    record = {"schema_version": 1, "status": "prepared_inputs_only", "run_id": c["run_id"],
              "created_utc": utc(), "gate": memory, "inputs": entries,
              "no_kernel_execution": True, "no_affinity_mutation": True, "no_binary_started": True}
    digest = write_new(prep_receipt, record)
    return {"receipt": str(prep_receipt), "sha256": digest, "inputs": entries, "status": record["status"]}


def _verify_sources(c: dict[str, Any]) -> dict[str, str]:
    result = {}
    for name, spec in c["source_files"].items():
        path = Path(spec["path"]); got = sha(path.read_bytes())
        if got != spec["sha256"]:
            raise SweepError(f"frozen source changed: {name}")
        result[name] = got
    return result


def _static_receipts(c: dict[str, Any]) -> dict[str, str]:
    checked = {}
    for label, path_key, sha_key in (
        ("inputs_manifest", "inputs_manifest_path", "inputs_manifest_sha256"),
        ("extraction_receipt", "extraction_receipt_path", "extraction_receipt_sha256"),
        ("identity", "expert_identity_path", "expert_identity_sha256"),
        ("build_receipt", "build_receipt_path", "build_receipt_sha256"),
        ("run32_frozen_source", "frozen_run32_source_path", "frozen_run32_source_sha256"),
        ("topology", "topology_receipt_path", "topology_receipt_sha256")):
        path = Path(c[path_key]); got = sha(path.read_bytes())
        if got != c[sha_key]: raise SweepError(f"frozen {label} receipt changed")
        checked[label] = got
    build = load(Path(c["build_receipt_path"]))
    if build.get("status") != "success" or build.get("processes", {}).get("ctest", {}).get("exit_code") != 0:
        raise SweepError("candidate CPU harness build/CTest receipt is not passing")
    source_rows = build.get("source_sha256_before", {})
    native_source = next((v for k,v in source_rows.items() if k.endswith("native_expert_bench.cpp")), None)
    if native_source != c["binary_source_sha256"]:
        raise SweepError("build receipt source does not match reviewed C++ source hash")
    identity = load(Path(c["expert_identity_path"]))
    if ((identity.get("layer"), identity.get("expert")) != (c["expert"]["layer"], c["expert"]["expert"]) or
        identity.get("source", {}).get("quantized_source_types") != {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"}):
        raise SweepError("selected real expert identity differs from the sweep contract")
    extraction = load(Path(c["extraction_receipt_path"]))
    if extraction.get("status") != "extracted" or extraction.get("selection") != {
            "layer": c["expert"]["layer"], "expert": c["expert"]["expert"], "profile_rank": 0}:
        raise SweepError("selected real expert extraction receipt differs from contract")
    manifest = load(Path(c["inputs_manifest_path"]))
    if manifest.get("blob", {}).get("sha256") != c["blob_sha256"] or manifest.get("blob", {}).get("bytes") != c["blob_bytes"]:
        raise SweepError("native blob identity differs from the frozen input manifest")
    binary = Path(c["binary_path"]); binary_sha, stat = file_sha(binary, c["binary_bytes"])
    if binary_sha != c["binary_sha256"]: raise SweepError("candidate binary SHA mismatch")
    checked["binary_sha256"] = binary_sha; checked["binary_stat"] = stat
    return checked


def static_preflight(c: dict[str, Any]) -> dict[str, Any]:
    """Read-only (except its caller-owned receipt) checks; never launches the C++ binary."""
    sources = _verify_sources(c)
    static = _static_receipts(c)
    blob_digest, blob_stat = file_sha(Path(c["blob_path"]), c["blob_bytes"])
    if blob_digest != c["blob_sha256"]: raise SweepError("native blob SHA mismatch")
    manifest = load(Path(c["inputs_manifest_path"]))
    manifest_inputs = {x["rows"]: x for x in manifest["inputs"]}
    data_root_exists = Path(c["data_root"]).exists()
    inputs_root_exists = Path(c["inputs_root"]).exists()
    output_root_exists = Path(c["output_root"]).exists()
    verified_frozen = {}
    for row_text, spec in c["frozen_outputs"].items():
        rows = int(row_text)
        raw, identity = checked_file({"path": spec["path"], "bytes": spec["bytes"], "sha256": spec["sha256"]})
        if not finite_f32le(raw, rows * 2560): raise SweepError(f"frozen output nonfinite: rows={rows}")
        receipt_path = Path(spec["receipt_path"])
        receipt = load(receipt_path)
        if (sha(receipt_path.read_bytes()) != spec["receipt_sha256"] or
            receipt.get("input", {}).get("sha256") != manifest_inputs[rows].get("sha256") or
            receipt.get("output", {}).get("sha256") != spec["sha256"]):
            raise SweepError(f"frozen output receipt mismatch: rows={rows}")
        verified_frozen[str(rows)] = {"path": spec["path"], "bytes": len(raw), "sha256": sha(raw), "identity": identity}
    return {"schema_version": 1, "status": "static_preflight_pass_no_kernel", "run_id": c["run_id"],
            "checked_utc": utc(), "runner_path": c["runner_path"], "runner_sha256": sources["runner"],
            "source_hashes": sources, "static_receipts": static,
            "candidate_binary_path": c["binary_path"], "candidate_binary_sha256": static["binary_sha256"],
            "selected_blob": {"path": c["blob_path"], "bytes": c["blob_bytes"], "sha256": blob_digest, "identity": blob_stat},
            "frozen_outputs": verified_frozen,
            "required_input_rows": [x["rows"] for x in c["input_specs"]],
            "missing_inputs_rows_not_read": [6, 10], "preparation_authorized": c["input_preparation_authorized"],
            "execution_authorized": c["execution_authorized"], "kernel_started": False,
            "affinity_mutated": False, "data_root_exists_at_check": data_root_exists,
            "inputs_root_exists_at_check": inputs_root_exists, "output_root_exists_at_check": output_root_exists}


def _prepared_inputs(c: dict[str, Any]) -> dict[int, dict[str, Any]]:
    import numpy as np
    manifest_path = Path(c["preparation_receipt"])
    if not manifest_path.is_file(): raise SweepError("authorized input-preparation receipt is missing")
    manifest = load(manifest_path)
    if manifest.get("status") != "prepared_inputs_only": raise SweepError("input preparation receipt not passing")
    found = {item["rows"]: item for item in manifest.get("inputs", [])}
    if set(found) != {6, 10}: raise SweepError("prepared rows must be exactly 6 and 10")
    for rows, item in found.items():
        planned = next(x for x in c["input_specs"] if x["rows"] == rows)
        if Path(item["path"]) != Path(planned["path"]) or item["bytes"] != planned["bytes"]:
            raise SweepError(f"prepared input path/size differs from contract: rows={rows}")
        expected, recipe = generate_input_bytes(rows, np)
        if item.get("recipe") != recipe or item.get("sha256") != sha(expected):
            raise SweepError(f"prepared input recipe/digest mismatch: rows={rows}")
        actual, _ = checked_file({"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]})
        if actual != expected:
            raise SweepError(f"prepared input bytes differ from deterministic recipe: rows={rows}")
    return found


def _fresh_topology(c: dict[str, Any]) -> dict[str, Any]:
    from tools.hetero_cpu_topology import collect_inventory
    fresh = collect_inventory()
    if fresh.get("platform") != "windows": raise SweepError("affinity runner requires the reviewed Windows CPU Set API")
    planned = c["affinity_candidates"]
    report = fresh["pool_affinity_candidates"]
    for name in ("all", "auto", "p-cores"):
        key = f"pool-affinity-{name}"
        current = report[key]
        want = planned[name]
        actual_worker = [(x["group"], x["logical_processor_index"]) for x in current["worker_candidates"]]
        expected_worker = [(x["group"], x["logical_processor_index"]) for x in want["worker_candidates"]]
        if (actual_worker != expected_worker or
                current["host_core_candidate"] != {k: want["host_core_candidate"][k] for k in ("group", "logical_processor_index")}):
            raise SweepError(f"fresh CPU affinity candidates differ from reviewed topology for {name}")
    return fresh


def run_sweep(c: dict[str, Any]) -> dict[str, Any]:
    if c.get("execution_authorized") is not True:
        raise SweepError("CPU affinity execution is not authorized in this contract")
    if Path(sys.executable).resolve() != Path(c["runner_interpreter"]).resolve():
        raise SweepError(f"sweep must use {c['runner_interpreter']}")
    sources = _verify_sources(c)
    static = _static_receipts(c)
    prepared = {}
    fresh = _fresh_topology(c)
    outroot = Path(c["output_root"])
    if outroot.exists(): raise FileExistsError(f"refusing existing output root: {outroot}")
    if shutil.disk_usage(outroot.parent).free < minimum_free_disk_bytes(c):
        raise SweepError("E: free-space reserve is below the reviewed floor")
    import psutil
    # Hash all bounded inputs/references once; subsequent cases compare retained byte buffers.
    input_bytes: dict[int, bytes] = {}; input_hashes: dict[int, str] = {}; frozen: dict[int, bytes] = {}
    inputs_manifest = load(Path(c["inputs_manifest_path"]))
    manifest_rows = {x["rows"]: x for x in inputs_manifest["inputs"]}
    for spec in c["input_specs"]:
        rows = int(spec["rows"])
        expected = prepared[rows] if rows in prepared else manifest_rows.get(rows)
        if not expected: raise SweepError(f"missing reviewed input evidence for rows={rows}")
        merged = {"path": spec["path"], "bytes": spec["bytes"], "sha256": expected["sha256"]}
        raw, identity = checked_file(merged)
        if rows in prepared and sha(raw) != expected["sha256"]: raise SweepError(f"prepared input hash changed: rows={rows}")
        input_bytes[rows], input_hashes[rows] = raw, sha(raw)
    for row_key, spec in c["frozen_outputs"].items():
        rows = int(row_key)
        receipt_path = Path(spec["receipt_path"])
        if sha(receipt_path.read_bytes()) != spec["receipt_sha256"]: raise SweepError(f"frozen native receipt changed: rows={rows}")
        native_receipt = load(receipt_path)
        input_manifest_row = manifest_rows.get(rows)
        if (native_receipt.get("status") != "success" or native_receipt.get("execution", {}).get("native_kernel_called") is not True or
                native_receipt.get("source", {}).get("native_harness_source_sha256") != c["old_native_source_sha256"] or
                native_receipt.get("output", {}).get("sha256") != spec["sha256"] or
                native_receipt.get("blob", {}).get("sha256") != c["blob_sha256"] or not input_manifest_row or
                native_receipt.get("input", {}).get("sha256") != input_manifest_row.get("sha256") or
                native_receipt.get("parameters") != {"rows": rows, "hidden": 2560, "intermediate": 640,
                    "gu_type": 12, "down_type": 7, "warmup": 1, "repeat": 5} or
                any(native_receipt.get("reference_controls", {}).get(k) != v
                    for k,v in c["reference_controls_expected"].items())):
            raise SweepError(f"frozen native output provenance mismatch: rows={rows}")
        raw, _ = checked_file({"path": spec["path"], "bytes": spec["bytes"], "sha256": spec["sha256"]})
        frozen[rows] = raw
    blob_sha, blob_stat = file_sha(Path(c["blob_path"]), c["blob_bytes"])
    if blob_sha != c["blob_sha256"]: raise SweepError("native quantized blob identity mismatch")

    outroot.mkdir(parents=False, exist_ok=False)
    plan = make_plan(c)
    write_new(outroot / "execution-plan.json", {"plan": plan, "current_topology": fresh,
               "static_identity": static, "source_hashes": sources, "prepared_input_receipt": str(c["preparation_receipt"])})
    results: list[dict[str, Any]] = []
    samples_path = outroot / "resource-samples.jsonl"
    try:
        with samples_path.open("x", encoding="utf-8", buffering=1) as samples:
            sample_resource("sweep-start", samples)
            exe_identity = {**current_identity(Path(c["binary_path"])), "sha256": c["binary_sha256"]}
            gates = [x for x in plan["cases"] if x["phase"] == "default_reference_gate"]
            for case in gates:
                results.append(run_case(case, c, input_bytes, input_hashes, frozen, exe_identity, samples))
            runnable_pool = [x for x in plan["cases"] if x["phase"] == "pool_matrix" and x["status"] == "planned"]
            for case in runnable_pool:
                results.append(run_case(case, c, input_bytes, input_hashes, frozen, exe_identity, samples))
        summary = {"schema_version": 1, "status": "complete", "run_id": c["run_id"],
                   "completed_utc": utc(), "results": results,
                   "counts": {"default_gates": len(gates), "pool_cases": len(runnable_pool),
                              "strict_pcore_not_run": plan["strict_pcore_overflow_not_run"]},
                   "claim_limit": "native CPU numerical parity/worker-affinity microbenchmark only; no model or throughput claim",
                   "worker_pin_success_observed": False, "p_e_labels_verified": False}
        write_new(outroot / "run-summary.json", summary)
        return summary
    except BaseException as exc:
        write_new(outroot / "run-failure.json", {"schema_version": 1, "status": "failed",
                   "failed_utc": utc(), "error": f"{type(exc).__name__}: {exc}",
                   "completed_case_count": len(results), "partial_outputs_retained": True,
                   "worker_pin_success_observed": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--prepare", action="store_true", help="generate only rows 6/10 inputs after explicit approval")
    parser.add_argument("--root-prepare-confirmed", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--root-start-confirmed", action="store_true")
    parser.add_argument("--models-terminal-confirmed", action="store_true")
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--root-static-preflight-confirmed", action="store_true")
    parser.add_argument("--static-preflight-output", type=Path, default=RUN / "static-preflight-01.json")
    args = parser.parse_args(argv)
    c = load(args.contract)
    if sum((args.prepare, args.run, args.static_preflight)) > 1: parser.error("choose one operation")
    if args.static_preflight:
        if not args.root_static_preflight_confirmed:
            parser.error("static checks require --root-static-preflight-confirmed")
        proof = static_preflight(c)
        contract_path = Path(args.contract).resolve()
        proof["contract_path"] = str(contract_path)
        proof["contract_sha256"] = sha(contract_path.read_bytes())
        target = args.static_preflight_output.resolve()
        if not target.is_relative_to(RUN):
            parser.error("static preflight receipt must stay under this run directory")
        write_new(target, proof)
        print(json.dumps({"status": proof["status"], "receipt": str(target),
                          "binary_sha256": proof["candidate_binary_sha256"],
                          "blob_sha256": proof["selected_blob"]["sha256"],
                          "frozen_rows": sorted(map(int, proof["frozen_outputs"]))}, indent=2))
        return 0
    if args.prepare:
        if not args.root_prepare_confirmed or c.get("input_preparation_authorized") is not True:
            parser.error("input preparation requires contract authorization and --root-prepare-confirmed")
        print(json.dumps(prepare_inputs(c), indent=2)); return 0
    if args.run:
        if not (args.root_start_confirmed and args.models_terminal_confirmed):
            parser.error("execution requires --root-start-confirmed and --models-terminal-confirmed")
        if c.get("execution_authorized") is not True:
            parser.error("execution_authorized is false in the reviewed contract")
        print(json.dumps(run_sweep(c), indent=2)); return 0
    print(json.dumps(make_plan(c), indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"CPU affinity sweep: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
