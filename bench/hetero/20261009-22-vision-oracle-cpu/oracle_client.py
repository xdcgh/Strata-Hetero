#!/usr/bin/env python3
"""Validate by default; --run explicitly starts the owned CPU vision oracle."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, math, os, queue, struct, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.hetero_resources import memory_snapshot
HELPER = Path(r"E:\Strata-Hetero-data\build\vision-oracle-cpu-20261009-20\bin\strata-vision.exe")
MM = Path(r"E:\Strata-Hetero-data\models\vision-ed59f92\mmproj-Qwen3.8-Flash-Next-BF16.gguf")
MODEL = Path(r"F:\Strata-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
FIXTURES = Path(r"E:\Strata-Hetero-data\vision-fixtures\20261009-22-vision-oracle-cpu")
RUNS = Path(r"E:\Strata-Hetero-data\vision-oracle-runs\20261009-22-vision-oracle-cpu")
HELPER_SHA = "7285dd08f24c3985fd6b97aaff7d9917b65e5ea8b263ae5567985d0dc04ba612"
MM_SHA = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
MODEL_SHA = "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"
MODEL_SIZE = 10946624
HIDDEN, HEADER = 2560, struct.Struct("<5i")
MAX_OUTPUT = HEADER.size + 256 * HIDDEN * 4
MAX_STDOUT_LINE = 16 * 1024

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def validate_sve(path, response):
    fields = response.strip().split()
    if len(fields) != 5 or fields[0] != "OK": raise ValueError(f"bad encoder response: {response[:160]!r}")
    n, rx, ry = map(int, fields[1:4]); ms = float(fields[4])
    if min(n, rx, ry) < 1 or n != rx * ry or not math.isfinite(ms) or ms < 0: raise ValueError("bad response dimensions/time")
    size = path.stat().st_size
    if not HEADER.size <= size <= MAX_OUTPUT: raise ValueError("SVE1 size outside bounded range")
    raw = path.read_bytes(); magic, hn, nx, ny, embd = HEADER.unpack_from(raw)
    if (magic, hn, nx, ny, embd) != (0x31455653, n, rx, ry, HIDDEN): raise ValueError("SVE1 header mismatch")
    if len(raw) != HEADER.size + n * embd * 4: raise ValueError("SVE1 body size is not exact")
    if any(not math.isfinite(x[0]) for x in struct.iter_unpack("<f", memoryview(raw)[HEADER.size:])):
        raise ValueError("SVE1 contains NaN or infinity")
    return {"n_tokens": n, "nx": nx, "ny": ny, "n_embd": embd, "size_bytes": size,
            "sha256": hashlib.sha256(raw).hexdigest(), "reported_encoder_ms": ms}

def reader(stream, out):
    try:
        while True:
            line = stream.readline(MAX_STDOUT_LINE + 1)
            if not line: out.put(None); return
            if len(line) > MAX_STDOUT_LINE: out.put(ValueError("stdout line exceeded 16 KiB")); return
            out.put(line.decode("utf-8", "replace").strip(), timeout=1)
    except Exception as exc:
        try: out.put(exc, timeout=1)
        except queue.Full: pass

def read_line(out, seconds):
    try: value = out.get(timeout=seconds)
    except queue.Empty as exc: raise TimeoutError(f"stdout timeout after {seconds}s") from exc
    if isinstance(value, Exception): raise RuntimeError(f"stdout reader: {value}")
    if value is None: raise RuntimeError("helper closed stdout")
    return value

def stop_owned(proc):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=3)

def idle(proc):
    if os.name == "nt":
        fn = ctypes.WinDLL("kernel32", use_last_error=True).SetPriorityClass
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int
        if not fn(ctypes.c_void_p(proc._handle), 0x40): raise OSError(ctypes.get_last_error(), "SetPriorityClass(IDLE)")

def process_created_utc(proc):
    class FILETIME(ctypes.Structure): _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]
    created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
    fn = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
    fn.restype = ctypes.c_int
    if not fn(ctypes.c_void_p(proc._handle), *(ctypes.byref(x) for x in (created, exited, kernel, user))):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed for owned helper")
    ticks = (created.high << 32) | created.low
    return datetime.fromtimestamp(ticks / 10_000_000 - 11_644_473_600, timezone.utc).isoformat()

def verify_model_identity():
    before = MODEL.stat()
    current_sha = sha(MODEL)
    after = MODEL.stat()
    before_id = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_dev, before.st_ino)
    after_id = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_dev, after.st_ino)
    if before_id != after_id or after.st_size != MODEL_SIZE or current_sha != MODEL_SHA:
        raise RuntimeError("vocab-only GGUF shard stat/SHA does not match model-integrity-02")
    return {"path": str(MODEL), "size_bytes": after.st_size, "sha256": current_sha,
            "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns,
            "volume": after.st_dev, "file_index": after.st_ino, "stat_stable_while_hashed": True,
            "identity_receipt": "bench/hetero/20261008-00-admission/model-integrity-02.json"}

def validate_only():
    manifest = json.loads(Path(__file__).with_name("fixture-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "strata-vision-fixtures-v1" or len(manifest.get("fixtures", [])) != 4:
        raise ValueError("fixture manifest schema/count mismatch")
    expected = {"gray-96": (96, 96), "quadrants-96": (96, 96), "checkerboard-96": (96, 96), "gradient-192x96": (192, 96)}
    for item in manifest["fixtures"]:
        p = Path(item["path"]).resolve(strict=True)
        if p.parent != FIXTURES.resolve() or (item["width"], item["height"]) != expected.get(item["name"]):
            raise ValueError(f"fixture path or dimensions outside fixed contract: {p}")
        if not p.is_file() or sha(p) != item["sha256"]: raise ValueError(f"fixture missing/SHA mismatch: {p}")
    if not HELPER.is_file() or sha(HELPER) != HELPER_SHA: raise ValueError("helper missing/SHA mismatch")
    return {"schema": "strata-vision-oracle-client-v1", "status": "validated_only", "helper_spawned": False,
            "mmproj_opened": False, "model_opened": False, "core_created": False, "device_touched": False,
            "fixture_count": 4}

def run():
    validate_only()
    if not MM.is_file() or not MODEL.is_file(): raise RuntimeError("fixed mmproj/model path missing")
    if any(" " in str(p) for p in (HELPER, MM, MODEL, FIXTURES, RUNS)): raise RuntimeError("protocol paths cannot contain spaces")
    run_dir = RUNS / time.strftime("run-%Y%m%dT%H%M%SZ", time.gmtime()); run_dir.mkdir(parents=True, exist_ok=False)
    progress = (run_dir / "progress.jsonl").open("x", encoding="utf-8")
    stderr = None; proc = None; phase = "resource-before-startup"; pid = None; created = None; exit_code = None
    model_identity = None; mm_sha = None
    results = []; gates = []
    argv = [str(HELPER), "--mmproj", str(MM), "--model", str(MODEL), "--threads", "10", "--max-tokens", "256", "--flash-attn", "off"]
    def record(kind, **fields):
        progress.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
                                   "event": kind, **fields}, separators=(",", ":")) + "\n")
        progress.flush(); os.fsync(progress.fileno())
    def gate(label):
        memory, errors = memory_snapshot()
        physical, commit = memory.get("physical_available_bytes"), memory.get("commit_available_bytes")
        passed = (not errors and type(physical) is int and type(commit) is int and
                  physical >= 12 * 1024**3 and commit >= 4 * 1024**3)
        info = {"label": label, "source": memory.get("source"), "physical_available_bytes": physical,
                "commit_available_bytes": commit, "errors": errors, "passed": passed}
        gates.append(info)
        record("resource_gate", **info)
        if not passed: raise RuntimeError(f"resource gate unknown/failed at {label}: {errors or info}")
        return info
    try:
        gate("before-startup")
        phase = "input-identity"
        model_identity = verify_model_identity()
        mm_sha = sha(MM)
        if mm_sha != MM_SHA: raise RuntimeError("fixed mmproj SHA-256 mismatch")
        record("input_identity", model=model_identity, mmproj={"path": str(MM), "sha256": mm_sha, "size_bytes": MM.stat().st_size})
        phase = "resource-immediately-before-helper-startup"; gate("before-helper-startup")
        stderr = (run_dir / "helper.stderr.log").open("xb")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        phase = "startup"
        t0 = time.perf_counter_ns()
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                shell=False, creationflags=flags, bufsize=0)
        pid, created = proc.pid, process_created_utc(proc); idle(proc)
        lines = queue.Queue(maxsize=8); threading.Thread(target=reader, args=(proc.stdout, lines), daemon=True).start()
        ready = read_line(lines, 180)
        if ready != f"READY {HIDDEN}": raise RuntimeError(f"unexpected READY line: {ready[:160]!r}")
        startup = time.perf_counter_ns() - t0; record("ready", line=ready, startup_wall_ns=startup, pid=pid, process_create_utc=created)
        images = json.loads(Path(__file__).with_name("fixture-manifest.json").read_text(encoding="utf-8"))["fixtures"]
        for image in images:
            for i in range(4):
                label = "warmup" if i == 0 else f"formal-{i}"
                phase = f"{image['name']}:{label}:before-ENC"; before = gate(phase)
                out = run_dir / f"{image['name']}-{label}.sve"
                if out.exists(): raise FileExistsError(out)
                command = f"ENC {image['path']} {out}"; replies = queue.Queue(maxsize=1); began = time.perf_counter_ns()
                def exchange():
                    try: proc.stdin.write((command + "\n").encode()); proc.stdin.flush(); replies.put(read_line(lines, 120))
                    except BaseException as exc: replies.put(exc)
                threading.Thread(target=exchange, daemon=True).start()
                try: response = replies.get(timeout=120)
                except queue.Empty as exc: raise TimeoutError("ENC write/read exceeded 120s") from exc
                if isinstance(response, BaseException): raise response
                response_ns = time.perf_counter_ns(); phase = f"{image['name']}:{label}:response"
                record("encoder_response", response=response, output_path=str(out))
                phase = f"{image['name']}:{label}:after-ENC"; after = gate(phase)
                parsed = validate_sve(out, response)
                row = {"image": image["name"], "iteration": label, "before_gate": before, "after_gate": after,
                       "client_rtt_ns_write_to_response_line": response_ns - began,
                       "client_wall_ns_through_sve_validation": time.perf_counter_ns() - began,
                       "response": response, **parsed}
                results.append(row); phase = f"{image['name']}:{label}:validated"; record("result", result=row)
        phase = "quit"; proc.stdin.write(b"QUIT\n"); proc.stdin.flush(); exit_code = proc.wait(timeout=10)
        record("process_exit", pid=pid, process_create_utc=created, exit_code=exit_code)
        if exit_code != 0: raise RuntimeError(f"owned helper exited {exit_code} after QUIT")
        proc = None
        comparison = {}
        for image in images:
            group = [r for r in results if r["image"] == image["name"]]
            hashes = [r["sha256"] for r in group if r["iteration"].startswith("formal-")]
            comparison[image["name"]] = {"warmup_sha256": group[0]["sha256"], "formal_sha256": hashes,
                                          "distinct_formal_hashes": len(set(hashes)), "all_formals_identical": len(set(hashes)) == 1}
        receipt = {"schema": "strata-vision-oracle-run-v1", "status": "completed", "argv": argv,
                   "helper_sha256": HELPER_SHA, "mmproj_sha256": mm_sha, "model_identity": model_identity,
                   "pid": pid, "process_create_utc": created, "exit_code": exit_code, "startup_wall_ns": startup,
                   "resource_gates": gates, "repeat_comparison": comparison, "results": results}
        record("completed", receipt="oracle-receipt.json", exit_code=exit_code)
        with (run_dir / "oracle-receipt.json").open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        return receipt
    except BaseException as exc:
        cleanup_error = None
        if proc is not None:
            try: stop_owned(proc)
            except Exception as stop_exc: cleanup_error = f"{type(stop_exc).__name__}: {stop_exc}"
            exit_code = proc.poll()
        failure = {"schema": "strata-vision-oracle-run-v1", "status": "failed", "phase": phase,
                   "argv": argv, "pid": pid, "process_create_utc": created, "exit_code": exit_code,
                   "model_identity": model_identity, "mmproj_sha256": mm_sha,
                   "resource_gates": gates, "partial_results": results,
                   "error": f"{type(exc).__name__}: {exc}", "cleanup_error": cleanup_error}
        record("failed", error=failure["error"], cleanup_error=cleanup_error, partial_result_count=len(results))
        with (run_dir / "failed-oracle-receipt.json").open("x", encoding="utf-8") as f:
            json.dump(failure, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        raise
    finally:
        if proc is not None:
            try: stop_owned(proc)
            except Exception: pass
        if stderr is not None: stderr.close()
        progress.close()

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", help="explicitly start owned CPU helper")
    args = parser.parse_args(argv)
    try: value = run() if args.run else validate_only()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr); return 1
    print(json.dumps(value, sort_keys=True, separators=(",", ":"))); return 0

if __name__ == "__main__": raise SystemExit(main())
