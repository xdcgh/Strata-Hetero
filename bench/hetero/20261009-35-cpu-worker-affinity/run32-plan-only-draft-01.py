#!/usr/bin/env python3
"""Plan or run bounded real-native CPU expert byte-parity cases."""
from __future__ import annotations
import argparse, datetime as dt, filecmp, hashlib, json, math, os, struct, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.hetero_copy_verified import write_receipt_exclusive
from tools.hetero_resources import memory_snapshot
from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged
CONTRACT = Path(__file__).with_name("contract.json")
PHYS_MIN, COMMIT_MIN, CHUNK = 12 * 1024**3, 4 * 1024**3, 1024 * 1024
def load(path: Path): return json.loads(path.read_text(encoding="utf-8-sig"))
def digest(path: Path):
    before = file_stat(path)
    h, size = hashlib.sha256(), 0
    with path.open("rb", buffering=0) as f:
        if not fd_matches_path(os.fstat(f.fileno()), before):
            raise RuntimeError(f"file identity changed before read: {path}")
        while block := f.read(CHUNK):
            h.update(block); size += len(block)
        final_fd = os.fstat(f.fileno())
    after = file_stat(path)
    if not stat_unchanged(before, after) or not fd_matches_path(final_fd, before) or size != before["size_bytes"]:
        raise RuntimeError(f"file identity changed during read: {path}")
    return h.hexdigest(), before
def checked(path: str, size: int, sha: str):
    got, identity = digest(Path(path))
    if identity["size_bytes"] != size or got != sha: raise RuntimeError(f"size/hash mismatch: {path} ({identity['size_bytes']}, {got})")
    return {**identity, "path": path, "sha256": got}
def static_receipts(c):
    if digest(Path(c["inputs_manifest_path"]))[0] != c["inputs_manifest_sha256"]: raise RuntimeError("inputs manifest changed")
    if digest(Path(c["expert_identity_path"]))[0] != c["expert_identity_sha256"] or digest(Path(c["extraction_receipt_path"]))[0] != c["extraction_receipt_sha256"]:
        raise RuntimeError("real-expert identity receipts changed")
    br = Path(c["build_receipt_path"])
    if digest(br)[0] != c["build_receipt_sha256"]: raise RuntimeError("reviewed candidate build receipt changed")
    build = load(br)
    source = next(x for x in build["source"]["tracked_and_untracked_source_hashes_before"] if x["path"].endswith("native_expert_bench.cpp"))
    if build.get("status") != "success" or source["sha256"] != c["new_harness_source_sha256"] or build["actual_exit_codes"]["ctest"] != 0: raise RuntimeError("candidate binary/source/CTest identity mismatch")
    inputs, pre, summary = load(Path(c["inputs_manifest_path"])), load(Path(c["old_native_preflight_path"])), load(Path(c["old_native_summary_path"]))
    if summary.get("status") != "complete_with_quality_failures" or pre["harness_binary"]["sha256"] != c["old_harness_sha256"]: raise RuntimeError("frozen native run/binary identity mismatch")
    if inputs["blob"]["path"] != c["blob_path"] or inputs["blob"]["bytes"] != c["blob_bytes"] or inputs["blob"]["sha256"] != c["blob_sha256"]: raise RuntimeError("native blob receipt mismatch")
    ident, extraction = load(Path(c["expert_identity_path"])), load(Path(c["extraction_receipt_path"]))
    if (ident.get("layer"), ident.get("expert")) != (c["expert_layer"], c["expert_id"]) or extraction["selection"] != {"layer": c["expert_layer"], "expert": c["expert_id"], "profile_rank": 0}: raise RuntimeError("expert tuple differs from selected real expert")
    for row, ref in c["frozen_outputs"].items():
        rows = int(row); old = load(Path(ref["native_receipt_path"]))
        entry = next(x for x in inputs["inputs"] if x["rows"] == rows)
        spec = next(x for x in c["inputs"] if x["rows"] == rows)
        done = next(x for x in summary["completed_rows"] if x["rows"] == rows)
        if (old.get("status") != "success" or old["output"]["path"] != ref["path"] or old["output"]["sha256"] != ref["sha256"] or
            old["source"]["native_harness_source_sha256"] != c["old_harness_source_sha256"] or old["blob"]["sha256"] != c["blob_sha256"] or
            old["input"]["sha256"] != spec["sha256"] or (entry["path"], entry["bytes"], entry["sha256"]) != (spec["path"], spec["bytes"], spec["sha256"]) or
            old["parameters"] != {"rows": rows, "hidden": 2560, "intermediate": 640, "gu_type": 12, "down_type": 7, "warmup": 1, "repeat": 5} or
            old["reference_controls"]["STRATA_KQ256"] != "0 (process-local)" or old["reference_controls"]["q8k_avx2_selected"] is not True or done["native_process_exit"] != 0 or
            digest(Path(ref["native_receipt_path"]))[0] != ref["native_receipt_sha256"]):
            raise RuntimeError(f"frozen row {rows} receipt mismatch")
def plan(c):
    if "sweep_plan" in c:
        return c["sweep_plan"]
    if "plan_cases" in c:
        return c["plan_cases"]
    root = Path(c["e_output_root"])
    return ([{"phase": "single_thread_gate", "rows": n, "workers": None,
              "output": str(root / "single-thread-default" / f"rows-{n:03d}" / "output.f32")} for n in c["gate_rows"]] +
            [{"phase": "pool_parity", "rows": n, "workers": w, "affinity": "none",
              "output": str(root / f"pool-workers-{w:02d}" / f"rows-{n:03d}" / "output.f32")}
             for w in c["pool_workers"] for n in c["gate_rows"]])

def sample(stream, label, enforce=True):
    mem, errors = memory_snapshot()
    row = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "label": label,
           "physical_available_bytes": mem.get("physical_available_bytes"),
           "commit_available_bytes": mem.get("commit_available_bytes"), "memory_source": mem.get("source"), "errors": errors}
    row["gate_ok"] = (not errors and type(row["physical_available_bytes"]) is int and type(row["commit_available_bytes"]) is int and
                      row["physical_available_bytes"] >= PHYS_MIN and row["commit_available_bytes"] >= COMMIT_MIN)
    stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
    if enforce and not row["gate_ok"]:
        raise RuntimeError(f"12/4 GiB resource gate failed: {row}")
    return row
def stdout_file(path, data):
    with path.open("xb") as f:
        f.write(data or b""); f.flush(); os.fsync(f.fileno())
def child(exe, argv, folder, samples, label, exe_identity):
    folder.mkdir(parents=True, exist_ok=False)
    before = sample(samples, f"{label}:prelaunch", enforce=False)
    env = os.environ.copy()
    keys = tuple(k for k in env if k.startswith("STRATA_"))
    inherited = {k: env[k] for k in keys}
    for key in keys: env.pop(key, None)
    attempt = {"label": label, "argv": argv, "executable": str(exe), "cwd": str(folder),
               "resource_before": before, "executable_identity_expected": exe_identity,
               "inherited_controls": inherited, "child_removed_controls": list(keys)}
    write_receipt_exclusive(folder / "attempt.json", attempt)
    write_receipt_exclusive(folder / "stage.json", {"stage": "launch_attempted", "time_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    current_exe = file_stat(exe)
    if any(current_exe[k] != exe_identity[k] for k in ("size_bytes", "mtime_ns", "ctime_ns", "file_id")):
        write_receipt_exclusive(folder / "failure.json", {"error": "candidate executable identity changed", "actual": current_exe})
        raise RuntimeError("candidate executable path identity changed")
    if not before["gate_ok"]:
        write_receipt_exclusive(folder / "failure.json", {"error": "prelaunch resource gate failed", "sample": before})
        raise RuntimeError(f"resource gate failed before {label}")
    proc = None
    try:
        proc = subprocess.Popen([str(exe), *argv], cwd=folder, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW, env=env)
        start = {"pid": proc.pid, "popen_owned_handle": True, "argv": argv,
                 "start_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_receipt_exclusive(folder / "process-start.json", start)
        out = err = b""
        deadline = time.monotonic() + 90
        while True:
            try:
                out, err = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded native microcase exceeded 90 seconds")
                current = sample(samples, f"{label}:running", enforce=False)
                if not current["gate_ok"]:
                    proc.kill()  # Terminate only the process represented by this owned Popen handle.
                    out, err = proc.communicate(timeout=10)
                    stdout_file(folder / "stdout.log", out); stdout_file(folder / "stderr.log", err)
                    raise RuntimeError(f"resource gate crossed; owned PID {proc.pid} was stopped: {current}")
        stdout_file(folder / "stdout.log", out); stdout_file(folder / "stderr.log", err)
        finish = sample(samples, f"{label}:finished", enforce=False)
        result = {**start, "exit_code": proc.returncode, "finish_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "resource_finish": finish}
        write_receipt_exclusive(folder / "process.json", result)
        if proc.returncode != 0:
            write_receipt_exclusive(folder / "failure.json", {"error": f"child exit {proc.returncode}", "process": result})
            raise RuntimeError(f"child exited {proc.returncode}; see retained stdout/stderr")
        if not finish["gate_ok"]:
            write_receipt_exclusive(folder / "failure.json", {"error": "post-child resource gate failed", "process": result})
            raise RuntimeError(f"resource gate failed after {label}")
        return out.decode("utf-8"), result
    except BaseException as original:
        detail = {"status": "failed", "original_exception": f"{type(original).__name__}: {original}",
                  "pid": proc.pid if proc else None, "exit_code": proc.poll() if proc else None,
                  "partial_stdout_stderr_retained": False}
        cleanup_error = None
        if proc and proc.poll() is None:
            try:
                proc.kill(); out, err = proc.communicate(timeout=10)
                stdout_file(folder / "stdout.log", out); stdout_file(folder / "stderr.log", err)
                detail["partial_stdout_stderr_retained"] = True
            except Exception as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
        if cleanup_error: detail["cleanup_error"] = cleanup_error
        if not (folder / "failure.json").exists():
            write_receipt_exclusive(folder / "failure.json", detail)
        raise

def finite_sha(path):
    h, tail, count = hashlib.sha256(), b"", 0
    finite = True
    with path.open("rb", buffering=0) as f:
        while block := f.read(CHUNK):
            h.update(block); block = tail + block
            usable = len(block) - len(block) % 4
            for (value,) in struct.iter_unpack("<f", block[:usable]):
                count += 1
                if not math.isfinite(value):
                    finite = False
            tail = block[usable:]
    return h.hexdigest(), count, finite and not tail
def equal_bytes(a, b): return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
def run_case(c, case, samples, defaults, exe_identity):
    rows, workers = case["rows"], case["workers"]
    folder = Path(case["output"]).parent
    inp = next(x for x in c["inputs"] if x["rows"] == rows)
    out, receipt = folder / "output.f32", folder / "native-receipt.json"
    argv = ["--run", "--blob", c["blob_path"], "--input", inp["path"], "--rows", str(rows),
            "--hidden", "2560", "--intermediate", "640", "--gu-type", "12", "--down-type", "7",
            "--output", str(out), "--receipt-json", str(receipt), "--warmup", "1", "--repeat", "3"]
    if workers is not None:
        argv += ["--pool-workers", str(workers), "--pool-affinity", "none"]
    text, process = child(Path(c["new_harness_path"]), argv, folder, samples, f"rows-{rows}-workers-{workers or 0}", exe_identity)
    native = load(receipt); sha, count, finite = finite_sha(out)
    if (native.get("status") != "success" or native["output"]["path"] != str(out) or native["output"]["sha256"] != sha or not finite or count != rows * 2560 or
        native["parameters"] != {"rows": rows, "hidden": 2560, "intermediate": 640, "gu_type": 12, "down_type": 7, "warmup": 1, "repeat": 3} or
        native["source"]["native_harness_source_sha256"] != c["new_harness_source_sha256"] or
        native["blob"]["sha256"] != c["blob_sha256"] or native["input"]["sha256"] != inp["sha256"] or
        native["reference_controls"]["STRATA_KQ256"] != "0 (process-local)" or native["reference_controls"]["q8k_avx2_selected"] is not True or
        native["reference_controls"]["down_activation_quantizer"] != "ggml Q8_1 from_float"):
        raise RuntimeError(f"receipt/output validation failed for {folder}")
    ref = Path(c["frozen_outputs"][str(rows)]["path"])
    same_old = equal_bytes(out, ref)
    same_default = True if workers is None else equal_bytes(out, Path(defaults[rows]))
    if workers is None:
        policy_ok = native["execution"]["engine_pool_called"] is False
        defaults[rows] = str(out)
    else:
        p = native["pool"]
        policy_ok = (p["requested_background_workers"] == workers and p["actual_background_workers"] == workers and
                     p["effective_compute_participants"] == workers + 1 and p["host_works"] is True and
                     p["pin"] is False and p["host_pin_applied"] is False and p["affinity"] == "none")
    result = {"status": "pass" if same_old and same_default and policy_ok else "fail", "case": case,
              "process": process, "stdout_json": text.strip(), "output_sha256": sha, "finite": finite,
              "output_identity": file_stat(out), "float_count": count, "matches_frozen_native_bitwise": same_old,
              "matches_new_default_bitwise": same_default, "execution_policy_ok": policy_ok}
    write_receipt_exclusive(folder / "case-result.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"bitwise parity or execution policy failed: {folder}")
    return result

def execute(c, args):
    if c.get("execution_supported") is False:
        raise RuntimeError("this affinity sweep contract is plan-only pending root review")
    if not (args.root_start_confirmed and args.vision_cpu_terminal_confirmed):
        raise RuntimeError("execution requires --root-start-confirmed and --vision-cpu-terminal-confirmed")
    static_receipts(c)
    root = Path(c["e_output_root"])
    if root.exists():
        raise RuntimeError(f"refusing existing E output root: {root}")
    root.mkdir(parents=False, exist_ok=False)
    cases, done = plan(c), []
    write_receipt_exclusive(root / "execution-plan.json", {"status": "prepared_for_run", "cases": cases})
    try:
        exe_identity = checked(c["new_harness_path"], c["new_harness_bytes"], c["new_harness_sha256"])
        with (root / "resource-samples.jsonl").open("x", encoding="utf-8", buffering=1) as samples:
            sample(samples, "run:initial")
            meta_dir = root / "metadata-preflight"
            meta, _ = child(Path(c["new_harness_path"]), ["--validate-only", "--rows", "1", "--hidden", "2560",
                "--intermediate", "640", "--gu-type", "12", "--down-type", "7", "--pool-workers", "1", "--pool-affinity", "none"],
                meta_dir, samples, "pool-metadata", exe_identity)
            m = json.loads(meta)["pool"]
            if m.get("kernel_started") is not False or m.get("topology_queried") is not False or m.get("requested_workers") != 1:
                raise RuntimeError("parser-only pool metadata preflight failed")
            write_receipt_exclusive(root / "metadata-preflight.json", {"pool": m, "executable_identity": exe_identity})
            checked_files = {c["blob_path"]: (c["blob_bytes"], c["blob_sha256"]),
                }
            for x in c["inputs"]: checked_files[x["path"]] = (x["bytes"], x["sha256"])
            for x in c["frozen_outputs"].values(): checked_files[x["path"]] = (x["bytes"], x["sha256"])
            identities = {p: checked(p, n, h) for p, (n, h) in checked_files.items()}
            write_receipt_exclusive(root / "data-identities-before-run.json", identities)
            for case in cases[:len(c["gate_rows"])]:
                done.append(run_case(c, case, samples, {}, exe_identity))
            defaults = {x["case"]["rows"]: x["case"]["output"] for x in done}
            for case in cases[len(c["gate_rows"]):]:
                done.append(run_case(c, case, samples, defaults, exe_identity))
        write_receipt_exclusive(root / "run-summary.json", {"status": "pass", "results": done})
    except BaseException as original:
        write_receipt_exclusive(root / "run-failure.json", {"status": "failed", "original_exception": f"{type(original).__name__}: {original}",
            "completed_cases": len(done), "partial_outputs_preserved": True})
        raise
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, default=CONTRACT)
    p.add_argument("--run", action="store_true"); p.add_argument("--root-start-confirmed", action="store_true")
    p.add_argument("--vision-cpu-terminal-confirmed", action="store_true")
    args = p.parse_args(); c = load(args.contract)
    if not args.run:
        print(json.dumps({"status": "plan_only", "cases": plan(c), "payload_read": False, "kernel_execution": False}, indent=2))
        return 0
    execute(c, args)
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native CPU pool parity: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
