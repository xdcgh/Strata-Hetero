#!/usr/bin/env python3
"""Bounded CPU-only MSVC/CMake build and CTest for worker-affinity observation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
OLD_RUN = ROOT / "bench/hetero/20261009-28-native-cpu-pool-build"
OLD_SWEEP = ROOT / "bench/hetero/20261009-35-cpu-worker-affinity/sweep-execution-contract-v2.json"
OLD_BUILD_CONTROLS = json.loads((OLD_RUN / "cmake-controls-verified.json").read_text(encoding="utf-8"))
GGML = Path(OLD_BUILD_CONTROLS["backend_options"]["STRATA_GGML_DIR"])
BUILD = Path(r"E:\Strata-Hetero-data\build\native-cpu-affinity-observe-20261009-48")
PHYS_MIN = 12 * 1024**3
COMMIT_MIN = 4 * 1024**3
FREE_DISK_MIN = 12 * 1024**3
COMMAND_TIMEOUT = 180.0
CHUNK = 1024 * 1024


class BuildError(RuntimeError):
    pass


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(raw: bytes): return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, obj: Any):
    path = path.resolve()
    if not path.parent.is_dir(): raise BuildError(f"receipt parent missing: {path.parent}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return sha(raw)


def write_bytes_new(path: Path, raw: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb", buffering=0) as f:
        view = memoryview(raw)
        while view:
            n = f.write(view)
            if not n: raise OSError(f"short write made no progress: {path}")
            view = view[n:]
        f.flush(); os.fsync(f.fileno())


def stable_file(path: Path) -> tuple[str, dict[str, Any]]:
    from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged
    before = file_stat(path); h = hashlib.sha256(); count = 0
    with path.open("rb", buffering=0) as f:
        if not fd_matches_path(os.fstat(f.fileno()), before): raise BuildError(f"file identity mismatch before hash: {path}")
        while block := f.read(CHUNK): h.update(block); count += len(block)
        last = os.fstat(f.fileno())
    after = file_stat(path)
    if count != before["size_bytes"] or not stat_unchanged(before, after) or not fd_matches_path(last, before):
        raise BuildError(f"file changed during hash: {path}")
    return h.hexdigest(), {**before, "path": str(path)}


def process_record(proc) -> dict[str, Any]:
    return {"pid": int(proc.pid), "parent_pid": int(proc.ppid()), "create_time": float(proc.create_time()),
            "executable": os.path.normcase(os.path.realpath(proc.exe())), "command_line": list(proc.cmdline())}


def same_process(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("pid", "create_time", "executable", "command_line"))


def memory_row(label: str) -> dict[str, Any]:
    from tools.hetero_resources import memory_snapshot
    memory, errors = memory_snapshot()
    phys = memory.get("physical_available_bytes"); commit = memory.get("commit_available_bytes")
    return {"time_utc": utc(), "label": label, "physical_available_bytes": phys,
            "commit_available_bytes": commit, "source": memory.get("source"), "errors": errors,
            "gate_ok": (not errors and type(phys) is int and type(commit) is int and
                        phys >= PHYS_MIN and commit >= COMMIT_MIN)}


def _seen_tree(proc, psutil, seen: dict[tuple[int, float], dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        root = psutil.Process(proc.pid)
        rows = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        rows = []
    for item in rows:
        try: current = process_record(item)
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
        key = (current["pid"], current["create_time"])
        previous = seen.get(key)
        if previous is not None and not same_process(previous, current):
            raise BuildError(f"owned build process identity changed: {current}")
        seen[key] = current
    return list(seen.values())


def _stop_owned(proc, psutil, seen) -> list[dict[str, Any]]:
    actions = []
    for identity in reversed(list(seen.values())):
        if identity["pid"] == proc.pid: continue  # Popen owns the command root.
        item = {"identity": identity, "action": "already_exited"}
        try:
            current_proc = psutil.Process(identity["pid"]); current = process_record(current_proc)
            if not same_process(identity, current): item["action"] = "pid_reused_or_identity_changed_not_touched"
            else:
                current_proc.terminate(); item["action"] = "terminate_owned_descendant"
                try: current_proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    again = process_record(psutil.Process(identity["pid"]))
                    if same_process(identity, again):
                        psutil.Process(identity["pid"]).kill(); item["action"] = "kill_owned_descendant_after_timeout"
                        psutil.Process(identity["pid"]).wait(timeout=3)
                item["exit_code"] = current_proc.wait(timeout=0.1)
        except psutil.NoSuchProcess: item["action"] = "exited_during_cleanup"
        except Exception as exc: item["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        actions.append(item)
    if proc.poll() is None:
        proc.kill()  # Exact Popen process handle; never resolve a bare PID for the command root.
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: actions.append({"action": "owned_popen_root_wait_timeout", "pid": proc.pid})
    return actions


def runner_main_source_paths() -> list[str]:
    prior = json.loads((OLD_RUN / "source-before.json").read_text(encoding="utf-8"))
    paths = {x["path"] for x in prior}
    paths.update(str((ROOT / rel).resolve()) for rel in (
        "src/hetero/native_expert_bench.cpp", "src/hetero/worker_affinity_observation_test.cpp",
        "src/program/generate.cpp", "tools/hetero_native_cpu/CMakeLists.txt",
        "include/strata/kernels/cpu/pool.hpp", "src/kernels/cpu/pool.cpp",
        "src/kernels/cpu/pool_affinity_win.hpp"))
    return sorted(paths, key=str.casefold)


def source_snapshot(paths: list[str], git_status: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for value in paths:
        path = Path(value)
        digest, identity = stable_file(path)
        try: rel = path.resolve().relative_to(ROOT).as_posix()
        except ValueError: rel = str(path)
        rows.append({**identity, "sha256": digest, "git_status": git_status.get(rel, "outside_or_clean_unknown")})
    return rows


def git_context() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    mapping = {}
    for line in status.splitlines():
        if len(line) > 3: mapping[line[3:].replace("\\", "/")] = line[:2]
    ggml = subprocess.check_output(["git", "-C", str(GGML), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if ggml != "3cf03257f219afbe7334045ff7c6a06ac68c627d": raise BuildError(f"wrong GGML pin: {ggml}")
    return {"head": head, "status_text": status, "status_by_path": mapping, "ggml_head": ggml}


def recover_previous_sources(receipt_dir: Path, git: dict[str, Any]) -> dict[str, Any]:
    previous_dir = receipt_dir / "previous-source"
    previous_dir.mkdir(parents=False, exist_ok=False)
    contract = json.loads(OLD_SWEEP.read_text(encoding="utf-8"))
    frozen = {"include/strata/kernels/cpu/pool.hpp": "4255fd01ae6e55a650b2e81316202c5b9206e0ae2271cd9fa81edc3b27d33c1a",
              "src/kernels/cpu/pool.cpp": "e54f0c2fe6a297c4aa49a3a9efabab3166d8644f05385b9f7e2fa5e5d34f8eff",
              "src/kernels/cpu/pool_affinity_win.hpp": "e179c53f0b02ddf250ae8244275d62db20efa5e1ac76a6d2dbd392bee09c8cd8",
              "src/hetero/native_expert_bench.cpp": "7082faf7df6a2fd578a96626b532bcdec4300aa6d02c669af80537d092f42a64"}
    extra = ("src/program/generate.cpp", "tools/hetero_native_cpu/CMakeLists.txt", "CMakeLists.txt")
    records = []
    for rel in list(frozen) + [x for x in extra if x not in frozen]:
        blob = subprocess.check_output(["git", "show", f"{git['head']}:{rel}"], cwd=ROOT)
        candidates = [("git_blob_lf", blob)]
        if b"\r\n" not in blob: candidates.append(("git_blob_crlf_worktree", blob.replace(b"\n", b"\r\n")))
        expected = frozen.get(rel)
        matched = [(fmt, raw) for fmt, raw in candidates if expected is None or hashlib.sha256(raw).hexdigest() == expected]
        if not matched: raise BuildError(f"cannot recover exact previous source for {rel}; hash candidates {[hashlib.sha256(x[1]).hexdigest() for x in candidates]}")
        fmt, raw = matched[0]
        target = previous_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_new(target, raw)
        records.append({"path": rel, "snapshot_path": str(target), "snapshot_format": fmt,
                        "sha256": hashlib.sha256(raw).hexdigest(), "run35_frozen_sha256": expected,
                        "matches_frozen": expected is None or hashlib.sha256(raw).hexdigest() == expected,
                        "git_head": git["head"]})
    new_test = "src/hetero/worker_affinity_observation_test.cpp"
    previous_test = subprocess.run(["git", "cat-file", "-e", f"{git['head']}:{new_test}"], cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if previous_test.returncode == 0: raise BuildError("new observation test unexpectedly exists at prior HEAD")
    receipt = {"schema_version": 1, "status": "previous_source_snapshots_match_run35_freeze",
               "git_head": git["head"], "ggml_head": git["ggml_head"], "files": records,
               "new_test_absent_at_previous_head": new_test}
    write_new(receipt_dir / "previous-source-manifest.json", receipt)
    return receipt


def get_vc_env(receipt_dir: Path, base_env: dict[str, str], observer) -> tuple[dict[str, str], dict[str, Any]]:
    import psutil
    vcvars = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
    if not vcvars.is_file(): raise BuildError(f"vcvars64.bat missing: {vcvars}")
    command = f'call "{vcvars}" >nul && set'
    argv = ["cmd.exe", "/d", "/s", "/c", command]
    before = observer.sample("vcvars-environment:prelaunch")
    if not before["gate_ok"]: raise BuildError(f"resource gate failed before vcvars: {before}")
    proc = subprocess.Popen(argv, cwd=ROOT, env=dict(base_env), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    identity = process_record(psutil.Process(proc.pid))
    expected_cmd = os.path.normcase(os.path.realpath(shutil.which("cmd.exe") or r"C:\Windows\System32\cmd.exe"))
    if identity["executable"] != expected_cmd or identity["parent_pid"] != os.getpid():
        proc.kill(); proc.wait(timeout=5)
        raise BuildError(f"vcvars command process identity mismatch: {identity}")
    try:
        seen = {(identity["pid"], identity["create_time"]): identity}
        deadline = time.monotonic() + 45
        while True:
            try:
                out, err = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                tree = _seen_tree(proc, psutil, seen)
                sample = observer.sample("vcvars-environment:running", tree)
                if not sample["gate_ok"] or time.monotonic() >= deadline:
                    cleanup = _stop_owned(proc, psutil, seen)
                    if not sample["gate_ok"]:
                        raise BuildError(f"vcvars environment resource gate failed: {sample}; cleanup={cleanup}")
                    raise BuildError(f"vcvars environment setup timed out; cleanup={cleanup}")
        _seen_tree(proc, psutil, seen)
        after = observer.sample("vcvars-environment:finished", list(seen.values()))
    except BaseException:
        if proc.poll() is None:
            try: proc.kill(); proc.wait(timeout=5)
            except Exception: pass
        raise
    if proc.returncode != 0 or not after["gate_ok"]:
        raise BuildError(f"vcvars environment setup failed: exit={proc.returncode}, resource={after}")
    if len(out) > 2 * 1024 * 1024: raise BuildError("vcvars environment output exceeded the bounded 2 MiB capture")
    meta = {"argv": argv, "process_identity": identity, "exit_code": proc.returncode,
            "resource_before": before, "resource_after": after,
            "stdout_bytes": len(out), "stdout_sha256": hashlib.sha256(out).hexdigest(),
            "stderr_bytes": len(err), "stderr_sha256": hashlib.sha256(err).hexdigest(),
            "stdout_environment_values_persisted": False}
    write_new(receipt_dir / "vcvars-environment-process.json", meta)
    env = dict(os.environ)
    for line in out.decode(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key and not key.startswith("="): env[key] = value
    removed = sorted(k for k in env if k.startswith("STRATA_") or k in {
        "CUDA_PATH", "CUDA_HOME", "CUDACXX", "CUDA_VISIBLE_DEVICES", "HIP_PATH", "ROCM_PATH",
        "ONEAPI_ROOT", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER"})
    for k in removed: env.pop(k, None)
    return env, {"vcvars_path": str(vcvars), "vcvars_sha256": stable_file(vcvars)[0],
                 "environment_keys_removed": removed, "env_command": command, "process": meta,
                 "environment_values_persisted": False}


def process_conflicts(psutil) -> list[dict[str, Any]]:
    blocked = {"cl.exe", "cmake.exe", "ninja.exe", "nvcc.exe", "hipcc.exe", "dpcpp.exe", "icx.exe", "icpx.exe"}
    current_pid = os.getpid(); matches = []
    for p in psutil.process_iter(["pid", "name", "exe", "cmdline", "create_time"]):
        try:
            if p.info["pid"] == current_pid: continue
            if (p.info.get("name") or "").casefold() in blocked:
                matches.append({"pid": p.info["pid"], "create_time": p.info.get("create_time"),
                                "name": p.info.get("name"), "exe": p.info.get("exe"), "cmdline": p.info.get("cmdline")})
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return matches


def run_command(name: str, argv: list[str], cwd: Path, env: dict[str, str], receipt_dir: Path,
                observer, timeout: float = COMMAND_TIMEOUT) -> tuple[bytes, dict[str, Any]]:
    import psutil
    log_path = receipt_dir / f"{name}.log"
    attempt_path = receipt_dir / f"{name}-attempt.json"
    if log_path.exists() or attempt_path.exists(): raise BuildError(f"command evidence path exists: {name}")
    before = observer.sample(f"{name}:prelaunch")
    if not before["gate_ok"]: raise BuildError(f"resource gate failed before {name}: {before}")
    attempt = {"name": name, "argv": argv, "cwd": str(cwd), "started_utc": utc(), "resource_before": before,
               "environment_stripped_gpu_or_test_controls": True}
    write_new(attempt_path, attempt)
    seen: dict[tuple[int, float], dict[str, Any]] = {}; output = b""; proc = None
    with log_path.open("xb") as log:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            root_identity = process_record(psutil.Process(proc.pid))
            expected_exe = os.path.normcase(os.path.realpath(argv[0]))
            if root_identity["executable"] != expected_exe or root_identity["parent_pid"] != os.getpid():
                raise BuildError(f"owned command process identity mismatch: {root_identity}")
            seen[(root_identity["pid"], root_identity["create_time"])] = root_identity
            write_new(receipt_dir / f"{name}-process-start.json", {"identity": root_identity,"argv": argv,"popen_owned_handle":True})
            deadline = time.monotonic() + timeout
            while True:
                try:
                    proc.wait(timeout=1.0); break
                except subprocess.TimeoutExpired:
                    tree = _seen_tree(proc, psutil, seen)
                    mem = observer.sample(f"{name}:running", tree)
                    forbidden = [x for x in tree if Path(x["executable"]).name.casefold() in {
                        "nvcc.exe", "hipcc.exe", "dpcpp.exe", "icx.exe", "icpx.exe"}]
                    if forbidden or not mem["gate_ok"] or time.monotonic() >= deadline:
                        cleanup = _stop_owned(proc, psutil, seen)
                        raise BuildError(f"{name} stopped fail-closed: resource={mem}, forbidden_compilers={forbidden}, timeout={time.monotonic() >= deadline}, cleanup={cleanup}")
            # A direct child should not leave exact descendants after it exits.
            live_descendants = []
            for record in list(seen.values()):
                if record["pid"] == proc.pid: continue
                try:
                    current = process_record(psutil.Process(record["pid"]))
                    if same_process(record,current): live_descendants.append(record)
                except psutil.NoSuchProcess: pass
            if live_descendants:
                cleanup = _stop_owned(proc, psutil, seen)
                raise BuildError(f"{name} left live owned descendants: {live_descendants}; cleanup={cleanup}")
            after = observer.sample(f"{name}:finished", list(seen.values()))
            log.flush(); os.fsync(log.fileno())
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                cleanup = _stop_owned(proc, psutil, seen)
            else: cleanup = []
            log.flush(); os.fsync(log.fileno())
            failure = {"name":name,"status":"failed","error":f"{type(exc).__name__}: {exc}",
                       "pid":proc.pid if proc else None,"exit_code":proc.poll() if proc else None,
                       "seen_processes":list(seen.values()),"cleanup":cleanup,"log_path":str(log_path)}
            failpath=receipt_dir/f"{name}-failure.json"
            if not failpath.exists(): write_new(failpath,failure)
            raise
    proc_receipt = {"name":name,"status":"success" if proc.returncode==0 and after["gate_ok"] else "failed",
                    "argv":argv,"cwd":str(cwd),"exit_code":proc.returncode,"finished_utc":utc(),
                    "resource_after":after,"process_tree":list(seen.values()),"log_path":str(log_path),
                    "log_bytes":log_path.stat().st_size,"log_sha256":stable_file(log_path)[0]}
    write_new(receipt_dir/f"{name}-process.json",proc_receipt)
    output=(receipt_dir/f"{name}.log").read_bytes()
    if proc_receipt["status"]!="success": raise BuildError(f"{name} exited {proc.returncode} or failed its postcommand gate")
    return output,proc_receipt


class ResourceObserver:
    def __init__(self, path: Path):
        self.path=path; self.stream=path.open("x",encoding="utf-8",buffering=1)
        self.last=None; self.minimum_phys=None; self.minimum_commit=None; self.samples=0
    def sample(self,label,tree=None):
        row=memory_row(label); row["owned_process_tree"]=tree
        self.stream.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+"\n"); self.stream.flush()
        self.samples+=1
        if row["gate_ok"]:
            self.minimum_phys=row["physical_available_bytes"] if self.minimum_phys is None else min(self.minimum_phys,row["physical_available_bytes"])
            self.minimum_commit=row["commit_available_bytes"] if self.minimum_commit is None else min(self.minimum_commit,row["commit_available_bytes"])
        self.last=row
        return row
    def close(self): self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()


def current_run_source_paths():
    from tools.hetero_verify_existing import file_stat
    prior=json.loads((OLD_RUN/"source-before.json").read_text())
    paths={Path(x["path"]) for x in prior}
    paths.update((ROOT/rel).resolve() for rel in (
        "src/hetero/native_expert_bench.cpp","src/hetero/worker_affinity_observation_test.cpp",
        "src/program/generate.cpp","tools/hetero_native_cpu/CMakeLists.txt",
        "include/strata/kernels/cpu/pool.hpp","src/kernels/cpu/pool.cpp","src/kernels/cpu/pool_affinity_win.hpp"))
    result=[]
    for path in sorted(paths,key=lambda p:str(p).casefold()):
        digest,identity=stable_file(path)
        try: rel=path.resolve().relative_to(ROOT).as_posix()
        except ValueError: rel=str(path)
        result.append({**identity,"sha256":digest,"repo_relative":rel})
    return result


def configure_argv(build_dir: Path, tools: dict[str,str]):
    cmake=tools["cmake"]; ninja=tools["ninja"]
    return [cmake,"-S",str(ROOT/"tools/hetero_native_cpu"),"-B",str(build_dir),"-G","Ninja",
      f"-DCMAKE_MAKE_PROGRAM={ninja}","-DCMAKE_BUILD_TYPE=Release",
      "-DSTRATA_ENABLE_CUDA=OFF","-DSTRATA_ENABLE_HIP=OFF","-DSTRATA_HIP_GFX906=OFF",
      "-DSTRATA_ENABLE_SYCL=OFF","-DSTRATA_PREFILL_MMQ=OFF","-DSTRATA_MMQ_KQUANTS=OFF",
      "-DSTRATA_ORCA_Q4KS_MMQ=OFF","-DSTRATA_NATIVE_EXPERTS=ON","-DSTRATA_BUILD_TESTS=OFF",
      "-DSTRATA_PORTABLE=ON","-DSTRATA_ISA_FLOOR=",f"-DSTRATA_GGML_DIR={GGML}",f"-DHETERO_NATIVE_GGML_DIR={GGML}"]


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--run",action="store_true")
    ap.add_argument("--root-compile-authorized",action="store_true"); a=ap.parse_args()
    if not a.run:
        print(json.dumps({"status":"plan_only","repo_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "build_dir":str(BUILD),"targets":["hetero_native_expert","hetero_native_expert_metadata_fixture","hetero_worker_affinity_observation"],
          "parallel_jobs":1,"GPU_backends":"all explicitly OFF","actual_harness_run":False,"existing_build_modified":False},indent=2)); return 0
    if not a.root_compile_authorized: ap.error("build requires --root-compile-authorized")
    run_build(a)
    return 0


def run_build(args):
    import psutil
    from tools.hetero_resources import memory_snapshot
    if BUILD.exists(): raise FileExistsError(f"refusing existing E build directory: {BUILD}")
    if (BENCH/"receipts").exists(): raise FileExistsError("run48 receipts already exist")
    if shutil.disk_usage(BUILD.parent).free<FREE_DISK_MIN: raise BuildError("E free space below 12GiB reserve")
    memory,errors=memory_snapshot()
    if errors or type(memory.get("physical_available_bytes")) is not int or type(memory.get("commit_available_bytes")) is not int or memory["physical_available_bytes"]<PHYS_MIN or memory["commit_available_bytes"]<COMMIT_MIN:
        raise BuildError(f"fresh 12/4 gate failed: {memory} {errors}")
    if subprocess.check_output(["git","-C",str(GGML),"rev-parse","HEAD"],cwd=ROOT,text=True).strip()!="3cf03257f219afbe7334045ff7c6a06ac68c627d":
        raise BuildError("GGML pin drift")
    conflict=process_conflicts(psutil)
    if conflict: raise BuildError(f"unowned compiler/accelerator compiler already active: {conflict}")
    BENCH.mkdir(parents=True,exist_ok=True); receipts=BENCH/"receipts"; receipts.mkdir(exist_ok=False)
    git=git_context(); paths=current_run_source_paths()
    previous=recover_previous_sources(BENCH,git)
    # Build invocation source identity and any working-tree changes are frozen before configuration.
    status_bytes=git["status_text"].encode()
    write_bytes_new(receipts/"git-status-before.txt",status_bytes)
    write_new(receipts/"source-before.json",{"head":git["head"],"ggml_head":git["ggml_head"],"files":paths})
    diff=subprocess.run(["git","diff","--binary","HEAD","--", "include/strata/kernels/cpu/pool.hpp","src/kernels/cpu/pool.cpp",
      "src/hetero/native_expert_bench.cpp","src/program/generate.cpp","tools/hetero_native_cpu/CMakeLists.txt"],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if diff.returncode: raise BuildError(f"git diff failed: {diff.stderr.decode(errors='replace')}")
    write_bytes_new(receipts/"source-diff.patch",diff.stdout)
    ctest_source=(ROOT/"src/hetero/worker_affinity_observation_test.cpp").read_bytes()
    write_bytes_new(receipts/"worker-affinity-test-source.cpp",ctest_source)
    controls=OLD_BUILD_CONTROLS
    old_sweep_contract=json.loads(OLD_SWEEP.read_text(encoding="utf-8"))
    tools={"cmake":controls["cmake"],"ninja":controls["ninja"],"compiler":controls["compiler"],
           "ctest":str(Path(controls["cmake"]).parent/"ctest.exe"),
           "vcvars":r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"}
    for key in ("cmake","ninja","compiler","ctest","vcvars"):
        if not Path(tools[key]).is_file(): raise BuildError(f"tool path absent: {key}={tools[key]}")
    tool_hashes={k:stable_file(Path(v))[0] for k,v in tools.items()}
    for k in ("cmake","ninja","compiler"):
        expected=controls.get({"cmake":"cmake_sha256","ninja":"ninja_sha256","compiler":"compiler_sha256"}[k])
        if tool_hashes[k]!=expected: raise BuildError(f"pinned tool hash mismatch for {k}")
    obs=ResourceObserver(receipts/"resource-samples.jsonl")
    try:
        start=obs.sample("run48-start")
        if not start["gate_ok"]: raise BuildError(f"initial resource gate failed: {start}")
        base_env=dict(os.environ)
        # Acquire a verified MSVC 64-bit environment from the pinned VS installation, without retaining its environment text.
        env_cmd=f'call "{tools["vcvars"]}" >nul && set'
        env_out,env_process=run_command("vcvars-environment",["cmd.exe","/d","/s","/c",env_cmd],ROOT,base_env,receipts,obs,timeout=45)
        env=dict(os.environ)
        for line in env_out.decode(errors="replace").splitlines():
            k,sep,v=line.partition("=")
            if sep and k and not k.startswith("="): env[k]=v
        stripped=sorted(k for k in env if k.startswith("STRATA_") or k in {"CUDA_PATH","CUDA_HOME","CUDACXX","CUDA_VISIBLE_DEVICES","HIP_PATH","ROCM_PATH","ONEAPI_ROOT","ONEAPI_DEVICE_SELECTOR","SYCL_DEVICE_FILTER"})
        for k in stripped: env.pop(k,None)
        tools_actual={k:shutil.which(Path(v).name,path=env.get("PATH")) for k,v in tools.items() if k in ("cmake","ninja","compiler","ctest")}
        if any(not v for v in tools_actual.values()): raise BuildError(f"pinned tool not found in MSVC environment: {tools_actual}")
        if any(os.path.normcase(os.path.realpath(tools_actual[k]))!=os.path.normcase(os.path.realpath(tools[k])) for k in tools_actual):
            raise BuildError(f"MSVC environment resolves a different tool path: {tools_actual}")
        version_logs={}
        for label,argv in (("cmake-version",[tools["cmake"],"--version"]),("ninja-version",[tools["ninja"],"--version"]),("cl-version",[tools["compiler"],"/Bv"])):
            raw,meta=run_command(label,argv,ROOT,env,receipts,obs,timeout=30)
            version_logs[label]={'stdout':raw.decode(errors='replace')[-5000:],'process':meta}
        configure=configure_argv(BUILD,tools)
        if any(not (isinstance(x,str) and x.strip()) for x in configure): raise BuildError("CMake configure argv contains empty/non-string values")
        bool_values=[x.split("=",1)[1] for x in configure if x.startswith("-DSTRATA_") and x.split("=",1)[0] in {
          "-DSTRATA_ENABLE_CUDA","-DSTRATA_ENABLE_HIP","-DSTRATA_HIP_GFX906","-DSTRATA_ENABLE_SYCL","-DSTRATA_PREFILL_MMQ","-DSTRATA_MMQ_KQUANTS","-DSTRATA_ORCA_Q4KS_MMQ","-DSTRATA_NATIVE_EXPERTS","-DSTRATA_BUILD_TESTS","-DSTRATA_PORTABLE"}]
        if not all(v in ("ON","OFF") for v in bool_values) or len(bool_values)!=10: raise BuildError(f"CMake boolean options are not literal ON/OFF: {bool_values}")
        write_new(receipts/"build-plan.json",{"repo_head":git["head"],"ggml_head":git["ggml_head"],"binary_old_path":old_sweep_contract["source"]["new_binary_path"],
          "build_dir":str(BUILD),"tool_paths":tools,"tool_sha256":tool_hashes,"tools_resolved_in_msvc_env":tools_actual,
          "cmake_argv":configure,"targets":["hetero_native_expert","hetero_native_expert_metadata_fixture","hetero_worker_affinity_observation"],
          "parallel_jobs":1,"static_msvc_runtime":"/MT (project CMakeLists sets MultiThreaded)","portable_avx2_expected":"/arch:AVX2",
          "literal_bool_values":bool_values,"gpu_backends":{"CUDA":"OFF","HIP":"OFF","SYCL":"OFF"},
          "removed_child_environment_names":stripped,"tool_versions":version_logs,"source_hashes_before":paths,
          "previous_source_manifest":previous,"memory_gate_initial":start,"compiler_idle_preflight":conflict})
        if not BUILD.parent.exists(): raise BuildError(f"E build parent missing: {BUILD.parent}")
        BUILD.mkdir(parents=False,exist_ok=False)
        config_out,config_meta=run_command("configure",configure,ROOT,env,receipts,obs,timeout=120)
        if config_meta["exit_code"]!=0: raise BuildError("CMake configure failed")
        build_argv=[tools["cmake"],"--build",str(BUILD),"--verbose","--parallel","1","--target",
                    "hetero_native_expert","hetero_native_expert_metadata_fixture","hetero_worker_affinity_observation"]
        build_out,build_meta=run_command("build",build_argv,ROOT,env,receipts,obs,timeout=180)
        if build_meta["exit_code"]!=0: raise BuildError("native CPU build failed")
        test_argv=[tools["ctest"],"--test-dir",str(BUILD),"--output-on-failure","--parallel","1"]
        test_out,test_meta=run_command("ctest",test_argv,ROOT,env,receipts,obs,timeout=120)
        if test_meta["exit_code"]!=0: raise BuildError("CTest failed")
        after_paths=current_run_source_paths()
        by_path={x["path"]:x for x in paths}; changed=[]
        for row in after_paths:
            prev=by_path.get(row["path"])
            if prev and row["sha256"]!=prev["sha256"]: changed.append({"before":prev,"after":row})
        if changed: raise BuildError(f"source changed during build: {changed}")
        exe_paths={name:BUILD/(name+".exe") for name in ("hetero_native_expert","hetero_native_expert_metadata_fixture","hetero_worker_affinity_observation")}
        binaries={}
        for name,p in exe_paths.items():
            h,st=stable_file(p); binaries[name]={"path":str(p),"sha256":h,"bytes":st["size_bytes"],"stat":st}
        build_log=(receipts/"build.log").read_text(encoding="utf-8",errors="replace")
        controls_found={"mt":(" /MT " in build_log or " /MT\n" in build_log),"avx2":"/arch:AVX2" in build_log}
        if not all(controls_found.values()): raise BuildError(f"build flags missing /MT or portable AVX2: {controls_found}")
        receipt={"schema_version":1,"status":"success","run_id":"native-cpu-affinity-observe-20261009-48",
          "started_utc":start["time_utc"],"finished_utc":utc(),"head":git["head"],"ggml_head":git["ggml_head"],
          "tools":tools,"tool_sha256":tool_hashes,"tool_version_processes":version_logs,"environment_removed_names":stripped,
          "configure_argv":configure,"build_argv":build_argv,"ctest_argv":test_argv,
          "processes":{"vcvars":env_process,"configure":config_meta,"build":build_meta,"ctest":test_meta},
          "exit_codes":{"vcvars":env_process["exit_code"],"configure":config_meta["exit_code"],"build":build_meta["exit_code"],"ctest":test_meta["exit_code"]},
          "targets":list(binaries),"binaries":binaries,"build_flags_verified":controls_found,
          "source_sha256_before":{x["repo_relative"]:x["sha256"] for x in paths},
          "source_sha256_after":{x["repo_relative"]:x["sha256"] for x in after_paths},
          "source_unchanged_during_build":not changed,"source_changes":changed,
          "old_run28_build_and_binary_untouched":True,"previous_source_manifest":previous,
          "resource_samples":obs.samples,"physical_available_min_bytes":obs.minimum_phys,
          "commit_available_min_bytes":obs.minimum_commit,"resource_sampling":"initial/inter-command/per-second while each owned command runs/after each command",
          "gpu_backends":{"CUDA":"OFF","HIP":"OFF","SYCL":"OFF"},
          "actual_native_blob_or_expert_kernel_run":"not run; CTest only validates parser/metadata and constructs a pool with two owned worker threads; no pool.run/kernel API is called",
          "affinity_mutation_scope":"CTest's dedicated owned worker threads only; all those threads exit with the fixture"}
        write_new(receipts/"build-receipt.json",receipt)
        write_new(receipts/"resource-sampler-receipt.json",{"status":"pass","samples":obs.samples,"phys_min":obs.minimum_phys,"commit_min":obs.minimum_commit,"path":str(receipts/"resource-samples.jsonl")})
        obs.close()
    except BaseException as exc:
        try: obs.close()
        except Exception: pass
        failure={"schema_version":1,"status":"failed","error":f"{type(exc).__name__}: {exc}","failed_utc":utc(),
                 "build_dir":str(BUILD),"old_build_dir_touched":False,"no_retry":True}
        target=receipts/"build-failure.json"
        if not target.exists(): write_new(target,failure)
        raise


if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(f"native CPU affinity observation build: {type(e).__name__}: {e}",file=sys.stderr)
        raise SystemExit(2)
