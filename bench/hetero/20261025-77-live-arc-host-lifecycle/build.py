"""Compile only the real Windows pipe and host fixture. No CUDA/ggml/device."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from hetero_resources import memory_snapshot

SOURCE = Path("E:/Strata-Hetero-data/source/hetero-live-arc-20261025")
BUILD = Path("E:/Strata-Hetero-data/build/live-arc-host77-01")
OUT = Path(__file__).resolve().parent
PIN = "238a452e0096b3078513298e819d3acb2324a030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    assert head == PIN and not subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE)
    memory, errors = memory_snapshot()
    assert not errors and memory["physical_available_bytes"] >= 12*2**30 and memory["commit_available_bytes"] >= 4*2**30
    BUILD.mkdir(exist_ok=False)
    files = ["src/kernels/cpu/live_helper.cpp", "src/kernels/cpu/live_helper_host_fixture.cpp",
             "include/strata/kernels/cpu/live_helper.hpp", "include/strata/kernels/cpu/pool.hpp",
             "include/strata/kernels/cpu/native_expert.hpp"]
    hashes = {name: sha(SOURCE / name) for name in files}
    batch = BUILD / "build.cmd"
    executable = BUILD / "live_helper_host_fixture.exe"
    lines = ["@echo off",
             'call "C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" > vcvars.log 2>&1',
             "if errorlevel 1 exit /b %errorlevel%",
             'cl /nologo /std:c++20 /EHsc /MT /O2 /I"' + str(SOURCE / "include") + '" "' +
             str(SOURCE / files[0]) + '" "' + str(SOURCE / files[1]) + '" /Fe"' + str(executable) + '" > build.log 2>&1',
             "exit /b %errorlevel%"]
    batch.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    receipt = {"status": "compiling_host_only", "pid": os.getpid(), "source": str(SOURCE), "source_commit": head,
               "source_sha256": hashes, "build": str(BUILD), "memory_before": memory,
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": ["cmd.exe", "/d", "/c", str(batch)], "cuda_compiled": False,
               "device_accessed": False, "model_loaded": False}
    print(json.dumps({"status": receipt["status"], "pid": os.getpid()}), flush=True)
    with (BUILD / "outer.stdout.txt").open("xb") as stdout, (BUILD / "outer.stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(receipt["command"], cwd=BUILD, stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        receipt["compiler_parent_pid"] = process.pid
        receipt["exit_code"] = process.wait()
    receipt["source_unchanged"] = hashes == {name: sha(SOURCE / name) for name in files}
    receipt["status"] = "host_fixture_compiled" if receipt["exit_code"] == 0 and receipt["source_unchanged"] and executable.is_file() else "failed_preserved"
    if executable.is_file():
        receipt["executable"] = {"path": str(executable), "sha256": sha(executable), "bytes": executable.stat().st_size}
    receipt["artifacts"] = []
    for path in (batch, BUILD / "build.log", BUILD / "vcvars.log", BUILD / "outer.stdout.txt", BUILD / "outer.stderr.txt"):
        if path.is_file():
            raw = path.read_bytes()
            receipt["artifacts"].append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
            (OUT / path.name).write_bytes(raw)
    receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"], "executable": receipt.get("executable")}), flush=True)
    return 0 if receipt["status"] == "host_fixture_compiled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
