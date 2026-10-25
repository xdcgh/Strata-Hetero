"""Run an already-built Windows helper fixture against stdlib mock children.

Never invokes a compiler, model, OpenVINO or device API. The timeout scenario
uses the production 120 second wait and is explicitly opt-in.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_identity(pid: int) -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows process handle check required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,)+(ctypes.POINTER(wintypes.FILETIME),)*4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE,wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x100000|0x1000,False,pid) # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return {"pid":pid,"state":"absent","winerror":error}
        raise ctypes.WinError(error)
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        stamp = (times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
        waited = kernel.WaitForSingleObject(handle,0)
        if waited not in (0,258):
            raise RuntimeError(f"process wait failed: {waited}")
        return {"pid":pid,"creation_filetime":stamp,"state":"exited" if waited==0 else "running"}
    finally:
        kernel.CloseHandle(handle)


def run(args) -> int:
    if os.name != "nt":
        raise RuntimeError("Windows fixture required")
    executable = args.fixture.resolve(strict=True)
    args.output.mkdir(parents=True,exist_ok=False)
    scenarios = ["success","reject","partial","short-output","nan"]
    if args.include_timeout:
        scenarios.append("timeout")
    mock = Path(__file__).with_name("hetero_live_mock_child.py").resolve()
    records = []
    for scenario in scenarios:
        pids = (args.output/f"{scenario}.pids.json").resolve()
        env = dict(os.environ)
        env["STRATA_EXPERT_HELPER"] = "forced"
        env["STRATA_EXPERT_HELPER_COMMAND"] = subprocess.list2cmdline(
            [sys.executable,str(mock),"--scenario",scenario,"--pids",str(pids)])
        command = [str(executable),"eligible-success" if scenario=="success" else "eligible-failure"]
        started = time.perf_counter()
        timed_out = False
        with (args.output/f"{scenario}.stdout.txt").open("xb") as out, (args.output/f"{scenario}.stderr.txt").open("xb") as err:
            try:
                result = subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    timeout=145 if scenario=="timeout" else 20,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run kills this exact owned fixture. Closing its
                # process handles must close its Job Object and reap descendants.
                timed_out = True
                exit_code = None
        elapsed = time.perf_counter()-started
        checks = []
        pid_record = json.loads(pids.read_text()) if pids.is_file() else None
        if pid_record:
            for name in ("mock","descendant"):
                original = pid_record[name]
                deadline=time.monotonic()+5
                observations=[]
                while True:
                    observed = process_identity(original["pid"])
                    observations.append(observed)
                    # A new process reusing the numeric PID is not the owned child.
                    gone = observed["state"] in ("absent","exited") or observed.get("creation_filetime") != original["creation_filetime"]
                    if gone or time.monotonic()>=deadline:
                        break
                    time.sleep(.05)
                checks.append({"role":name,"original":original,"observations":observations,"owned_process_gone":gone})
        passed = exit_code==0 and not timed_out and len(checks)==2 and all(row["owned_process_gone"] for row in checks)
        records.append({"scenario":scenario,"command":command,"exit_code":exit_code,"runner_timed_out":timed_out,
            "wall_seconds":elapsed,"process_cleanup":checks,"passed":passed})
    files = {}
    for path in [executable,mock,Path(__file__).resolve(),*sorted(args.output.glob("*.txt")),*sorted(args.output.glob("*.pids.json"))]:
        files[str(path)] = {"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
    receipt = {"schema":"strata-live-helper-eligible-host-fixture-v1","records":records,"files":files,
        "device_accessed":False,"model_loaded":False,"compiler_invoked":False,
        "timeout_scenario_run":args.include_timeout,"all_passed":all(row["passed"] for row in records)}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))
    return 0 if receipt["all_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",required=True,type=Path,help="already-built live_helper_host_fixture.exe")
    parser.add_argument("--output",required=True,type=Path,help="new exclusive evidence directory")
    parser.add_argument("--include-timeout",action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
