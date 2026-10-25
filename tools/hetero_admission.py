#!/usr/bin/env python3
"""Read-only resource and process gate for a possible GPU run; this is not a model test."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import importlib.util
import io
import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

TIMEOUT_SECONDS = 60
Runner = Callable[..., subprocess.CompletedProcess[Any]]
DISPLAY_ALLOWLIST = {
    "system", "system.exe", "dwm.exe", "explorer.exe", "msedge.exe",
    "applicationframehost.exe", "logonui.exe",
}


def _idle_codex_desktop(path: Any, gpus: list[dict[str, Any]]) -> bool:
    """The installed desktop running this task may hold a small graphics/compute context."""
    if not isinstance(path, str):
        return False
    apps = ntpath.normpath(ntpath.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "WindowsApps"))
    pattern = (re.escape(apps) + r"\\OpenAI\.Codex_\d+(?:\.\d+){3}_x64__2p2nqsd0c76g0\\app\\ChatGPT\.exe")
    if re.fullmatch(pattern, ntpath.normpath(path), flags=re.IGNORECASE) is None:
        return False
    return bool(gpus) and all(
        type(g.get("utilization_percent")) in (int, float) and 0 <= g["utilization_percent"] <= 5
        and type(g.get("vram_used_mib")) is int and 0 <= g["vram_used_mib"] <= 2048 for g in gpus)


def _execute(args: list[str], runner: Runner, *, text: bool = True) -> subprocess.CompletedProcess[Any]:
    command = Path(args[0]).name if args else "command"
    try:
        result = runner(args, capture_output=True, text=text, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{command} timed out after {TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise RuntimeError(f"{command} unavailable ({type(exc).__name__})") from exc
    if result.returncode:
        raise RuntimeError(f"{command} exited with code {result.returncode}")
    return result


def parse_nvidia_csv(gpu_csv: str, process_csv: str) -> dict[str, Any]:
    """Parse only the exact requested nvidia-smi columns; unknown rows fail closed."""
    rows = list(csv.reader(io.StringIO(gpu_csv), skipinitialspace=True))
    if not rows:
        raise ValueError("NVIDIA returned no GPU telemetry")
    gpus = []
    for index, row in enumerate(rows):
        if len(row) != 6 or not row[0].strip():
            raise ValueError(f"invalid NVIDIA GPU telemetry row {index + 1}")
        try:
            total, free, used = (int(row[i].strip()) for i in (1, 2, 3))
            utilization = float(row[4].strip())
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA numeric telemetry row {index + 1}") from exc
        if min(total, free, used) < 0 or not 0 <= utilization <= 100:
            raise ValueError(f"out-of-range NVIDIA telemetry row {index + 1}")
        gpus.append({"name": row[0].strip(), "vram_total_mib": total, "vram_free_mib": free,
                     "vram_used_mib": used, "utilization_percent": utilization,
                     "driver_version": row[5].strip()})
    pids: list[int] = []
    process_unknown = False
    for line in process_csv.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.casefold() in {"n/a", "[insufficient permissions]", "insufficient permissions"}:
            process_unknown = True
            continue
        if not re.fullmatch(r"\d+", value):
            process_unknown = True
            continue
        pids.append(int(value))
    return {"gpus": gpus, "compute_pids": sorted(set(pids)), "process_telemetry_unknown": process_unknown,
            "raw_gpu_csv": gpu_csv, "raw_process_csv": process_csv}


def decode_wsl_list(data: bytes | str) -> list[str]:
    if isinstance(data, bytes):
        if data.startswith(b"\xff\xfe"):
            text = data[2:].decode("utf-16le", errors="strict")
        elif b"\x00" in data:
            text = data.decode("utf-16le", errors="strict")
        else:
            text = data.decode("utf-8-sig", errors="strict")
    else:
        text = data.lstrip("\ufeff")
    return [line.strip().lstrip("\ufeff") for line in text.splitlines() if line.strip()]


def verify_controller_chain(proof: dict[str, Any], live: list[dict[str, Any]],
                            observer_parent_pid: int, now: dt.datetime) -> list[dict[str, Any]]:
    """Allow only fresh, exact Python ancestors of the owned launch controller.

    This does not exempt any NVIDIA compute process or an unrelated Python PID.
    The controller records its identity before accepting the launch script.
    """
    if proof.get("schema") != "strata-hetero-controller-chain-v1":
        raise ValueError("unsupported controller proof")
    created = dt.datetime.fromisoformat(proof["created_utc"].replace("Z", "+00:00"))
    expires = dt.datetime.fromisoformat(proof["expires_utc"].replace("Z", "+00:00"))
    if any(t.tzinfo is None or t.utcoffset() != dt.timedelta(0) for t in (created, expires, now)):
        raise ValueError("controller proof clocks must be UTC")
    if not 0 <= (now - created).total_seconds() <= 120 or not created < now < expires or (expires-created).total_seconds() > 120:
        raise ValueError("controller proof expired or future dated")
    if re.fullmatch(r"[0-9a-f]{32}", str(proof.get("nonce", ""))) is None:
        raise ValueError("controller proof nonce missing")
    controller = proof.get("controller", {})
    owners = proof.get("python_controllers")
    if not isinstance(owners, list) or not 1 <= len(owners) <= 2:
        raise ValueError("one actual coordinator and optional venv launcher required")
    by_pid = {r["pid"]: r for r in live}
    all_owners = [controller, *owners]
    if len({r.get("pid") for r in all_owners}) != len(all_owners):
        raise ValueError("duplicate controller PID")
    fields = ("pid", "parent_pid", "name", "path", "create_utc", "command_line")
    for index, owner in enumerate(all_owners):
        if type(owner.get("pid")) is not int or type(owner.get("parent_pid")) is not int:
            raise ValueError("controller PIDs must be known integers")
        actual = by_pid.get(owner["pid"])
        if actual is None or any(actual.get(k) != owner.get(k) for k in fields):
            raise ValueError("controller identity changed or no longer live")
        if not isinstance(owner.get("create_utc"), str) or not owner["create_utc"].endswith("Z"):
            raise ValueError("controller CreateTime must retain UTC text")
        if index + 1 < len(all_owners) and owner["parent_pid"] != all_owners[index+1]["pid"]:
            raise ValueError("controller is not an immediate ancestor chain")
    expected_pwsh = r"C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe"
    if (controller["pid"] != observer_parent_pid or controller["path"].casefold() != expected_pwsh.casefold()
            or controller["name"].casefold() != "pwsh.exe" or "ReadToEnd" not in controller["command_line"]
            or "-Start" not in controller["command_line"]):
        raise ValueError("observer is not a child of the owned launch controller")
    allowed_python = (r"C:\Python314\python.exe", r"E:\Strata-Hetero-data\venv\Scripts\python.exe")
    for index, owner in enumerate(owners):
        if (owner["name"].casefold() != "python.exe" or owner["path"].casefold() != allowed_python[index].casefold()
                or " -m tools.hetero_pool_sweep " not in owner["command_line"]
                or " --execute" not in owner["command_line"] or " --adapter run41" not in owner["command_line"]):
            raise ValueError("ancestor is not the reviewed Python sweep coordinator")
    return owners


def evaluate(raw: dict[str, Any], required_vram_mib: int, minimum_ram_gib: float,
             check_wsl: bool = True, *, inspect_running_wsl: bool = False) -> dict[str, Any]:
    reasons: list[str] = []
    gates: dict[str, Any] = {"required_vram_mib": required_vram_mib,
                             "minimum_available_ram_gib": minimum_ram_gib,
                             "check_wsl_current": check_wsl}
    memory = raw.get("windows_memory")
    if not isinstance(memory, dict) or not isinstance(memory.get("available_bytes"), int):
        gates["ram"] = {"pass": False, "reason": "Windows available RAM telemetry unknown"}
        reasons.append(gates["ram"]["reason"])
    else:
        available_gib = memory["available_bytes"] / (1024 ** 3)
        passed = available_gib >= minimum_ram_gib
        gates["ram"] = {"pass": passed, "available_gib": round(available_gib, 3),
                        "reason": None if passed else "available RAM below reserve"}
        if not passed: reasons.append(gates["ram"]["reason"])

    available_commit = memory.get("available_commit_bytes") if isinstance(memory, dict) else None
    commit_pass = type(available_commit) is int and available_commit >= 4 * 1024**3
    gates["system_commit"] = {"pass": commit_pass, "minimum_gib": 4,
                              "available_bytes": available_commit,
                              "reason": None if commit_pass else "PSAPI system commit reserve missing or below 4 GiB"}
    if not commit_pass: reasons.append(gates["system_commit"]["reason"])

    gpu_data = raw.get("nvidia")
    if not isinstance(gpu_data, dict) or gpu_data.get("telemetry_error") or not gpu_data.get("gpus"):
        gates["vram"] = {"pass": False, "reason": "NVIDIA GPU telemetry unknown"}
        reasons.append(gates["vram"]["reason"])
    else:
        selected = max(gpu_data["gpus"], key=lambda g: g.get("vram_free_mib", -1))
        passed = isinstance(selected.get("vram_free_mib"), int) and selected["vram_free_mib"] >= required_vram_mib
        gates["vram"] = {"pass": passed, "selected_gpu": selected.get("name"),
                         "available_mib": selected.get("vram_free_mib"),
                         "reason": None if passed else "free VRAM below requirement"}
        if not passed: reasons.append(gates["vram"]["reason"])
        if gpu_data.get("process_telemetry_unknown"):
            gates["gpu_processes"] = {"pass": False, "reason": "NVIDIA compute process telemetry unknown"}
            reasons.append(gates["gpu_processes"]["reason"])
        else:
            process_rows = gpu_data.get("processes")
            if process_rows is None and gpu_data.get("compute_pids"):
                process_rows = [{"pid": p, "name": None, "path": None} for p in gpu_data["compute_pids"]]
            process_rows = process_rows or []
            blocked = []
            for row in process_rows:
                name = (row.get("name") or "").casefold()
                path = row.get("path")
                base = ntpath.basename(path or "").casefold()
                if not path and name in DISPLAY_ALLOWLIST:
                    base = name
                if not path and not base:
                    blocked.append({"pid": row.get("pid"), "reason": "process identity unknown"})
                elif re.search(r"strata|python|comfy", name + " " + (path or ""), re.IGNORECASE):
                    blocked.append({"pid": row.get("pid"), "reason": "Strata/Python/ComfyUI process present"})
                elif base not in DISPLAY_ALLOWLIST and not _idle_codex_desktop(path, gpu_data.get("gpus", [])):
                    blocked.append({"pid": row.get("pid"), "reason": "unapproved GPU process"})
            passed = not blocked
            gates["gpu_processes"] = {"pass": passed, "blocked": blocked,
                                      "reason": None if passed else "GPU process admission blocked"}
            if not passed: reasons.append(gates["gpu_processes"]["reason"])

    risky_processes = raw.get("risky_windows_processes")
    if risky_processes is None:
        gates["windows_processes"] = {"pass": False, "reason": "Windows process identity telemetry unknown"}
        reasons.append(gates["windows_processes"]["reason"])
    else:
        proof = raw.get("owned_controller_validation")
        allowed = proof.get("verified_python_controllers", []) if isinstance(proof, dict) and proof.get("status") == "pass" else []
        allowed_by_pid = {r["pid"]: r for r in allowed}
        identity_fields = ("pid", "parent_pid", "name", "path", "create_utc")
        blocked = [r for r in risky_processes if r.get("pid") not in allowed_by_pid or
                   any(r.get(k) != allowed_by_pid[r["pid"]].get(k) for k in identity_fields)]
        passed = len(blocked) == 0 and (proof is None or isinstance(proof, dict) and proof.get("status") == "pass")
        gates["windows_processes"] = {"pass": passed, "found": blocked,
                                      "exact_owned_controllers": allowed,
                                      "reason": None if passed else "Strata/Python/ComfyUI process present"}
        if not passed: reasons.append(gates["windows_processes"]["reason"])

    if check_wsl:
        wsl = raw.get("wsl")
        if not isinstance(wsl, dict) or wsl.get("error"):
            gates["wsl"] = {"pass": False, "reason": "current WSL state unknown"}
            reasons.append(gates["wsl"]["reason"])
        else:
            blocked_wsl = []
            for distro in wsl.get("running_distros", wsl.get("ubuntu_running", [])):
                if distro.get("error"):
                    blocked_wsl.append({"distro": distro.get("name"), "reason": "WSL inspection unknown"})
                    continue
                if inspect_running_wsl and verified_wsl_idle(distro):
                    continue
                blocked_wsl.append({"distro": distro.get("name"),
                                    "reason": "WSL device users unknown: running Linux distro"})
            passed = not blocked_wsl
            gates["wsl"] = {"pass": passed, "blocked": blocked_wsl,
                            "reason": None if passed else "WSL device users unknown: running Linux distro"}
            if not passed: reasons.append(gates["wsl"]["reason"])

    return {"status": "pass" if not reasons else "blocked", "observer_only": True,
            "is_model_test": False, "gates": gates, "reasons": reasons}


def _windows_memory() -> dict[str, Any]:
    # Reuse the sampler's tested WinAPI implementation. GMS pagefile
    # availability is a diagnostic, never the system commit gate.
    source = Path(__file__).resolve().with_name("hetero_resources.py")
    spec = importlib.util.spec_from_file_location("_hetero_admission_resources", source)
    if spec is None or spec.loader is None: raise RuntimeError("resource observer unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    memory = module.windows_memory_global()
    fields = ("physical_total_bytes", "physical_available_bytes", "commit_limit_bytes", "commit_available_bytes")
    if any(type(memory.get(key)) is not int for key in fields):
        raise RuntimeError("physical RAM or PSAPI system commit telemetry unavailable")
    return {"total_bytes": memory["physical_total_bytes"], "available_bytes": memory["physical_available_bytes"],
            "commit_limit_bytes": memory["commit_limit_bytes"], "available_commit_bytes": memory["commit_available_bytes"],
            "source": "GlobalMemoryStatusEx physical RAM; PSAPI GetPerformanceInfo system commit"}


def _process_records(pids: list[int], runner: Runner) -> list[dict[str, Any]]:
    if not pids:
        return []
    ps = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        raise RuntimeError("PowerShell process identity query unavailable")
    pidlist = ",".join(str(int(pid)) for pid in sorted(set(pids)))
    script = f"Get-CimInstance Win32_Process | Where-Object {{$_.ProcessId -in @({pidlist})}} | Select-Object ProcessId,Name,ExecutablePath | ConvertTo-Json -Compress"
    result = _execute([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], runner)
    data = json.loads(result.stdout or "null")
    if isinstance(data, dict): data = [data]
    records = []
    for item in data or []:
        records.append({"pid": int(item["ProcessId"]), "name": item.get("Name"),
                        "path": item.get("ExecutablePath")})
    found = {r["pid"] for r in records}
    records.extend({"pid": pid, "name": None, "path": None} for pid in pids if pid not in found)
    return records


def _risky_windows_processes(runner: Runner) -> list[dict[str, Any]]:
    ps = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        raise RuntimeError("PowerShell process identity query unavailable")
    observer_pid = os.getpid()
    script = f"Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -ne {observer_pid} -and ($_.Name -match '(?i)strata|python|comfy' -or $_.ExecutablePath -match '(?i)strata|python|comfy') }} | Select-Object ProcessId,Name,ExecutablePath | ConvertTo-Json -Compress"
    result = _execute([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], runner)
    data = json.loads(result.stdout or "null")
    if isinstance(data, dict): data = [data]
    return [{"pid": int(item["ProcessId"]), "name": item.get("Name"), "path": item.get("ExecutablePath")}
            for item in data or [] if int(item["ProcessId"]) != observer_pid]


WSL_GPU_PROBE = r'''import datetime,errno,json,os,stat,subprocess
result={"schema":"strata-wsl-gpu-idle-v1","uid":os.geteuid(),"clients":[],"errors":[],"scanned_pids":0,"scanned_fds":0,"complete":False}
def error(message):
 if len(result["errors"])<32: result["errors"].append(str(message))
try:
 result["boot_id"]=open("/proc/sys/kernel/random/boot_id").read().strip()
 device=os.stat("/dev/dxg")
 result["device"]={"path":"/dev/dxg","rdev":device.st_rdev,"character":stat.S_ISCHR(device.st_mode)}
 if result["uid"]!=0: raise RuntimeError("root required for complete fd observation")
 for entry in os.listdir("/proc"):
  if not entry.isdigit(): continue
  root="/proc/"+entry
  try: descriptors=os.listdir(root+"/fd")
  except FileNotFoundError: continue
  except OSError as exc: error("pid "+entry+" fd list: "+str(exc));continue
  result["scanned_pids"]+=1
  for fd in descriptors:
   path=root+"/fd/"+fd
   try: item=os.stat(path)
   except FileNotFoundError: continue
   except OSError as exc: error("pid "+entry+" fd "+fd+": "+str(exc));continue
   result["scanned_fds"]+=1
   if stat.S_ISCHR(item.st_mode) and item.st_rdev==device.st_rdev:
    try: command=open(root+"/cmdline","rb").read(8192).replace(b"\0",b" ").decode("utf-8","replace")
    except OSError: command=None
    result["clients"].append({"pid":int(entry),"fd":fd,"command":command})
 status=subprocess.run(["systemctl","show","comfyui.service","-p","ActiveState","-p","MainPID","-p","LoadState"],capture_output=True,text=True,timeout=5)
 values=dict(line.split("=",1) for line in status.stdout.splitlines() if "=" in line)
 result["comfy"]={"active":values.get("ActiveState"),"main_pid":int(values.get("MainPID","-1")),"load":values.get("LoadState"),"exit_code":status.returncode}
 if result["boot_id"]!=open("/proc/sys/kernel/random/boot_id").read().strip(): error("boot changed during fd scan")
 result["complete"]=not result["errors"] and result["scanned_pids"]>0
except Exception as exc: error(type(exc).__name__+": "+str(exc))
print(json.dumps(result,separators=(",",":"),ensure_ascii=True))
'''


def verified_wsl_idle(distro: dict[str, Any], now: dt.datetime | None = None) -> bool:
    """Accept only a fresh, complete root fd inspection with no GPU clients.

    Opt-in callers still use the independent NVIDIA/process/RAM/commit gates.
    An idle service or low utilization alone is never sufficient.
    """
    import hashlib
    proof = distro.get("gpu_inspection")
    if not isinstance(proof, dict) or distro.get("inspection_script_sha256") != hashlib.sha256(WSL_GPU_PROBE.encode()).hexdigest():
        return False
    try:
        at = dt.datetime.fromisoformat(distro["inspected_at_utc"].replace("Z", "+00:00"))
        if at.tzinfo is None: return False
        age = ((now or dt.datetime.now(dt.timezone.utc)) - at).total_seconds()
        if not 0 <= age <= 10: return False
    except (KeyError, TypeError, ValueError):
        return False
    device, comfy = proof.get("device"), proof.get("comfy")
    return bool(
        proof.get("schema") == "strata-wsl-gpu-idle-v1" and type(proof.get("uid")) is int and proof["uid"] == 0
        and proof.get("complete") is True and proof.get("errors") == [] and proof.get("clients") == []
        and type(proof.get("scanned_pids")) is int and proof["scanned_pids"] > 0
        and type(proof.get("scanned_fds")) is int and proof["scanned_fds"] > 0
        and isinstance(proof.get("boot_id"), str)
        and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", proof["boot_id"])
        and isinstance(device, dict) and device.get("path") == "/dev/dxg" and device.get("character") is True
        and type(device.get("rdev")) is int and device["rdev"] > 0
        and isinstance(comfy, dict) and comfy.get("active") in ("inactive", "failed")
        and type(comfy.get("main_pid")) is int and comfy["main_pid"] == 0
        and comfy.get("load") in ("loaded", "not-found")
        and type(comfy.get("exit_code")) is int
        and (comfy["exit_code"] == 0 or comfy["load"] == "not-found" and comfy["exit_code"] == 4)
    )


def _probe_wsl(runner: Runner, *, inspect_running: bool = False) -> dict[str, Any]:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return {"available": False, "running_list_raw": "", "running_distros": []}
    listed = _execute([wsl, "--list", "--running", "--quiet"], runner, text=False)
    raw_bytes = listed.stdout or b""
    names = decode_wsl_list(raw_bytes)
    records = [{"name": name, "comfy_active": None, "dxg_users": None,
                "reason": "WSL device users unknown: running Linux distro"} for name in names]
    if inspect_running:
        import hashlib
        for record in records:
            # Recheck before exec so the opt-in inspection is requested only
            # for a distro observed running now, never every installed distro.
            current = decode_wsl_list(_execute([wsl, "--list", "--running", "--quiet"], runner, text=False).stdout or b"")
            if record["name"] not in current:
                record["error"] = "distro state changed before inspection"
                continue
            try:
                observed = _execute([wsl, "--distribution", record["name"], "--user", "root", "--exec",
                                     "python3", "-I", "-S", "-B", "-c", WSL_GPU_PROBE], runner)
                proof = json.loads(observed.stdout)
                record.update({"gpu_inspection": proof,
                               "inspection_script_sha256": hashlib.sha256(WSL_GPU_PROBE.encode()).hexdigest(),
                               "inspected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
                if verified_wsl_idle(record):
                    record.update({"comfy_active": False, "dxg_users": [], "reason": None})
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        after = decode_wsl_list(_execute([wsl, "--list", "--running", "--quiet"], runner, text=False).stdout or b"")
        if set(after) != set(names):
            return {"available": True, "running_list_raw": after, "running_distros": records,
                    "error": "running distro set changed during inspection"}
    return {"available": True, "running_list_raw": names, "running_distros": records}


def _controller_process_observation(path: Path, runner: Runner) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError("duplicate controller proof key")
            result[key] = value
        return result
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("bounded absolute regular controller proof required")
    with path.open("rb") as stream:
        raw_proof = stream.read(65537)
    if len(raw_proof) > 65536: raise ValueError("controller proof too large")
    proof = json.loads(raw_proof, object_pairs_hook=unique_object)
    source = proof["source"]
    expected_source = Path(__file__).resolve().with_name("hetero_pool_sweep.py")
    if Path(source["path"]).resolve(strict=True) != expected_source or hashlib.sha256(expected_source.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("controller proof source hash mismatch")
    owner_pids = [os.getpid(), proof["controller"]["pid"], *(r["pid"] for r in proof["python_controllers"])]
    if any(type(pid) is not int or pid <= 0 for pid in owner_pids): raise ValueError("invalid controller PID")
    pidlist = ",".join(str(pid) for pid in owner_pids)
    ps = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not ps: raise RuntimeError("PowerShell process identity query unavailable")
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(" + pidlist + ") -or "
              "$_.Name -match '(?i)strata|python|comfy' -or $_.ExecutablePath -match '(?i)strata|python|comfy' } | "
              "ForEach-Object { [ordered]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;"
              "name=$_.Name;path=$_.ExecutablePath;create_utc=$_.CreationDate.ToUniversalTime().ToString('o');"
              "command_line=$_.CommandLine} } | ConvertTo-Json -Compress")
    result = _execute([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], runner)
    live = json.loads(result.stdout or "null")
    if isinstance(live, dict): live = [live]
    own = next((r for r in live or [] if r["pid"] == os.getpid()), None)
    if own is None: raise ValueError("admission observer identity missing")
    allowed = verify_controller_chain(proof, live, own["parent_pid"], dt.datetime.now(dt.timezone.utc))
    risky = [{k: v for k, v in r.items() if k != "command_line"} for r in live
             if r["pid"] != os.getpid() and re.search(r"strata|python|comfy", str(r["name"]) + " " + str(r["path"]), re.I)]
    return risky, {"status": "pass", "proof_sha256": hashlib.sha256(raw_proof).hexdigest(),
                   "verified_python_controllers": allowed, "launch_controller": proof["controller"]}


def observe(runner: Runner = subprocess.run, check_wsl: bool = True,
             owned_controller_proof: Path | None = None, inspect_running_wsl: bool = False) -> dict[str, Any]:
    raw: dict[str, Any] = {"observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "errors": []}
    if os.name != "nt":
        raw["windows_memory"] = None
        raw["risky_windows_processes"] = None
        raw["nvidia"] = {"telemetry_error": "Windows host required for this admission observer", "gpus": []}
        if check_wsl: raw["wsl"] = {"error": "Windows host required"}
        return raw
    try: raw["windows_memory"] = _windows_memory()
    except Exception as exc:
        raw["windows_memory"] = None; raw["errors"].append({"section": "windows_memory", "error": str(exc)})
    try:
        if owned_controller_proof is None:
            raw["risky_windows_processes"] = _risky_windows_processes(runner)
        else:
            raw["risky_windows_processes"], raw["owned_controller_validation"] = _controller_process_observation(owned_controller_proof, runner)
    except Exception as exc:
        raw["risky_windows_processes"] = None; raw["errors"].append({"section": "windows_processes", "error": str(exc)})

    smi = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if not smi:
        raw["nvidia"] = {"telemetry_error": "nvidia-smi not found", "gpus": []}
    else:
        try:
            gpu_out = _execute([smi, "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"], runner).stdout
            proc_out = _execute([smi, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], runner).stdout
            nvidia = parse_nvidia_csv(gpu_out, proc_out)
            records = _process_records(nvidia["compute_pids"], runner)
            nvidia["processes"] = records
            nvidia["telemetry_error"] = None
            raw["nvidia"] = nvidia
        except Exception as exc:
            raw["nvidia"] = {"telemetry_error": str(exc), "gpus": [], "raw_gpu_csv": None, "raw_process_csv": None}
            raw["errors"].append({"section": "nvidia", "error": str(exc)})
    if check_wsl:
        try: raw["wsl"] = _probe_wsl(runner, inspect_running=inspect_running_wsl)
        except Exception as exc:
            raw["wsl"] = {"error": str(exc), "running_list_raw": None, "ubuntu_running": []}
            raw["errors"].append({"section": "wsl", "error": str(exc)})
    home = str(Path.home())
    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(home, "%USERPROFILE%") if home and home != "/" else value
        if isinstance(value, list): return [scrub(v) for v in value]
        if isinstance(value, dict): return {k: scrub(v) for k, v in value.items()}
        return value
    raw = scrub(raw)
    return raw


def _check_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    if not path.parent.exists() or not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")


def main(argv: list[str] | None = None, *, observer: Callable[..., dict[str, Any]] = observe) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-vram-mib", type=int, default=46000)
    parser.add_argument("--minimum-ram-gib", type=float, default=12)
    parser.add_argument("--check-wsl-current", action=argparse.BooleanOptionalAction, default=(os.name == "nt"))
    parser.add_argument("--inspect-running-wsl", action="store_true",
                        help="inspect already-running WSL GPU file descriptors; unknown or non-idle state still blocks")
    parser.add_argument("--owned-controller-proof", type=Path,
                        help="fresh exact launch-controller ancestry proof; never exempts GPU workloads")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--observe", action="store_true")
    args = parser.parse_args(argv)
    if args.required_vram_mib < 1 or args.minimum_ram_gib < 0:
        parser.error("resource thresholds must be positive")
    output = Path(args.output)
    try:
        _check_output(output)
        if args.validate_only:
            print(json.dumps({"status": "valid", "observer_only": True, "observe_performed": False,
                              "output_available": str(output)}, ensure_ascii=False))
            return 0
        kwargs = {"check_wsl": args.check_wsl_current}
        if args.inspect_running_wsl: kwargs["inspect_running_wsl"] = True
        if args.owned_controller_proof is not None: kwargs["owned_controller_proof"] = args.owned_controller_proof
        raw = observer(**kwargs)
        decision = evaluate(raw, args.required_vram_mib, args.minimum_ram_gib, args.check_wsl_current,
                            inspect_running_wsl=args.inspect_running_wsl)
        manifest = {"schema_version": 1, "observed_at_utc": raw.get("observed_at_utc"),
                    "observer_only": True, "is_model_test": False,
                    "raw": raw, "chosen_gates": decision["gates"], "status": decision["status"],
                    "pass": decision["status"] == "pass", "reasons": decision["reasons"]}
        payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        # Exclusive creation prevents accidentally replacing a prior admission receipt.
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        print(json.dumps({"status": manifest["status"], "output": str(output), "reasons": manifest["reasons"]}, ensure_ascii=False))
        return 0 if manifest["pass"] else 3
    except FileExistsError as exc:
        print("output already exists", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"admission collection failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
