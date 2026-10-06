#!/usr/bin/env python3
"""Read-only resource and process gate for a possible GPU run; this is not a model test."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
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


def evaluate(raw: dict[str, Any], required_vram_mib: int, minimum_ram_gib: float,
             check_wsl: bool = True) -> dict[str, Any]:
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
                elif base not in DISPLAY_ALLOWLIST:
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
        passed = len(risky_processes) == 0
        gates["windows_processes"] = {"pass": passed, "found": risky_processes,
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
                blocked_wsl.append({"distro": distro.get("name"),
                                    "reason": "WSL device users unknown: running Linux distro"})
            passed = not blocked_wsl
            gates["wsl"] = {"pass": passed, "blocked": blocked_wsl,
                            "reason": None if passed else "WSL device users unknown: running Linux distro"}
            if not passed: reasons.append(gates["wsl"]["reason"])

    return {"status": "pass" if not reasons else "blocked", "observer_only": True,
            "is_model_test": False, "gates": gates, "reasons": reasons}


def _windows_memory() -> dict[str, int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_bytes": int(status.ullTotalPhys), "available_bytes": int(status.ullAvailPhys),
            "commit_limit_bytes": int(status.ullTotalPageFile), "available_commit_bytes": int(status.ullAvailPageFile)}


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


def _probe_wsl(runner: Runner) -> dict[str, Any]:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return {"available": False, "running_list_raw": "", "running_distros": []}
    listed = _execute([wsl, "--list", "--running", "--quiet"], runner, text=False)
    raw_bytes = listed.stdout or b""
    names = decode_wsl_list(raw_bytes)
    records = [{"name": name, "comfy_active": None, "dxg_users": None,
                "reason": "WSL device users unknown: running Linux distro"} for name in names]
    return {"available": True, "running_list_raw": names, "running_distros": records}


def observe(runner: Runner = subprocess.run, check_wsl: bool = True) -> dict[str, Any]:
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
    try: raw["risky_windows_processes"] = _risky_windows_processes(runner)
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
        try: raw["wsl"] = _probe_wsl(runner)
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
        raw = observer(check_wsl=args.check_wsl_current)
        decision = evaluate(raw, args.required_vram_mib, args.minimum_ram_gib, args.check_wsl_current)
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
