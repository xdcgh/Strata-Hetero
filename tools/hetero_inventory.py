#!/usr/bin/env python3
"""Collect a small, read-only hardware inventory on Windows or Linux."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

TIMEOUT_SECONDS = 60
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(args: list[str], runner: Runner = subprocess.run, *, input_text: str | None = None) -> str:
    result = runner(args, input=input_text, capture_output=True, text=True,
                    timeout=TIMEOUT_SECONDS, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or f"exit {result.returncode}").strip())
    return result.stdout


def _powershell_inventory(runner: Runner = subprocess.run) -> dict[str, Any]:
    script = r'''
$ErrorActionPreference = 'Stop'
function Safe-Section($name, [scriptblock]$action) {
  try { $script:result[$name] = @(& $action) }
  catch { $script:errors += [ordered]@{section=$name; error=$_.Exception.Message} }
}
$result = [ordered]@{}
$errors = @()
Safe-Section 'os' { $o=Get-CimInstance Win32_OperatingSystem; [ordered]@{name=$o.Caption; version=$o.Version; build=$o.BuildNumber} }
Safe-Section 'cpu' { Get-CimInstance Win32_Processor | ForEach-Object { [ordered]@{name=$_.Name; cores=$_.NumberOfCores; logical_processors=$_.NumberOfLogicalProcessors; max_clock_mhz=$_.MaxClockSpeed} } }
Safe-Section 'memory' {
  $o=Get-CimInstance Win32_OperatingSystem
  $m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory
  [ordered]@{total_bytes=[int64]$o.TotalVisibleMemorySize*1024; available_bytes=[int64]$o.FreePhysicalMemory*1024; commit_limit_bytes=$(if($m){[int64]$m.CommitLimit}); committed_bytes=$(if($m){[int64]$m.CommittedBytes})}
}
Safe-Section 'gpus' { Get-CimInstance Win32_VideoController | ForEach-Object { [ordered]@{name=$_.Name; driver_version=$_.DriverVersion; status=$_.Status} } }
Safe-Section 'accelerators' { Get-PnpDevice -PresentOnly | Where-Object { $_.Class -eq 'ComputeAccelerator' -or $_.FriendlyName -match '(?i)\bNPU\b|Neural|AI Boost' } | ForEach-Object { $d=$_; $v=Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceID='$($d.InstanceId.Replace('\','\\'))'" -ErrorAction SilentlyContinue | Select-Object -First 1; [ordered]@{name=$d.FriendlyName; class=$d.Class; status=$d.Status; driver_version=$v.DriverVersion} } }
Safe-Section 'storage' {
  [ordered]@{disks=@(Get-Disk | ForEach-Object {[ordered]@{number=$_.Number;model=$_.FriendlyName;bus_type=[string]$_.BusType}}); partitions=@(Get-Partition | ForEach-Object {[ordered]@{disk_number=$_.DiskNumber;partition_number=$_.PartitionNumber;drive_letter=$(if($_.DriveLetter){[string]$_.DriveLetter}else{$null});access_paths=@($_.AccessPaths);size_bytes=[int64]$_.Size}}); volumes=@(Get-Volume | ForEach-Object {[ordered]@{drive_letter=$(if($_.DriveLetter){[string]$_.DriveLetter}else{$null});path=$_.Path;label=$_.FileSystemLabel;filesystem=$_.FileSystem;total_bytes=$(if($_.Size){[int64]$_.Size}else{0});free_bytes=$(if($_.SizeRemaining){[int64]$_.SizeRemaining}else{0})}})}
}
Safe-Section 'toolchain' {
  $names=@('git','python','python3','cmake','nvcc','clang','clang++','cl','vswhere','icx','ovc','benchmark_app')
  $commands=@(foreach($n in $names){$c=Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){[ordered]@{name=$n;path=$c.Source}}})
  $roots=@("$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools", "$env:ProgramFiles\Microsoft Visual Studio\2022\Community", "${env:ProgramFiles(x86)}\Intel\oneAPI", "$env:ProgramFiles\Intel\oneAPI", "$env:ProgramFiles\OpenVINO")
  [ordered]@{commands=$commands; installed_roots=@($roots | Where-Object {Test-Path -LiteralPath $_})}
}
[ordered]@{data=$result; errors=$errors} | ConvertTo-Json -Depth 8 -Compress
'''
    shell = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        raise RuntimeError("PowerShell was not found")
    raw = _run([shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], runner)
    return decode_windows_payload(raw)


def decode_windows_payload(raw: str) -> dict[str, Any]:
    """Decode the PowerShell envelope while retaining partial section results."""
    decoded = json.loads(raw.lstrip("\ufeff"))
    data = decoded.get("data", {})
    data = data or {}
    for key in ("os", "memory", "toolchain"):
        if isinstance(data.get(key), list) and len(data[key]) == 1:
            data[key] = data[key][0]
    raw_storage = data.get("storage")
    if isinstance(raw_storage, list) and len(raw_storage) == 1:
        raw_storage = raw_storage[0]
    if isinstance(raw_storage, dict):
        data["storage"] = map_windows_storage(raw_storage.get("volumes", []),
                                              raw_storage.get("partitions", []),
                                              raw_storage.get("disks", []))
    return {**data, "errors": decoded.get("errors", [])}


def map_windows_storage(volumes: list[dict[str, Any]], partitions: list[dict[str, Any]],
                        disks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join Windows volume roots through partitions to physical disk identity."""
    by_number = {int(d["number"]): d for d in disks if d.get("number") is not None}
    result = []
    for volume in volumes:
        letter = (volume.get("drive_letter") or "").strip()
        volume_path = (volume.get("path") or "").rstrip("\\/").casefold()
        matched = []
        for part in partitions:
            part_letter = (part.get("drive_letter") or "").strip()
            paths = [str(x).rstrip("\\/").casefold() for x in (part.get("access_paths") or []) if x]
            if (letter and part_letter.casefold() == letter.casefold()) or (volume_path and volume_path in paths):
                disk = by_number.get(int(part["disk_number"]))
                if disk:
                    matched.append({"disk_number": int(part["disk_number"]), "model": disk.get("model"),
                                    "bus_type": disk.get("bus_type"),
                                    "partition_number": int(part["partition_number"]),
                                    "partition_bytes": int(part["size_bytes"])})
        result.append({"root": f"{letter}:\\" if letter else None, "label": volume.get("label"),
                       "filesystem": volume.get("filesystem"), "total_bytes": int(volume.get("total_bytes") or 0),
                       "free_bytes": int(volume.get("free_bytes") or 0), "physical_disks": matched})
    return result


def parse_nvidia_gpus(text: str) -> list[dict[str, Any]]:
    """Parse strict nvidia-smi CSV rows; malformed rows are rejected."""
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    if rows and [x.strip().lower() for x in rows[0]] == [
        "name", "driver_version", "memory.total [mib]", "memory.used [mib]",
        "memory.free [mib]", "utilization.gpu [%]",
    ]:
        rows = rows[1:]
    parsed = []
    for number, row in enumerate(rows, 1):
        if len(row) != 6 or not row[0].strip():
            raise ValueError(f"invalid NVIDIA GPU CSV row {number}")
        try:
            values = [int(row[i].strip()) for i in range(2, 6)]
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA numeric value on row {number}") from exc
        if min(values[:3]) < 0 or not 0 <= values[3] <= 100:
            raise ValueError(f"out-of-range NVIDIA numeric value on row {number}")
        parsed.append({"name": row[0].strip(), "driver_version": row[1].strip(),
                       "vram_total_bytes": values[0] * 1024**2,
                       "vram_used_bytes": values[1] * 1024**2,
                       "vram_free_bytes": values[2] * 1024**2,
                       "utilization_percent": values[3], "source": "nvidia-smi"})
    return parsed


def _nvidia(runner: Runner) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    exe = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not exe:
        return [], []
    errors: list[dict[str, str]] = []
    gpus: list[dict[str, Any]] = []
    try:
        output = _run([exe, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], runner)
        gpus = parse_nvidia_gpus(output)
    except Exception as exc:
        errors.append({"section": "nvidia_smi.gpu", "error": str(exc)})
        return [], errors
    try:
        proc_text = _run([exe, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], runner)
        pids = []
        for line in proc_text.splitlines():
            v = line.strip()
            if v and v.lower() not in {"n/a", "[insufficient permissions]"}:
                if not re.fullmatch(r"\d+", v):
                    raise ValueError("invalid NVIDIA compute PID row")
                pids.append(int(v))
        # Process paths are queried without command lines or account names.
        if pids and os.name == "nt":
            ps = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
            if ps:
                ids = ",".join(str(p) for p in sorted(set(pids)))
                script = f"Get-CimInstance Win32_Process | Where-Object {{$_.ProcessId -in @({ids})}} | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
                obj = json.loads(_run([ps, "-NoProfile", "-NonInteractive", "-Command", script], runner) or "null")
                if isinstance(obj, dict): obj = [obj]
                path_by_pid = {int(x["ProcessId"]): x.get("ExecutablePath") for x in (obj or [])}
            else: path_by_pid = {}
        elif pids:
            path_by_pid = {p: os.readlink(f"/proc/{p}/exe") for p in pids if Path(f"/proc/{p}/exe").exists()}
        else:
            path_by_pid = {}
        # query-compute-apps without a device selector covers every NVIDIA GPU.
        # Keep that scope explicit; assigning these PIDs to GPU zero is wrong on
        # a multi-card host. This is an observation, not an admission decision.
        processes = [{"pid": p, "path": path_by_pid.get(p)} for p in sorted(set(pids))]
        for gpu in gpus:
            gpu["compute_processes"] = processes
            gpu["compute_processes_scope"] = "all_nvidia_devices"
    except Exception as exc:
        errors.append({"section": "nvidia_smi.processes", "error": str(exc)})
    return gpus, errors


def _linux_inventory(runner: Runner) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    def section(name: str, fn: Callable[[], Any]) -> None:
        try: result[name] = fn()
        except Exception as exc: result["errors"].append({"section": name, "error": str(exc)})
    def read(path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    section("os", lambda: {"name": platform.system(), "version": platform.release(), "build": platform.version()})
    def cpu() -> list[dict[str, Any]]:
        text = read("/proc/cpuinfo")
        model = next((x.split(":",1)[1].strip() for x in text.splitlines() if x.lower().startswith("model name")), platform.processor() or "unknown")
        logical = sum(1 for x in text.splitlines() if x.startswith("processor\t")) or os.cpu_count() or 0
        return [{"name": model, "cores": None, "logical_processors": logical}]
    section("cpu", cpu)
    def memory() -> dict[str, Any]:
        vals = {}
        for line in read("/proc/meminfo").splitlines():
            m = re.match(r"(MemTotal|MemAvailable|CommitLimit|Committed_AS):\s+(\d+) kB", line)
            if m: vals[m.group(1)] = int(m.group(2)) * 1024
        return {"total_bytes": vals["MemTotal"], "available_bytes": vals["MemAvailable"],
                "commit_limit_bytes": vals.get("CommitLimit"), "committed_bytes": vals.get("Committed_AS")}
    section("memory", memory)
    def storage() -> list[dict[str, Any]]:
        data = json.loads(_run([shutil.which("lsblk") or "lsblk", "--json", "--bytes", "--output", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,PKNAME,MODEL,TRAN"], runner))
        entries = []
        def visit(node: dict[str, Any], physical: dict[str, Any] | None) -> None:
            current = physical
            if node.get("type") == "disk":
                current = {"name": node.get("name"), "model": node.get("model"), "bus_type": node.get("tran")}
            if node.get("mountpoint"):
                stat = os.statvfs(node["mountpoint"])
                entries.append({"root":node["mountpoint"], "filesystem":node.get("fstype"), "total_bytes":int(node.get("size") or 0),
                                "free_bytes":stat.f_bavail*stat.f_frsize, "physical_disks":[current] if current else []})
            for child in node.get("children", []): visit(child, current)
        for node in data.get("blockdevices", []): visit(node, None)
        return entries
    section("storage", storage)
    def linux_devices() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        gpus = []
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            if not re.fullmatch(r"card\d+", card.name):
                continue
            device = card / "device"
            if not device.exists():
                continue
            vendor = (device / "vendor").read_text().strip() if (device / "vendor").exists() else None
            driver_link = device / "driver"
            driver = os.path.basename(os.path.realpath(driver_link)) if driver_link.exists() else None
            name_file = next((device / name for name in ("product_name", "product") if (device / name).exists()), None)
            name = name_file.read_text(errors="replace").strip() if name_file else None
            if not name:
                name = f"PCI GPU ({vendor})" if vendor else "Unknown GPU"
            gpus.append({"name": name, "vendor_id": vendor, "driver": driver, "device_class": "DRM"})
        accelerators = []
        for dev in sorted(Path("/sys/class/accel").glob("accel*")):
            driver_link = dev / "device/driver"
            accelerators.append({"name": dev.name, "driver": os.path.basename(os.path.realpath(driver_link)) if driver_link.exists() else None})
        return gpus, accelerators
    try:
        result["gpus"], result["accelerators"] = linux_devices()
    except Exception as exc:
        result["errors"].append({"section": "devices", "error": str(exc)})
    section("toolchain", lambda: [{"name":n,"path":shutil.which(n)} for n in ("git","python3","python","cmake","nvcc","clang","clang++","icx","ovc","benchmark_app") if shutil.which(n)])
    return result


def collect(runner: Runner = subprocess.run) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if os.name == "nt":
        data = _powershell_inventory(runner)
        errors.extend(data.pop("errors", []))
    else:
        data = _linux_inventory(runner)
        errors.extend(data.pop("errors", []))
    nvidia, nvidia_errors = _nvidia(runner)
    errors.extend(nvidia_errors)
    if nvidia:
        known = data.get("gpus") or []
        # Avoid duplicating a device already named by the platform inventory.
        for device in nvidia:
            match = next((x for x in known if x.get("name", "").casefold() == device["name"].casefold()), None)
            if match: match.update(device)
            else: known.append(device)
        data["gpus"] = known
    home = str(Path.home())
    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(home, "%USERPROFILE%") if home and home != "/" else value
        if isinstance(value, list):
            return [scrub(x) for x in value]
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        return value
    data = scrub(data)
    errors = scrub(errors)
    return {"schema_version": 1, "collected_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            **data, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="write the JSON inventory to this path")
    args = parser.parse_args(argv)
    try:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"snapshot already exists: {path}")
        inventory = collect()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Evidence snapshots are immutable. A fresh output name is required for
        # a second observation, including a retry after a failed launch.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
        print(path)
        return 0
    except Exception as exc:
        print(f"hardware inventory failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
