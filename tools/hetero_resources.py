#!/usr/bin/env python3
"""Read-only Windows resource JSONL sampler for an already-running owner process."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - host-dependent fallback
    psutil = None

RAM_RESERVE_BYTES = 12 * 1024**3
DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_SECONDS = 3600
MAX_ALLOWED_SECONDS = 86400
COMMAND_TIMEOUT_SECONDS = 5

BASE_GPU_FIELDS = ("name", "memory.used", "memory.free", "utilization.gpu", "power.draw", "temperature.gpu")
OPTIONAL_GPU_FIELDS = ("clocks.current.graphics", "clocks.current.memory", "pcie.link.gen.current", "pcie.link.width.current")


def _finite_number(text: str, *, integer: bool = False) -> int | float | None:
    value = text.strip()
    if value.casefold() in {"n/a", "na", "not supported", "[not supported]", ""}:
        return None
    try:
        parsed = int(value) if integer else float(value)
    except ValueError:
        raise ValueError("invalid numeric NVIDIA field") from None
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError("non-finite NVIDIA field")
    return parsed


def parse_nvidia_base(text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Strictly parse six no-header/no-units CSV fields, retaining N/A as unknown."""
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    if not rows:
        raise ValueError("NVIDIA returned no GPU rows")
    devices: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        if len(row) != len(BASE_GPU_FIELDS) or not row[0].strip():
            raise ValueError(f"NVIDIA CSV row {index} has wrong field count")
        used = _finite_number(row[1], integer=True)
        free = _finite_number(row[2], integer=True)
        util = _finite_number(row[3])
        power = _finite_number(row[4])
        temp = _finite_number(row[5])
        if (used is not None and used < 0) or (free is not None and free < 0):
            raise ValueError(f"NVIDIA CSV row {index} has negative memory")
        if util is not None and not 0 <= util <= 100:
            raise ValueError(f"NVIDIA CSV row {index} has invalid utilization")
        if power is not None and power < 0:
            raise ValueError(f"NVIDIA CSV row {index} has invalid power")
        if temp is not None and temp < 0:
            raise ValueError(f"NVIDIA CSV row {index} has invalid temperature")
        known = used is not None and free is not None and util is not None
        device = {"name": row[0].strip(), "memory_used_mib": used, "memory_free_mib": free,
                  "utilization_percent": util, "power_watts": power, "temperature_c": temp,
                  "telemetry_known": known, "optional": {}}
        if not known:
            errors.append({"field": f"devices[{index - 1}]", "error": "required NVIDIA telemetry contains N/A"})
        devices.append(device)
    return devices, errors


def parse_nvidia_optional(text: str, count: int) -> tuple[list[dict[str, Any]], str | None]:
    if not text.strip():
        return [{} for _ in range(count)], None
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    if len(rows) != count:
        return [{} for _ in range(count)], "optional NVIDIA row count did not match base telemetry"
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if len(row) != len(OPTIONAL_GPU_FIELDS):
            return [{} for _ in range(count)], f"optional NVIDIA row {index} has wrong field count"
        values = [_finite_number(cell, integer=True) for cell in row]
        parsed.append(dict(zip(OPTIONAL_GPU_FIELDS, values)))
    return parsed, None


def _error(section: str, exc: BaseException | str) -> dict[str, str]:
    # Keep errors short and avoid storing command lines, stderr, environment, or secrets.
    msg = str(exc) if isinstance(exc, str) else type(exc).__name__
    return {"section": section, "error": msg[:160]}


def _utc_iso(now: dt.datetime | None = None) -> str:
    value = now or dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _scrub_executable(path: str | None) -> str | None:
    if not path:
        return None
    home = os.environ.get("USERPROFILE") or str(Path.home())
    if home and home != ".":
        norm_home = os.path.normcase(os.path.normpath(home))
        norm_path = os.path.normcase(os.path.normpath(path))
        if norm_path == norm_home or norm_path.startswith(norm_home + os.sep):
            suffix = os.path.normpath(path)[len(os.path.normpath(home)):].lstrip("\\/")
            return "%USERPROFILE%" + ("\\" + suffix.replace("/", "\\") if suffix else "")
    return path


def windows_memory_global() -> dict[str, Any]:
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
    return {"source": "GlobalMemoryStatusEx", "physical_total_bytes": int(status.ullTotalPhys),
            "physical_available_bytes": int(status.ullAvailPhys),
            "physical_used_bytes": int(status.ullTotalPhys - status.ullAvailPhys),
            "commit_limit_bytes": int(status.ullTotalPageFile),
            "commit_available_bytes": int(status.ullAvailPageFile),
            "commit_used_bytes": int(status.ullTotalPageFile - status.ullAvailPageFile),
            "pagefile_total_bytes": None, "pagefile_used_bytes": None}


def memory_snapshot(global_memory: Callable[[], dict[str, Any]] = windows_memory_global,
                    psutil_module: Any = psutil) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    try:
        return global_memory(), errors
    except Exception as exc:
        errors.append(_error("memory_global", exc))
    if psutil_module is None:
        errors.append(_error("memory_fallback", "psutil unavailable"))
        return {"source": None, "physical_total_bytes": None, "physical_available_bytes": None,
                "physical_used_bytes": None, "commit_limit_bytes": None, "commit_available_bytes": None,
                "commit_used_bytes": None, "pagefile_total_bytes": None, "pagefile_used_bytes": None}, errors
    try:
        vm = psutil_module.virtual_memory()
        sw = psutil_module.swap_memory()
        return {"source": "psutil.virtual_memory+swap_memory (commit capacity unavailable)",
                "physical_total_bytes": int(vm.total), "physical_available_bytes": int(vm.available),
                "physical_used_bytes": int(vm.total - vm.available), "commit_limit_bytes": None,
                "commit_available_bytes": None, "commit_used_bytes": None,
                "pagefile_total_bytes": int(sw.total), "pagefile_used_bytes": int(sw.used)}, errors
    except Exception as exc:
        errors.append(_error("memory_fallback", exc))
        return {"source": None, "physical_total_bytes": None, "physical_available_bytes": None,
                "physical_used_bytes": None, "commit_limit_bytes": None, "commit_available_bytes": None,
                "commit_used_bytes": None, "pagefile_total_bytes": None, "pagefile_used_bytes": None}, errors


def _nvidia_commands(runner: Callable[..., Any], executable: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    base_args = [executable, "--query-gpu=" + ",".join(BASE_GPU_FIELDS), "--format=csv,noheader,nounits"]
    try:
        result = runner(base_args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, check=False)
    except Exception as exc:
        return [], [_error("nvidia", exc)]
    if result.returncode != 0:
        return [], [_error("nvidia", f"nvidia-smi base query exited {result.returncode}")]
    try:
        devices, parse_errors = parse_nvidia_base(result.stdout or "")
        errors.extend(parse_errors)
    except Exception as exc:
        return [], [_error("nvidia", exc)]
    optional_args = [executable, "--query-gpu=" + ",".join(OPTIONAL_GPU_FIELDS), "--format=csv,noheader,nounits"]
    try:
        optional = runner(optional_args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, check=False)
        if optional.returncode == 0:
            rows, optional_error = parse_nvidia_optional(optional.stdout or "", len(devices))
            for device, row in zip(devices, rows):
                device["optional"].update(row)
            if optional_error:
                errors.append(_error("nvidia_optional", optional_error))
        else:
            errors.append(_error("nvidia_optional", f"optional query exited {optional.returncode}"))
    except Exception as exc:
        errors.append(_error("nvidia_optional", exc))
    return devices, errors


class ResourceSampler:
    def __init__(self, owner_pid: int | None, *, psutil_module: Any = psutil,
                 nvidia_runner: Callable[..., Any] = subprocess.run,
                 memory_global: Callable[[], dict[str, Any]] = windows_memory_global,
                 monotonic: Callable[[], float] = time.monotonic,
                 utc_now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
                 nvidia_path: str | None = None):
        self.owner_pid = owner_pid
        self.psutil_module = psutil_module
        self.nvidia_runner = nvidia_runner
        self.memory_global = memory_global
        self.monotonic = monotonic
        self.utc_now = utc_now
        self.nvidia_path = nvidia_path if nvidia_path is not None else (shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi"))
        self.previous_sample_start: float | None = None
        self._owner_create_time: float | None = None
        self._prime_cpu()

    def _prime_cpu(self) -> None:
        if self.psutil_module is not None:
            try:
                self.psutil_module.cpu_percent(interval=None)
            except Exception:
                pass

    def _process_tree(self) -> tuple[list[dict[str, Any]] | None, list[dict[str, str]]]:
        if self.owner_pid is None:
            return None, []
        if self.psutil_module is None:
            return None, [_error("process_tree", "psutil unavailable")]
        errors: list[dict[str, str]] = []
        try:
            root = self.psutil_module.Process(self.owner_pid)
            create_time = float(root.create_time())
            if self._owner_create_time is None:
                self._owner_create_time = create_time
            elif create_time != self._owner_create_time:
                return None, [_error("process_tree", "owner PID identity changed")]
            processes = [root] + root.children(recursive=True)
        except Exception as exc:
            return None, [_error("process_tree", exc)]
        records: list[dict[str, Any]] = []
        for proc in processes:
            row: dict[str, Any] = {"pid": None, "parent_pid": None, "name": None, "executable_path": None,
                                   "rss_bytes": None, "io_read_bytes": None, "io_write_bytes": None,
                                   "cpu_user_seconds": None, "cpu_system_seconds": None, "errors": []}
            try: row["pid"] = int(proc.pid)
            except Exception as exc: row["errors"].append(_error("pid", exc))
            try: row["parent_pid"] = int(proc.ppid())
            except Exception as exc: row["errors"].append(_error("parent_pid", exc))
            try: row["name"] = proc.name()
            except Exception as exc: row["errors"].append(_error("name", exc))
            try: row["executable_path"] = _scrub_executable(proc.exe())
            except Exception as exc: row["errors"].append(_error("executable_path", exc))
            try: row["rss_bytes"] = int(proc.memory_info().rss)
            except Exception as exc: row["errors"].append(_error("rss", exc))
            try:
                io = proc.io_counters()
                row["io_read_bytes"] = int(io.read_bytes)
                row["io_write_bytes"] = int(io.write_bytes)
            except Exception as exc: row["errors"].append(_error("io_counters", exc))
            try:
                cpu = proc.cpu_times()
                row["cpu_user_seconds"] = float(cpu.user)
                row["cpu_system_seconds"] = float(cpu.system)
            except Exception as exc: row["errors"].append(_error("cpu_times", exc))
            records.append(row)
        return records, errors

    def sample(self) -> dict[str, Any]:
        started = self.monotonic()
        utc = _utc_iso(self.utc_now())
        errors: list[dict[str, str]] = []
        try:
            cpu_percent = self.psutil_module.cpu_percent(interval=None) if self.psutil_module is not None else None
            if cpu_percent is not None and not math.isfinite(float(cpu_percent)):
                cpu_percent = None
                errors.append(_error("cpu", "non-finite CPU percent"))
            cpu = {"percent": cpu_percent, "source": "psutil.cpu_percent (system-wide; interval since prior call)" if self.psutil_module else None}
        except Exception as exc:
            cpu = {"percent": None, "source": "psutil.cpu_percent"}
            errors.append(_error("cpu", exc))
        memory, mem_errors = memory_snapshot(self.memory_global, self.psutil_module)
        errors.extend(mem_errors)
        available = memory.get("physical_available_bytes")
        if available is None:
            ram_gate = {"status": "unknown", "available_gib": None, "required_gib": 12}
        else:
            available_gib = float(available) / 1024**3
            ram_gate = {"status": "pass" if available >= RAM_RESERVE_BYTES else "block",
                        "available_gib": round(available_gib, 3), "required_gib": 12}

        process_tree, process_errors = self._process_tree()
        errors.extend(process_errors)
        disks: dict[str, Any] | None = None
        if self.psutil_module is not None:
            try:
                counters = self.psutil_module.disk_io_counters(perdisk=True)
                disks = {str(name): {"read_bytes": int(value.read_bytes), "write_bytes": int(value.write_bytes),
                                     "read_count": int(value.read_count), "write_count": int(value.write_count),
                                     "device_key_source": "psutil.disk_io_counters(perdisk=True)",
                                     "volume_mapping": "not_inferred"}
                         for name, value in (counters or {}).items()}
            except Exception as exc:
                errors.append(_error("disk_io", exc))
        else:
            errors.append(_error("disk_io", "psutil unavailable"))

        nvidia: dict[str, Any] = {"source": "nvidia-smi", "telemetry_known": False, "devices": []}
        if not self.nvidia_path:
            nvidia["error"] = "nvidia-smi not found"
            errors.append(_error("nvidia", nvidia["error"]))
        else:
            devices, nvidia_errors = _nvidia_commands(self.nvidia_runner, self.nvidia_path)
            nvidia["devices"] = devices
            nvidia["telemetry_known"] = bool(devices) and all(d["telemetry_known"] for d in devices)
            if nvidia_errors:
                nvidia["errors"] = nvidia_errors
                errors.extend(nvidia_errors)
            if not devices:
                nvidia["error"] = "required NVIDIA telemetry unavailable"

        alerts: list[str] = []
        if ram_gate["status"] == "block": alerts.append("available_ram_below_12_gib")
        if ram_gate["status"] == "unknown": alerts.append("available_ram_unknown")
        if not nvidia["telemetry_known"]: alerts.append("nvidia_telemetry_unknown")
        if process_tree is None and self.owner_pid is not None: alerts.append("owner_process_tree_unknown")
        ended = self.monotonic()
        gap = None if self.previous_sample_start is None else started - self.previous_sample_start
        self.previous_sample_start = started
        return {"schema_version": 1, "record_type": "resource_sample", "observed_at_utc": utc,
                "monotonic_seconds": started, "sample_gap_seconds": gap,
                "sampler_elapsed_seconds": max(0.0, ended - started), "owner_pid": self.owner_pid,
                "cpu": cpu, "memory": memory, "ram_gate": ram_gate,
                "owner_process_tree": process_tree, "disk_io": disks, "nvidia": nvidia,
                "alerts": alerts, "errors": errors}


def _write_jsonl_line(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def run_sampling(stream: Any, sampler: ResourceSampler, *, interval_seconds: float, max_seconds: int,
                 stop_file: str | None, monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 stop_exists: Callable[[str], bool] = os.path.exists) -> str:
    start = monotonic()
    next_sample = start
    while True:
        now = monotonic()
        if stop_file and stop_exists(stop_file):
            _write_jsonl_line(stream, {"schema_version": 1, "record_type": "sampler_stopped",
                                       "observed_at_utc": _utc_iso(), "monotonic_seconds": now,
                                       "reason": "stop_file_exists"})
            return "stop_file"
        if now - start >= max_seconds:
            _write_jsonl_line(stream, {"schema_version": 1, "record_type": "sampler_stopped",
                                       "observed_at_utc": _utc_iso(), "monotonic_seconds": now,
                                       "reason": "max_seconds_reached"})
            return "max_seconds"
        _write_jsonl_line(stream, sampler.sample())
        next_sample += interval_seconds
        delay = max(0.0, next_sample - monotonic())
        if delay:
            sleep(min(delay, max(0.0, start + max_seconds - monotonic())))


def _check_output(path: Path) -> str | None:
    if path.exists(): return "output already exists"
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.exists() or not parent.is_dir(): return "output directory must exist"
    return None


def main(argv: list[str] | None = None, *, sampler_factory: Callable[[int | None], ResourceSampler] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--stop-file")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.pid is not None and args.pid <= 0: parser.error("--pid must be positive")
    if not math.isfinite(args.interval) or args.interval <= 0: parser.error("--interval must be finite and positive")
    if args.max_seconds <= 0 or args.max_seconds > MAX_ALLOWED_SECONDS:
        parser.error(f"--max-seconds must be 1..{MAX_ALLOWED_SECONDS}")
    output = Path(args.output)
    error = _check_output(output)
    if error: print(error, file=sys.stderr); return 2
    if args.validate_only:
        print(json.dumps({"status": "valid", "queries_started": False, "output_written": False}, allow_nan=False))
        return 0
    try:
        stream = output.open("x", encoding="utf-8", newline="\n")
    except FileExistsError:
        print("output already exists", file=sys.stderr); return 2
    except OSError as exc:
        print(f"cannot create output ({type(exc).__name__})", file=sys.stderr); return 2
    with stream:
        sampler = (sampler_factory or (lambda pid: ResourceSampler(pid)))(args.pid)
        reason = run_sampling(stream, sampler, interval_seconds=args.interval, max_seconds=args.max_seconds,
                              stop_file=args.stop_file)
    print(json.dumps({"status": "stopped", "reason": reason, "output_written": True}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
