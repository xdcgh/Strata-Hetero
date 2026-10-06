#!/usr/bin/env python3
"""Read-only logical CPU topology inventory and affinity candidate report."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WIN_CPU_SET_TYPE = 0
WIN_CPU_SET_ENTRY_MIN_SIZE = 32
WIN_CPU_SET_HEADER_SIZE = 8
WIN_SDK_SOURCE = "https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information"
WIN_API_SOURCE = "https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemcpusetinformation"


def parse_windows_cpu_set_buffer(data: bytes) -> list[dict[str, Any]]:
    """Decode SDK SYSTEM_CPU_SET_INFORMATION records (32-byte CPU-set payload, extensible entry size).

    SDK offsets: outer Size/Type at 0/4; CpuSet begins at 8; AllocationTag is at 24.
    Unknown types are skipped by their Size, as required by the variable-size API contract.
    """
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        remaining = len(data) - offset
        if remaining < WIN_CPU_SET_HEADER_SIZE:
            raise ValueError(f"truncated record header at byte {offset}: {remaining} bytes remain")
        size, kind = struct.unpack_from("<II", data, offset)
        if size < WIN_CPU_SET_HEADER_SIZE:
            raise ValueError(f"invalid record size {size} at byte {offset}")
        if size > remaining:
            raise ValueError(f"record size {size} exceeds {remaining} remaining bytes at byte {offset}")
        if kind == WIN_CPU_SET_TYPE:
            if size < WIN_CPU_SET_ENTRY_MIN_SIZE:
                raise ValueError(f"CPU-set record too small at byte {offset}: {size} < 32")
            base = offset + WIN_CPU_SET_HEADER_SIZE
            cpu_id, group = struct.unpack_from("<IH", data, base)
            logical, core, llc, numa, efficiency, flags = struct.unpack_from("<6B", data, base + 6)
            scheduling_class = data[base + 12]  # BYTE member of the SDK's Reserved/SchedulingClass union
            allocation_tag = struct.unpack_from("<Q", data, base + 16)[0]
            records.append({"cpu_set_id": cpu_id, "group": group, "logical_processor_index": logical,
                            "core_index": core, "last_level_cache_index": llc, "numa_node_index": numa,
                            "efficiency_class": efficiency, "flags_byte": flags,
                            "parked": bool(flags & 0x01), "allocated": bool(flags & 0x02),
                            "allocated_to_target_process": bool(flags & 0x04), "realtime": bool(flags & 0x08),
                            "scheduling_class": scheduling_class,
                            "allocation_tag": allocation_tag,
                            "core_identity": {"group": group, "core_index": core},
                            "logical_identity": {"group": group, "logical_processor_index": logical}})
        offset += size
    return records


def apply_windows_allowed_masks(rows: list[dict[str, Any]], group_masks: dict[int, int] | None,
                                known_groups: set[int] | None = None,
                                process_default_cpu_set_ids: set[int] | None = None,
                                defaults_known: bool = True) -> None:
    """Annotate reservation and known primary-group process-mask constraints without guessing other groups."""
    known_groups = known_groups or set(group_masks or {})
    for row in rows:
        group = row["group"]
        lp = row["logical_processor_index"]
        reserved_elsewhere = row["allocated"] and not row["allocated_to_target_process"]
        mask_known = group_masks is not None and group in known_groups
        mask_allows = bool(group_masks[group] & (1 << lp)) if mask_known else None
        default_allows = (row["cpu_set_id"] in process_default_cpu_set_ids
                          if defaults_known and process_default_cpu_set_ids else
                          True if defaults_known else None)
        row["process_affinity_mask_allows"] = mask_allows
        row["process_default_cpu_set_allows"] = default_allows
        if reserved_elsewhere or mask_allows is False:
            allowed = False
        elif default_allows is False:
            allowed = False
        elif mask_allows is True:
            allowed = True if default_allows is True else None
        else:
            allowed = None
        row["allowed_cpuset"] = allowed
        row["allowed_status"] = ("excluded_cpu_set_allocated_elsewhere" if reserved_elsewhere else
                                 "excluded_process_affinity_mask" if mask_allows is False else
                                 "excluded_process_default_cpu_sets" if default_allows is False else
                                 "allowed_by_known_masks" if mask_allows is True and default_allows is True else
                                 "process_default_cpu_sets_unknown" if default_allows is None else
                                 "process_mask_unknown_for_group")


def parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    text = text.strip()
    if not text:
        return result
    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo or hi - lo > 1_000_000:
                raise ValueError(f"invalid CPU range: {part}")
            result.update(range(lo, hi + 1))
        else:
            v = int(part)
            if v < 0:
                raise ValueError(f"invalid CPU number: {part}")
            result.add(v)
    return result


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def linux_inventory(sysfs_root: Path = Path("/sys/devices/system/cpu"),
                    allowed_cpus: set[int] | None = None) -> dict[str, Any]:
    online_text = (sysfs_root / "online").read_text(encoding="ascii")
    online = parse_cpu_list(online_text)
    allowed_error = None
    if allowed_cpus is None:
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except (AttributeError, OSError) as e:
            allowed_error = f"{type(e).__name__}: {e}"
            allowed_cpus = set()
    rows = []
    for cpu in sorted(online):
        root = sysfs_root / f"cpu{cpu}"
        pkg = _read_int(root / "topology" / "physical_package_id")
        core = _read_int(root / "topology" / "core_id")
        capacity = _read_int(root / "cpu_capacity")
        nodes = sorted(int(p.name[4:]) for p in root.glob("node[0-9]*") if p.name[4:].isdigit())
        numa = nodes[0] if len(nodes) == 1 else None
        allowed = (cpu in allowed_cpus) if allowed_error is None else None
        rows.append({"logical_processor": cpu, "processor_group": None,
                     "physical_core": {"package_id": pkg, "core_id": core} if pkg is not None and core is not None else None,
                     "numa_node": numa, "efficiency_class": None, "efficiency_class_source": "unknown",
                     "core_type": "unknown", "classify_confidence": "unknown",
                     "linux_cpu_capacity": capacity, "allowed_cpuset": allowed,
                     "allowed_status": "sched_getaffinity" if allowed_error is None else "unknown",
                     "online": True})
    candidates = candidate_pool_workers(rows, "linux", sysfs_root)
    return {"platform": "linux", "allowed_cpuset_source": "os.sched_getaffinity(0)",
            "allowed_cpuset_scope": "calling thread mask (new worker threads normally inherit this mask)",
            "allowed_cpuset_error": allowed_error, "logical_processors": rows,
            "pool_affinity_candidates": candidates}


def _win_query() -> tuple[bytes, dict[int, int] | None, set[int] | None, set[int] | None, bool, dict[str, Any]]:
    if os.name != "nt":
        raise RuntimeError("Windows CPU Set collection requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetCurrentThread.restype = ctypes.c_void_p
    get_info = kernel.GetSystemCpuSetInformation
    get_info.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_ulong]
    get_info.restype = ctypes.c_int
    needed = ctypes.c_ulong(0)
    get_info(None, 0, ctypes.byref(needed), kernel.GetCurrentProcess(), 0)
    size = needed.value
    if size <= 0:
        err = ctypes.get_last_error()
        if err:
            raise OSError(err, "GetSystemCpuSetInformation size query failed")
        return b"", None, None, None, False, {"api": "GetSystemCpuSetInformation", "bytes": 0}
    buf = (ctypes.c_ubyte * size)()
    returned = ctypes.c_ulong(0)
    ok = get_info(ctypes.cast(buf, ctypes.c_void_p), size, ctypes.byref(returned), kernel.GetCurrentProcess(), 0)
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, "GetSystemCpuSetInformation data query failed")
    data = bytes(buf[:returned.value])

    default_ids: set[int] | None = None
    defaults_known = False
    default_api = getattr(kernel, "GetProcessDefaultCpuSets", None)
    if default_api:
        default_api.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
                                ctypes.POINTER(ctypes.c_ulong)]
        default_api.restype = ctypes.c_int
        required = ctypes.c_ulong(0)
        ok = default_api(kernel.GetCurrentProcess(), None, 0, ctypes.byref(required))
        if ok and required.value == 0:
            default_ids, defaults_known = set(), True
        elif ctypes.get_last_error() == 122 and required.value > 0:
            ids = (ctypes.c_ulong * required.value)()
            filled = ctypes.c_ulong(0)
            if default_api(kernel.GetCurrentProcess(), ids, required.value, ctypes.byref(filled)):
                default_ids, defaults_known = set(int(ids[i]) for i in range(filled.value)), True

    group_masks: dict[int, int] | None = None
    known_groups: set[int] | None = None
    mask_source = "unknown"
    try:
        class GROUP_AFFINITY(ctypes.Structure):
            _fields_ = [("Mask", ctypes.c_size_t), ("Group", ctypes.c_ushort), ("Reserved", ctypes.c_ushort * 3)]
        get_thread_group = kernel.GetThreadGroupAffinity
        get_thread_group.argtypes = [ctypes.c_void_p, ctypes.POINTER(GROUP_AFFINITY)]
        get_thread_group.restype = ctypes.c_int
        ga = GROUP_AFFINITY()
        if get_thread_group(kernel.GetCurrentThread(), ctypes.byref(ga)):
            p_mask, s_mask = ctypes.c_size_t(0), ctypes.c_size_t(0)
            get_aff = kernel.GetProcessAffinityMask
            get_aff.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            get_aff.restype = ctypes.c_int
            if get_aff(kernel.GetCurrentProcess(), ctypes.byref(p_mask), ctypes.byref(s_mask)):
                group_masks = {int(ga.Group): int(p_mask.value)}
                known_groups = {int(ga.Group)}
                mask_source = "GetProcessAffinityMask_primary_group_only"
    except (AttributeError, OSError):
        pass
    return data, group_masks, known_groups, default_ids, defaults_known, {
        "api": "GetSystemCpuSetInformation", "bytes": returned.value,
        "process_mask_source": mask_source,
        "process_default_cpu_set_source": "GetProcessDefaultCpuSets" if defaults_known else "unknown",
        "process_default_cpu_set_count": len(default_ids) if default_ids is not None else None,
        "api_source": WIN_API_SOURCE, "structure_source": WIN_SDK_SOURCE,
        "process_default_sets_source": "https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocessdefaultcpusets"}


def windows_inventory() -> dict[str, Any]:
    data, masks, known_groups, default_ids, defaults_known, api_info = _win_query()
    rows = parse_windows_cpu_set_buffer(data)
    apply_windows_allowed_masks(rows, masks, known_groups, default_ids, defaults_known)
    for row in rows:
        row.update({"logical_processor": row["logical_identity"], "processor_group": row["group"],
                    "physical_core": row["core_identity"], "numa_node": row["numa_node_index"],
                    "core_type": "unknown", "classify_confidence": "unknown",
                    "efficiency_class_source": "SYSTEM_CPU_SET_INFORMATION.CpuSet.EfficiencyClass"})
    return {"platform": "windows", "api": api_info, "logical_processors": rows,
            "pool_affinity_candidates": candidate_pool_workers(rows, "windows")}


def candidate_pool_workers(rows: list[dict[str, Any]], platform_name: str,
                           sysfs_root: Path = Path("/sys/devices/system/cpu")) -> dict[str, Any]:
    eligible = [r for r in rows if r.get("allowed_cpuset") is True]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in eligible:
        physical = r.get("physical_core") or r.get("core_identity")
        if not physical:
            # Unknown physical identity: do not pretend that logical siblings are unique cores.
            continue
        if platform_name == "windows":
            key = (physical.get("group"), physical.get("core_index"))
        else:
            key = (physical.get("package_id"), physical.get("core_id"))
        grouped[key].append(r)
    cores = []
    for key, siblings in grouped.items():
        def lp_number(r):
            ident = r.get("logical_processor")
            if ident is None:
                ident = r.get("logical_identity")
            if isinstance(ident, dict):
                return int(ident["logical_processor_index"])
            return int(ident)
        siblings.sort(key=lp_number)
        first = siblings[0]
        cores.append({"key": key, "primary": first, "siblings": siblings,
                      "class": first.get("efficiency_class"), "capacity": first.get("linux_cpu_capacity")})
    class_vals = {c["class"] for c in cores if isinstance(c["class"], int)}
    hybrid = len(class_vals) > 1
    basis = "windows EfficiencyClass ordering (higher class first; P/E names unverified)" if platform_name == "windows" else "unknown"
    p_cores: list[dict[str, Any]] = []
    e_cores: list[dict[str, Any]] = []
    p_siblings: list[dict[str, Any]] = []
    if platform_name == "windows":
        max_class = max(class_vals) if class_vals else None
        if hybrid:
            # std::stable_sort in pool.cpp prioritizes class while preserving OS enumeration order within a class.
            cores.sort(key=lambda c: (-c["class"] if isinstance(c["class"], int) else 0))
        p_cores = [c for c in cores if max_class is not None and c["class"] == max_class]
        e_cores = [c for c in cores if max_class is not None and c["class"] != max_class]
        p_siblings = [lp for c in p_cores for lp in c["siblings"][1:]]
        if not hybrid:
            basis = "single/unknown EfficiencyClass; physical-core primaries only"
    else:
        pmu_core, pmu_atom = set(), set()
        root = sysfs_root.parent
        try:
            pmu_core = parse_cpu_list((root / "cpu_core" / "cpus").read_text(encoding="ascii"))
            pmu_atom = parse_cpu_list((root / "cpu_atom" / "cpus").read_text(encoding="ascii"))
        except (OSError, ValueError):
            pmu_core, pmu_atom = set(), set()
        have_pmu = bool(pmu_core and pmu_atom)
        capacities = [c["capacity"] for c in cores if isinstance(c["capacity"], int) and c["capacity"] > 0]
        max_capacity = max(capacities) if capacities else None
        for c in cores:
            cpu = int(c["primary"]["logical_processor"])
            if have_pmu:
                is_e = cpu in pmu_atom
            elif max_capacity and isinstance(c["capacity"], int):
                is_e = c["capacity"] * 10 < max_capacity * 9
            else:
                is_e = False
            (e_cores if is_e else p_cores).append(c)
        hybrid = bool(p_cores and e_cores)
        p_siblings = [lp for c in p_cores for lp in c["siblings"][1:]]
        basis = "sysfs cpu_core/cpu_atom sets" if have_pmu else ("sysfs cpu_capacity < 90% max heuristic" if max_capacity else "unknown; no class-based filtering")
    primary_order = [c["primary"] for c in (p_cores + e_cores if hybrid else cores)]
    all_host = primary_order[0] if primary_order else None
    all_workers = primary_order[1:] if primary_order else []
    auto_primary = [c["primary"] for c in p_cores]
    auto_host = auto_primary[0] if auto_primary else None
    auto_workers = auto_primary[1:] + p_siblings + [c["primary"] for c in e_cores]
    p_primary = [c["primary"] for c in p_cores]
    p_host = p_primary[0] if p_primary else None
    p_workers = p_primary[1:] + p_siblings
    if platform_name == "linux" and basis == "unknown; no class-based filtering":
        auto_workers = list(all_workers)
        if not hybrid:
            p_workers = list(all_workers)
            p_host = all_host
        else:
            p_workers = []
            p_host = None
        auto_host = all_host
    def ident(r):
        if r is None:
            return None
        value = r.get("logical_processor")
        return value if value is not None else r.get("logical_identity")
    return {"status": "candidate_only_no_performance_measurement", "classification_basis": basis,
            "classification_confidence": "unknown", "core_type_claims": "unknown; no P/E/LP labels inferred",
            "candidate_coverage": ("partial_unknown_allowed_set" if any(r.get("allowed_cpuset") is None for r in rows)
                                   else "all_known_allowed_processors"),
            "implementation_reference": "src/kernels/cpu/pool.cpp topology order; candidates are not measured best settings",
            "pool-affinity-all": {"host_core_candidate": ident(all_host), "worker_candidates": [ident(x) for x in all_workers]},
            "pool-affinity-auto": {"host_core_candidate": ident(auto_host), "worker_candidates": [ident(x) for x in auto_workers]},
            "pool-affinity-p-cores": {"host_core_candidate": ident(p_host), "worker_candidates": [ident(x) for x in p_workers]}}


def collect_inventory() -> dict[str, Any]:
    if os.name == "nt":
        result = windows_inventory()
    elif sys.platform.startswith("linux"):
        result = linux_inventory()
    else:
        raise RuntimeError(f"unsupported platform: {sys.platform}")
    result["collected_utc"] = datetime.now(timezone.utc).isoformat()
    result["python"] = platform.python_version()
    result["mutation_policy"] = "read-only; no affinity, priority, power, or process state changed"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", help="exclusive destination JSON; will not overwrite an existing file")
    ap.add_argument("--validate-only", action="store_true", help="validate destination only; no hardware query or write")
    args = ap.parse_args(argv)
    if not args.output:
        ap.error("--output PATH is required for collection or validation")
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        print(json.dumps({"valid": False, "error": "output_exists", "path": str(target)}))
        return 2
    if args.validate_only:
        print(json.dumps({"valid": True, "observation_performed": False, "write_performed": False,
                          "path": str(target)}))
        return 0
    try:
        payload = collect_inventory()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except FileExistsError:
        print(json.dumps({"valid": False, "error": "output_exists", "path": str(target)}))
        return 2
    except Exception as e:
        print(json.dumps({"valid": False, "error_type": type(e).__name__, "error": str(e)}))
        return 1
    print(json.dumps({"written": str(target), "platform": payload["platform"],
                      "logical_processor_count": len(payload.get("logical_processors", [])),
                      "candidate_status": payload["pool_affinity_candidates"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
