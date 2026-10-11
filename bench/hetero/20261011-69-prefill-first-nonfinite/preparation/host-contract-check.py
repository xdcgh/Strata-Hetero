from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

WORKTREE = Path(r"E:\Strata-Hetero-data\source\hetero-prefill-probe-69")
BASE_REV = "b50ba7702531d0184bc89be13bb9d69b0194f71c"
RECEIPT = Path(__file__).with_name("host-contract-receipt-02.json")
SOURCE_FILES = (
    "CMakeLists.txt",
    "src/prefill/prefill.cpp",
    "include/strata/prefill/finite_probe.hpp",
    "src/prefill/finite_probe.cu",
    "tests/cuda/prefill_finite_probe_test.cpp",
)
TARGET = re.compile(r"^\s*#\s*if\s+defined\s*\(\s*STRATA_PREFILL_FINITE_PROBE\s*\)\s*$")
PP = re.compile(r"^\s*#\s*(if|ifdef|ifndef|endif)\b")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalized(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def without_probe_blocks(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        if not TARGET.match(lines[i].rstrip("\r\n")):
            kept.append(lines[i])
            i += 1
            continue
        removed += 1
        depth = 1
        i += 1
        while i < len(lines) and depth:
            directive = PP.match(lines[i])
            if directive:
                kind = directive.group(1)
                if kind in ("if", "ifdef", "ifndef"):
                    depth += 1
                elif kind == "endif":
                    depth -= 1
            i += 1
        if depth:
            raise ValueError("unterminated STRATA_PREFILL_FINITE_PROBE block")
    return "".join(kept), removed


def check(condition: bool, detail: str) -> dict[str, object]:
    return {"status": "pass" if condition else "fail", "detail": detail}


def main() -> int:
    if RECEIPT.exists():
        raise SystemExit(f"refusing to overwrite exclusive receipt: {RECEIPT}")
    source_paths = {name: WORKTREE / name for name in SOURCE_FILES}
    raw = {name: path.read_bytes() for name, path in source_paths.items()}
    prefill = raw["src/prefill/prefill.cpp"].decode("utf-8")
    cmake = raw["CMakeLists.txt"].decode("utf-8")
    probe_cu = raw["src/prefill/finite_probe.cu"].decode("utf-8")
    baseline = subprocess.run(
        ["git", "show", f"{BASE_REV}:src/prefill/prefill.cpp"], cwd=WORKTREE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, shell=False,
    )
    if baseline.returncode:
        raise RuntimeError("git show of pinned baseline failed: " + baseline.stderr.decode("utf-8", "replace"))
    stripped, block_count = without_probe_blocks(prefill)
    stripped_bytes = normalized(stripped.encode("utf-8"))
    baseline_bytes = normalized(baseline.stdout)
    diff_offset = next((i for i, (a, b) in enumerate(zip(stripped_bytes, baseline_bytes)) if a != b),
                       min(len(stripped_bytes), len(baseline_bytes)))
    stripped_equal = stripped_bytes == baseline_bytes

    reports = [match.start() for match in re.finditer(r"pf_probe\.report_after_sync\s*\(", prefill)]
    report_order = []
    for position in reports:
        sync = prefill.rfind("cudaStreamSynchronize(m.cs)", 0, position)
        read = prefill.rfind("pf_probe.enqueue_read(err)", 0, position)
        report_order.append({
            "report_line": prefill.count("\n", 0, position) + 1,
            "enqueue_read_line": prefill.count("\n", 0, read) + 1 if read >= 0 else None,
            "existing_sync_line": prefill.count("\n", 0, sync) + 1 if sync >= 0 else None,
            "ordered_enqueue_sync_host_report": 0 <= read < sync < position,
        })
    report_order_ok = len(reports) == 3 and all(item["ordered_enqueue_sync_host_report"] for item in report_order)
    bound_compute_stream = bool(re.search(r"pf_probe\.init\([^;]*\bm\.cs\s*,", prefill, re.S))
    kernel_stream = "scan_kernel<<<blocks, 256, 0, (cudaStream_t) stream_>>>" in probe_cu
    copy_stream = "cudaMemcpyAsync(host_record_, device_record_, sizeof(Record), cudaMemcpyDeviceToHost,\n                                    (cudaStream_t) stream_)" in probe_cu
    host_report = "const Record& r = *host_record_;" in probe_cu and "void Probe::report_after_sync" in probe_cu
    host_report_body = probe_cu.split("void Probe::report_after_sync", 1)[-1].split("\n}", 1)[0]
    host_report_after_sync_only = "cudaMemcpy" not in host_report_body and "cudaStreamSynchronize" not in host_report_body

    option_off = bool(re.search(
        r'^\s*option\(STRATA_PREFILL_FINITE_PROBE\s+"[^"]*"\s+OFF\)\s*$', cmake, re.M))
    cuda_only_gate = bool(re.search(
        r"if\(STRATA_PREFILL_FINITE_PROBE AND \(NOT STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_HIP_GFX906\)\)",
        cmake))
    guarded_target = bool(re.search(
        r"add_library\(strata_prefill STATIC[^\n]*\)\s*\n\s*if\(STRATA_PREFILL_FINITE_PROBE\)\s*\n\s*add_library\(strata_prefill_finite_probe STATIC src/prefill/finite_probe\.cu\)",
        cmake))
    compile_switch = "target_compile_definitions(strata_prefill PRIVATE STRATA_PREFILL_FINITE_PROBE=1)" in cmake
    no_device_wide_sync = "cudaDeviceSynchronize" not in probe_cu

    checks = {
        "probe_blocks_removed_match_pinned_b50ba770": check(
            stripped_equal, f"removed_blocks={block_count}; normalized_bytes_equal={stripped_equal}; first_difference_offset={None if stripped_equal else diff_offset}"),
        "probe_scan_read_order_on_existing_compute_stream": check(
            bound_compute_stream and kernel_stream and copy_stream and no_device_wide_sync,
            "Prefill binds Probe to m.cs; scan/read use stream_; no cudaDeviceSynchronize"),
        "host_record_reads_follow_existing_stream_sync": check(
            report_order_ok and host_report and host_report_after_sync_only,
            "all report_after_sync sites queue one read then follow cudaStreamSynchronize(m.cs); report only dereferences host record"),
        "cmake_cuda_only_default_off": check(
            option_off and cuda_only_gate and guarded_target and compile_switch,
            "option defaults OFF; rejects non-CUDA backends; new target and macro are inside opt-in guard"),
    }
    receipt = {
        "schema_version": 1,
        "status": "host_source_contract_pass" if all(x["status"] == "pass" for x in checks.values()) else "host_source_contract_fail",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "static source/file structure only; no compile, test binary, CUDA runtime, model, or GPU execution",
        "worktree": str(WORKTREE),
        "base_revision": BASE_REV,
        "source_file_sha256": {name: sha(value) for name, value in raw.items()},
        "base_prefill_sha256": sha(baseline.stdout),
        "probe_blocks_removed": block_count,
        "report_after_sync_sites": report_order,
        "checks": checks,
    }
    encoded = (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with RECEIPT.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "receipt": str(RECEIPT), "sha256": sha(encoded)}, sort_keys=True))
    return 0 if receipt["status"] == "host_source_contract_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
