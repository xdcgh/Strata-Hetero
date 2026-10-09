#!/usr/bin/env python3
"""Download one fixed HF vision-projector asset with bounded I/O and exclusive evidence."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_ID = "20261009-15-01"
REPO_ID = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
FILENAME = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
EXPECTED_BYTES = 907_543_008
EXPECTED_SHA256 = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
CANONICAL_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{FILENAME}"
PROXY_URL = "http://127.0.0.1:10808"
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 64 * 1024 * 1024
RESERVE_BYTES = 12 * 1024**3
RAM_MIN_BYTES = 12 * 1024**3
COMMIT_MIN_BYTES = 4 * 1024**3

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
SOURCE_METADATA = ROOT / "bench" / "hetero" / "20261009-13-vision-inventory" / "source-metadata.json"
MODELS_PARENT = Path(r"E:\Strata-Hetero-data\models")
TARGET_DIR = MODELS_PARENT / "vision-ed59f92"
TARGET_FILE = TARGET_DIR / FILENAME
PART_FILE = TARGET_DIR / f"{FILENAME}.part.{RUN_ID}"
PROGRESS_FILE = RUN_DIR / "progress.jsonl"
PREFLIGHT_FILE = RUN_DIR / "preflight.json"
RECEIPT_FILE = RUN_DIR / "download-01.json"


class DownloadFailure(RuntimeError):
    pass


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_progress(stream, *, phase: str, status: str, **fields: Any) -> None:
    row = {"at_utc": utc_now(), "phase": phase, "status": status, **fields}
    stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def memory_gates() -> dict[str, Any]:
    if os.name != "nt":
        raise DownloadFailure("this guarded download procedure is Windows-only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(mem)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise DownloadFailure("GlobalMemoryStatusEx did not return a physical-memory measurement")
    perf = PERFORMANCE_INFORMATION()
    perf.cb = ctypes.sizeof(perf)
    psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), ctypes.c_ulong]
    psapi.GetPerformanceInfo.restype = ctypes.c_int
    if not psapi.GetPerformanceInfo(ctypes.byref(perf), ctypes.sizeof(perf)):
        raise DownloadFailure("PSAPI GetPerformanceInfo did not return a system-commit measurement")
    commit_available = (int(perf.CommitLimit) - int(perf.CommitTotal)) * int(perf.PageSize)
    return {
        "observed_at_utc": utc_now(),
        "physical_available_bytes": int(mem.ullAvailPhys),
        "physical_available_gib": round(int(mem.ullAvailPhys) / 1024**3, 3),
        "physical_gate": "pass" if int(mem.ullAvailPhys) >= RAM_MIN_BYTES else "fail",
        "commit_available_bytes": commit_available,
        "commit_available_gib": round(commit_available / 1024**3, 3),
        "commit_gate": "pass" if commit_available >= COMMIT_MIN_BYTES else "fail",
        "commit_source": "PSAPI GetPerformanceInfo",
    }


def file_identity(path: Path) -> dict[str, Any]:
    path_stat = path.stat()
    fd = os.open(path, os.O_RDONLY)
    try:
        fd_stat = os.fstat(fd)
    finally:
        os.close(fd)
    # Windows can report a zero ctime through fstat. Compare only stable descriptor fields.
    for name in ("st_size", "st_mtime_ns", "st_dev", "st_ino"):
        if getattr(path_stat, name) != getattr(fd_stat, name):
            raise DownloadFailure(f"path stat/fstat differ on {name}")
    return {
        "size_bytes": int(path_stat.st_size), "mtime_ns": int(path_stat.st_mtime_ns),
        "ctime_ns_path": int(path_stat.st_ctime_ns),
        "file_id": {"volume": int(fd_stat.st_dev), "index": int(fd_stat.st_ino)},
        "fstat": {"size_bytes": int(fd_stat.st_size), "mtime_ns": int(fd_stat.st_mtime_ns),
                  "volume": int(fd_stat.st_dev), "index": int(fd_stat.st_ino)},
    }


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


class BoundedHFRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.last_host: str | None = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        host = urllib.parse.urlsplit(newurl).hostname
        self.last_host = host
        allowed = bool(host) and (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"))
        if self.count > 3 or not allowed:
            # Never include newurl in the exception text: redirects may carry temporary signed query values.
            raise urllib.error.HTTPError(req.full_url, code, "redirect rejected by pinned-source policy", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_source_metadata() -> tuple[dict[str, Any], str]:
    raw = SOURCE_METADATA.read_bytes()
    metadata = json.loads(raw)
    api = metadata.get("api_metadata", {})
    if (metadata.get("source_repository") != REPO_ID or metadata.get("fixed_revision") != REVISION or
            metadata.get("filename") != FILENAME or metadata.get("canonical_resolve_url") != CANONICAL_URL or
            api.get("size_bytes") != EXPECTED_BYTES or api.get("lfs_size_bytes") != EXPECTED_BYTES or
            str(api.get("lfs_oid_sha256", "")).lower() != EXPECTED_SHA256):
        raise DownloadFailure("source metadata does not match the authorized fixed repository/revision/file/size/LFS SHA")
    return metadata, hashlib.sha256(raw).hexdigest()


def main() -> int:
    if os.name != "nt":
        print(json.dumps({"status": "failed", "error_type": "PlatformError", "detail": "Windows-only no-replace rename is required"}))
        return 1
    if not RUN_DIR.is_dir() or not MODELS_PARENT.is_dir():
        print(json.dumps({"status": "failed", "error_type": "MissingParent", "detail": "required output parent is absent"}))
        return 1
    if PROGRESS_FILE.exists() or RECEIPT_FILE.exists():
        print(json.dumps({"status": "failed", "error_type": "ExistsError", "detail": "exclusive progress or receipt already exists"}))
        return 1

    result: dict[str, Any] = {
        "schema_version": 1, "run_id": RUN_ID, "status": "failed",
        "source_repository": REPO_ID, "fixed_revision": REVISION, "filename": FILENAME,
        "canonical_resolve_url": CANONICAL_URL,
        "source_query_or_signed_redirect_url_saved": False,
        "expected_size_bytes": EXPECTED_BYTES, "expected_lfs_sha256": EXPECTED_SHA256,
        "destination": str(TARGET_FILE), "temporary_path": str(PART_FILE),
        "buffer_bytes": CHUNK_BYTES, "proxy_route": PROXY_URL,
        "stream_bytes": 0, "stream_sha256": None, "readback_bytes": None,
        "readback_sha256": None, "partial_preserved": False, "final_file_present": False,
        "error": None,
    }
    stage = "preflight"
    part_created = False
    final_identity = None
    progress = None
    try:
        progress = PROGRESS_FILE.open("x", encoding="utf-8", newline="\n")
        metadata, metadata_sha = verify_source_metadata()
        preflight = json.loads(PREFLIGHT_FILE.read_text(encoding="utf-8"))
        preflight_sha = hashlib.sha256(PREFLIGHT_FILE.read_bytes()).hexdigest()
        f_readers = preflight.get("host_build_processes_reading_F_model")
        if (preflight.get("status") != "pass" or
                preflight.get("source_metadata_sha256") != metadata_sha or
                preflight.get("target_path") != str(TARGET_FILE) or
                not isinstance(f_readers, list) or len(f_readers) != 0):
            raise DownloadFailure("prepared host/source/target preflight is missing, failed or does not match")
        result["source_metadata_path"] = str(SOURCE_METADATA)
        result["source_metadata_sha256"] = metadata_sha
        result["preflight_receipt"] = str(PREFLIGHT_FILE)
        result["preflight_receipt_sha256"] = preflight_sha
        result["host_build_process_snapshot"] = preflight.get("host_build_processes")
        if TARGET_FILE.exists() or TARGET_FILE.is_symlink() or TARGET_DIR.exists() or TARGET_DIR.is_symlink():
            raise DownloadFailure("exclusive vision target already exists; refusing overwrite or reuse")
        if not MODELS_PARENT.is_dir():
            raise DownloadFailure("selected E: models parent directory disappeared")
        proxy_sock = socket.create_connection(("127.0.0.1", 10808), timeout=3)
        proxy_sock.close()
        result["proxy_listener_preflight"] = "127.0.0.1:10808 reachable"
        free = shutil.disk_usage(MODELS_PARENT).free
        result["destination_space_preflight"] = {
            "free_bytes": int(free), "required_bytes": EXPECTED_BYTES + RESERVE_BYTES,
            "reserve_bytes": RESERVE_BYTES, "pass": free >= EXPECTED_BYTES + RESERVE_BYTES,
        }
        if free < EXPECTED_BYTES + RESERVE_BYTES:
            raise DownloadFailure("E: free-space gate failed before download")
        gates = memory_gates()
        result["system_gates_before"] = gates
        if gates["physical_gate"] != "pass" or gates["commit_gate"] != "pass":
            raise DownloadFailure("current physical RAM or system-commit gate failed before download")
        append_progress(progress, phase=stage, status="pass", source_metadata_sha256=metadata_sha,
                        physical_ram_gib=gates["physical_available_gib"],
                        commit_available_gib=gates["commit_available_gib"],
                        destination_free_bytes=int(free), expected_size_bytes=EXPECTED_BYTES)

        TARGET_DIR.mkdir(parents=False, exist_ok=False)
        with PART_FILE.open("xb", buffering=0) as destination:
            part_created = True
            digest = hashlib.sha256()
            received = 0
            next_progress = PROGRESS_BYTES
            stage = "http_stream"
            append_progress(progress, phase=stage, status="started", canonical_url=CANONICAL_URL)
            redirects = BoundedHFRedirect()
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": PROXY_URL, "https": PROXY_URL}), redirects)
            request = urllib.request.Request(CANONICAL_URL,
                headers={"User-Agent": "Strata-Hetero-verified-asset/1", "Accept-Encoding": "identity"},
                method="GET")
            with opener.open(request, timeout=30) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_length_header = response.headers.get("Content-Length")
                content_length = int(content_length_header) if content_length_header is not None else None
                final_host = urllib.parse.urlsplit(response.geturl()).hostname
                result["http_status"] = status
                result["http_content_length_bytes"] = content_length
                result["redirect_count"] = redirects.count
                result["response_final_host_only"] = final_host
                if status != 200 or content_length != EXPECTED_BYTES:
                    raise DownloadFailure("HTTP response status or Content-Length differs from pinned metadata")
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    view = memoryview(block)
                    offset = 0
                    while offset < len(view):
                        written = destination.write(view[offset:])
                        if written is None or written <= 0 or written > len(view) - offset:
                            raise DownloadFailure("destination write failed or reported an invalid short write")
                        digest.update(view[offset:offset + written])
                        offset += written
                        received += written
                        result["stream_bytes"] = received
                        result["stream_sha256"] = digest.copy().hexdigest()
                    if received >= next_progress:
                        current_gates = memory_gates()
                        free_now = shutil.disk_usage(MODELS_PARENT).free
                        if current_gates["physical_gate"] != "pass" or current_gates["commit_gate"] != "pass":
                            raise DownloadFailure("physical RAM or commit gate failed during download")
                        if free_now < (EXPECTED_BYTES - received) + RESERVE_BYTES:
                            raise DownloadFailure("E: free-space reserve gate failed during download")
                        append_progress(progress, phase=stage, status="progress", received_bytes=received,
                                        expected_bytes=EXPECTED_BYTES, stream_sha256=digest.copy().hexdigest(),
                                        physical_ram_gib=current_gates["physical_available_gib"],
                                        commit_available_gib=current_gates["commit_available_gib"],
                                        destination_free_bytes=int(free_now))
                        next_progress += PROGRESS_BYTES
            destination.flush()
            os.fsync(destination.fileno())
        stream_hash = digest.hexdigest()
        result["stream_bytes"] = received
        result["stream_sha256"] = stream_hash
        result["partial_preserved"] = PART_FILE.is_file()
        if received != EXPECTED_BYTES or stream_hash != EXPECTED_SHA256:
            raise DownloadFailure("downloaded stream size or SHA-256 differs from pinned LFS metadata; keeping .part")

        stage = "readback"
        part_before = file_identity(PART_FILE)
        readback_hash, readback_bytes = hash_file(PART_FILE)
        part_after = file_identity(PART_FILE)
        result["part_identity_before_readback"] = part_before
        result["part_identity_after_readback"] = part_after
        result["readback_bytes"] = readback_bytes
        result["readback_sha256"] = readback_hash
        result["readback_path"] = str(PART_FILE)
        if part_before != part_after:
            raise DownloadFailure("temporary file identity changed during fresh readback; keeping .part")
        if readback_bytes != EXPECTED_BYTES or readback_hash != EXPECTED_SHA256:
            raise DownloadFailure("fresh .part readback size or SHA-256 differs; final path not published")

        gates_after = memory_gates()
        free_after = shutil.disk_usage(MODELS_PARENT).free
        result["system_gates_before_publish"] = gates_after
        result["destination_free_bytes_before_publish"] = int(free_after)
        if gates_after["physical_gate"] != "pass" or gates_after["commit_gate"] != "pass":
            raise DownloadFailure("physical RAM or commit gate failed before publishing final file")
        if free_after < RESERVE_BYTES:
            raise DownloadFailure("E: minimum 12 GiB reserve failed before publishing final file")
        if TARGET_FILE.exists() or TARGET_FILE.is_symlink():
            raise DownloadFailure("final target appeared before publish; refusing overwrite and keeping .part")

        stage = "publish"
        os.rename(PART_FILE, TARGET_FILE)  # Windows same-volume rename refuses an existing destination.
        result["partial_preserved"] = False
        final_identity = file_identity(TARGET_FILE)
        result["final_identity"] = final_identity
        same_verified_file = (
            final_identity["file_id"] == part_after["file_id"] and
            final_identity["size_bytes"] == part_after["size_bytes"] and
            final_identity["mtime_ns"] == part_after["mtime_ns"]
        )
        result["final_is_same_verified_file"] = same_verified_file
        result["final_identity_scope"] = "file ID, size and mtime match the independently SHA-256-verified .part; final path bytes were not rehashed after atomic rename"
        if not same_verified_file:
            raise DownloadFailure("final path is not the same file as the verified temporary file")
        result["status"] = "pass"
        append_progress(progress, phase=stage, status="pass", final_path=str(TARGET_FILE),
                        sha256=readback_hash, bytes=readback_bytes, final_identity=final_identity)
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__,
                           "http_status": int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None,
                           "stage": stage,
                           "detail": "URL/query and exception text intentionally omitted"}
        result["partial_preserved"] = PART_FILE.is_file()
        result["final_file_present"] = TARGET_FILE.is_file()
        if part_created and PART_FILE.is_file():
            try:
                result["partial_identity"] = file_identity(PART_FILE)
            except Exception as identity_exc:
                result["partial_identity_error_type"] = type(identity_exc).__name__
        if progress is not None:
            append_progress(progress, phase=stage, status="failed", error_type=type(exc).__name__,
                            partial_preserved=result["partial_preserved"], stream_bytes=result["stream_bytes"],
                            stream_sha256=result["stream_sha256"])
    finally:
        if progress is not None:
            progress.flush()
            os.fsync(progress.fileno())
            progress.close()
        # Receipt is exclusive for both success and failure; no partial or failed payload is deleted.
        with RECEIPT_FILE.open("x", encoding="utf-8", newline="\n") as receipt:
            json.dump(result, receipt, indent=2, ensure_ascii=True)
            receipt.write("\n")

    print(json.dumps({"status": result["status"], "receipt": str(RECEIPT_FILE),
                      "destination": str(TARGET_FILE) if result["status"] == "pass" else None,
                      "expected_bytes": EXPECTED_BYTES, "stream_bytes": result.get("stream_bytes"),
                      "stream_sha256": result.get("stream_sha256"), "readback_sha256": result.get("readback_sha256"),
                      "partial_preserved": result.get("partial_preserved"), "error": result.get("error")}, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
