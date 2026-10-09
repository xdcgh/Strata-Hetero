#!/usr/bin/env python3
"""Plan or explicitly copy a model shard using a prior passing integrity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTEGRITY = ROOT / "bench" / "hetero" / "20261008-00-admission" / "model-integrity-02.json"
DEFAULT_INVENTORY = ROOT / "bench" / "hetero" / "20261006-00-inventory" / "hardware_manifest.json"
BUFFER_BYTES = 8 * 1024 * 1024
RESERVE_BYTES = 12 * 1024 * 1024 * 1024


class CopyError(ValueError):
    pass


def _path_identity(path: Path) -> dict[str, Any]:
    """Capture path stat plus fd identity; Windows fstat ctime is intentionally ignored."""
    path_stat = path.stat()
    fd = os.open(path, os.O_RDONLY)
    try:
        fd_stat = os.fstat(fd)
    finally:
        os.close(fd)
    for name in ("st_size", "st_mtime_ns", "st_dev", "st_ino"):
        if getattr(path_stat, name) != getattr(fd_stat, name):
            raise CopyError(f"path stat and fstat disagree for {path} on {name}")
    return {
        "size_bytes": path_stat.st_size,
        "mtime_ns": path_stat.st_mtime_ns,
        "ctime_ns": path_stat.st_ctime_ns,
        "file_id": {"volume": fd_stat.st_dev, "index": fd_stat.st_ino},
        "fstat": {"size_bytes": fd_stat.st_size, "mtime_ns": fd_stat.st_mtime_ns,
                  "volume": fd_stat.st_dev, "index": fd_stat.st_ino},
    }


def _same_identity(current: dict[str, Any], prior: dict[str, Any]) -> bool:
    fid = prior.get("file_id", {})
    return (current.get("size_bytes") == prior.get("size_bytes") and
            current.get("mtime_ns") == prior.get("mtime_ns") and
            current.get("ctime_ns") == prior.get("ctime_ns") and
            current.get("file_id", {}).get("volume") == fid.get("volume") and
            current.get("file_id", {}).get("index") == fid.get("index"))


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CopyError(f"cannot read integrity receipt: {exc}") from exc
    if value.get("status") != "pass" or value.get("claims", {}).get("model_weight_integrity") is not True:
        raise CopyError("integrity receipt does not establish a passing full-model SHA check")
    return value


def _inventory_mapping(inventory_path: Path, drive: str) -> dict[str, Any]:
    try:
        inv = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "unknown", "reason": f"inventory unavailable: {type(exc).__name__}"}
    for row in inv.get("storage", []):
        root = row.get("root")
        if isinstance(root, str) and root.casefold() == drive.casefold():
            return {"status": "mapped_in_inventory", "inventory_collected_at_utc": inv.get("collected_at_utc"),
                    "root": root, "label": row.get("label"), "filesystem": row.get("filesystem"),
                    "physical_disks": row.get("physical_disks", []),
                    "scope": "as recorded in the supplied inventory; not a live physical-device query"}
    return {"status": "unknown", "reason": f"no storage root {drive!r} in supplied inventory"}


def _validate_destination(source: Path, destination: Path) -> tuple[Path, Path]:
    if not source.is_absolute() or not destination.is_absolute():
        raise CopyError("source and destination paths must be absolute")
    if source.is_symlink():
        raise CopyError("source must be a regular non-symlink file")
    source_abs = source.resolve(strict=True)
    dest_abs = destination.resolve(strict=False)
    if source_abs == dest_abs:
        raise CopyError("destination must differ from source")
    if not dest_abs.parent.is_dir():
        raise CopyError(f"destination parent must already exist: {dest_abs.parent}")
    if destination.exists() or destination.is_symlink():
        raise CopyError(f"destination already exists; refusing overwrite: {destination}")
    if source_abs.parent == dest_abs.parent and source_abs.name == dest_abs.name:
        raise CopyError("destination resolves to source")
    return source_abs, dest_abs


def validate_receipt_path(receipt: Path, source: Path, destination: Path, run_id: str) -> Path:
    if not receipt.is_absolute():
        raise CopyError("receipt path must be absolute")
    receipt_abs = receipt.resolve(strict=False)
    if not receipt_abs.parent.is_dir():
        raise CopyError(f"receipt parent must already exist: {receipt_abs.parent}")
    if receipt.exists() or receipt.is_symlink():
        raise CopyError(f"exclusive receipt path already exists: {receipt_abs}")
    dest_abs = destination.resolve(strict=False)
    source_abs = source.resolve(strict=True)
    part = dest_abs.with_name(dest_abs.name + ".part." + run_id)
    if receipt_abs in (source_abs, dest_abs, part):
        raise CopyError("receipt path collides with source, destination or owned temporary path")
    return receipt_abs


def build_plan(source: Path, destination: Path, integrity_path: Path = DEFAULT_INTEGRITY,
               inventory_path: Path = DEFAULT_INVENTORY, reserve_bytes: int = RESERVE_BYTES,
               free_bytes: int | None = None) -> dict[str, Any]:
    source_abs, dest_abs = _validate_destination(source, destination)
    if source_abs.is_symlink() or not source_abs.is_file():
        raise CopyError("source must be a regular non-symlink file")
    receipt = _load_receipt(integrity_path)
    source_record = next((x for x in receipt.get("model_shards", [])
                          if Path(x.get("path", "")).resolve(strict=False) == source_abs), None)
    if not isinstance(source_record, dict):
        raise CopyError("source path is not a shard in the supplied integrity receipt")
    if source_record.get("matches_expected") is not True:
        raise CopyError("integrity receipt does not record a passing expected SHA for this source")
    prior_after = source_record.get("after", {})
    current_identity = _path_identity(source_abs)
    if not _same_identity(current_identity, prior_after):
        raise CopyError("source size/mtime/path-ctime/file identity differs from prior full-hash receipt")
    expected_size = source_record.get("expected_size_bytes", source_record.get("size_bytes"))
    expected_sha = source_record.get("expected_sha256", source_record.get("sha256"))
    if (current_identity["size_bytes"] != expected_size or source_record.get("size_bytes") != expected_size or
            source_record.get("sha256") != expected_sha or not isinstance(expected_sha, str) or len(expected_sha) != 64):
        raise CopyError("source expected size/SHA is missing or does not match its prior identity")
    if any(c not in "0123456789abcdef" for c in expected_sha.lower()):
        raise CopyError("source expected SHA is malformed")
    drive, _ = os.path.splitdrive(str(dest_abs))
    if not drive:
        drive = str(dest_abs.anchor)
    elif not drive.endswith(("\\", "/")):
        drive += "\\"
    available = shutil.disk_usage(dest_abs.parent).free if free_bytes is None else free_bytes
    required = int(expected_size) + int(reserve_bytes)
    if available < required:
        raise CopyError(f"destination free-space gate failed: available={available}, required={required}")
    return {
        "source": str(source_abs), "source_identity": current_identity,
        "destination": str(dest_abs), "destination_parent": str(dest_abs.parent),
        "expected_size_bytes": int(expected_size), "expected_sha256": expected_sha.lower(),
        "prior_integrity_receipt": {"path": str(integrity_path.resolve()), "run_id": receipt.get("run_id"),
                                    "status": receipt.get("status"),
                                    "matches_expected": source_record.get("matches_expected")},
        "prior_full_hash_scope_note": "The source SHA is taken from the prior passing full-shard integrity receipt. Plan validation checks current source identity only and does not read/hash source payload bytes.",
        "destination_drive_mapping": _inventory_mapping(inventory_path, drive),
        "space_gate": {"free_bytes_at_plan": int(available), "source_bytes": int(expected_size),
                       "required_free_bytes": required, "minimum_free_after_copy_bytes": int(reserve_bytes),
                       "pass": True},
    }


def _write_all(stream: BinaryIO, data: bytes, on_chunk=None) -> int:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = stream.write(view[written:])
        if count is None or count <= 0:
            raise OSError("destination stream made no forward progress on write")
        if count > len(view) - written:
            raise OSError("destination stream reported writing more bytes than supplied")
        if on_chunk is not None:
            on_chunk(view[written:written + count])
        written += count
    return written


def _stream_copy_and_hash(source: Path, part: Path, buffer_bytes: int = BUFFER_BYTES,
                          progress: dict[str, Any] | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb", buffering=0) as src, part.open("xb", buffering=0) as dst:
        while True:
            block = src.read(buffer_bytes)
            if not block:
                break
            def recorded_chunk(chunk: memoryview) -> None:
                nonlocal copied
                digest.update(chunk)
                copied += len(chunk)

            try:
                _write_all(dst, block, recorded_chunk)
            except Exception:
                if progress is not None:
                    progress["source_stream_sha256"] = digest.copy().hexdigest()
                    progress["copied_bytes"] = copied
                raise
            if progress is not None:
                progress["source_stream_sha256"] = digest.copy().hexdigest()
                progress["copied_bytes"] = copied
        dst.flush()
        os.fsync(dst.fileno())
    return digest.hexdigest(), copied


def _stream_hash(path: Path, buffer_bytes: int = BUFFER_BYTES) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(buffer_bytes)
            if not block:
                break
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def _require_windows_publication(platform: str | None = None) -> str:
    actual = os.name
    if actual != "nt" or (platform is not None and platform != actual):
        raise CopyError("actual copy publication is supported only on Windows (no-clobber rename semantics)")
    return actual


def publish_verified_part(part: Path, destination: Path, platform: str | None = None) -> None:
    _require_windows_publication(platform)
    if part.parent.resolve() != destination.parent.resolve():
        raise CopyError("verified temporary file must be in the destination parent for atomic publication")
    if destination.exists() or destination.is_symlink():
        raise CopyError("destination appeared before final rename; preserving temporary copy")
    os.rename(part, destination)


def execute_copy(plan: dict[str, Any], run_id: str, reserve_bytes: int = RESERVE_BYTES,
                 buffer_bytes: int = BUFFER_BYTES, platform: str | None = None) -> dict[str, Any]:
    source = Path(plan["source"])
    destination = Path(plan["destination"])
    part = destination.with_name(destination.name + ".part." + run_id)
    result: dict[str, Any] = {"run_id": run_id, "status": "failed", "copy_method": "sequential_bounded_buffer",
                              "buffer_bytes": buffer_bytes, "temporary_path": str(part),
                              "source_stream_sha256": None, "destination_readback_sha256": None,
                              "destination_readback_path": str(part),
                              "part_identity_before_readback": None, "part_identity_after_readback": None,
                              "source_identity_after": None, "destination_identity": None,
                              "partial_preserved": False, "copy_error": None,
                              "publication_platform": os.name,
                              "requested_platform_override": platform}
    try:
        _require_windows_publication(platform)
    except CopyError:
        result["copy_error"] = {"type": "CopyError", "message": "actual copy is supported only on Windows; refusing before payload I/O"}
        return result
    if part.exists() or part.is_symlink():
        raise CopyError(f"exclusive copy refuses existing temporary path: {part}")
    if destination.exists() or destination.is_symlink():
        raise CopyError(f"destination appeared before copy; refusing overwrite: {destination}")
    source_before = _path_identity(source)
    result["source_identity_before"] = source_before
    if source_before != plan["source_identity"]:
        result["copy_error"] = {"type": "CopyError", "message": "source identity changed after plan validation"}
        return result
    try:
        copied_hash, copied_bytes = _stream_copy_and_hash(source, part, buffer_bytes, result)
        result["source_stream_sha256"] = copied_hash
        result["copied_bytes"] = copied_bytes
        result["partial_preserved"] = part.exists()
        if copied_bytes != plan["expected_size_bytes"] or copied_hash != plan["expected_sha256"]:
            raise CopyError("source stream size/SHA differs from prior expected size/SHA; preserving temporary copy")
        part_before = _path_identity(part)
        result["part_identity_before_readback"] = part_before
        readback_hash, readback_bytes = _stream_hash(part, buffer_bytes)
        result["destination_readback_sha256"] = readback_hash
        result["destination_readback_bytes"] = readback_bytes
        part_after = _path_identity(part)
        result["part_identity_after_readback"] = part_after
        if part_after != part_before:
            raise CopyError("owned temporary file identity changed during fresh readback; preserving it")
        if readback_bytes != plan["expected_size_bytes"] or readback_hash != plan["expected_sha256"]:
            raise CopyError("fresh temporary-file readback size/SHA differs from expected; final destination not published")
        source_after = _path_identity(source)
        result["source_identity_after"] = source_after
        if source_after != source_before:
            raise CopyError("source path/fd identity changed during copy; preserving temporary copy")
        free_after_write = shutil.disk_usage(destination.parent).free
        result["free_bytes_before_rename"] = free_after_write
        if free_after_write < reserve_bytes:
            raise CopyError(f"post-copy reserve gate failed: free={free_after_write}, reserve={reserve_bytes}")
        # Windows os.rename is same-volume and fails if the destination already exists.
        publish_verified_part(part, destination)
        result["partial_preserved"] = False
        result["destination_identity"] = _path_identity(destination)
        candidate_id = result["part_identity_after_readback"]
        final_id = result["destination_identity"]
        same_file = (candidate_id["file_id"] == final_id["file_id"] and
                     candidate_id["size_bytes"] == final_id["size_bytes"] and
                     candidate_id["mtime_ns"] == final_id["mtime_ns"])
        result["final_is_same_verified_file"] = same_file
        result["final_identity_check_scope"] = "file ID, size and mtime match the verified temporary file; final path was not rehashed"
        if not same_file:
            raise CopyError("published final path does not identify the verified temporary file")
        result["status"] = "pass"
        result["copy_error"] = None
    except Exception as exc:
        result["copy_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["partial_preserved"] = part.exists()
        if destination.exists():
            try:
                result["destination_identity"] = _path_identity(destination)
            except Exception as stat_exc:
                result["destination_identity_error"] = f"{type(stat_exc).__name__}: {stat_exc}"
        if source.exists():
            try:
                result["source_identity_after"] = _path_identity(source)
            except Exception as stat_exc:
                result["source_identity_after_error"] = f"{type(stat_exc).__name__}: {stat_exc}"
    return result


def write_receipt_exclusive(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise CopyError("receipt path must be absolute")
    if not path.parent.is_dir():
        raise CopyError(f"receipt parent must already exist: {path.parent}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="default metadata plan only; no payload reads or writes")
    mode.add_argument("--copy", action="store_true", help="explicitly copy and independently hash-readback the destination")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--integrity-receipt", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--reserve-bytes", type=int, default=RESERVE_BYTES)
    parser.add_argument("--buffer-bytes", type=int, default=BUFFER_BYTES)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--receipt", type=Path, help="exclusive JSON copy receipt; required with --copy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.copy and args.receipt is None:
        parser.error("--copy requires --receipt")
    if not args.copy and args.receipt is not None:
        parser.error("--receipt is only valid with --copy")
    if args.reserve_bytes < 0 or args.buffer_bytes <= 0:
        parser.error("reserve bytes must be non-negative and buffer bytes must be positive")
    if args.copy and (not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id)):
        parser.error("--run-id may contain only letters, digits, hyphens and underscores")
    if not 1 <= args.buffer_bytes <= 64 * 1024 * 1024:
        parser.error("buffer bytes must be between 1 and 67108864")
    try:
        receipt_path = None
        if args.copy:
            receipt_path = validate_receipt_path(args.receipt, args.source, args.destination, args.run_id)
        plan = build_plan(args.source, args.destination, args.integrity_receipt,
                          args.inventory, args.reserve_bytes)
        output: dict[str, Any] = {"schema_version": 1, "status": "validated_only", "payload_read": False,
                                  "files_written": False, "plan": plan}
        if args.copy:
            copy_result = execute_copy(plan, args.run_id, args.reserve_bytes, args.buffer_bytes)
            output.update(copy_result)
            output["status"] = copy_result["status"]
            output["payload_read"] = True
            output["files_written"] = True
            output["claims"] = {
                "source_stream_sha256_matches_prior_expected": copy_result.get("source_stream_sha256") == plan["expected_sha256"],
                "fresh_part_readback_matches_prior_expected": copy_result.get("destination_readback_sha256") == plan["expected_sha256"],
                "final_path_is_same_verified_file": copy_result.get("final_is_same_verified_file") is True,
                "source_identity_stable_during_copy": copy_result.get("source_identity_before") == copy_result.get("source_identity_after"),
                "claim_scope": "This run copied only the selected shard. Its prior expected SHA came from the passing integrity receipt; the source stream and owned temporary copy were freshly hashed in this run. After no-replace publication, final path identity (file ID, size, mtime) was checked against the verified temporary file; final path bytes were not rehashed.",
            }
            output["receipt_path"] = str(receipt_path)
            write_receipt_exclusive(receipt_path, output)
        print(json.dumps(output, indent=2, ensure_ascii=True))
        return 0 if output["status"] in ("validated_only", "pass") else 1
    except Exception as exc:
        print(f"hetero_copy_verified: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
