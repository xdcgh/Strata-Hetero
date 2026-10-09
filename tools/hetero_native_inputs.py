"""Prepare original quantized expert bytes and identified OpenVINO-compatible inputs.

The default validates metadata and source identities only. --prepare reads the
three selected payloads, checks their earlier scoped hashes, and creates a new
output directory. It never requantizes decoded weights or starts a device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

if __package__:
    from . import hetero_extract_expert as expert
else:
    import hetero_extract_expert as expert

ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MAX_PAYLOAD_BYTES = 256 << 20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect(receipt_path: Path) -> tuple[dict, dict]:
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("status") != "extracted" or receipt.get("source_file_stat_unchanged") is not True:
        raise ValueError("source receipt must describe a completed, stable expert extraction")
    identity_path = Path(receipt["identity_path"])
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    if (identity.get("layer"), identity.get("expert")) != (
            receipt["selection"]["layer"], receipt["selection"]["expert"]):
        raise ValueError("source identity and receipt select different experts")
    hidden, intermediate = (receipt["dimensions"][name] for name in ("hidden", "intermediate"))
    if not all(type(x) is int and 1 <= x <= 65536 for x in (hidden, intermediate)):
        raise ValueError("expert dimensions are outside bounded integer widths")
    if hidden % 256 or intermediate % 32:
        raise ValueError("Q4_K/Q5_1 dimensions must be block aligned")
    expected_sizes = {"gate": intermediate * (hidden // 256) * 144,
                      "up": intermediate * (hidden // 256) * 144,
                      "down": hidden * (intermediate // 32) * 24}
    if sum(expected_sizes.values()) > MAX_PAYLOAD_BYTES:
        raise ValueError("selected native blob exceeds bounded size")
    source = identity["source"]
    if receipt["raw_selected_payload_sha256"] != source["raw_selected_payload_sha256"]:
        raise ValueError("source scoped hashes disagree")
    if receipt["raw_selected_payload_bytes"] != expected_sizes or source["raw_selected_payload_bytes"] != expected_sizes:
        raise ValueError("source payload lengths disagree with the native format")
    snapshots = source["source_file_stat_before_after"]
    shards = []
    selection = receipt["selection"]
    if type(selection["expert"]) is not int or not 0 <= selection["expert"] < receipt["profile"]["experts"]:
        raise ValueError("selected expert index is out of range")
    for role in expert.TENSOR_ROLES:
        tensor = receipt["tensors"][role]
        if tensor["type_id"] != (7 if role == "down" else 12):
            raise ValueError("this native input preparer supports Q4_K/Q5_1 only")
        matches = [Path(p) for p in snapshots if Path(p).name == tensor["shard"]]
        if len(matches) != 1 or not matches[0].is_absolute():
            raise ValueError("source shard must have one absolute identity path")
        path = matches[0]
        current = expert.file_stat(path)
        if current != snapshots[str(path)]:
            raise ValueError(f"source identity changed since extraction: {path}")
        offset = tensor["data_offset"] + selection["expert"] * expected_sizes[role]
        if type(offset) is not int or offset < 0 or offset + expected_sizes[role] > current["size_bytes"]:
            raise ValueError("selected payload range is outside source file")
        shards.append({"role": role, "path": str(path), "offset": offset,
                       "bytes": expected_sizes[role], "sha256": receipt["raw_selected_payload_sha256"][role],
                       "identity": current})
    result = {"schema_version": 1, "status": "validated_only", "payload_read": False,
              "files_written": False, "device_created": False,
              "source_receipt": str(receipt_path.resolve()), "source_receipt_sha256": digest(receipt_bytes),
              "source_identity": str(identity_path), "source_identity_sha256": digest(identity_bytes),
              "selection": selection, "dimensions": receipt["dimensions"], "payloads": shards,
              "layout": "[original gate bytes][original up bytes][original down bytes]",
              "quantization": {"gate_up": "Q4_K", "down": "Q5_1", "requantized": False},
              "integrity_scope": "fresh scoped payload hashes plus current source identities; no full-shard rehash"}
    return result, receipt


def _write_new(path: Path, data: bytes) -> dict:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # Independent readback of the small owned artifact, not a source-shard hash.
    reread = path.read_bytes()
    if reread != data:
        raise ValueError(f"owned artifact readback differs: {path}")
    return {"path": str(path), "bytes": len(data), "sha256": digest(reread)}


def prepare(result: dict, output_dir: Path, seed: int = 42) -> dict:
    if not output_dir.is_absolute() or not output_dir.parent.is_dir():
        raise ValueError("output directory must be absolute with an existing parent")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"exclusive preparation refuses existing directory: {output_dir}")
    payloads = []
    for record in result["payloads"]:
        path = Path(record["path"])
        if expert.file_stat(path) != record["identity"]:
            raise ValueError("source identity changed before payload read")
        with path.open("rb", buffering=0) as stream:
            fd = os.fstat(stream.fileno())
            if (fd.st_dev, fd.st_ino, fd.st_size, fd.st_mtime_ns) != (
                    record["identity"]["device"], record["identity"]["file_index"],
                    record["identity"]["size_bytes"], record["identity"]["mtime_ns"]):
                raise ValueError("opened source file differs from checked identity")
            stream.seek(record["offset"])
            data = stream.read(record["bytes"])
        if len(data) != record["bytes"] or digest(data) != record["sha256"]:
            raise ValueError("selected original quantized payload hash/length differs")
        if expert.file_stat(path) != record["identity"]:
            raise ValueError("source identity changed during scoped read")
        payloads.append(data)
    output_dir.mkdir()  # Exclusive creation after source verification; failures preserve owned partials.
    blob = _write_new(output_dir / "expert.native.bin", b"".join(payloads))
    inputs = []
    hidden = result["dimensions"]["hidden"]
    for rows in ROWS:
        rng = np.random.default_rng(seed + rows)
        x = rng.standard_normal((rows, hidden), dtype=np.float32) * np.float32(0.25)
        info = _write_new(output_dir / f"input-{rows:03d}.f32", x.astype("<f4", copy=False).tobytes(order="C"))
        inputs.append({"rows": rows, "shape": [rows, hidden], **info})
    return {**result, "status": "prepared", "payload_read": True, "files_written": True,
            "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "blob": blob, "inputs": inputs,
            "numpy_version": np.__version__, "seed": seed,
            "input_recipe": "default_rng(seed+rows).standard_normal(shape,dtype=float32)*float32(0.25)",
            "input_format": "token-major float32 little endian; same recipe as the prior OpenVINO probe",
            "claim_limit": "prepared bytes only; no native kernel, OpenVINO or model inference run"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.prepare and (not args.output_dir or not args.receipt):
        parser.error("--prepare requires --output-dir and --receipt")
    if not args.prepare and (args.output_dir or args.receipt):
        parser.error("write paths require explicit --prepare")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in 0..2**32-1")
    try:
        if args.prepare:
            if not args.receipt.is_absolute() or not args.receipt.parent.is_dir():
                raise ValueError("receipt must be absolute with an existing parent")
            if args.receipt.exists() or args.receipt.is_symlink():
                raise FileExistsError("receipt already exists")
            out = args.output_dir.resolve(strict=False)
            receipt = args.receipt.resolve(strict=False)
            if out == receipt or out in receipt.parents or receipt == args.source_receipt.resolve():
                raise ValueError("receipt must be outside output directory and distinct from source receipt")
        result, _ = inspect(args.source_receipt)
        if args.prepare:
            result = prepare(result, args.output_dir, args.seed)
            _write_new(args.receipt, (json.dumps(result, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "owned_partials_preserved": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
