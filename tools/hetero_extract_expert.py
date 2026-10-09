#!/usr/bin/env python3
"""Inspect or explicitly extract one real expert from a sharded GGUF with pinned gguf-py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "data" / "expert-profile.bin"
DEFAULT_PACK = Path(r"F:\Strata-data\packs\unsloth-ud-q4_k_xl")
DEFAULT_MODEL = Path(r"F:\Strata-data\models\unsloth-UD-Q4_K_XL")
DEFAULT_GGUF_PY = Path(r"E:\Strata-Hetero-data\source\llama-3cf0325\gguf-py")
DEFAULT_INTEGRITY = ROOT / "bench" / "hetero" / "20261008-00-admission" / "model-integrity-02.json"
EXPECTED_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
TENSOR_ROLES = ("gate", "up", "down")
GGUF_NAMES = {
    "gate": "ffn_gate_exps.weight",
    "up": "ffn_up_exps.weight",
    "down": "ffn_down_exps.weight",
}


class ExtractError(ValueError):
    pass


def sha256(data: bytes | memoryview | np.ndarray) -> str:
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def read_profile(path: Path) -> tuple[dict[str, int], list[tuple[int, int]]]:
    data = path.read_bytes()
    if len(data) < 24:
        raise ExtractError("expert profile is shorter than its 24-byte header")
    magic, version, n_layer, n_expert, slots, count = struct.unpack_from("<4s5I", data)
    if magic != b"STRP" or version != 1:
        raise ExtractError(f"unsupported expert profile magic/version: {magic!r}/{version}")
    if count > slots or len(data) != 24 + count * 4 + n_layer * n_expert * 4:
        raise ExtractError("expert profile count/size does not match its header")
    pairs = [struct.unpack_from("<HH", data, 24 + i * 4) for i in range(count)]
    if not pairs:
        raise ExtractError("expert profile contains no ranked pairs")
    if len(set(pairs)) != len(pairs) or any(l >= n_layer or e >= n_expert for l, e in pairs):
        raise ExtractError("expert profile has duplicate or out-of-range ranked pairs")
    return {"version": version, "layers": n_layer, "experts": n_expert, "slots": slots,
            "ranked_count": count, "sha256": hashlib.sha256(data).hexdigest()}, pairs


def read_native_entry(path: Path, layer: int) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# strata native experts v4:"):
        raise ExtractError("expected native_experts.txt v4")
    found = []
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ExtractError(f"malformed native expert entry: {line}")
        if int(fields[0]) == layer:
            found.append(fields)
    if len(found) != 1:
        raise ExtractError(f"expected one native entry for layer {layer}, found {len(found)}")
    fields = found[0]
    shards = fields[8].split(",") if len(fields) > 8 else []
    if len(shards) not in (1, 3):
        raise ExtractError("native entry must name one shard or one shard per role")
    if len(shards) == 1:
        shards *= 3
    return {"layer": int(fields[0]), "gate_type_id": int(fields[1]), "down_type_id": int(fields[2]),
            "layer_blob_offset": int(fields[3]), "blob_bytes": int(fields[4]),
            "gate_offset": int(fields[5]), "up_offset": int(fields[6]), "down_offset": int(fields[7]),
            "shards": dict(zip(TENSOR_ROLES, shards))}


def _load_gguf(gguf_py: Path):
    if not gguf_py.is_dir():
        raise ExtractError(f"pinned gguf-py directory not found: {gguf_py}")
    sys.path.insert(0, str(gguf_py))
    try:
        import gguf
        from gguf import quants
    except Exception as exc:
        raise ExtractError(f"cannot import pinned gguf-py: {type(exc).__name__}: {exc}") from exc
    module_path = Path(gguf.__file__).resolve()
    if gguf_py.resolve() not in module_path.parents:
        raise ExtractError(f"gguf import resolved outside pinned checkout: {module_path}")
    revision = subprocess.run(["git", "-C", str(gguf_py), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(gguf_py), "status", "--porcelain=v1"],
                           check=True, capture_output=True, text=True).stdout
    if dirty.strip():
        raise ExtractError("pinned gguf-py checkout is dirty")
    if revision != EXPECTED_REVISION:
        raise ExtractError(f"gguf-py revision {revision} != pinned {EXPECTED_REVISION}")
    quants_sha256 = hashlib.sha256((module_path.parent / "quants.py").read_bytes()).hexdigest()
    return gguf, quants, module_path, revision, quants_sha256


def _identity_matches_manifest(current: dict[str, Any], after: dict[str, Any]) -> bool:
    file_id = after.get("file_id", {})
    return (current["size_bytes"] == after.get("size_bytes") and
            current["mtime_ns"] == after.get("mtime_ns") and
            current["ctime_ns"] == after.get("ctime_ns") and
            current["device"] == file_id.get("volume") and
            current["file_index"] == file_id.get("index"))


def verify_integrity_manifest(path: Path, model: Path, shard_names: set[str]) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractError(f"cannot read integrity manifest: {exc}") from exc
    if manifest.get("status") != "pass" or manifest.get("claims", {}).get("model_weight_integrity") is not True:
        raise ExtractError("integrity manifest does not record a passing full-model integrity check")
    records = {Path(item.get("path", "")).name: item for item in manifest.get("model_shards", [])}
    verified: dict[str, Any] = {}
    for name in sorted(shard_names):
        record = records.get(name)
        if not isinstance(record, dict) or record.get("matches_expected") is not True:
            raise ExtractError(f"integrity manifest has no passing expected-hash record for {name}")
        current_path = model / name
        if not current_path.is_file() or not _identity_matches_manifest(file_stat(current_path), record.get("after", {})):
            raise ExtractError(f"current shard identity does not match integrity receipt after-state: {name}")
        verified[name] = {"expected_sha256": record.get("expected_sha256", record.get("sha256")),
                          "matches_expected_in_prior_full_hash_run": True,
                          "current_identity_matches_prior_after_state": True,
                          "prior_after": record["after"]}
    return {"path": str(path), "run_id": manifest.get("run_id"), "status": manifest["status"],
            "scope_note": "The receipt records a prior full-shard hash. This tool checks only the selected shard's current size, timestamps, volume and file index against that receipt; it does not rehash the full shard.",
            "selected_shards": verified}


def inspect_expert(profile: Path, pack: Path, model: Path, gguf_py: Path,
                   pair: tuple[int, int] | None = None, profile_rank: int = 0,
                   integrity_manifest: Path = DEFAULT_INTEGRITY) -> dict[str, Any]:
    profile_meta, ranked = read_profile(profile)
    if profile_rank < 0 or profile_rank >= len(ranked):
        raise ExtractError(f"profile rank {profile_rank} outside 0..{len(ranked) - 1}")
    selected_pair = pair if pair is not None else ranked[profile_rank]
    if selected_pair not in ranked:
        raise ExtractError(f"requested pair {selected_pair} is not present in the profile")
    layer, expert = selected_pair
    if layer >= profile_meta["layers"] or expert >= profile_meta["experts"]:
        raise ExtractError("requested layer/expert is outside profile bounds")
    native = read_native_entry(pack / "native_experts.txt", layer)
    gguf, quants, module_path, revision, quants_sha256 = _load_gguf(gguf_py)
    tensors: dict[str, Any] = {}
    readers: dict[str, Any] = {}
    for role in TENSOR_ROLES:
        shard_name = native["shards"][role]
        shard_path = model / shard_name
        if not shard_path.is_file():
            raise ExtractError(f"GGUF shard named by pack is missing: {shard_path}")
        reader = gguf.GGUFReader(shard_path)
        name = f"blk.{layer}.{GGUF_NAMES[role]}"
        matches = [t for t in reader.tensors if t.name == name]
        if len(matches) != 1:
            raise ExtractError(f"expected exactly one {name} in {shard_name}; found {len(matches)}")
        tensor = matches[0]
        expected_type = native["down_type_id"] if role == "down" else native["gate_type_id"]
        if int(tensor.tensor_type) != expected_type:
            raise ExtractError(f"{name} type id {int(tensor.tensor_type)} != native pack type {expected_type}")
        expected_offset = native[f"{role}_offset"]
        if int(tensor.data_offset) != expected_offset:
            raise ExtractError(f"{name} GGUF data offset {tensor.data_offset} != native pack offset {expected_offset}")
        dims = tuple(int(x) for x in tensor.shape)
        if len(dims) != 3 or dims[2] != profile_meta["experts"]:
            raise ExtractError(f"{name} has unexpected GGUF dims {dims}")
        # GGUF dims are in storage order. ReaderTensor.data is reversed for the Python view;
        # its first axis is the expert index, and each expert slice can be dequantized independently.
        if tensor.data.shape[0] != profile_meta["experts"]:
            raise ExtractError(f"{name} mapped expert axis mismatch: {tensor.data.shape}")
        tensors[role] = {"name": name, "shard": shard_name, "type": tensor.tensor_type.name,
                         "type_id": int(tensor.tensor_type), "gguf_dims": list(dims),
                         "reader_data_shape": list(tensor.data.shape), "data_offset": int(tensor.data_offset),
                         "n_bytes": int(tensor.n_bytes)}
        readers[role] = (reader, tensor)
    gate_dims = tensors["gate"]["gguf_dims"]
    up_dims = tensors["up"]["gguf_dims"]
    down_dims = tensors["down"]["gguf_dims"]
    if tuple(gate_dims) != tuple(up_dims) or tuple(down_dims) != (gate_dims[1], gate_dims[0], gate_dims[2]):
        raise ExtractError(f"gate/up/down GGUF dimension relation is unexpected: {gate_dims}, {up_dims}, {down_dims}")
    hidden, intermediate = gate_dims[0], gate_dims[1]
    if gate_dims[2] != profile_meta["experts"]:
        raise ExtractError("expert count mismatch in GGUF")
    qtypes = {"gate": readers["gate"][1].tensor_type, "up": readers["up"][1].tensor_type,
              "down": readers["down"][1].tensor_type}
    dequant_support = {}
    for role, qtype in qtypes.items():
        dequant_support[role] = {"type": qtype.name, "official_trait_available": qtype in quants._type_traits or
                                 qtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)}
    if not all(x["official_trait_available"] for x in dequant_support.values()):
        unsupported = [r for r, v in dequant_support.items() if not v["official_trait_available"]]
        raise ExtractError(f"pinned official dequantizer has no implementation for: {unsupported}")
    integrity = verify_integrity_manifest(integrity_manifest, model, {x["shard"] for x in tensors.values()})
    return {"selection": {"layer": layer, "expert": expert, "profile_rank": ranked.index((layer, expert))},
            "profile": profile_meta, "native_entry": native, "gguf_py": {"path": str(module_path),
            "expected_revision": EXPECTED_REVISION, "actual_revision": revision,
            "quants_py_sha256": quants_sha256}, "dequant_support": dequant_support,
            "integrity_manifest": integrity,
            "dimensions": {"hidden": hidden, "intermediate": intermediate,
                           "worker_shapes": {"gate": [intermediate, hidden], "up": [intermediate, hidden],
                                             "down": [hidden, intermediate]}},
            "tensors": tensors, "_readers": readers, "_qtypes": qtypes}


def file_stat(path: Path) -> dict[str, int]:
    st = path.stat()
    fd = os.open(path, os.O_RDONLY)
    try:
        fst = os.fstat(fd)
    finally:
        os.close(fd)
    # Windows may return zero ctime from fstat while path.stat has the real timestamp.
    # Compare fd size/mtime/file identity only; keep path ctime for before/after checks.
    fields = ("st_size", "st_mtime_ns", "st_dev", "st_ino")
    if any(getattr(st, name) != getattr(fst, name) for name in fields):
        raise ExtractError(f"path stat/fstat differ for {path}")
    return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "device": fst.st_dev, "file_index": fst.st_ino,
            "fstat_size_bytes": fst.st_size, "fstat_mtime_ns": fst.st_mtime_ns,
            "fstat_device": fst.st_dev, "fstat_file_index": fst.st_ino}


def extract(result: dict[str, Any], output_dir: Path, model: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.npz"
    identity_path = output_dir / "identity.json"
    if weights_path.exists() or identity_path.exists():
        raise ExtractError("exclusive extraction refuses to overwrite weights.npz or identity.json")
    selected = result["selection"]
    before: dict[str, dict[str, int]] = {}
    arrays: dict[str, np.ndarray] = {}
    raw_hashes: dict[str, str] = {}
    raw_bytes: dict[str, int] = {}
    decoded_hashes: dict[str, str] = {}
    for role in TENSOR_ROLES:
        _, tensor = result["_readers"][role]
        shard_path = model / result["tensors"][role]["shard"]
        before.setdefault(str(shard_path), file_stat(shard_path))
        raw = np.ascontiguousarray(tensor.data[selected["expert"]]).reshape(-1)
        raw_hashes[role] = sha256(raw)
        raw_bytes[role] = int(raw.nbytes)
        decoded = result["_dequantize"](raw, result["_qtypes"][role])
        shape = tuple(result["dimensions"]["worker_shapes"][role])
        if decoded.size != int(np.prod(shape)):
            raise ExtractError(f"{role}: official dequantizer returned {decoded.size} values, expected {shape}")
        matrix = np.asarray(decoded, dtype=np.float32).reshape(shape)
        if not np.isfinite(matrix).all():
            raise ExtractError(f"{role}: dequantized values contain NaN or infinity")
        arrays[role] = np.ascontiguousarray(matrix)
        decoded_hashes[role] = sha256(arrays[role])
    after = {path: file_stat(Path(path)) for path in before}
    if before != after:
        raise ExtractError("source shard stat identity changed during selected payload read")
    identity = {
        "schema_version": 1,
        "model": "Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL",
        "shard": result["tensors"]["gate"]["shard"],
        "layer": selected["layer"], "expert": selected["expert"],
        "weights_sha256": decoded_hashes,
        "source": {"model": "Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL",
                   "quantization": "UD-Q4_K_XL", "quantized_source_types": {
                       role: result["tensors"][role]["type"] for role in TENSOR_ROLES},
                   "raw_selected_payload_sha256": raw_hashes,
                   "raw_selected_payload_bytes": raw_bytes,
                   "decoded_float32_sha256": decoded_hashes,
                   "scope": "hashes cover only the selected expert's three tensor payload slices and decoded float32 matrices; not full GGUF shards",
                   "dequantizer": "official pinned gguf-py gguf.quants.dequantize",
                   "gguf_py_path": result["gguf_py"]["path"],
                   "gguf_py_revision": result["gguf_py"]["actual_revision"],
                   "gguf_quants_py_sha256": result["gguf_py"]["quants_py_sha256"],
                   "integrity_manifest": result["integrity_manifest"],
                   "source_file_stat_before_after": before,
                   "source_file_stat_unchanged": True,
                   "limitations": ["dequantized weights preserve quantized source values but not Strata native CPU Q8 activation arithmetic",
                                   "no model inference or task quality claim is made"]},
    }
    # Use exclusive file creation so a rerun cannot replace prior evidence.
    try:
        with weights_path.open("xb") as stream:
            np.savez(stream, **arrays)
        with identity_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(identity, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
    except Exception:
        # Preserve a partial output for forensic review; never silently delete evidence.
        raise
    return {"weights_path": str(weights_path), "identity_path": str(identity_path),
            "raw_selected_payload_sha256": raw_hashes, "decoded_float32_sha256": decoded_hashes,
            "raw_selected_payload_bytes": raw_bytes,
            "decoded_float32_bytes": {name: int(array.nbytes) for name, array in arrays.items()},
            "source_file_stat_unchanged": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="metadata checks only; no tensor payload read or writes")
    mode.add_argument("--extract", action="store_true", help="explicitly read/dequantize one expert and write an exclusive NPZ+identity")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument("--integrity-manifest", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--profile-rank", type=int, default=0)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--expert", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path, help="exclusive JSON receipt path; required with --extract")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.layer is None) != (args.expert is None):
        parser.error("--layer and --expert must be supplied together")
    if args.profile_rank < 0 or (args.layer is not None and (args.layer < 0 or args.expert < 0)):
        parser.error("profile rank, layer and expert must be non-negative")
    if args.extract and args.output_dir is None:
        parser.error("--extract requires --output-dir")
    if args.extract and args.receipt is None:
        parser.error("--extract requires --receipt")
    if not args.extract and args.output_dir is not None:
        parser.error("--output-dir is only valid with --extract")
    if not args.extract and args.receipt is not None:
        parser.error("--receipt is only valid with --extract")
    try:
        pair = (args.layer, args.expert) if args.layer is not None else None
        result = inspect_expert(args.profile, args.pack, args.model, args.gguf_py, pair,
                                args.profile_rank, args.integrity_manifest)
        output = {k: v for k, v in result.items() if not k.startswith("_")}
        output.update({"status": "metadata_validated", "payload_read": False, "files_written": False})
        if args.extract:
            if args.receipt.exists():
                raise ExtractError(f"exclusive extraction refuses to overwrite receipt: {args.receipt}")
            _, quants, _, _, _ = _load_gguf(args.gguf_py)
            result["_dequantize"] = quants.dequantize
            output.update(extract(result, args.output_dir, args.model))
            output.update({"status": "extracted", "payload_read": True, "files_written": True})
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(output, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
            output["receipt_path"] = str(args.receipt)
        print(json.dumps(output, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(f"hetero_extract_expert: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
