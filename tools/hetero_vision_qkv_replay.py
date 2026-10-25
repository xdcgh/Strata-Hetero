#!/usr/bin/env python3
"""Selected-column saved QKV replay; default is a no-payload, no-runtime plan.

--run reads three captured input rows, seven exported weight rows and saved
native QKV values. It never imports NumPy/OpenVINO, loads a GGUF, or invokes an
encoder. Selected-element differences are diagnostics, never acceptance gates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

from hetero_vision_arithmetic import SOURCE_REVISION, bf16_avx2_dot, bf16_rne, f32

EXPORT_SHA = "9293d413d80a89bea202d9c79c480fb3e8c07ca79e8fae740ca52cfca0ff4e01"
NATIVE_SHA = "005b73df9b3182252256d89e6b71f9a3e005a1bb01d01d78fd74e952c75230fc"
ASSET_SHA = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
DATA = Path("E:/Strata-Hetero-data")
COLUMNS = (0, 1, 1151, 1152, 2303, 2304, 3455)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document(path: Path) -> dict:
    if path.stat().st_size > 4 << 20:
        raise ValueError("receipt exceeds diagnostic byte bound")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def npy_rows(item: dict, rows: tuple[int, ...], shape: tuple[int, ...]) -> dict[int, tuple]:
    """Hash existing NPY then seek only selected rows; no full array allocation."""
    path = Path(item["path"])
    if item["storage_type"] not in ("BF16", "F32") or tuple(item["shape"]) != shape:
        raise ValueError("exported tensor descriptor differs")
    if path.stat().st_size != item["file_bytes"] or sha(path) != item["file_sha256"]:
        raise ValueError("exported NPY file identity differs")
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("NPY magic differs")
        version = stream.read(2)
        if version not in (b"\x01\x00", b"\x02\x00"):
            raise ValueError("unsupported NPY version")
        size = struct.unpack("<H" if version[0] == 1 else "<I", stream.read(2 if version[0] == 1 else 4))[0]
        if size > 4096:
            raise ValueError("NPY header exceeds byte bound")
        header = ast.literal_eval(stream.read(size).decode("latin1"))
        if header != {"descr": "<f4", "fortran_order": False, "shape": shape}:
            raise ValueError("NPY shape/dtype/layout differs")
        offset = stream.tell()
        width = shape[-1] if len(shape) == 2 else 1
        count = shape[0]
        if path.stat().st_size != offset + math.prod(shape) * 4:
            raise ValueError("NPY payload byte count differs")
        result = {}
        for row in rows:
            if not 0 <= row < count:
                raise ValueError("selected row outside tensor")
            stream.seek(offset + row * width * 4)
            values = struct.unpack("<" + "f" * width, stream.read(width * 4))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite selected tensor row")
            if item["storage_type"] == "BF16" and any(value != bf16_rne(value) for value in values):
                raise ValueError("selected weights are not exact decoded BF16")
            result[row] = values
        return result


def raw_rows(item: dict, rows: tuple[int, ...], width: int) -> dict[int, tuple]:
    path = Path(item["path"])
    ne = item["ne"]
    if item["actual_dtype"] != "F32" or ne[0] != width or ne[1] not in (36, 72) or ne[2:] != [1, 1]:
        raise ValueError("saved stage dimensions differ")
    if item["nb"][:2] != [4, width * 4] or item["bytes"] != width * ne[1] * 4:
        raise ValueError("saved stage not contiguous")
    if path.stat().st_size != item["bytes"] or sha(path) != item["raw_sha256"]:
        raise ValueError("saved stage hash differs")
    result = {}
    with path.open("rb") as stream:
        for row in rows:
            stream.seek(row * width * 4)
            result[row] = struct.unpack("<" + "f" * width, stream.read(width * 4))
            if not all(math.isfinite(value) for value in result[row]):
                raise ValueError("nonfinite captured row")
    return result


def run(args) -> dict:
    if sha(args.export) != EXPORT_SHA:
        raise ValueError("pinned export receipt changed")
    if sha(args.native) != NATIVE_SHA:
        raise ValueError("frozen native stage receipt changed")
    export, native = document(args.export), document(args.native)
    if export["source_revision"] != SOURCE_REVISION or export["asset_sha256"] != ASSET_SHA:
        raise ValueError("pinned asset/source differs")
    if native["status"] != "native_stage_fingerprint_pass":
        raise ValueError("native oracle fingerprint gate did not pass")
    weights = npy_rows(export["tensors"]["v.blk.0.attn_qkv.weight"], COLUMNS, (3456, 1152))
    biases = npy_rows(export["tensors"]["v.blk.0.attn_qkv.bias"], COLUMNS, (3456,))
    results = []
    for capture in native["results"]:
        if capture["iteration"] != "formal-1":
            continue
        stages = {stage["stage"]: stage for stage in capture["manifest"]["stages"]}
        rows = (0, stages["ln1.0"]["ne"][1] // 2, stages["ln1.0"]["ne"][1] - 1)
        inputs = raw_rows(stages["ln1.0"], rows, 1152)
        targets = raw_rows(stages["qkv.0"], rows, 3456)
        for row in rows:
            for column in COLUMNS:
                source = f32(bf16_avx2_dot(weights[column], inputs[row]) + biases[column][0])
                rounded_fsum = f32(math.fsum(bf16_rne(x) * w for x, w in zip(inputs[row], weights[column])))
                rounded_fsum = f32(rounded_fsum + biases[column][0])
                target = targets[row][column]
                results.append({"fixture": capture["fixture"], "row": row, "column": column,
                                "native": target, "avx2_source": source,
                                "source_minus_native": source - target,
                                "source_bits_equal": struct.pack("<f", source) == struct.pack("<f", target),
                                "rounded_fsum": rounded_fsum})
    if len(results) != 84:
        raise ValueError("expected four fixture groups and 84 selected QKV elements")
    return {"status": "selected_qkv_source_replay", "source_revision": SOURCE_REVISION,
            "export_sha256": EXPORT_SHA, "native_receipt_sha256": sha(args.native),
            "script_sha256": sha(Path(__file__)),
            "arithmetic_sha256": sha(Path(__file__).with_name("hetero_vision_arithmetic.py")),
            "results": results, "npu_accepted": False, "original_reference_replaced": False,
            "gate_relaxed": False, "scope": "Source AVX2 operation order; compiler fusion/dispatch unproven"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--export", type=Path, default=DATA / "vision-payloads/20261009-30-native-vision-export/export-receipt.json")
    parser.add_argument("--native", type=Path, default=DATA / "vision-stage-oracle/20261009-43-native/native-stage-receipt.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "plan_only", "columns": COLUMNS, "rows_per_fixture": 3,
                          "payload_read": False, "array_runtime_imported": False,
                          "encoder_run": False, "source_revision": SOURCE_REVISION}))
        return 0
    if args.output is None or args.output.exists():
        raise ValueError("--run requires a new --output receipt path")
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "failed_preserved", "error": f"{type(exc).__name__}: {exc}",
                  "original_reference_replaced": False, "gate_relaxed": False, "npu_accepted": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "selected_elements": len(result.get("results", []))}))
    return 1 if result["status"] == "failed_preserved" else 0


if __name__ == "__main__":
    raise SystemExit(main())
