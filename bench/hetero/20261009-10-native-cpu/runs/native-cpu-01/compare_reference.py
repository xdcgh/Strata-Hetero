#!/usr/bin/env python3
"""Load verified F32 weights after a native process and score its saved output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Keep this quality-only NumPy reference bounded to one CPU thread.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from tools.hetero_xpu_worker import load_verified_weights, numpy_ffn, quality_report  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-f32-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    input_manifest = json.loads(args.input_receipt.read_text(encoding="utf-8"))
    input_entry = next((x for x in input_manifest["inputs"] if x["rows"] == args.rows), None)
    if input_entry is None or Path(input_entry["path"]).resolve() != args.input.resolve():
        fail("input path/rows do not match the approved input receipt")
    input_raw = args.input.read_bytes()
    input_sha = sha256(input_raw)
    if len(input_raw) != input_entry["bytes"] or input_sha != input_entry["sha256"]:
        fail("input size or SHA-256 no longer matches its approved receipt")

    native = json.loads(args.native_receipt.read_text(encoding="utf-8"))
    if native.get("status") != "success":
        fail("native receipt does not report success")
    params = native.get("parameters", {})
    if params != {"rows": args.rows, "hidden": 2560, "intermediate": 640, "gu_type": 12,
                  "down_type": 7, "warmup": 1, "repeat": 5}:
        fail(f"native receipt parameters differ from the approved run: {params}")
    if native.get("execution", {}).get("gpu_executed") is not False or native["execution"].get("engine_pool_called") is not False:
        fail("native receipt has an unexpected GPU or engine-pool scope")
    if native.get("reference_controls", {}).get("STRATA_KQ256") != "0 (process-local)":
        fail("native receipt did not hold the KQ row kernels off")
    controls = native.get("reference_controls", {})
    if controls.get("STRATA_NO_Q8K_AVX2") != "unset (model-default Q8_K activation path, process-local)":
        fail("native receipt did not use the model-default Q8_K activation setting")
    if controls.get("q8k_avx2_selected") is not True:
        fail("native receipt shows Q8_K AVX2 was not selected on this intended AVX2 host")

    output_raw = args.output.read_bytes()
    output_sha = sha256(output_raw)
    output_info = native.get("output", {})
    if len(output_raw) != args.rows * 2560 * 4 or output_info.get("bytes") != len(output_raw) or output_info.get("sha256") != output_sha:
        fail("output size or SHA-256 does not match the native receipt")
    if native.get("input", {}).get("sha256") != input_sha:
        fail("native receipt input hash differs from the approved input bytes")

    cpu_f32 = json.loads(args.cpu_f32_receipt.read_text(encoding="utf-8"))
    if cpu_f32.get("status") != "complete" or not cpu_f32.get("weights_loaded_and_hash_verified"):
        fail("prior CPU F32 reference receipt is not complete and weight-verified")
    tolerances = cpu_f32["quality_tolerances"]
    strict = {"max_abs": 1e-2, "relative_rmse": 1e-3, "row_norm_relative": 1e-3}
    if tolerances != strict:
        fail(f"prior strict tolerances differ from the approved values: {tolerances}")
    identity = cpu_f32["weights_identity"]
    header = cpu_f32["weights_header_info"]
    weights_path = Path(header["path"])
    before = weights_path.stat()
    file_id = header["file_id"]
    if (before.st_size != header["size_bytes"] or before.st_mtime_ns != header["mtime_ns"] or
            before.st_dev != file_id["volume"] or before.st_ino != file_id["index"]):
        fail("verified F32 weight NPZ identity differs from the prior CPU receipt")

    # This is deliberately after the native process: load and hash-check the previous official-dequantized NPZ now.
    weights = load_verified_weights(weights_path, identity, header)
    x = np.frombuffer(input_raw, dtype="<f4").reshape(args.rows, 2560).astype(np.float32, copy=False)
    actual = np.frombuffer(output_raw, dtype="<f4").reshape(args.rows, 2560).astype(np.float32, copy=False)
    reference = numpy_ffn(x, weights)
    quality = quality_report(actual, reference, tolerances)

    after = weights_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_dev, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_dev, after.st_ino
    ):
        fail("F32 weight NPZ identity changed during the comparison")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        fail("native or F32 reference output contains non-finite values")

    report = {
        "schema": "strata-hetero-native-cpu-quality-v1",
        "rows": args.rows,
        "input_path": str(args.input.resolve()),
        "input_bytes": len(input_raw),
        "input_sha256": input_sha,
        "native_receipt_path": str(args.native_receipt.resolve()),
        "output_path": str(args.output.resolve()),
        "output_sha256": output_sha,
        "reference": "NumPy F32 FFN over the previously hash-verified official gguf-py dequantized weights",
        "weights_npz_path": str(weights_path.resolve()),
        "weights_npz_stat": {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                              "device": after.st_dev, "file_index": after.st_ino},
        "weights_tensor_sha256_verified": identity["weights_sha256"],
        "weights_loaded_and_hash_verified": True,
        "numpy_version": np.__version__,
        "threads": 1,
        "quality_tolerances": tolerances,
        "quality": quality,
        "comparison_scope": "Native quantized CPU Q4_K/Q5_1 with Q8 activation arithmetic versus the same expert dequantized to F32; this is not task-quality or engine-throughput evidence.",
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": args.rows, "quality_pass": quality.get("pass"),
                      "max_abs_error": quality.get("max_abs_error"),
                      "relative_rmse": quality.get("relative_rmse"),
                      "max_row_norm_relative_error": quality.get("max_row_norm_relative_error")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
