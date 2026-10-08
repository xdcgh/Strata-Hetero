#!/usr/bin/env python3
"""Offline validation and explicit OpenVINO FFN microbench for an identified expert."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROW_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TENSOR_NAMES = ("gate", "up", "down")
DEVICES = ("CPU", "GPU.0", "NPU")
PRECISIONS = ("f32", "f16")
DEFAULT_TOLERANCES = {"max_abs": 1e-2, "relative_rmse": 1e-3, "row_norm_relative": 1e-3}
MAX_TENSOR_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_TENSOR_DIMENSION = 131_072


class WorkerError(ValueError):
    pass


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise WorkerError("identity must be a schema_version 1 JSON object")
    for field in ("model", "shard"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise WorkerError(f"identity.{field} must be a non-empty string")
    for field in ("layer", "expert"):
        v = value.get(field)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise WorkerError(f"identity.{field} must be a non-negative integer")
    hashes = value.get("weights_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(TENSOR_NAMES):
        raise WorkerError("identity.weights_sha256 must contain exactly gate/up/down")
    for name in TENSOR_NAMES:
        digest = hashes[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise WorkerError(f"identity.weights_sha256.{name} must be lowercase SHA-256")
    return value


def _npy_header(stream) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise WorkerError(f"unsupported .npy header version {version}")
    return tuple(int(x) for x in shape), bool(fortran), np.dtype(dtype), stream.tell()


def inspect_npz_headers(npz_path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Inspect NPZ/NPY metadata; header reads may decompress a prefix, but arrays are never materialized here."""
    if not npz_path.is_file() or npz_path.is_symlink():
        raise WorkerError(f"weights NPZ is missing or not a regular file: {npz_path}")
    expected_names = {f"{name}.npy" for name in TENSOR_NAMES}
    tensors: dict[str, Any] = {}
    total_payload_bytes = 0
    with zipfile.ZipFile(npz_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise WorkerError(f"NPZ must contain exactly {sorted(expected_names)}")
        for name in TENSOR_NAMES:
            info = next(x for x in infos if x.filename == f"{name}.npy")
            with archive.open(info, "r") as entry:
                shape, fortran, dtype, header_bytes = _npy_header(entry)
            if len(shape) != 2 or any(dim <= 0 or dim > MAX_TENSOR_DIMENSION for dim in shape):
                raise WorkerError(f"{name} dimensions must be positive and <= {MAX_TENSOR_DIMENSION}, got {shape}")
            if dtype.kind != "f" or dtype.itemsize != 4 or dtype.name != "float32" or not dtype.isnative:
                raise WorkerError(f"{name} must be native-endian float32, got {dtype}")
            payload_bytes = math.prod(shape) * dtype.itemsize
            total_payload_bytes += payload_bytes
            if total_payload_bytes > MAX_TENSOR_PAYLOAD_BYTES:
                raise WorkerError(f"tensor payload total exceeds {MAX_TENSOR_PAYLOAD_BYTES} bytes")
            if info.file_size != header_bytes + payload_bytes:
                raise WorkerError(f"{name} NPY member size does not match its header")
            tensors[name] = {"shape": list(shape), "dtype": dtype.name, "fortran_order": fortran,
                             "uncompressed_member_bytes": info.file_size, "tensor_payload_bytes": payload_bytes,
                             "sha256_expected": identity["weights_sha256"][name]}
    gate, up, down = (tensors[x]["shape"] for x in TENSOR_NAMES)
    if len(gate) != 2 or up != gate or len(down) != 2:
        raise WorkerError("gate/up must share [intermediate, hidden], down must be [hidden, intermediate]")
    if gate[0] <= 0 or gate[1] <= 0 or down != [gate[1], gate[0]]:
        raise WorkerError("gate/up/down dimensions are inconsistent")
    stat = npz_path.stat()
    return {"path": str(npz_path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "file_id": {"volume": stat.st_dev, "index": stat.st_ino}, "tensors": tensors,
            "hidden": gate[1], "intermediate": gate[0], "tensor_payload_bytes": total_payload_bytes,
            "max_tensor_payload_bytes": MAX_TENSOR_PAYLOAD_BYTES, "payload_hashes_verified": False,
            "tensor_arrays_materialized": False}


def validate_inputs(weights_path: Path, identity_path: Path) -> dict[str, Any]:
    try:
        identity = _safe_identity(json.loads(identity_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot read identity JSON: {exc}") from exc
    info = inspect_npz_headers(weights_path, identity)
    return {"identity_path": str(identity_path), "identity": identity, "weights": info,
            "validation_only_note": "NPZ/NPY metadata was checked (header reads may decompress a prefix); tensor arrays were not materialized and payload hashes were not checked."}


def validate_npz_metadata(weights_path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata without importing OpenVINO or materializing tensor arrays."""
    return inspect_npz_headers(weights_path, identity)


def assess_precision_policy(actual_precision: Any, requested_precision: Any,
                            query_error: str | None = None) -> dict[str, Any]:
    """Assess the reported property only; it cannot prove per-operation execution precision."""
    if query_error is not None or actual_precision is None or actual_precision == "":
        return {"precision_policy_status": "unknown", "precision_policy_matches": False}
    matches = bool(actual_precision == requested_precision)
    return {"precision_policy_status": "match" if matches else "mismatch",
            "precision_policy_matches": matches}


def execution_devices_match(requested_device: str, actual_devices: list[str]) -> bool:
    """Refuse any actual execution-device identifier that differs from the requested identifier."""
    return bool(actual_devices) and all(name == requested_device for name in actual_devices)


def device_full_name_matches(requested_device: str, full_name: str) -> bool:
    """Accept OpenVINO's observed Intel NPU alias while keeping execution ID matching exact."""
    label = full_name.casefold()
    if requested_device.startswith("GPU."):
        return "intel" in label
    if requested_device == "NPU":
        return "npu" in label or "ai boost" in label
    return bool(label.strip())


def compile_property_policy(device: str, precision: str,
                            supported_property_names: set[str]) -> dict[str, Any]:
    """Return requested/submitted/omitted compile hints from advertised device properties."""
    requested = {"INFERENCE_PRECISION_HINT": precision, "EXECUTION_MODE_HINT": "ACCURACY"}
    omitted: dict[str, str] = {}
    submitted: dict[str, Any] = {}
    if device == "NPU":
        omitted["EXECUTION_MODE_HINT"] = (
            "Omitted: the prior NPU attempt was rejected with NOT_FOUND for this option, and the device's "
            "SUPPORTED_PROPERTIES did not advertise it."
        )
        if "INFERENCE_PRECISION_HINT" in supported_property_names:
            submitted["INFERENCE_PRECISION_HINT"] = precision
        else:
            omitted["INFERENCE_PRECISION_HINT"] = (
                "Omitted: NPU SUPPORTED_PROPERTIES did not advertise INFERENCE_PRECISION_HINT; reported policy "
                "remains unknown. The F16 model/input dtype does not prove per-operation precision."
            )
    else:
        missing = {name for name in requested if name not in supported_property_names}
        if missing:
            raise WorkerError(f"{device} does not advertise required compile properties: {sorted(missing)}")
        submitted.update(requested)
    return {"requested": requested, "submitted": submitted, "omitted": omitted}


def load_verified_weights(weights_path: Path, identity: dict[str, Any],
                          header_info: dict[str, Any]) -> dict[str, np.ndarray]:
    before = weights_path.stat()
    arrays: dict[str, np.ndarray] = {}
    with np.load(weights_path, allow_pickle=False) as archive:
        for name in TENSOR_NAMES:
            value = np.asarray(archive[name])
            if value.dtype != np.float32 or list(value.shape) != header_info["tensors"][name]["shape"]:
                raise WorkerError(f"{name} payload dtype/shape differs from its validated NPY header")
            if not np.isfinite(value).all():
                raise WorkerError(f"{name} contains non-finite values")
            contiguous = np.ascontiguousarray(value)
            digest = _sha256_bytes(memoryview(contiguous).cast("B"))
            if digest != identity["weights_sha256"][name]:
                raise WorkerError(f"{name} SHA-256 does not match identity; input preserved")
            arrays[name] = contiguous
    after = weights_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev
    ):
        raise WorkerError("NPZ changed while loading its tensors")
    return arrays


def stable_silu(values: np.ndarray) -> np.ndarray:
    """Numerically stable float32 SiLU used by the NumPy reference."""
    z = np.asarray(values, dtype=np.float32)
    sigmoid = np.empty_like(z)
    positive = z >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    sigmoid[~positive] = exp_z / (1.0 + exp_z)
    return z * sigmoid


def numpy_ffn(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    gate = np.asarray(x, dtype=np.float32) @ weights["gate"].T
    up = np.asarray(x, dtype=np.float32) @ weights["up"].T
    return (stable_silu(gate) * up) @ weights["down"].T


def quality_report(actual: np.ndarray, reference: np.ndarray,
                   tolerances: dict[str, float] = DEFAULT_TOLERANCES) -> dict[str, Any]:
    got = np.asarray(actual, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    shape_ok = got.shape == ref.shape
    finite = bool(np.isfinite(got).all() and np.isfinite(ref).all())
    if not shape_ok or not finite:
        return {"shape_ok": shape_ok, "finite": finite, "pass": False}
    diff = got - ref
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64)))) if ref.size else 0.0
    rel_rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))) / max(denom, 1e-12))
    got_norm = np.linalg.norm(got.astype(np.float64), axis=1)
    ref_norm = np.linalg.norm(ref.astype(np.float64), axis=1)
    row_norm_rel = float(np.max(np.abs(got_norm - ref_norm) / np.maximum(ref_norm, 1e-12)))
    ok = (max_abs <= tolerances["max_abs"] and rel_rmse <= tolerances["relative_rmse"]
          and row_norm_rel <= tolerances["row_norm_relative"])
    return {"shape_ok": True, "finite": True, "max_abs_error": max_abs,
            "relative_rmse": rel_rmse, "max_row_norm_relative_error": row_norm_rel,
            "tolerances": tolerances, "pass": bool(ok)}


def make_ffn_model(weights: dict[str, np.ndarray], rows: int, precision: str):
    """Lazy OpenVINO model creation; never called by validation-only mode."""
    import openvino as ov
    ops = ov.opset13
    dtype = np.float32 if precision == "f32" else np.float16
    ov_type = ov.Type.f32 if precision == "f32" else ov.Type.f16
    hidden = int(weights["gate"].shape[1])
    x = ops.parameter([rows, hidden], ov_type, name="X")
    gate = ops.constant(np.asarray(weights["gate"], dtype=dtype))
    up = ops.constant(np.asarray(weights["up"], dtype=dtype))
    down = ops.constant(np.asarray(weights["down"], dtype=dtype))
    gate_linear = ops.matmul(x, gate, transpose_a=False, transpose_b=True)
    up_linear = ops.matmul(x, up, transpose_a=False, transpose_b=True)
    one = ops.constant(np.asarray(1.0, dtype=dtype))
    gate_silu = ops.divide(gate_linear, ops.add(one, ops.exp(ops.negative(gate_linear))))
    mixed = ops.multiply(gate_silu, up_linear)
    result = ops.matmul(mixed, down, transpose_a=False, transpose_b=True)
    result = ops.convert(result, ov.Type.f32)  # stable output contract for both F32 and explicit F16 experiments
    return ov.Model([result], [x], f"strata_expert_ffn_rows_{rows}")


def _atomic_update(stream, record: dict[str, Any]) -> None:
    stream.seek(0)
    json.dump(record, stream, indent=2, ensure_ascii=True)
    stream.write("\n")
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def run_bench(args: argparse.Namespace, validated: dict[str, Any]) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x+", encoding="utf-8", newline="\n") as receipt:
        record: dict[str, Any] = {
            "schema_version": 1, "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "device_requested": args.device, "precision_requested": args.precision,
            "precision_experiment": args.precision == "f16", "weights_identity": validated["identity"],
            "weights_header_info": validated["weights"], "rows": [], "crossover_eligible": False,
            "precision_policy_scope": "Reported OpenVINO inference_precision property only; not proof of actual per-operation precision.",
            "native_cpu_comparison_note": (
                "This NumPy/OpenVINO F32 FFN is not Strata's native quantized CPU kernel; do not claim a native CPU "
                "or model-level speedup from these timings."
            ),
        }
        _atomic_update(receipt, record)
        try:
            import openvino as ov  # Import/Core/device queries are restricted to explicit --run.
            from openvino import properties

            core = ov.Core()
            full_name = str(core.get_property(args.device, properties.device.full_name))
            if not full_name.strip():
                raise WorkerError("OpenVINO returned an empty FULL_DEVICE_NAME")
            if not device_full_name_matches(args.device, full_name):
                raise WorkerError(f"{args.device} resolved to an unexpected device: {full_name}")
            record["device_full_name"] = full_name
            record["openvino_version"] = ov.__version__
            supported_properties_raw = core.get_property(args.device, properties.supported_properties)
            if isinstance(supported_properties_raw, dict):
                supported_property_names = {str(name) for name in supported_properties_raw}
            else:
                supported_property_names = {str(name) for name in supported_properties_raw}
            record["device_supported_property_names"] = sorted(supported_property_names)
            record["seed"] = args.seed
            record["row_counts"] = list(ROW_COUNTS)
            record["warmups_per_shape"] = args.warmups
            record["repeats_per_shape"] = args.repeat
            record["quality_tolerances"] = {"max_abs": args.max_abs_tol,
                                            "relative_rmse": args.relative_rmse_tol,
                                            "row_norm_relative": args.row_norm_rel_tol}
            _atomic_update(receipt, record)

            weights_path = Path(args.weights)
            weights = load_verified_weights(weights_path, validated["identity"], validated["weights"])
            record["weights_loaded_and_hash_verified"] = True
            record["weight_tensor_sha256"] = {
                name: _sha256_bytes(memoryview(np.ascontiguousarray(weights[name])).cast("B"))
                for name in TENSOR_NAMES
            }
            _atomic_update(receipt, record)

            hint = properties.hint
            requested_type = ov.Type.f32 if args.precision == "f32" else ov.Type.f16
            policy = compile_property_policy(args.device, args.precision, supported_property_names)
            compile_property_keys = {
                "INFERENCE_PRECISION_HINT": hint.inference_precision,
                "EXECUTION_MODE_HINT": hint.execution_mode,
            }
            compile_properties = {}
            for name, value in policy["submitted"].items():
                compile_properties[compile_property_keys[name]] = (
                    requested_type if name == "INFERENCE_PRECISION_HINT" else hint.ExecutionMode.ACCURACY
                )
            record["compile_property_policy"] = policy
            record["compile_properties_note"] = (
                "Submitted only device-advertised properties, except EXECUTION_MODE_HINT is explicitly omitted "
                "for NPU because the prior attempt was rejected. An omitted inference precision hint leaves policy "
                "unknown; F16 model/input dtype does not prove per-operation precision."
            )
            _atomic_update(receipt, record)
            for rows in ROW_COUNTS:
                row_result: dict[str, Any] = {"rows": rows, "status": "compiling"}
                record["rows"].append(row_result)
                _atomic_update(receipt, record)
                rng = np.random.default_rng(args.seed + rows)
                x = (rng.standard_normal((rows, validated["weights"]["hidden"]), dtype=np.float32)
                     * np.float32(0.25))
                reference = numpy_ffn(x, weights)
                row_result["input_finite"] = bool(np.isfinite(x).all())
                compile_start = time.perf_counter()
                model = make_ffn_model(weights, rows, args.precision)
                compiled = core.compile_model(model, args.device, compile_properties)
                row_result["compile_seconds"] = round(time.perf_counter() - compile_start, 6)
                actual_devices = compiled.get_property(properties.execution_devices)
                if isinstance(actual_devices, str):
                    actual_device_names = [x.strip() for x in actual_devices.split(",") if x.strip()]
                else:
                    actual_device_names = [str(x) for x in actual_devices]
                if not execution_devices_match(args.device, actual_device_names):
                    raise WorkerError(f"explicit device {args.device} compiled on {actual_device_names}; "
                                      "AUTO/HETERO/fallback execution is refused")
                row_result["execution_devices"] = actual_device_names
                if "INFERENCE_PRECISION_HINT" not in policy["submitted"]:
                    row_result["reported_inference_precision"] = None
                    row_result.update(assess_precision_policy(None, requested_type))
                    row_result["precision_policy_reason"] = policy["omitted"].get(
                        "INFERENCE_PRECISION_HINT", "No precision hint was submitted; policy is unknown."
                    )
                else:
                    try:
                        actual_precision = compiled.get_property(hint.inference_precision)
                        row_result["reported_inference_precision"] = str(actual_precision)
                        row_result.update(assess_precision_policy(actual_precision, requested_type))
                    except Exception as exc:
                        row_result["reported_inference_precision"] = None
                        row_result.update(assess_precision_policy(None, requested_type, str(exc)))
                        row_result["precision_query_error"] = f"{type(exc).__name__}: {exc}"

                request = compiled.create_infer_request()
                input_port, output_port = compiled.input(0), compiled.output(0)

                def infer_once() -> tuple[np.ndarray, float]:
                    start = time.perf_counter()
                    input_copy = np.array(x, dtype=(np.float32 if args.precision == "f32" else np.float16),
                                          copy=True, order="C")
                    request.infer({input_port: input_copy})
                    output_copy = np.array(request.get_output_tensor(0).data, dtype=np.float32, copy=True)
                    return output_copy, (time.perf_counter() - start) * 1000.0

                warmup_times = []
                for _ in range(args.warmups):
                    _, elapsed = infer_once()
                    warmup_times.append(elapsed)
                row_result["warmup_wall_ms"] = warmup_times
                latencies = []
                quality = []
                for _ in range(args.repeat):
                    actual, elapsed = infer_once()
                    latencies.append(elapsed)
                    quality.append(quality_report(actual, reference, {
                        "max_abs": args.max_abs_tol,
                        "relative_rmse": args.relative_rmse_tol,
                        "row_norm_relative": args.row_norm_rel_tol,
                    }))
                row_result["repeat_wall_ms_includes_input_copy_and_output_retrieval"] = latencies
                row_result["latency_median_ms"] = float(np.median(latencies))
                row_result["latency_p95_ms"] = float(np.percentile(latencies, 95))
                row_result["quality_by_repeat"] = quality
                row_result["quality_pass"] = all(x["pass"] for x in quality)
                row_result["precision_policy_scope"] = "reported property only; does not prove per-operation precision"
                row_result["crossover_eligible"] = (row_result["quality_pass"] and
                                                     row_result["precision_policy_matches"] and
                                                     execution_devices_match(args.device, actual_device_names))
                row_result["status"] = "complete"
                _atomic_update(receipt, record)
                del request, compiled, model
                import gc
                gc.collect()
            record["status"] = "complete"
            record["crossover_eligible_all_rows"] = all(x["crossover_eligible"] for x in record["rows"])
            record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_update(receipt, record)
            return record
        except Exception as exc:
            record["status"] = "failed"
            record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_update(receipt, record)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true",
                       help="default: inspect NPZ headers/identity only; no OpenVINO import or file writes")
    modes.add_argument("--run", action="store_true",
                       help="explicitly import OpenVINO and run the selected device microbench")
    parser.add_argument("--weights", required=True, help="NPZ with gate/up/down float32 matrices")
    parser.add_argument("--identity", required=True, help="JSON identifying model/shard/layer/expert and hashes")
    parser.add_argument("--device", choices=DEVICES, help="required with --run; exact device, no AUTO/HETERO")
    parser.add_argument("--precision", choices=PRECISIONS, default="f32",
                        help="f32 baseline; f16 is a separately labelled experiment")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", help="exclusive JSON receipt path for --run")
    parser.add_argument("--max-abs-tol", type=float, default=DEFAULT_TOLERANCES["max_abs"])
    parser.add_argument("--relative-rmse-tol", type=float, default=DEFAULT_TOLERANCES["relative_rmse"])
    parser.add_argument("--row-norm-rel-tol", type=float, default=DEFAULT_TOLERANCES["row_norm_relative"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run:
        if args.device is None:
            parser.error("--run requires an exact --device CPU|GPU.0|NPU")
        if args.repeat < 3:
            parser.error("--repeat must be at least 3")
        if args.warmups < 1:
            parser.error("--warmups must be at least 1")
        if not args.output:
            parser.error("--run requires --output (created exclusively)")
    try:
        identity = _safe_identity(json.loads(Path(args.identity).read_text(encoding="utf-8")))
        validated = validate_npz_metadata(Path(args.weights), identity)
        if not args.run:
            result = {"status": "validated_only", "openvino_imported": False, "device_queried": False,
                      "file_written": False, "tensor_arrays_materialized": False,
                      "payload_hashes_verified": False,
                      "validation_note": "NPZ/NPY metadata was checked; header reads may decompress a prefix, but tensor arrays were not materialized and payload hashes were not checked.",
                      "identity": identity, "weights_header_info": validated}
            print(json.dumps(result, indent=2, ensure_ascii=True))
            return 0
        run_bench(args, {"identity": identity, "weights": validated})
        return 0
    except Exception as exc:
        print(f"hetero_xpu_worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
