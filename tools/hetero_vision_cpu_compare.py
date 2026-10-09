#!/usr/bin/env python3
"""JSON-only vision contract check; --run compares both policies on exact CPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import time
from pathlib import Path

if __package__:
    from . import hetero_vision_payload as payload
else:
    import hetero_vision_payload as payload
port = payload.port
DEFAULT_INDEX = payload.ROOT / "bench/hetero/20261009-23-vision-oracle-cpu-results/sve-output-index.json"
ORACLE_HELPER_SHA256 = "7285dd08f24c3985fd6b97aaff7d9917b65e5ea8b263ae5567985d0dc04ba612"


def inspect(inventory_path=port.DEFAULT_INVENTORY, fixtures_path=payload.DEFAULT_FIXTURES,
            index_path=DEFAULT_INDEX, export_path=None, taps=()):
    spec, inventory, fixtures, metadata = payload.inspect(inventory_path, fixtures_path)
    index, index_sha = payload.read_json(index_path)
    if index.get("schema") != "strata-vision-oracle-sve-output-index-v1":
        raise ValueError("native index schema differs")
    oracle, oracle_sha = payload.read_json(index_path.parent / index["run_receipt"])
    argv = oracle.get("argv", [])
    if (oracle.get("status") != "completed" or oracle.get("exit_code") != 0
            or oracle.get("helper_sha256") != ORACLE_HELPER_SHA256
            or oracle.get("mmproj_sha256") != spec.asset_sha256 or "--gpu" in argv
            or "--flash-attn" not in argv or argv[argv.index("--flash-attn") + 1] != "off"):
        raise ValueError("native oracle must be completed CPU, selected asset, flash-attn off")
    references = {}
    for fixture in fixtures["fixtures"]:
        matches = [item for item in index["outputs"] if item["image"] == fixture["name"] and item["iteration"] == "formal-1"]
        if len(matches) != 1:
            raise ValueError("each fixture needs exactly one identified formal-1 native reference")
        item = matches[0]
        plan = port.image_shape_plan(spec, fixture["height"], fixture["width"])
        n, embd = plan["output"]
        ny, nx = plan["merger_grid"]
        if (item["n_tokens"], item["n_embd"], item["nx"], item["ny"], item["size_bytes"]) != (n, embd, nx, ny, 20 + 4 * n * embd):
            raise ValueError("native reference metadata shape/grid/body size differs")
        port._digest(item["sha256"], "native_sve_sha256")
        references[fixture["name"]] = item
    allowed_taps = {"patch_merge", "positioned", "pre_norm", "post_norm", "merged", "projection"}
    allowed_taps.update(f"block.{i}" for i in range(spec.blocks))
    allowed_taps.update(f"ln1.{i}" for i in (0, 1, 26) if i < spec.blocks)
    allowed_taps.add("qkv.0")
    if len(set(taps)) != len(taps) or set(taps) - allowed_taps:
        raise ValueError("unknown or duplicate diagnostic tap")
    exported = None
    if export_path is not None:
        exported, export_sha = payload.read_json(export_path)
        if (exported.get("status") != "exported" or exported.get("asset_sha256") != spec.asset_sha256
                or exported.get("inventory_sha256") != metadata["inventory_sha256"]
                or exported.get("fixture_manifest_sha256") != metadata["fixture_manifest_sha256"]
                or exported.get("asset_sha256_verified") is not True):
            raise ValueError("export receipt does not bind the current contracts and verified selected asset")
        identity = exported["identity"]
        for name, expected in (("schema_version", 1), ("kind", "gguf_decoded"), ("asset_sha256", spec.asset_sha256),
                               ("source_revision", spec.source_revision), ("layout", port.ARRAY_LAYOUT), ("decoded_dtype", "float32")):
            if identity.get(name) != expected:
                raise ValueError("exported decoded identity differs: " + name)
        if (identity.get("decoder") != f"llama.cpp:{port.SOURCE_REVISION}:gguf.quants.dequantize"
                or exported.get("decoder_hashes") != payload.DECODER_HASHES):
            raise ValueError("exported decoder is not the pinned source contract")
        if set(exported["tensors"]) != set(spec.headers) or set(identity["weights_sha256"]) != set(spec.headers):
            raise ValueError("export receipt must identify all and only 334 selected tensors")
        for name, header in spec.headers.items():
            item = exported["tensors"][name]
            source = next(t for t in inventory["tensor_descriptors"] if t["name"] == name)
            if (item["shape"], item["dtype"], item["storage_type"], item["raw_offset"], item["raw_bytes"]) != (
                    list(header.numpy_shape), "float32", header.storage_type, source["data_offset_absolute"], source["bytes"]):
                raise ValueError("raw/decoded descriptor binding differs: " + name)
            if item["decoded_sha256"] != identity["weights_sha256"][name]:
                raise ValueError("decoded identity hash differs from export artifact")
            for key in ("raw_sha256", "decoded_sha256", "file_sha256"):
                port._digest(item[key], name + "." + key)
            contract_artifact_path(item, exported, name + ".npy")
        if {item["name"] for item in exported["inputs"]} != set(references) or len(exported["inputs"]) != 4:
            raise ValueError("export must include exactly the four oracle fixture inputs")
        for item in exported["inputs"]:
            fixture = next(f for f in fixtures["fixtures"] if f["name"] == item["name"])
            plan = port.image_shape_plan(spec, fixture["height"], fixture["width"])
            if item["sha256"] != fixture["sha256"] or item["input"]["shape"] != plan["input"] or item["input"]["dtype"] != "float32":
                raise ValueError("preprocessed fixture identity/shape differs")
            contract_artifact_path(item["input"], exported, item["name"] + ".input.npy")
            for key in ("decoded_sha256", "file_sha256"):
                port._digest(item["input"][key], "input." + key)
        metadata["export_receipt_sha256"] = export_sha
    metadata.update(native_index_sha256=index_sha, native_receipt_sha256=oracle_sha,
        device_requested="CPU", arithmetic_policies=list(port.ARITHMETIC_POLICIES),
        compile_properties_required={"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"},
        quality_contract="hetero_xpu_worker.DEFAULT_TOLERANCES, unchanged; no tolerance override flags",
        native_sve_read=False, exported_arrays_read=False, native_parity="NOT_RUN", output_taps=list(taps),
        warmups_per_image=1, formal_repeats_per_image=3)
    return spec, fixtures, references, exported, metadata


def contract_artifact_path(item, exported, filename):
    root, path = Path(exported["output_directory"]), Path(item["path"])
    if not root.is_absolute() or not path.is_absolute() or path != root / filename or ".." in root.parts:
        raise ValueError("artifact path is outside its declared export directory")
    if not root.is_relative_to(payload.DATA_ROOT):
        raise ValueError("real exported artifacts must remain under the selected E data root")
    if type(item["file_bytes"]) is not int or not 4 * math.prod(item["shape"]) < item["file_bytes"] <= 4 * math.prod(item["shape"]) + 1024:
        raise ValueError("declared NPY size is outside its exact-shape bound")
    return path


def load_array(item):
    import numpy as np
    if __package__:
        from .hetero_xpu_worker import _npy_header
    else:
        from hetero_xpu_worker import _npy_header
    path = Path(item["path"])
    if path.stat().st_size != item["file_bytes"] or payload.hash_file(path) != item["file_sha256"]:
        raise ValueError("exported NPY file size/hash changed")
    with path.open("rb") as stream:
        shape, fortran, dtype, header_bytes = _npy_header(stream)
    if list(shape) != item["shape"] or fortran or dtype != np.dtype("float32") or not dtype.isnative:
        raise ValueError("NPY header does not match the bounded declared F32 C-order contract")
    if header_bytes + 4 * int(np.prod(shape)) != item["file_bytes"]:
        raise ValueError("NPY body size differs")
    array = np.load(path, allow_pickle=False)
    if port.decoded_array_sha256(array) != item["decoded_sha256"] or not np.isfinite(array).all():
        raise ValueError("exported decoded payload hash/finite check failed")
    return array


def load_reference(item):
    import numpy as np
    path = Path(item["path"])
    if path.stat().st_size != item["size_bytes"]:
        raise ValueError("native SVE size changed")
    with path.open("rb") as stream:
        raw = stream.read(item["size_bytes"] + 1)
    if len(raw) != item["size_bytes"]:
        raise ValueError("native SVE changed during bounded read")
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("native SVE SHA-256 changed")
    header = struct.unpack_from("<5i", raw)
    if header != (0x31455653, item["n_tokens"], item["nx"], item["ny"], item["n_embd"]):
        raise ValueError("native SVE header differs from the identified reference")
    array = np.frombuffer(raw, dtype="<f4", offset=20).reshape(item["n_tokens"], item["n_embd"]).copy()
    if not np.isfinite(array).all():
        raise ValueError("native SVE contains NaN or infinity")
    return array


def run(spec, fixtures, references, exported, metadata, output, taps=()):
    if exported is None:
        raise ValueError("explicit --run requires --export-receipt")
    journal = payload.Journal(output)
    result = {**metadata, "status": "running", "arms": [], "files_written": True, "npu_accepted": False,
              "claim_limit": "CPU/native encoder comparison; no production route or performance acceptance"}
    try:
        journal.gate("before-core")
        import numpy as np
        import openvino as ov
        from openvino import properties as props
        if __package__:
            from .hetero_xpu_worker import DEFAULT_TOLERANCES, quality_report, execution_devices_match, compile_property_policy
        else:
            from hetero_xpu_worker import DEFAULT_TOLERANCES, quality_report, execution_devices_match, compile_property_policy
        result["core_creation_attempted"] = True
        core = ov.Core()
        result.update(core_created=True, openvino_version=ov.__version__, quality_tolerances=dict(DEFAULT_TOLERANCES))
        result["cpu_full_name"] = str(core.get_property("CPU", props.device.full_name))
        if not result["cpu_full_name"].strip():
            raise ValueError("CPU full device name is empty")
        supported = {str(p) for p in core.get_property("CPU", props.supported_properties)}
        result["compile_property_policy"] = compile_property_policy("CPU", "f32", supported)
        settings = {props.hint.inference_precision: ov.Type.f32, props.hint.execution_mode: props.hint.ExecutionMode.ACCURACY}
        weights = {}
        for name, item in exported["tensors"].items():
            journal.gate("before-weight-load:" + name)
            weights[name] = load_array(item)
        port.validate_decoded_weights(spec, weights, exported["identity"])
        inputs, native = {}, {}
        for item in exported["inputs"]:
            journal.gate("before-input-reference:" + item["name"])
            inputs[item["name"]] = load_array(item["input"])
            native[item["name"]] = load_reference(references[item["name"]])
        result.update(exported_arrays_read=True, native_sve_read=True)
        shapes = sorted({(f["height"], f["width"]) for f in fixtures["fixtures"]})
        for policy in port.ARITHMETIC_POLICIES:
            for height, width in shapes:
                arm = {"policy": policy, "height": height, "width": width, "status": "building", "pass": False, "images": []}
                result["arms"].append(arm)
                model = compiled = request = None
                try:
                    journal.gate(f"before-build:{policy}:{height}x{width}")
                    model, arm["graph"] = port.build_openvino_model(spec, weights, exported["identity"],
                        image_height=height, image_width=width, arithmetic_policy=policy, output_taps=taps)
                    journal.gate(f"before-compile:{policy}:{height}x{width}")
                    started = time.perf_counter()
                    arm["compile_attempted"] = True
                    compiled = core.compile_model(model, "CPU", settings)
                    arm["compile_seconds"] = time.perf_counter() - started
                    result["compiled"] = True
                    journal.gate(f"after-compile:{policy}:{height}x{width}")
                    devices = compiled.get_property(props.execution_devices)
                    devices = [devices] if isinstance(devices, str) else list(map(str, devices))
                    arm["execution_devices"] = devices
                    precision = compiled.get_property(props.hint.inference_precision)
                    mode = compiled.get_property(props.hint.execution_mode)
                    arm.update(reported_inference_precision=str(precision), reported_execution_mode=str(mode))
                    if not execution_devices_match("CPU", devices) or precision != ov.Type.f32 or mode != props.hint.ExecutionMode.ACCURACY:
                        raise ValueError("CPU execution/F32/ACCURACY reported properties differ or are unknown")
                    request = compiled.create_infer_request()
                    for fixture in fixtures["fixtures"]:
                        if (fixture["height"], fixture["width"]) != (height, width):
                            continue
                        image_result = {"fixture": fixture["name"], "formal": [], "pass": False}
                        arm["images"].append(image_result)
                        for iteration in range(4):  # One excluded warmup, three formal comparisons.
                            label = "warmup" if iteration == 0 else f"formal-{iteration}"
                            stage = f"{policy}:{fixture['name']}:{label}"
                            journal.gate("before-inference:" + stage)
                            arm["inference_attempted"] = True
                            started = time.perf_counter()
                            request.infer({compiled.input(0): inputs[fixture["name"]]})
                            infer_ms = (time.perf_counter() - started) * 1000
                            result["inference_run"] = True
                            journal.gate("after-inference:" + stage)
                            actual = np.array(request.get_output_tensor(0).data, dtype=np.float32, order="C", copy=True)
                            if iteration == 0:
                                image_result["warmup"] = {"inference_ms": infer_ms, "output_sha256": port.decoded_array_sha256(actual)}
                                journal.event("image-warmed", policy=policy, fixture=fixture["name"])
                                continue
                            formal = {"iteration": iteration, "inference_ms": infer_ms,
                                      "quality": quality_report(actual, native[fixture["name"]]), "taps": {}}
                            image_result["formal"].append(formal)
                            prefix = f"{policy}-{height}x{width}-{fixture['name']}-{label}"
                            formal["output"] = payload.write_array(journal.output / (prefix + ".npy"), actual)
                            for index, tap in enumerate(taps, 1):
                                array = np.array(request.get_output_tensor(index).data, dtype=np.float32, order="C", copy=True)
                                formal["taps"][tap] = payload.write_array(journal.output / (prefix + ".tap-" + tap + ".npy"), array)
                            journal.event("image-compared", policy=policy, fixture=fixture["name"], iteration=iteration, quality=formal["quality"])
                        image_result["repeat_hashes"] = [f["output"]["decoded_sha256"] for f in image_result["formal"]]
                        image_result["repeat_identical"] = len(set(image_result["repeat_hashes"])) == 1
                        image_result["pass"] = len(image_result["formal"]) == 3 and all(f["quality"]["pass"] for f in image_result["formal"])
                    arm.update(status="compared", **{"pass": bool(arm["images"]) and all(i["pass"] for i in arm["images"])})
                except payload.AdmissionError:
                    raise
                except Exception as exc:
                    arm.update(status="failed", error=f"{type(exc).__name__}: {exc}", failure_stage=journal.stage)
                finally:
                    request = compiled = model = None
                    gc.collect()
                journal.event("arm-terminal", arm=arm)
        passed = len(result["arms"]) == 4 and all(arm["pass"] for arm in result["arms"])
        result["policy_quality_pass"] = {
            policy: len([a for a in result["arms"] if a["policy"] == policy]) == 2
            and all(a["pass"] for a in result["arms"] if a["policy"] == policy)
            for policy in port.ARITHMETIC_POLICIES}
        result.update(status="compared", all_quality_pass=passed, native_parity="PASS" if passed else "FAIL")
        journal.gate("comparison-complete")
    except Exception as exc:
        result.update(status="failed", all_quality_pass=False, native_parity="FAIL_INCOMPLETE",
                      failure_stage=journal.stage, error=f"{type(exc).__name__}: {exc}")
    journal.finish(result, "comparison-receipt.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=port.DEFAULT_INVENTORY)
    parser.add_argument("--fixtures", type=Path, default=payload.DEFAULT_FIXTURES)
    parser.add_argument("--oracle-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tap", action="append", default=[])
    args = parser.parse_args(argv)
    if args.run != bool(args.output_dir) or (args.run and args.export_receipt is None):
        parser.error("--run requires --export-receipt and --output-dir; write paths require --run")
    try:
        spec, fixtures, refs, exported, result = inspect(args.inventory, args.fixtures, args.oracle_index, args.export_receipt, args.tap)
        if args.run:
            result = run(spec, fixtures, refs, exported, result, args.output_dir, args.tap)
        print(json.dumps(result, indent=2))
        return 1 if result["status"] == "failed" or result.get("all_quality_pass") is False else 0
    except Exception as exc:
        print(json.dumps({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
