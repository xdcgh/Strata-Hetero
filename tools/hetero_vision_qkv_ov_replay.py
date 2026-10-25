#!/usr/bin/env python3
"""Explicit bounded CPU OV MatMul replay; never a full encoder or NPU run."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path

import hetero_vision_qkv_replay as saved
from hetero_vision_arithmetic import bf16_avx2_dot, f32


def execute(args, receipt):
    from hetero_resources import memory_snapshot
    def gate(label):
        memory, errors = memory_snapshot()
        receipt.setdefault("memory_gates", []).append({"stage": label, "memory": memory, "errors": errors})
        if errors or memory["physical_available_bytes"] < 12 * 2**30 or memory["commit_available_bytes"] < 4 * 2**30:
            raise ValueError("12/4 GiB memory gate failed")
    gate("before-runtime")
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops, properties as props
    if saved.sha(args.export) != saved.EXPORT_SHA or saved.sha(args.native) != saved.NATIVE_SHA:
        raise ValueError("frozen receipts changed")
    export, native = saved.document(args.export), saved.document(args.native)
    if export["asset_sha256"] != saved.ASSET_SHA or export["source_revision"] != saved.SOURCE_REVISION:
        raise ValueError("pinned source/asset changed")
    if native["status"] != "native_stage_fingerprint_pass":
        raise ValueError("native fingerprints incomplete")
    columns = saved.COLUMNS
    weights = saved.npy_rows(export["tensors"]["v.blk.0.attn_qkv.weight"], columns, (3456, 1152))
    biases = saved.npy_rows(export["tensors"]["v.blk.0.attn_qkv.bias"], columns, (3456,))
    weight = np.asarray([weights[i] for i in columns], np.float32)
    bias = np.asarray([biases[i][0] for i in columns], np.float32)
    receipt.update(openvino_version=ov.__version__, numpy_version=np.__version__,
                   export_receipt_sha256=saved.EXPORT_SHA, native_receipt_sha256=saved.NATIVE_SHA,
                   source_revision=saved.SOURCE_REVISION, columns=list(columns),
                   geometry={"input": [3, 1152], "weights": [7, 1152], "output": [3, 7],
                             "full_encoder_qkv_output_width": 3456},
                   claim_limit="Seven selected output columns can change kernel dispatch; no full encoder, latency or NPU acceptance")
    def save(name, array):
        path = args.output / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
        return {"path": str(path), "shape": list(array.shape), "dtype": "float32",
                "file_bytes": path.stat().st_size, "file_sha256": saved.sha(path),
                "decoded_sha256": saved.hashlib.sha256(array.tobytes()).hexdigest()}
    def compare(actual, reference):
        difference = actual.astype(np.float64) - reference.astype(np.float64)
        denominator = float(np.sum(reference.astype(np.float64)**2))
        return {"max_abs": float(np.max(np.abs(difference))),
                "relative_rmse": float(np.sqrt(np.sum(difference**2) / denominator)),
                "bit_equal_elements": int(np.count_nonzero(actual.view(np.uint32) == reference.view(np.uint32))),
                "elements": int(actual.size)}
    gate("before-core")
    core = ov.Core()
    settings = {props.hint.inference_precision: ov.Type.f32,
                props.hint.execution_mode: props.hint.ExecutionMode.ACCURACY,
                props.inference_num_threads: 1}
    receipt["device"] = {"requested": "CPU", "full_name": str(core.get_property("CPU", props.device.full_name)),
                         "settings": {"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY",
                                      "INFERENCE_NUM_THREADS": 1}}
    receipt["arms"] = []
    for policy in ("cpu_vec_dot_rounding", "graph_f32"):
        inp = ops.parameter([3, 1152], np.float32, name="native_ln1_0")
        activation = ops.convert(ops.convert(inp, "bf16"), "f32") if policy == "cpu_vec_dot_rounding" else inp
        result = ops.add(ops.matmul(activation, ops.constant(weight), False, True), ops.constant(bias))
        model = ov.Model([result], [inp], "bounded_selected_qkv")
        gate("before-compile:" + policy)
        compiled = core.compile_model(model, "CPU", settings)
        devices = [str(device) for device in compiled.get_property(props.execution_devices)]
        if devices != ["CPU"]:
            raise ValueError("CPU execution-device identity differs")
        arm = {"policy": policy, "execution_devices": devices, "images": []}
        receipt["arms"].append(arm)
        for capture in native["results"]:
            if capture["iteration"] != "formal-1":
                continue
            stages = {stage["stage"]: stage for stage in capture["manifest"]["stages"]}
            rows = (0, stages["ln1.0"]["ne"][1]//2, stages["ln1.0"]["ne"][1]-1)
            inputs = saved.raw_rows(stages["ln1.0"], rows, 1152)
            targets = saved.raw_rows(stages["qkv.0"], rows, 3456)
            array = np.asarray([inputs[row] for row in rows], np.float32)
            target = np.asarray([[targets[row][column] for column in columns] for row in rows], np.float32)
            source = np.asarray([[f32(bf16_avx2_dot(weights[column], inputs[row]) + biases[column][0])
                                  for column in columns] for row in rows], np.float32)
            image = {"fixture": capture["fixture"], "rows": list(rows), "iterations": [],
                     "native": save(policy + "-" + capture["fixture"] + "-native", target),
                     "source": save(policy + "-" + capture["fixture"] + "-source", source),
                     "source_vs_native": compare(source, target)}
            arm["images"].append(image)
            repeated = []
            for iteration in ("warmup", "formal-1", "formal-2", "formal-3"):
                gate(policy + ":" + capture["fixture"] + ":" + iteration)
                actual = np.asarray(compiled([array])[0], np.float32).copy()
                if actual.shape != (3, 7) or not np.isfinite(actual).all():
                    raise ValueError("output shape/nonfinite failure")
                artifact = save(policy + "-" + capture["fixture"] + "-" + iteration, actual)
                image["iterations"].append({"iteration": iteration, "output": artifact,
                                            "vs_native": compare(actual, target), "vs_source": compare(actual, source)})
                if iteration != "warmup":
                    repeated.append(artifact["decoded_sha256"])
            image["formal_hashes_identical"] = len(set(repeated)) == 1
            if not image["formal_hashes_identical"]:
                raise ValueError("formal hashes differ")
        del compiled, model
    gate("complete")
    receipt["status"] = "bounded_cpu_qkv_compared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path, default=saved.DATA / "vision-payloads/20261009-30-native-vision-export/export-receipt.json")
    parser.add_argument("--native", type=Path, default=saved.DATA / "vision-stage-oracle/20261009-43-native/native-stage-receipt.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "pid": os.getpid(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "script_sha256": saved.sha(Path(__file__)), "npu_accepted": False,
               "original_reference_replaced": False, "gate_relaxed": False, "encoder_run": False}
    print(json.dumps({"pid": os.getpid(), "status": "started", "device": "CPU"}), flush=True)
    try:
        execute(args, receipt)
    except Exception as exc:
        receipt.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (args.output / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
    print(json.dumps({"pid": os.getpid(), "status": receipt["status"], "error": receipt.get("error")}), flush=True)
    return 1 if receipt["status"] == "failed_preserved" else 0


if __name__ == "__main__":
    raise SystemExit(main())
