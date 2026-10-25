"""Read-only saved-stage diagnostic. No runtime, model or device execution."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "bench/hetero/20261009-43-vision-stage-native/native-stage-receipt.json"
OV = ROOT / "bench/hetero/20261009-43-vision-stage-openvino/comparison-receipt.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def saved(item, native):
    data = Path(item["path"]).read_bytes()
    if native:
        assert sha(data) == item["raw_sha256"] and len(data) == item["bytes"]
        shape = (item["ne"][1], item["ne"][0])
        assert item["actual_dtype"] == "F32" and item["nb"][:2] == [4, shape[1]*4]
        raw = data
    else:
        assert sha(data) == item["file_sha256"] and len(data) == item["file_bytes"]
        assert data[:8] == b"\x93NUMPY\x01\x00"
        size = struct.unpack_from("<H", data, 8)[0]
        header = ast.literal_eval(data[10:10+size].decode("latin1"))
        shape = tuple(item["shape"])
        assert header == {"descr": "<f4", "fortran_order": False, "shape": shape}
        raw = data[10+size:]
        assert sha(raw) == item["decoded_sha256"]
    assert len(raw) == math.prod(shape)*4
    result = struct.unpack("<" + str(math.prod(shape)) + "f", raw)
    assert all(map(math.isfinite, result))
    return shape, result


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def main():
    native_bytes, ov_bytes = NATIVE.read_bytes(), OV.read_bytes()
    assert sha(native_bytes) == "005b73df9b3182252256d89e6b71f9a3e005a1bb01d01d78fd74e952c75230fc"
    native, ov = json.loads(native_bytes), json.loads(ov_bytes)
    assert native["status"] == "native_stage_fingerprint_pass" and ov["all_quality_pass"] is False
    captures = {x["fixture"]: {s["stage"]: s for s in x["manifest"]["stages"]}
                for x in native["results"] if x["iteration"] == "formal-1"}
    results = []
    for arm in ov["arms"]:
        for image in arm["images"]:
            stages = image["formal"][0]["taps"]
            for name in ("ln1.0", "qkv.0", "block.0", "ln1.1"):
                shape, n = saved(captures[image["fixture"]][name], True)
                oshape, actual = saved(stages[name], False)
                assert shape == oshape
                selected = [r*shape[1]+c for r in (0, shape[0]//2, shape[0]-1) for c in range(shape[1])]
                delta = [actual[i]-n[i] for i in selected]
                row = {"policy": arm["policy"], "fixture": image["fixture"], "stage": name,
                       "selected_rows": [0, shape[0]//2, shape[0]-1], "elements": len(selected),
                       "max_abs": max(map(abs, delta)),
                       "relative_rmse": math.sqrt(math.fsum(x*x for x in delta)/math.fsum(n[i]*n[i] for i in selected)),
                       "f32_equal_elements": sum(struct.pack("<f", actual[i]) == struct.pack("<f", n[i]) for i in selected)}
                if name == "ln1.0":
                    row["bf16_bucket_mismatches"] = sum(bf16(actual[i]) != bf16(n[i]) for i in selected)
                results.append(row)
    out = {"status": "saved_selected_rows_compared", "script_sha256": sha(Path(__file__).read_bytes()),
           "native_receipt_sha256": sha(native_bytes), "ov_receipt_sha256": sha(ov_bytes), "results": results,
           "original_reference_replaced": False, "npu_accepted": False, "gate_relaxed": False,
           "claim_limit": "Prior instrumented CPU stages only; no new encoder, NPU, latency or causality acceptance."}
    path = Path(__file__).with_name("saved-stages-review.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2)
    print(json.dumps({"status": out["status"], "comparisons": len(results),
                      "ln0_buckets": [r for r in results if r["stage"] == "ln1.0"]}, indent=2))


if __name__ == "__main__":
    main()
