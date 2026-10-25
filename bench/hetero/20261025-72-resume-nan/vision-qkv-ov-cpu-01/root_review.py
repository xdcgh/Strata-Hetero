"""Independently rehash and recompute the retained tiny CPU QKV comparison.

Uses only stdlib; never imports OpenVINO, NumPy or starts a worker/device.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent
RECEIPT = BASE / "receipt.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array(record):
    path = Path(record["path"])
    assert path.parent.resolve() == BASE
    data = path.read_bytes()
    assert len(data) == record["file_bytes"] and sha(data) == record["file_sha256"]
    assert data[:8] == b"\x93NUMPY\x01\x00", path
    hsize = struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10:10+hsize].decode("latin1"))
    assert header == {"descr": "<f4", "fortran_order": False, "shape": (3, 7)}, header
    raw = data[10+hsize:]
    assert len(raw) == 84 and sha(raw) == record["decoded_sha256"], path
    assert record["shape"] == [3, 7] and record["dtype"] == "float32"
    result = struct.unpack("<21f", raw)
    assert all(math.isfinite(v) for v in result), path
    return result


def metrics(candidate, native):
    squared = math.fsum((a-b)**2 for a, b in zip(candidate, native))
    norm = math.fsum(b*b for b in native)
    return {"max_abs": max(abs(a-b) for a, b in zip(candidate, native)),
            "relative_rmse": math.sqrt(squared/norm),
            "bit_equal_elements": sum(a == b for a, b in zip(candidate, native)),
            "elements": len(native)}


def check_metrics(actual, recorded):
    for key in ("elements", "bit_equal_elements"):
        assert actual[key] == recorded[key], (actual, recorded)
    for key in ("max_abs", "relative_rmse"):
        assert math.isclose(actual[key], recorded[key], rel_tol=1e-10, abs_tol=1e-15), (actual, recorded)


def main():
    receipt_bytes = RECEIPT.read_bytes()
    r = json.loads(receipt_bytes)
    terminal = json.loads((BASE / "terminal-verification.json").read_text())
    assert terminal["receipt_sha256"] == sha(receipt_bytes)
    assert terminal["exit_code"] == 0 and terminal["pid_present"] is False
    assert terminal["exact_python_processes"] == []
    assert r["device"]["requested"] == "CPU"
    assert r["device"]["settings"]["INFERENCE_NUM_THREADS"] == 1
    assert r["npu_accepted"] is r["encoder_run"] is r["gate_relaxed"] is r["original_reference_replaced"] is False
    assert len(r["memory_gates"]) > 0
    for gate in r["memory_gates"]:
        assert gate["errors"] == []
        assert gate["memory"]["physical_available_bytes"] >= 12*1024**3
        assert gate["memory"]["commit_available_bytes"] >= 4*1024**3
    verified_paths = set()
    def load(record):
        result = array(record)
        verified_paths.add(record["path"])
        return result
    summaries = []
    for arm in r["arms"]:
        assert arm["execution_devices"] == ["CPU"]
        assert len(arm["images"]) == 4
        summary = {"policy": arm["policy"], "formal_groups": 0, "selected_values_per_formal": 0,
                   "equal_values_per_formal": 0, "max_abs": 0, "image_relative_rmse": []}
        for image in arm["images"]:
            native, source = load(image["native"]), load(image["source"])
            check_metrics(metrics(source, native), image["source_vs_native"])
            assert [it["iteration"] for it in image["iterations"]] == ["warmup", "formal-1", "formal-2", "formal-3"]
            formal = []
            for iteration in image["iterations"]:
                out = load(iteration["output"])
                actual = metrics(out, native)
                check_metrics(actual, iteration["vs_native"])
                check_metrics(metrics(out, source), iteration["vs_source"])
                if iteration["iteration"] != "warmup":
                    formal.append((iteration["output"]["file_sha256"], actual))
                    summary["formal_groups"] += 1
            assert len(set(digest for digest, _ in formal)) == 1
            assert image["formal_hashes_identical"] is True
            one = formal[0][1]
            summary["max_abs"] = max(summary["max_abs"], one["max_abs"])
            summary["selected_values_per_formal"] += one["elements"]
            summary["equal_values_per_formal"] += one["bit_equal_elements"]
            summary["image_relative_rmse"].append(one["relative_rmse"])
        summaries.append(summary)
    assert len(verified_paths) == 48
    output = {"status": "retained_files_and_metrics_independently_verified", "receipt_sha256": sha(receipt_bytes),
              "review_script_sha256": sha(Path(__file__).read_bytes()), "unique_npy_files": len(verified_paths),
              "memory_gate_count": len(r["memory_gates"]), "summaries": summaries,
              "claim_limit": "Tiny CPU 3x1152 by 7x1152 QKV only. No full encoder, NPU, latency or end-to-end acceptance."}
    (BASE / "root-review.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
