"""Independent root raw-output/numerical/cleanup review after run61 terminal."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import numpy as np
import psutil
from tools import hetero_xpu_worker as worker

BASE = Path(__file__).resolve().parent
def digest(data): return hashlib.sha256(data).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
contract = load(BASE / "compare-contract.json")
receipt_path = BASE / "receipts/compare-receipt.json"
receipt = load(receipt_path)
assert digest(receipt_path.read_bytes()) == "930b3d03400aa70f0681d92743864260b1ee350f85144eeda425219bd20fba08"
assert receipt["status"] == "complete" and receipt.get("failure") is None
assert receipt["source_unchanged_during_run"] is True
assert receipt["resource_gate_all_samples_passed"] is True
assert receipt["physical_available_min_bytes"] >= 12 * 2**30
assert receipt["commit_available_min_bytes"] >= 4 * 2**30
baseline = load(contract["paired_only"]["baseline_receipt_path"])
cases = {c["rows"]: c for c in contract["row_cases"]}
gold = {}
for rows, case in cases.items():
    data = Path(case["native_cpu_output_path"]).read_bytes()
    assert digest(data) == case["native_cpu_output_sha256"]
    gold[rows] = (data, np.frombuffer(data, dtype="<f4").reshape(rows, 2560))
seen = set(); checked = []; table = []
for arm in receipt["arms"]:
    assert arm["status"] == "complete" and len(arm["rows"]) == 9
    for cleanup in arm["cleanup"].values():
        assert cleanup["terminal"] is True and cleanup["remaining_owned_descendants"] == []
    solo_cpu = next(a for a in baseline["arms"] if a["id"] == arm["backend"] + "-solo")
    solo_arc = next(a for a in baseline["arms"] if a["id"] == "arc-gpu0-f32-accuracy-solo")
    for row in arm["rows"]:
        rows = row["rows"]
        assert row["status"] == "complete" and len(row["formals"]) == 5
        pairs = [row["warmup"], *row["formals"]]
        for pair in pairs:
            for side in ("cpu", "arc"):
                trial = pair[side]; raw = trial["raw_output_storage"]
                assert trial["comparison_pass"] is True and raw["complete"] is True
                path = Path(raw["path"])
                assert path.is_relative_to(Path(contract["output"]["private_output_root"]))
                assert path not in seen
                seen.add(path)
                data = path.read_bytes()
                assert len(data) == rows * 2560 * 4 == raw["bytes"]
                assert digest(data) == raw["sha256"] == trial["output_sha256"]
                arr = np.frombuffer(data, dtype="<f4").reshape(rows, 2560)
                assert np.isfinite(arr).all()
                if side == "cpu":
                    assert data == gold[rows][0]
                    quality = {"bitwise_gold_match": True}
                else:
                    quality = worker.quality_report(arr, gold[rows][1], contract["quality_tolerances"])
                    assert quality["pass"] is True
                    assert quality == trial["numerical_quality_vs_native_cpu"]
                checked.append({"arm": arm["id"], "rows": rows, "label": pair["label"],
                    "side": side, "path": str(path), "bytes": len(data), "sha256": digest(data), "quality": quality})
        for side, solo in (("cpu", solo_cpu), ("arc", solo_arc)):
            solo_row = next(r for r in solo["rows"] if r["rows"] == rows)
            solo_ms = statistics.median(t["caller_boundary_rtt_ns"] for t in solo_row["formals"]) / 1e6
            joint_ms = statistics.median(p[side]["caller_boundary_rtt_ns"] for p in row["formals"]) / 1e6
            table.append({"backend": arm["backend"], "rows": rows, "side": side,
                "solo_median_ms": solo_ms, "joint_median_ms": joint_ms,
                "delta_pct": (joint_ms / solo_ms - 1) * 100,
                "comparison_scope": "cross-run baseline; not interleaved"})
assert len(seen) == len(checked) == 216
terminal = load(BASE / "run61-actual-01.terminal.json")
review = {"status": "root_raw_and_numerical_review_pass_joint_only", "receipt_sha256": digest(receipt_path.read_bytes()),
    "checked_response_files": 216, "checked_bytes": sum(p["bytes"] for p in checked),
    "cpu_bitwise_outputs": 108, "independent_arc_numerical_outputs": 108,
    "quality_tolerances": contract["quality_tolerances"], "model_gain_unrun": True,
    "cross_run_not_interleaved": True, "memory_bandwidth_unmeasured": True,
    "terminal_receipt_sha256": digest((BASE / "run61-actual-01.terminal.json").read_bytes()),
    "resource_samples": receipt["resource_gate_samples"], "points": checked, "median_comparison": table}
with (BASE / "receipts/root-output-review.json").open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(review, stream, indent=2, allow_nan=False); stream.write("\n")
with (BASE / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
print(json.dumps({"status": review["status"], "outputs": len(checked), "bytes": review["checked_bytes"],
    "rows1_256": [p for p in table if p["rows"] in (1, 256)]}, indent=2))
