from pathlib import Path
import datetime
import hashlib
import json
import re
import sys

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tools"))
from hetero_compare_quality import extract_final_sse

run = Path(__file__).resolve().parent
arm = run / "arms/probe69/quality-on"
base = repo / "bench/hetero/20261009-04-hetero04-direct-quality/requests"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
matrix = json.loads((run / "matrix.json").read_text())
short = []
for prompt in matrix["quality_recipe"]["prompt_ids"][:5]:
    for i in range(3):
        path = arm / "requests" / prompt / f"formal-{i:04d}.sse.txt"
        actual = extract_final_sse(path)
        reference = extract_final_sse(base / prompt / path.name)
        raw = path.with_name(path.name.replace(".sse.txt", ".text.txt")).read_bytes()
        assert actual["valid"] and reference["valid"]
        assert actual["actual_generated_token_ids"] == reference["actual_generated_token_ids"]
        assert raw == actual["visible_text_from_sse"].encode("utf-8")
        short.append({"prompt_id": prompt, "repeat": i, "sse_sha256": sha(path),
                      "text_sha256": hashlib.sha256(raw).hexdigest(),
                      "ids_sha256": actual["actual_generated_token_ids_sha256"], "h4_ids_equal": True})
long = []
for stem in ["warmup-0000", "formal-0000", "formal-0001", "formal-0002"]:
    path = arm / "requests/long_1k" / (stem + ".sse.txt")
    actual = extract_final_sse(path)
    raw = path.with_name(stem + ".text.txt").read_bytes()
    assert actual["actual_generated_token_ids"] == [0] * 128
    assert actual["finish_reason"] == "length" and actual["include_stop"] and actual["sse_done_marker"]
    assert raw == actual["visible_text_from_sse"].encode("utf-8") == b"!" * 128
    assert actual["errors"] == ["final finish reason is not stop: length"]
    long.append({"stem": stem, "sse_sha256": sha(path), "text_sha256": hashlib.sha256(raw).hexdigest(),
                 "actual_id_count": 128, "actual_unique_ids": [0],
                 "ids_sha256": actual["actual_generated_token_ids_sha256"], "finish_reason": "length",
                 "wire_artifact_readable_with_quality_failure": True})
log = (arm / "logs/engine.log").read_text(encoding="utf-8")
probe = [json.loads(line.split("strata pf_nonfinite: ", 1)[1]) for line in log.splitlines()
         if line.startswith("strata pf_nonfinite: ")]
begins = [row for row in probe if row["event"] == "begin"]
checks = [row for row in probe if row["event"] == "checkpoint"]
bad = [row for row in checks if row["first_bad"] is not None]
assert len(bad) == 4
expected = {"ordinal": 277, "stage": "moe_output", "chunk": 0, "pos0": 0, "layer": 45, "half": 1,
            "rows": 1035, "width": 2560, "index": 394240, "row": 154, "column": 0,
            "nan": 2560, "positive_inf": 0, "negative_inf": 0}
for row in bad:
    assert row["first_bad"] == expected
heads = []
pattern = (r"position (\d+) row (\d+) selected (\d+); vocab (\d+) finite (\d+) nan (\d+) "
           r"posinf (\d+) neginf (\d+) zeros (\d+); finite_argmax (-?\d+)")
for line in log.splitlines():
    if line.startswith("strata zero-logits diag:"):
        match = re.search(pattern, line)
        assert match, line
        values = list(map(int, match.groups()))
        assert values[2:] == [0, 248320, 0, 248320, 0, 0, 0, -1]
        heads.append({"position": values[0], "row": values[1], "selected": 0, "vocab": 248320,
                      "finite": 0, "nan": 248320, "finite_argmax": -1})
assert heads
all_rows = [json.loads(line) for line in (arm / "resource/samples.jsonl").read_text().splitlines() if line]
resource = [row for row in all_rows if row["record_type"] == "resource_sample"]
assert len(resource) == 780 and len(all_rows) == 781
for row in resource:
    assert not row["errors"] and not row["alerts"]
    assert row["ram_gate"]["status"] == row["commit_gate"]["status"] == "pass"
    assert row["memory"]["physical_available_bytes"] >= 12 * 1024**3
    assert row["memory"]["commit_available_bytes"] >= 4 * 1024**3
post = json.loads((arm / "cleanup/post-unload-status.json").read_text())
assert post["loaded"] is False and post["activity"]["in_flight"] == 0 and post["activity"]["requests"] == 24
cleanup = json.loads((arm / "cleanup/final.json").read_text())
assert cleanup["engine_server_sampler_terminal"] and cleanup["listener8081_released"]
diag = json.loads((arm / "quality/F-long_1k-diagnostics.json").read_text())
assert diag["status"] == "failclosed_fatal_diagnostic" and not diag["reader_error"] and diag["partial_line_bytes"] == 0
assert sum(row["classification"] == "fatal_nonfinite" for row in diag["records"]) == 4
output = {"schema_version": 1, "status": "root_first_instrumented_nonfinite_boundary_verified_model_quality_failed",
          "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "engine_sha256": sha(Path(matrix["hetero"]["binary_path"])),
          "source_head": matrix["hetero"]["source_commit"], "short_formals": short, "long_warm_and_formals": long,
          "probe": {"begin_records": len(begins), "checkpoint_records": len(checks),
                    "finite_checkpoints": sum(row["first_bad"] is None for row in checks), "bad_checkpoints": bad,
                    "scope": "earliest scanned boundary; unscanned intermediates may fail earlier; probe work may perturb races"},
          "zero_head_readback": {"rows": len(heads), "positions": sorted({row["position"] for row in heads}),
                                 "all_rows_vocab_nan": True, "finite_argmax": -1},
          "resource": {"samples": len(resource), "terminal_rows": 1, "all_pass": True,
                       "physical_min_gib": min(row["memory"]["physical_available_bytes"] for row in resource) / 1024**3,
                       "commit_min_gib": min(row["memory"]["commit_available_bytes"] for row in resource) / 1024**3},
          "cleanup": {"exit_code": 0, "loaded_after_unload": False, "inflight_after_unload": 0,
                      "owners_and_8081_released": True}, "later_long_and_performance": "NOT RUN",
          "next": "instrument layer45 MoE routing/shared/routed buffers before combine; no unchanged costly retry",
          "math_fix_established": False, "performance_claim": False,
          "initial_review_attempts": ["module import failed before output; fixed by explicit tools import path",
                                      "guessed engine_stats field absent; rebound actual activity.in_flight and requests=24; no device action"]}
with (run / "root-model-output-review.json").open("x", encoding="utf-8") as stream:
    json.dump(output, stream, indent=2)
    stream.write("\n")
print(json.dumps({"status": output["status"], "short_formals": len(short), "long_outputs": len(long),
                  "first_bad": expected, "head_rows": len(heads), "resource": output["resource"],
                  "cleanup": output["cleanup"]}))
