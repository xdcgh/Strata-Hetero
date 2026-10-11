"""Independent offline review of captured run63 failure, logits and resource history."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import re

import psutil

BASE = Path(__file__).resolve().parent
ARM = BASE / "arms/w10-all/quality-on"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))

assert sha(ARM / "logs/engine.log") == "878d0608bdfe4fda130dcd476e4160b1c007d45857c53cfded67d0df2254f9d3"
assert sha(ARM / "resource/samples.jsonl") == "c84084370d6298443125c132f562d4c2ba4813f4ade472314e7fbfc9539435fe"
bad_outputs = []
for label in ["warmup-0000", "formal-0000", "formal-0001", "formal-0002"]:
    base = ARM / "requests/long_1k" / label
    assert base.with_suffix(".text.txt").read_bytes() == b"!" * 128
    wire = base.with_suffix(".sse.txt").read_text(encoding="utf-8-sig")
    chunks = [json.loads(line[6:]) for line in wire.splitlines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
    ids = chunks[-1]["strata_diagnostics"]["actual_generated_token_ids"]
    assert ids == [0] * 128
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    visible = "".join(c["choices"][0].get("delta", {}).get("content", "") for c in chunks)
    assert visible == "!" * 128
    bad_outputs.append({"label": label, "actual_token_count": len(ids), "actual_ids_unique": sorted(set(ids)),
        "text_sha256": sha(base.with_suffix(".text.txt")), "sse_sha256": sha(base.with_suffix(".sse.txt"))})
lines = (ARM / "logs/engine.log").read_text(encoding="utf-8-sig").splitlines()
rx = re.compile(r"strata zero-logits diag: position (\d+) row (\d+) selected 0; vocab (\d+) finite (\d+) nan (\d+) posinf (\d+) neginf (\d+) zeros (\d+); finite_argmax (-?\d+)")
logits = []
for i, line in enumerate(lines, 1):
    match = rx.search(line)
    if match:
        values = list(map(int, match.groups()))
        position, row, vocab, finite, nan, posinf, neginf, zeros, best = values
        assert vocab == nan == 248320 and finite == posinf == neginf == zeros == 0 and best == -1
        logits.append({"line": i, "position": position, "row": row, "vocab": vocab,
            "finite": finite, "nan": nan, "finite_argmax": best, "selected_token": 0})
assert logits and len(logits) <= 80
progress = [json.loads(line) for line in (ARM / "quality/quality-progress.jsonl").read_text().splitlines()]
passed = [r["event"]["result"] for r in progress if r["event"].get("event") == "prompt_pass"]
assert len(passed) == 5
for p in passed:
    assert p["actual_sse_token_ids_valid"] is True
    assert p["actual_token_id_hashes"] == p["h4_token_id_hashes"]
    assert p["quality_statuses"] == "pass pass pass"
rows = [json.loads(line) for line in (ARM / "resource/samples.jsonl").read_text().splitlines()]
samples = [r for r in rows if r.get("record_type") == "resource_sample"]
assert len(samples) == 268
assert all(r["ram_gate"]["status"] == "pass" and r["commit_gate"]["status"] == "pass" for r in samples)
unknown = [r for r in samples if r.get("errors") or r.get("alerts")]
assert len(unknown) == 1 and unknown[0] is samples[-1]
assert "owner_process_tree_unknown" in unknown[0]["alerts"]
assert len([r for r in rows if r.get("record_type") == "sampler_stopped"]) == 1
terminal = read(ARM / "cleanup/terminal-verification-01.json")
assert terminal["cleanup_exit_code"] == 1 and terminal["cleanup_reported_success"] is False
assert terminal["post_unload_loaded"] is False and terminal["post_unload_in_flight"] == 0
assert not [c for c in psutil.net_connections("tcp") if c.status == "LISTEN" and c.laddr.port == 8081]
assert not [p.info for p in psutil.process_iter(["pid", "name", "cmdline"])
            if (p.info["name"] or "").lower().startswith("strata")]
report = {"status": "root_failure_logit_and_wire_review_pass_no_model_acceptance", "passed_prompts": 5,
    "short_formals_matching_H4": 15, "failed_prompt": "long_1k", "failed_formals": 3,
    "bad_warmups": 1, "actual_zero_id_outputs": bad_outputs, "zero_logit_rows": logits,
    "sampled_position_range": [min(r["position"] for r in logits), max(r["position"] for r in logits)],
    "all_captured_logit_rows_entirely_NaN": True, "first_nonfinite_layer_unresolved": True,
    "resource_sample_count": 268, "physical_and_commit_gates_all_pass": True,
    "minimum_physical_gib": min(r["memory"]["physical_available_bytes"] for r in samples) / 2**30,
    "minimum_commit_gib": min(r["memory"]["commit_available_bytes"] for r in samples) / 2**30,
    "owner_telemetry_failure_samples": 1, "failed_owner_sample": unknown[0],
    "aggregate_cleanup_exit_code": 1, "exact_owners_and_listener_terminal": True,
    "quality_pass": False, "performance_NOT_RUN": True, "later_prompts_NOT_RUN": ["long_4k", "long_16k", "long_30k7"],
    "later_phases_NOT_RUN": 9, "engine_log_sha256": sha(ARM / "logs/engine.log"),
    "resource_log_sha256": sha(ARM / "resource/samples.jsonl"),
    "terminal_addendum_sha256": sha(ARM / "cleanup/terminal-verification-01.json")}
with (BASE / "root-output-review.json").open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
print(json.dumps({key: report[key] for key in ("status", "passed_prompts", "failed_formals", "sampled_position_range", "minimum_physical_gib", "minimum_commit_gib", "owner_telemetry_failure_samples")}, indent=2))
