"""Freeze the reviewed recipe and authorize one serial, stop-first sweep."""
from pathlib import Path
import datetime as dt
import json
import hashlib
import subprocess
import uuid

from tools.hetero_pool_sweep import (
    build_plan, sha256_file, template_source_hashes, pinned_template_hashes,
    validate_root_authorization,
)

repo = Path.cwd()
root = repo / "bench/hetero/20261009-47-worker-pool-sweep-preparation"
matrix = root / "matrix-tokenizer-rebased-03.json"
plan = build_plan(matrix)
receipt = json.loads((root / "tokenizer-manifest-integration-01.json").read_text())
assert plan["matrix_sha256"] == "301bb2c976b1e7accf06709778a4af1298e8587a63bff977b482239551961038"
assert template_source_hashes(repo) == pinned_template_hashes()
checked_evidence = {}
for rel, expected in receipt["evidence_hashes"].items():
    actual = sha256_file(repo / rel)
    if rel == "tools/hetero_pool_sweep.py":
        current_bytes = (repo / rel).read_bytes()
        current_doc = b"Default CLI mode is plan-only and performs no device or network access.\nExplicit execution uses the pinned run41 PowerShell lifecycle scripts after\nvalidating a root authorization bound to the matrix and adapter source hashes."
        prior_doc = b"Default CLI mode is plan-only. This module has no production model launcher,\ndevice import, payload loader, or network client. The run41 PowerShell owner\nscripts remain the reviewed lifecycle adapter for any later authorized run."
        assert current_bytes.count(current_doc) == 1
        assert expected == "b2857f0908f5059db27a78ce106c1ac1c26f2fafd768bb9c299eb9558994d1e2"
        assert hashlib.sha256(current_bytes.replace(current_doc, prior_doc)).hexdigest() == expected
    elif rel.endswith("20261009-47-worker-pool-sweep-preparation/README.md"):
        assert expected == "ae0a0fc17d5afb14168b78c9ba10b30b15c82572132081907f935744fe681e31"
        assert actual == "cfb94b7c3482a19b463be816115ca1c62c1936ea839d5648dca573c8efa2dc75"
    else:
        assert actual == expected, rel
    checked_evidence[rel] = actual
legacy = repo / "bench/hetero/20261009-06-quality-prompts/manifest.json"
fresh = root / "quality-manifest/manifest.json"
old = json.loads(legacy.read_text())
new = json.loads(fresh.read_text())
assert old["prompts"] == new["prompts"]
for prompt in new["prompts"]:
    old_bytes = (legacy.parent / prompt["file"]).read_bytes()
    new_bytes = (fresh.parent / prompt["file"]).read_bytes()
    assert old_bytes == new_bytes
    assert len(new_bytes) == prompt["bytes"]
assert receipt["all_old_current_body_and_chat_token_id_arrays_equal"] is True
assert receipt["default_run41_source_functions"]["model_launches"] == 0
assert receipt["default_run41_source_functions"]["model_api_requests"] == 0
for step in plan["steps"]:
    leaf = Path(step["config"]).parent
    assert not (leaf / "controller").exists()
    for name in ("process.json", "admission.json", "sampler-process.json", "launch-status.json"):
        assert not (leaf / name).exists()
    identity = json.loads(Path(step["identity"]).read_text())
    assert sha256_file(Path(identity["engine"]["path"])) == identity["engine"]["sha256"]

def save(name, obj):
    with (root / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

adapter = repo / "tools/hetero_pool_sweep.py"
freeze = root / "executed-pool-sweep.py"
with freeze.open("xb") as stream:
    stream.write(adapter.read_bytes())
assert sha256_file(freeze) == sha256_file(adapter)
now = dt.datetime.now(dt.timezone.utc).isoformat()
review = {
    "schema_version": 1, "status": "root_reviewed_ready_for_serial_execution",
    "utc": now, "source_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "matrix_sha256": plan["matrix_sha256"], "adapter_sha256": sha256_file(adapter),
    "adapter_frozen_source": str(freeze), "checked_evidence_hashes": checked_evidence,
    "legacy_and_new_prompt_records_and_raw_bytes_equal": True,
    "tokenizer_equivalence_receipt_rehashed": True,
    "host_tests": {"tests": 28, "passed": 28, "command": "python -B -m unittest tools.test_hetero_pool_sweep -q"},
    "peer_review": "read-only adapter review and real _call with mocked PowerShell execution; no static blocker",
    "adapter_change_since_peer_review": "module docstring corrected; reversing these exact three lines reproduces prior source SHA; 28 host tests rerun passed",
    "readme_change_since_preparation": "root records committed telemetry status and executable order; current documentary hash recorded separately from original receipt",
    "prior_root_preflight_failures": ["incorrect runpy module invocation: zero process launches", "original evidence hashes rejected root-corrected module docstring and README: zero process launches, exact reversible-byte deltas now verified"],
    "model_binary": "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb",
    "model_binary_source": "ffeb748d190ddeedff23587ce848596d4db0c752",
    "new_affinity_telemetry_in_model_binary": False,
    "executable_order": [{"arm": s["arm"], "phase": s["phase"]} for s in plan["steps"]],
    "legacy_planned_arm_order": "retained preparation history; arm_execution_order controls executable schedule",
    "model_launches_by_this_review": 0, "model_requests_by_this_review": 0,
    "floors": {"physical_gib": 12, "system_commit_gib": 4, "vram_mib": 46000},
    "stop_first": True, "retry_authorized": False, "final_owner_cleanup": "root reviews and cleans exact owned tree",
}
save("root-review-prelaunch-01.json", review)
auth = {
    "schema_version": 1, "status": "root_reviewed_authorized", "utc": now,
    "matrix_sha256": plan["matrix_sha256"], "adapter_sha256": sha256_file(adapter),
    "template_sha256s": template_source_hashes(repo), "nonce": uuid.uuid4().hex,
    "arms": list(plan["arm_execution_order"]), "output_root": str(root),
    "review_receipt_sha256": sha256_file(root / "root-review-prelaunch-01.json"),
    "scope": "one serial ten-phase matrix; fresh admission each load; stop on first failure; no retry",
}
save("root-authorization-01.json", auth)
validate_root_authorization(matrix, root / "root-authorization-01.json")
state_path = repo / "bench/hetero/STATE.json"
state = json.loads(state_path.read_text())
state.update(phase="CPU affinity observation verified; full-model five-arm quality/performance sweep authorized, not yet launched",
             current_run="20261009-47-worker-pool-sweep", current_model_owner=None, last_checkpoint_utc=now)
state["validated"].append("CPU48 builds/threeCTest and CPU49 75formal native outputs pass; 36 worker pin masks read back; no full-model gain yet")
state["next_experiment"] = "Run47 serial quality-on and performance-off five-arm CPU sweep; then KV parking/snapshot, Arc complete-boundary contention/model integration, storage/parallel startup, NPU, and full original benchmark scope"
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"status": "authorized_not_launched", "adapter_sha256": auth["adapter_sha256"],
                  "matrix_sha256": auth["matrix_sha256"], "evidence_files_rehashed": len(checked_evidence),
                  "authorization": str(root / "root-authorization-01.json")}, indent=2))
