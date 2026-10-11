"""Freeze the corrected controller and authorize its admission probe and sweep."""
from pathlib import Path
import datetime as dt
import json
import uuid
from tools import hetero_pool_sweep as sweep

repo = Path.cwd()
root = repo / "bench/hetero/20261011-51-worker-pool-sweep"
matrix = root / "matrix-tokenizer-rebased-04.json"
plan = sweep.build_plan(matrix)
assert plan["matrix_sha256"] == "dce95ce74eca894a5378789df7848bb2591a22a82cd98858d559122e1ecf58db"
assert len(plan["steps"]) == 10
assert plan["steps"][0]["arm"] == plan["steps"][5]["arm"] == "w10-all"
snapshot = root / "executed-sources"
snapshot.mkdir(exist_ok=False)
hashes = {}
for name in ("hetero_pool_sweep.py", "hetero_admission.py", "hetero_resources.py"):
    source = repo / "tools" / name
    with (snapshot / name).open("xb") as stream: stream.write(source.read_bytes())
    hashes[name] = sweep.sha256_file(source)
    assert sweep.sha256_file(snapshot / name) == hashes[name]
for step in plan["steps"]:
    folder = Path(step["config"]).parent
    assert not (folder / "controller").exists()
    assert not (folder / "admission.json").exists()
    identity = json.loads(Path(step["identity"]).read_text())
    assert sweep.sha256_file(Path(identity["engine"]["path"])) == identity["engine"]["sha256"]
now = dt.datetime.now(dt.timezone.utc).isoformat()
review = {
    "status": "root_reviewed_serial_sweep_and_admission_probe", "utc": now,
    "source_hashes": hashes, "matrix_sha256": plan["matrix_sha256"],
    "host_tests": {"admission": 18, "pool_sweep": 28, "failed": 0},
    "prior_failure": "run47 blocked its own coordinator before model Popen; external wrapper also used unavailable QuoteArgument",
    "correction": "fresh exact PS launch-controller and C Python/optional E venv ancestor identities; no blanket Python or GPU exemptions",
    "new_commit_gate": "4 GiB PSAPI GetPerformanceInfo reserve, independent from 12 GiB physical RAM",
    "model_source": "ffeb748d190ddeedff23587ce848596d4db0c752",
    "model_binary_sha256": "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb",
    "telemetry_worker_source_in_candidate_binary": False,
    "no_model_or_request_in_this_review": True,
    "scope": "one zero-model admission probe, then one serial ten-phase matrix; stop first; no unreasoned retries",
}
with (root / "root-review.json").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(review, indent=2) + "\n")
auth = {
    "status": "root_reviewed_authorized", "utc": now,
    "matrix_sha256": plan["matrix_sha256"], "adapter_sha256": hashes["hetero_pool_sweep.py"],
    "template_sha256s": sweep.template_source_hashes(repo), "gate_source_sha256s": sweep.gate_source_hashes(repo),
    "nonce": uuid.uuid4().hex, "arms": plan["arm_execution_order"], "output_root": str(root.resolve()),
    "review_sha256": sweep.sha256_file(root / "root-review.json"), "scope": review["scope"],
}
with (root / "root-authorization.json").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(auth, indent=2) + "\n")
sweep.validate_root_authorization(matrix, root / "root-authorization.json")
print(json.dumps({"status": "authorized_not_launched", "source_hashes": hashes, "phases": 10}, indent=2))
