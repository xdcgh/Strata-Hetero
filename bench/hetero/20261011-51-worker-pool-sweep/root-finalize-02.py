"""Bind completed dry validation and fresh file identities to a new matrix."""
from pathlib import Path
import json
import datetime as dt
import uuid
from tools import hetero_pool_sweep as s

repo = Path.cwd()
root = repo / "bench/hetero/20261011-51-worker-pool-sweep"
old = root / "matrix-tokenizer-rebased-04.json"
value = json.loads(old.read_text())
fresh_hash = "a64cb90f8d7302980453d9cf97bef50d63ae2f48d5813b6c2a042a27d99e2568"
for arm in value["arms"]:
    for phase in ("quality_on", "performance_off"):
        entry = arm[phase]
        path = Path(entry["path"]) / "fresh-identities.json"
        assert s.sha256_file(path) == fresh_hash
        assert not (path.parent / "controller").exists()
        entry["fresh_identities_sha256"] = fresh_hash
value["predecessor_matrix"] = {"path": str(old), "sha256": s.sha256_file(old)}
matrix = root / "matrix-tokenizer-rebased-05.json"
with matrix.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
plan = s.build_plan(matrix)
receipt = root / "receipts/run51-dry-validation-03/supplement-receipt.json"
assert s.sha256_file(receipt) == "b069a8695563f831c5ce29a94267e8415d37e2653b37298289e6a00157cfea17"
snapshot = root / "executed-sources-02"
snapshot.mkdir(exist_ok=False)
hashes = {}
for name in ("hetero_pool_sweep.py", "hetero_admission.py", "hetero_resources.py"):
    path = repo / "tools" / name
    with (snapshot / name).open("xb") as stream: stream.write(path.read_bytes())
    hashes[name] = s.sha256_file(path)
now = dt.datetime.now(dt.timezone.utc).isoformat()
review = {"status": "root_reviewed_full_dry_validation", "utc": now, "source_hashes": hashes,
          "matrix_sha256": plan["matrix_sha256"], "dry_functions_passed": 30,
          "host_tests": {"admission": 18, "pool_sweep": 30, "failures_final": 0},
          "supplement_receipt_sha256": s.sha256_file(receipt), "fresh_identities_sha256": fresh_hash,
          "previous_probe": "stopped at missing prepared file before WSL/admission/model; preserved",
          "scope": "one admission-only probe after corrected metadata, then one serial ten-phase sweep; stop first",
          "model_binary": "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb",
          "model_source": "ffeb748d190ddeedff23587ce848596d4db0c752", "new_worker_telemetry_in_model_binary": False}
with (root / "root-review-02.json").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(review, indent=2) + "\n")
auth = {"status": "root_reviewed_authorized", "utc": now,
        "matrix_sha256": plan["matrix_sha256"], "adapter_sha256": hashes["hetero_pool_sweep.py"],
        "gate_source_sha256s": s.gate_source_hashes(repo), "template_sha256s": s.template_source_hashes(repo),
        "nonce": uuid.uuid4().hex, "arms": plan["arm_execution_order"], "output_root": str(root.resolve()),
        "admission_probe_id": "controller-admission-probe-02", "review_sha256": s.sha256_file(root / "root-review-02.json"),
        "scope": review["scope"]}
with (root / "root-authorization-02.json").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(auth, indent=2) + "\n")
s.validate_root_authorization(matrix, root / "root-authorization-02.json")
print(json.dumps({"status": "authorized_not_launched", "matrix": str(matrix),
                  "matrix_sha256": plan["matrix_sha256"], "source_hashes": hashes}, indent=2))
