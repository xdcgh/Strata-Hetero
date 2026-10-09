from pathlib import Path
import datetime
import hashlib
import json
import subprocess

repo = Path(r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero")
base = repo / "bench" / "hetero"
run = base / "20261009-47-worker-pool-sweep-preparation"
matrix_path = run / "matrix.json"
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if (run / "README.md").exists() or (run / "CONTROLLER-CHANGE-PROPOSAL.json").exists():
    raise FileExistsError("run47 review artifact already exists")

expected_arms = {("w10-all", 10, "all"), ("w04-all", 4, "all"),
                 ("w06-all", 6, "all"), ("w16-all", 16, "all"),
                 ("w06-auto", 6, "auto")}
if len(matrix.get("arms", [])) != 5:
    raise ValueError("worker-pool matrix must describe five arms")
seen = set()
leaf_records = []
for arm in matrix["arms"]:
    seen.add((arm["arm"], arm["workers"], arm["affinity"]))
    for phase, key, capture in (("quality-on", "quality_on", True),
                                ("performance-off", "performance_off", False)):
        leaf = Path(arm[key]["path"])
        cfg = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
        identity = json.loads((leaf / "identity.json").read_text(encoding="utf-8"))
        provenance = json.loads((leaf / "provenance.json").read_text(encoding="utf-8"))
        argv = cfg["args"]
        if sha(leaf / "config.json") != arm[key]["config_sha256"]:
            raise ValueError(f"config hash mismatch: {leaf}")
        if argv[argv.index("--pool-workers") + 1] != str(arm["workers"]):
            raise ValueError(f"worker count mismatch: {leaf}")
        if argv[argv.index("--pool-affinity") + 1] != arm["affinity"]:
            raise ValueError(f"affinity mismatch: {leaf}")
        if cfg["hetero_capture_token_ids"] is not capture:
            raise ValueError(f"capture phase mismatch: {leaf}")
        if cfg["env"].get("STRATA_PREFILL_CPU_SHARE") != "0" or cfg["env"].get("STRATA_STAGE_PIN") != "0":
            raise ValueError(f"v1.41 opt-out env mismatch: {leaf}")
        if argv[argv.index("--resident-budget-gib") + 1] != "20" or \
           argv[argv.index("--max-context") + 1] != "32768" or \
           argv[argv.index("--kv") + 1] != "int8" or \
           argv[argv.index("--ple-io") + 1] != "direct":
            raise ValueError(f"shared controls mismatch: {leaf}")
        if not argv[argv.index("--ple-gguf") + 1].startswith("F:\\Strata-data\\models\\"):
            raise ValueError(f"PLE file is not the original F shard: {leaf}")
        if identity["engine"]["sha256"] != matrix["candidate_binary_sha256"] or \
           provenance["engine_cpp_sha"] != matrix["candidate_source_sha"]:
            raise ValueError(f"candidate build binding mismatch: {leaf}")
        for folder in ("logs", "quality", "requests", "resource"):
            path = leaf / folder
            if not path.is_dir() or any(path.iterdir()):
                raise ValueError(f"planned output directory is missing or not empty: {path}")
        leaf_records.append({"arm": arm["arm"], "phase": phase,
                             "config_sha256": sha(leaf / "config.json"),
                             "identity_sha256": sha(leaf / "identity.json"),
                             "provenance_sha256": sha(leaf / "provenance.json")})
if seen != expected_arms or len(leaf_records) != 10:
    raise ValueError("worker-pool arm/count matrix mismatch")

pair_path = base / "20261009-39-upstream41-build" / "pair" / "cuda-build-pair-41.json"
pair = json.loads(pair_path.read_text(encoding="utf-8"))
candidate_root = Path(r"E:\Strata-Hetero-data\source\hetero-0-1-41")
candidate_paths = ("src/program/generate.cpp", "src/kernels/cpu/pool.cpp",
                   "include/strata/kernels/cpu/pool.hpp")
candidate_blobs = {p: subprocess.check_output(["git", "-C", str(candidate_root),
                 "rev-parse", "HEAD:" + p], text=True).strip() for p in candidate_paths}
main_blobs = matrix.get("source_blobs", {})
current_status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).splitlines()
root_source_diffs = [x for x in current_status if x.startswith(" M ")]
matrix["candidate_build_source_tree"] = pair["builds"][1]["source_tree"]
matrix["candidate_build_source_clean_before_after"] = bool(pair["builds"][1]["source_clean_before"] and pair["builds"][1]["source_clean_after"])
matrix["candidate_build_source_blobs"] = candidate_blobs
matrix["merged_head_committed_blobs"] = main_blobs
matrix["root_worktree_tracked_source_changes_not_in_candidate_binary"] = root_source_diffs
matrix["binary_scope_note"] = (
    "All planned arms bind the already validated 6bbc binary built from clean ffeb748. "
    "Root's current uncommitted worker-affinity telemetry and Win32 readback changes are not compiled into it. "
    "Do not substitute current HEAD for this binary's source identity."
)
matrix["telemetry48_note"] = (
    "A separate CPU-only telemetry48 build is planned by root; this preparation does not bind or build its future binary. "
    "Re-review source/binary/config hashes if root later chooses to substitute it."
)
matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

proposal = {
    "schema_version": 1,
    "status": "root_review_proposal_only",
    "run_id": matrix["run_id"],
    "single_parameterized_controller_not_per_arm_scripts": True,
    "reuse": [
        "bench/hetero/20261009-41-hetero41-direct-quality/launch-owned.ps1",
        "bench/hetero/20261009-41-hetero41-direct-quality/startup-watch.ps1",
        "bench/hetero/20261009-41-hetero41-direct-quality/run-quality-after-ready.ps1",
        "tools/hetero_bench.py", "tools/hetero_resources.py", "tools/hetero_admission.py"
    ],
    "controller_changes": {
        "launch": "Parameterize the run41 launch guard with a run directory/RunId; keep its exact fresh admission, source/binary/bridge and F asset identity checks, local env clearing, listener/process ownership and sampler launch.",
        "quality": "Parameterize the run41 fixed nine-prompt controller; preserve strict H4 actual-token-ID comparison and caps for the quality-on configuration.",
        "performance": "For performance-off configs, call hetero_bench sequentially for the six matrix scenarios with warmup 1/formal 3, capture off, and allow-capped-performance; check exact child ownership, fresh resource sample and timing metadata per request.",
        "coordinator": "One coordinator reads matrix.json and executes each RunId in phase order; per-load fresh 12/4/46k admission and topology snapshot; stop at first failure, retain partial evidence, no retry/restart, leave last server live for root review."
    },
    "per_arm_scripts": False,
    "implementation_or_launch_in_this_task": False
}
proposal_path = run / "CONTROLLER-CHANGE-PROPOSAL.json"
proposal_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

readme = f'''# CPU worker-pool full-model comparison preparation

Status: `prepared_for_root_review_not_launched`; five arms and ten phase configurations. No model/API/CPU benchmark, build, or source edit was performed by this preparation.

Arms: `w10-all` baseline, `w04-all`, `w06-all`, `w16-all`, and `w06-auto`. The generate CLI accepts `all`, `auto`, and `p-cores`; this matrix uses only requested modes `all`/`auto`. Harness-only `none` is excluded. Worker count is N background pool workers plus the participating caller under the reviewed host-works default.

All arms bind the same validated Hetero41 binary SHA `{matrix['candidate_binary_sha256']}` from clean source `{matrix['candidate_source_sha']}`, the v1.41 shared capture bridge, original F native-dense shards/direct PLE F shard2, 20 GiB expert budget, 32K context, int8 KV, same pack/MTP/tokenizer/sampling/cache and environment. `STRATA_PREFILL_CPU_SHARE=0` and `STRATA_STAGE_PIN=0` are explicit. Between arms only pool workers and affinity mode differ. Each quality-on/performance-off pair also differs by capture flag and phase log.

Quality-on enables token-ID capture: nine fixed prompts, warmup 1 and formal 3 each; strict quality plus H4 token-ID comparison for every formal. Performance-off disables capture: code prompt output caps 256/512/1024 and long prompts 4K/16K/32K with output cap 128, warmup 1 and formal 3 through `tools/hetero_bench.py --allow-capped-performance`. Keep the two timing populations separate. Four matched prompt/cap cases permit a labeled capture-overhead diagnostic.

Each of ten model loads requires a fresh 12 GiB RAM / 4 GiB commit / 46,000 MiB VRAM admission and fresh read-only CPU topology snapshot. Keep the one-second resource sampler and exact owner PID/exe/parent/CreateTime/full-argv checks; stop on the first error, preserve partial outputs, never retry/restart. The candidate server stays live for root review at the end.

The future controller proposal reuses run41's reviewed launch, watcher and quality controls plus the generic benchmark runner, with one parameterized coordinator rather than per-arm scripts. Root review is required before implementation or launch.

Root has uncommitted affinity telemetry edits in the current main worktree. They are not in the planned binary. All arms remain pinned to the validated 6bbc/ffeb pair unless root reviews a later binary substitution.
'''
(run / "README.md").write_text(readme, encoding="utf-8")

validation = {
    "schema_version": 1, "status": "static_configs_and_identity_bindings_passed",
    "run_id": matrix["run_id"], "arms": 5, "phase_configs": 10,
    "all_phase_output_dirs_empty": True, "supported_affinity_modes_only": True,
    "all_non_pool_settings_equal": True, "capture_modes_separated": True,
    "candidate_binary_sha256": matrix["candidate_binary_sha256"],
    "candidate_source_sha": matrix["candidate_source_sha"],
    "root_worktree_source_changes_not_in_binary": root_source_diffs,
    "normalized_controls_sha256": matrix["canonical_config_hashes"]["all_modes_capture_placeholder"],
    "quality_capture_on_hash": matrix["canonical_config_hashes"]["quality_capture_on"],
    "performance_capture_off_hash": matrix["canonical_config_hashes"]["performance_capture_off"],
    "model_or_api_launched": False, "compiled": False,
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "leaves": leaf_records
}
validation_path = run / "preparation-validation.json"
validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
prep_path = run / "preparation.json"
prep = json.loads(prep_path.read_text(encoding="utf-8"))
prep.update({"status": "prepared_for_root_review_not_launched",
             "matrix_sha256": sha(matrix_path), "controller_proposal_sha256": sha(proposal_path),
             "readme_sha256": sha(run / "README.md"), "validation_sha256": sha(validation_path),
             "arm_count": 5, "phase_config_count": 10, "controller_implementation_started": False,
             "model_or_api_launched": False, "compiled": False})
prep_path.write_text(json.dumps(prep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": validation["status"], "arms": 5, "phase_configs": 10,
                  "matrix_sha256": sha(matrix_path), "proposal_sha256": sha(proposal_path),
                  "root_uncommitted_source_paths_not_in_binary": root_source_diffs}, indent=2))
