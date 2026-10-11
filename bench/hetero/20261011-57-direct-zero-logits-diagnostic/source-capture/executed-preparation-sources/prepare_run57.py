from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as sweep

RUN_ID = "20261011-57-direct-zero-logits-diagnostic"
SOURCE_ROOT = REPO / "bench" / "hetero" / "20261011-51-worker-pool-sweep"
SOURCE_MATRIX = SOURCE_ROOT / "matrix-tokenizer-rebased-05.json"
RUN = REPO / "bench" / "hetero" / RUN_ID
MATRIX = RUN / "matrix.json"
STAMP = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def sha(path: Path) -> str:
    return sweep.sha256_file(path)

def dump(path: Path, value) -> bytes:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return raw

def overwrite_json(path: Path, value) -> bytes:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw

def replace_run_strings(value):
    if isinstance(value, str):
        return value.replace(str(SOURCE_ROOT), str(RUN)).replace(
            "20261011-51-worker-pool-sweep", RUN_ID
        ).replace("20261011-51-", "20261011-57-")
    if isinstance(value, list):
        return [replace_run_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: replace_run_strings(v) for k, v in value.items()}
    return value

def validate_script(label: str, script: str, out_dir: Path) -> dict:
    ps = sweep.PWsh7
    argv = [str(ps), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            "$s=[Console]::In.ReadToEnd();$t=$null;$e=$null;[void][System.Management.Automation.Language.Parser]::ParseInput($s,[ref]$t,[ref]$e);if($e.Count){$e|ForEach-Object{[Console]::Error.WriteLine($_.Message)};exit 2};Write-Output syntax_ok"]
    script_path = out_dir / f"rendered-{label}.ps1"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    started = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    proc = subprocess.Popen(argv, cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=False,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    stdout, stderr = proc.communicate(script.encode("utf-8"))
    (out_dir / f"{label}.stdout.bin").write_bytes(stdout)
    (out_dir / f"{label}.stderr.bin").write_bytes(stderr)
    result = {
        "action": "validate_only_parse_no_script_evaluation",
        "template": label,
        "status": "pass" if proc.returncode == 0 and b"syntax_ok" in stdout else "fail",
        "process": {
            "pid": proc.pid,
            "parent_pid": os.getpid(),
            "executable": str(ps),
            "argv": argv,
            "started_utc": started,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "exit_code": proc.returncode,
        },
        "rendered_script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "rendered_script_bytes": len(script.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_path": str(out_dir / f"{label}.stdout.bin"),
        "stderr_path": str(out_dir / f"{label}.stderr.bin"),
        "script_path": str(script_path),
        "model_or_lifecycle_script_executed": False,
    }
    if result["status"] != "pass":
        raise RuntimeError(f"{label} parse-only validation failed; raw output retained")
    return result

if RUN.exists() and {p.name for p in RUN.iterdir()} not in (
    {Path(__file__).name}, {Path(__file__).name, "draft-01"}
):
    raise SystemExit(f"refusing to overwrite nonempty run directory: {RUN}")
if not SOURCE_MATRIX.is_file():
    raise SystemExit(f"missing run51 v05 source matrix: {SOURCE_MATRIX}")
RUN.mkdir(parents=True, exist_ok=True)

source_matrix_raw = SOURCE_MATRIX.read_bytes()
source_matrix = json.loads(source_matrix_raw)
if source_matrix.get("candidate_binary_sha256") != "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb":
    raise SystemExit("run51 v05 binary pin differs from the requested validated binary")
if source_matrix.get("candidate_source_sha") != "ffeb748d190ddeedff23587ce848596d4db0c752":
    raise SystemExit("run51 v05 source pin differs from the requested validated source")
if source_matrix.get("fixed_controls", {}).get("spec") != 4:
    raise SystemExit("run51 v05 MTP/spec control differs")
if source_matrix.get("fixed_controls", {}).get("prompt_cache") != 0:
    raise SystemExit("run51 v05 prompt-cache control differs")

# Copy frozen text/protocol assets only. Never copy output, server, admission,
# process, resource, or model/pack runtime artifacts.
for rel in ("quality-manifest", "tokenizer-source-01"):
    shutil.copytree(SOURCE_ROOT / rel, RUN / rel)
for rel in ("tokenizer-equivalence.json", "tokenizer-manifest-provenance.json"):
    shutil.copy2(SOURCE_ROOT / rel, RUN / rel)
(RUN / "prompts").mkdir()
for rel in ("prompts/sustained_decode.prompt.txt", "prompts/sustained_decode.prompt.json"):
    shutil.copy2(SOURCE_ROOT / rel, RUN / rel)

# Snapshot fixed controller templates and the exact engine-side diagnostic source.
capture = RUN / "source-capture"
template_map = {
    "launch": Path("bench/hetero/20261009-41-hetero41-direct-quality/launch-owned.ps1"),
    "watch": Path("bench/hetero/20261009-41-hetero41-direct-quality/startup-watch.ps1"),
    "quality": Path("bench/hetero/20261009-41-hetero41-direct-quality/run-quality-after-ready.ps1"),
    "cleanup": Path("bench/hetero/20261009-41-hetero41-direct-quality/cleanup-owned-model.ps1"),
    "jsonl_reader": Path("tools/hetero_jsonl_tail.ps1"),
}
captured_templates = {}
for name, rel in template_map.items():
    src = REPO / rel
    dest = capture / "run41-templates" / rel.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    captured_templates[name] = {
        "source_path": str(src),
        "captured_path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": sha(dest),
        "expected_sha256": sweep.pinned_template_hashes()[name],
        "matches_pinned_source": sha(dest) == sweep.pinned_template_hashes()[name],
    }
    if not captured_templates[name]["matches_pinned_source"]:
        raise SystemExit(f"fixed {name} template is not at its pinned SHA")

engine_source = Path(r"E:\Strata-Hetero-data\source\hetero-0-1-41\src\core\verify.cpp")
if not engine_source.is_file():
    raise SystemExit("pinned 0.1.41 verify.cpp source is unavailable")
captured_verify = capture / "engine-source" / "src" / "core" / "verify.cpp"
captured_verify.parent.mkdir(parents=True)
shutil.copy2(engine_source, captured_verify)
verify_text = captured_verify.read_text(encoding="utf-8")
if "STRATA_DIAG_ZERO_LOGITS" not in verify_text or "Read only after sampling has already returned ID 0" not in verify_text:
    raise SystemExit("validated binary source does not contain the expected zero-logit diagnostic path")
if "cudaStreamSynchronize(cs_) != cudaSuccess" not in verify_text:
    raise SystemExit("source readback synchronization anchor missing")
if verify_text.index("Read only after sampling has already returned ID 0") < verify_text.index("cudaStreamSynchronize(cs_) != cudaSuccess"):
    raise SystemExit("diagnostic anchor unexpectedly precedes the sampling-window sync")
source_capture = {
    "schema_version": 1,
    "status": "captured_fixed_sources_read_only",
    "created_utc": STAMP,
    "run_id": RUN_ID,
    "binary": {
        "path": source_matrix["candidate_binary_path"],
        "sha256": source_matrix["candidate_binary_sha256"],
        "bytes": source_matrix["candidate_binary_bytes"],
        "source_commit": source_matrix["candidate_source_sha"],
    },
    "server_bridge_sha256": source_matrix["shared_bridge_sha256"],
    "templates": captured_templates,
    "diagnostic_source": {
        "path": str(engine_source),
        "captured_path": str(captured_verify),
        "sha256": sha(captured_verify),
        "bytes": captured_verify.stat().st_size,
        "diagnostic_env": "STRATA_DIAG_ZERO_LOGITS=1",
        "readback_after_sampling_selected_id_zero_and_window_sync": True,
        "adds_prefill_gemm_synchronization": False,
        "source_lines": [2103, 2127, 2129],
        "note": "source inspection only; no build or device execution",
    },
}
dump(capture / "source-template-capture.json", source_capture)
for name, rel in {
    "pool_sweep": Path("tools/hetero_pool_sweep.py"),
    "quality": Path("tools/hetero_quality.py"),
    "bench": Path("tools/hetero_bench.py"),
    "resources": Path("tools/hetero_resources.py"),
    "file_identity": Path("tools/hetero_verify_existing.py"),
}.items():
    src = REPO / rel
    dest = capture / "executed-preparation-sources" / rel.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
# Save the preparation recipe itself as executed source.
shutil.copy2(Path(__file__), capture / "executed-preparation-sources" / Path(__file__).name)

# Copy and rebind the frozen prompt manifest to this run's tokenizer snapshot.
manifest_path = RUN / "quality-manifest" / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
tok_rel = Path("tokenizer-source-01") / "strata_tokenizer-current-461b8afb.py"
tok_path = RUN / tok_rel
tok_sha = sha(tok_path)
tok_bytes = tok_path.stat().st_size
manifest["tokenizer"]["implementation_source"].update({
    "path": str(tok_path.resolve()), "sha256": tok_sha, "bytes": tok_bytes
})
manifest_raw = dump(manifest_path.with_name("manifest.json.tmp"), manifest)
manifest_path.unlink()
(manifest_path.with_name("manifest.json.tmp")).replace(manifest_path)
manifest_sha = sha(manifest_path)

# Rebind both tokenizer evidence records to this run's copied source paths and
# the exact rebased manifest while retaining the source equivalence evidence.
equivalence_path = RUN / "tokenizer-equivalence.json"
equivalence = replace_run_strings(json.loads(equivalence_path.read_text(encoding="utf-8")))
equivalence["current_implementation_snapshot"].update({"path": str(tok_path.resolve()), "sha256": tok_sha, "bytes": tok_bytes})
equivalence_raw = overwrite_json(equivalence_path, equivalence)
equivalence_sha = hashlib.sha256(equivalence_raw).hexdigest()
manifest_provenance_path = RUN / "tokenizer-manifest-provenance.json"
manifest_provenance = replace_run_strings(json.loads(manifest_provenance_path.read_text(encoding="utf-8")))
manifest_provenance["manifest_path"] = str(manifest_path.resolve())
manifest_provenance["manifest_sha256"] = manifest_sha
manifest_provenance["tokenizer_equivalence_receipt"] = str(equivalence_path.resolve())
manifest_provenance["tokenizer_equivalence_receipt_sha256"] = equivalence_sha
manifest_provenance["implementation_snapshot"].update({"path": str(tok_path.resolve()), "sha256": tok_sha, "bytes": tok_bytes})
legacy_tok_path = RUN / "tokenizer-source-01" / "strata_tokenizer-legacy-b8af8e56.py"
manifest_provenance["legacy_implementation"].update({"path": str(legacy_tok_path.resolve()), "sha256": sha(legacy_tok_path), "bytes": legacy_tok_path.stat().st_size})
manifest_provenance_raw = overwrite_json(manifest_provenance_path, manifest_provenance)
manifest_provenance_sha = hashlib.sha256(manifest_provenance_raw).hexdigest()

# Fresh identities are preserved as prior identity references only. Runtime
# admission must revalidate them; no current host/asset observation occurs here.
matrix = replace_run_strings(source_matrix)
matrix["source_worktree_status_at_matrix_finalization"] = source_matrix["source_worktree_status_at_matrix_finalization"]
matrix["run_id"] = RUN_ID
matrix["status"] = "prepared_for_root_review_not_launched"
matrix["prepared_utc"] = STAMP
matrix["source_run51_v05"] = {
    "path": str(SOURCE_MATRIX),
    "sha256": hashlib.sha256(source_matrix_raw).hexdigest(),
    "candidate_source_sha": source_matrix["candidate_source_sha"],
    "candidate_binary_sha256": source_matrix["candidate_binary_sha256"],
    "note": "recipe reference only; no run51 runtime, admission, result, or process files copied",
}
matrix["diagnostic_plan"] = {
    "only_extra_environment": {"STRATA_DIAG_ZERO_LOGITS": "1"},
    "applies_to": "first quality phase only: w10-all",
    "other_arms_and_phases": "UNRUN and diagnostic environment absent",
    "maximum_authorized_phase_intent": 1,
    "diagnostic_is_not_a_quality_acceptance_or_cause_claim": True,
}
matrix["quality_prompt_manifest"].update({"path": str(manifest_path.resolve()), "sha256": manifest_sha})
matrix["tokenizer_implementation_snapshot"].update({
    "path": str(tok_path.resolve()), "sha256": tok_sha, "bytes": tok_bytes
})
matrix["tokenizer_manifest_provenance"].update({
    "path": str((RUN / "tokenizer-manifest-provenance.json").resolve()),
    "sha256": manifest_provenance_sha,
})
matrix["preparation_source"] = {
    "path": str((RUN / Path(__file__).name).resolve()),
    "sha256": sha(Path(__file__)),
}
matrix["source_template_capture"] = {
    "path": str((capture / "source-template-capture.json").resolve()),
    "sha256": sha(capture / "source-template-capture.json"),
}
matrix.pop("predecessor_matrix", None)
source_history = {
    "schema_version": 1,
    "status": "source_recipe_reference_only",
    "source_run_id": "20261011-51-worker-pool-sweep",
    "source_matrix_path": str(SOURCE_MATRIX.resolve()),
    "source_matrix_sha256": hashlib.sha256(source_matrix_raw).hexdigest(),
    "source_matrix_variant": "matrix-tokenizer-rebased-05.json",
    "runtime_evidence_copied": False,
    "process_admission_or_result_artifacts_copied": False,
    "note": "Only configuration, identity, provenance, freshness-reference, source, and prompt recipe inputs were used.",
}
history_raw = dump(RUN / "source-run51-recipe.json", source_history)
matrix["source_preparation_history"] = {
    "path": str((RUN / "source-run51-recipe.json").resolve()),
    "sha256": hashlib.sha256(history_raw).hexdigest(),
    "source_matrix_path": str(SOURCE_MATRIX.resolve()),
    "source_matrix_sha256": hashlib.sha256(source_matrix_raw).hexdigest(),
}

arms_by_id = {a["arm"]: a for a in matrix["arms"]}
for phase_name, phase_leaf, capture_on in (
    ("quality_on", "quality-on", True), ("performance_off", "performance-off", False)
):
    for arm in matrix["arms"]:
        base_leaf = SOURCE_ROOT / "arms" / arm["arm"] / phase_leaf
        leaf = RUN / "arms" / arm["arm"] / phase_leaf
        leaf.mkdir(parents=True, exist_ok=False)
        config = json.loads((base_leaf / "config.json").read_text(encoding="utf-8"))
        config = replace_run_strings(config)
        config["log"] = str((leaf / "logs" / "engine.log").resolve())
        if arm["arm"] == "w10-all" and phase_name == "quality_on":
            config["env"]["STRATA_DIAG_ZERO_LOGITS"] = "1"
        normalized_config = json.loads(json.dumps(config))
        normalized_config["exe"] = "@SAME_ENGINE@"
        normalized_config["log"] = "@RUN_LOG@"
        normalized_args = list(normalized_config["args"])
        normalized_args[normalized_args.index("--pool-workers") + 1] = "@WORKERS@"
        normalized_args[normalized_args.index("--pool-affinity") + 1] = "@AFFINITY@"
        normalized_config["args"] = normalized_args
        normalized_config_sha = hashlib.sha256(
            json.dumps(normalized_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        config_raw = dump(leaf / "config.json", config)
        config_sha = hashlib.sha256(config_raw).hexdigest()

        identity = replace_run_strings(json.loads((base_leaf / "identity.json").read_text(encoding="utf-8")))
        identity["config"]["sha256"] = config_sha
        identity["config"]["diagnostic_environment_delta"] = (
            {"STRATA_DIAG_ZERO_LOGITS": "1"} if arm["arm"] == "w10-all" and phase_name == "quality_on" else {}
        )
        identity["config"]["only_runtime_difference"] = (
            "STRATA_DIAG_ZERO_LOGITS=1 is the only additional environment value for this first quality phase; other recipe controls match run51 v05."
            if arm["arm"] == "w10-all" and phase_name == "quality_on" else
            "Pool workers/affinity, capture phase, and run-local log follow the run51 v05 recipe; zero-logit diagnostic is unset."
        )
        identity["quality_prompt_manifest"] = {
            **identity.get("quality_prompt_manifest", {}),
            "path": str(manifest_path.resolve()), "sha256": manifest_sha
        }
        identity["tokenizer_implementation"].update({
            "path": str(tok_path.resolve()), "sha256": tok_sha, "bytes": tok_bytes
        })
        identity["tokenizer_implementation"]["equivalence_receipt"] = str(equivalence_path.resolve())
        identity["tokenizer_implementation"]["equivalence_receipt_sha256"] = equivalence_sha
        identity_raw = dump(leaf / "identity.json", identity)
        identity_sha = hashlib.sha256(identity_raw).hexdigest()

        provenance = replace_run_strings(json.loads((base_leaf / "provenance.json").read_text(encoding="utf-8")))
        provenance["prepared_utc"] = STAMP
        provenance["run_id"] = f"20261011-57-{arm['arm']}-{'quality-on' if capture_on else 'performance-off'}"
        provenance["acceptance_status"] = "prepared_not_launched"
        provenance["process_receipt"] = None
        provenance["quality_prompt_manifest"] = str(manifest_path.resolve())
        provenance["quality_prompt_manifest_sha256"] = manifest_sha
        provenance["diagnostic_environment_delta"] = (
            {"STRATA_DIAG_ZERO_LOGITS": "1"} if arm["arm"] == "w10-all" and phase_name == "quality_on" else {}
        )
        provenance["diagnostic_scope"] = "first quality phase only; no output-quality or causal claim"
        provenance.setdefault("shared_controls", {})["quality_prompt_manifest"] = str(manifest_path.resolve())
        provenance["shared_controls"]["quality_prompt_manifest_sha256"] = manifest_sha
        provenance.get("comparison_contract", {})["normalized_config_sha256"] = normalized_config_sha
        provenance["comparison_contract"]["diagnostic_environment_delta"] = (
            {"STRATA_DIAG_ZERO_LOGITS": "1"} if arm["arm"] == "w10-all" and phase_name == "quality_on" else {}
        )
        provenance.get("tokenizer_manifest_rebase", {}).update({
            "new_manifest_path": str(manifest_path.resolve()),
            "new_manifest_sha256": manifest_sha,
            "equivalence_receipt_sha256": equivalence_sha,
        })
        provenance_raw = dump(leaf / "provenance.json", provenance)
        provenance_sha = hashlib.sha256(provenance_raw).hexdigest()

        fresh_src = base_leaf / "fresh-identities.json"
        fresh = json.loads(fresh_src.read_text(encoding="utf-8"))
        fresh["preparation_reference_only"] = {
            "source_run_id": "20261011-51-worker-pool-sweep",
            "source_sha256": sha(fresh_src),
            "current_admission_recheck_required": True,
            "this_preparation_performed_host_or_asset_observation": False,
        }
        fresh_raw = dump(leaf / "fresh-identities.json", fresh)
        fresh_sha = hashlib.sha256(fresh_raw).hexdigest()

        prep = {
            "schema_version": 1,
            "run_id": provenance["run_id"],
            "status": "prepared_not_launched",
            "config_sha256": config_sha,
            "identity_sha256": identity_sha,
            "provenance_sha256": provenance_sha,
            "fresh_identities_sha256": fresh_sha,
            "source_fresh_identities_sha256": sha(fresh_src),
            "worker_count": arm["workers"],
            "affinity": arm["affinity"],
            "quality_capture": capture_on,
            "diagnostic_environment_delta": provenance["diagnostic_environment_delta"],
            "source_run51_recipe": str(SOURCE_MATRIX),
            "current_admission_required": True,
        }
        prep_raw = dump(leaf / "preparation.json", prep)

        row = arm[phase_name]
        row.update({
            "run_id": provenance["run_id"], "path": str(leaf.resolve()),
            "config_sha256": config_sha, "identity_sha256": identity_sha,
            "provenance_sha256": provenance_sha, "fresh_identities_sha256": fresh_sha,
        })

        for sub in ("logs", "resource", "requests", "quality", "performance", "cleanup"):
            (leaf / sub).mkdir()
        for prompt in manifest["prompts"]:
            (leaf / "requests" / prompt["id"]).mkdir()

# Rebind matrix manifest SHA to its exact copied/rebased source; no prompt text or
# quality expectations are changed by the tokenizer path rebase.
matrix["quality_prompt_manifest"]["sha256"] = manifest_sha
matrix["tokenizer_implementation_snapshot"]["sha256"] = tok_sha
matrix["tokenizer_implementation_snapshot"]["equivalence_receipt_sha256"] = equivalence_sha
matrix["tokenizer_manifest_provenance"]["sha256"] = manifest_provenance_sha
# The canonical per-phase hashes are recomputed using the same normalizer as the planner.
for arm in matrix["arms"]:
    for phase_name, leaf_name in (("quality_on", "quality-on"), ("performance_off", "performance-off")):
        armrow = arm[phase_name]
        config = json.loads((RUN / "arms" / arm["arm"] / leaf_name / "config.json").read_text(encoding="utf-8"))
        config["log"] = "@RUN_LOG@"
        if arm["arm"] == "w10-all" and phase_name == "quality_on":
            # Diagnostics are a deliberate one-phase runtime delta and stay visible
            # in the canonical plan rather than being normalized away.
            pass
        normalized = dict(config)
        normalized["exe"] = "@SAME_ENGINE@"
        normalized["log"] = "@RUN_LOG@"
        args = list(normalized["args"])
        args[args.index("--pool-workers") + 1] = "@WORKERS@"
        args[args.index("--pool-affinity") + 1] = "@AFFINITY@"
        normalized["args"] = args
        armrow["normalized_config_sha256"] = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
matrix["canonical_config_hashes"]["quality_capture_on_first_diagnostic_phase"] = next(
    arm["quality_on"]["normalized_config_sha256"] for arm in matrix["arms"] if arm["arm"] == "w10-all"
)
matrix["canonical_config_hashes"]["quality_capture_on"] = next(
    arm["quality_on"]["normalized_config_sha256"] for arm in matrix["arms"] if arm["arm"] == "w04-all"
)
matrix_raw = dump(MATRIX, matrix)

# Directory contract: no controller exists before root authorization; all output
# locations are fresh and empty.
for name in ("controller", "resources", "resource", "logs", "quality", "performance", "requests", "cleanup", "receipts"):
    if name == "controller":
        continue
    (RUN / name).mkdir()
(RUN / "README.md").write_text(
    "# Run57 direct zero-logits diagnostic preparation\n\n"
    "Prepared from the run51 v05 small recipe. Only the first quality phase (w10-all) adds "
    "STRATA_DIAG_ZERO_LOGITS=1; the other nine phases are UNRUN with the diagnostic unset. "
    "The matrix is prepared for root review and has no authorization. This preparation did not "
    "run admission, start/monitor a service, send requests, build, or call a GPU.\n",
    encoding="utf-8", newline="\n"
)

# Planner validation is local/static. It reads only the prepared source/config/prompt
# artifacts and does not inspect host hardware, process trees, network, or devices.
plan = sweep.build_plan(MATRIX)
if len(plan["steps"]) != 10 or plan["steps"][0]["run_id"] != "20261011-57-w10-all-quality-on":
    raise SystemExit("run57 full plan is not the expected ten-step plan with w10 first")
first = plan["steps"][0]
first_cfg = json.loads(Path(first["config"]).read_text(encoding="utf-8"))
if first_cfg.get("env", {}).get("STRATA_DIAG_ZERO_LOGITS") != "1":
    raise SystemExit("first quality config did not bind the diagnostic environment")
for step in plan["steps"][1:]:
    cfg = json.loads(Path(step["config"]).read_text(encoding="utf-8"))
    if "STRATA_DIAG_ZERO_LOGITS" in cfg.get("env", {}):
        raise SystemExit(f"diagnostic environment leaked into non-first phase {step['run_id']}")

receipt_dir = RUN / "receipts" / "dry-validation-01"
receipt_dir.mkdir(parents=True, exist_ok=False)
validation = []
for name in ("launch", "watch", "quality"):
    rendered = sweep.render_run41_template(name, first, REPO)
    validation.append(validate_script(name, rendered, receipt_dir))
dry = {
    "schema_version": 1,
    "status": "all_default_first_phase_scripts_parse_only_validated",
    "run_id": RUN_ID,
    "matrix_sha256": sha(MATRIX),
    "plan_step_count": len(plan["steps"]),
    "validated_step": first["run_id"],
    "actions": validation,
    "model_or_lifecycle_script_executed": False,
    "admission_or_hardware_observed": False,
    "service_started_or_monitored": False,
    "requests_sent": False,
}
dump(receipt_dir / "dry-validation.json", dry)

# Final append-only-preparation hashes; runtime evidence directories remain empty.
inventory = {}
for p in sorted(RUN.rglob("*")):
    if p.is_file() and p not in (RUN / "preparation.json", RUN / "preparation-receipt.json"):
        inventory[p.relative_to(RUN).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha(p)}
preparation = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "status": "prepared_for_root_review_not_launched",
    "prepared_utc": STAMP,
    "matrix_path": str(MATRIX.resolve()),
    "matrix_sha256": sha(MATRIX),
    "source_recipe_matrix": str(SOURCE_MATRIX),
    "source_recipe_matrix_sha256": hashlib.sha256(source_matrix_raw).hexdigest(),
    "binary_sha256": source_matrix["candidate_binary_sha256"],
    "source_commit": source_matrix["candidate_source_sha"],
    "bridge_sha256": source_matrix["shared_bridge_sha256"],
    "quality_prompt_count": 9,
    "all_quality_prompt_bytes_and_expectations_preserved": True,
    "fixed_controls": {
        "pool_workers": 10, "pool_affinity": "all", "spec": 4,
        "prefill_cpu_share": "0", "stage_pin": "0", "prompt_cache": 0,
        "sampling": {"temperature": 0, "top_p": 1, "seed": 42},
    },
    "only_diagnostic_env_delta": {"first_phase": "20261011-57-w10-all-quality-on",
                                  "STRATA_DIAG_ZERO_LOGITS": "1"},
    "other_nine_phases": "diagnostic variable absent; unrun",
    "fresh_identity_files": "copied/rebound prior identity only; root must require fresh admission recheck",
    "source_template_capture_sha256": sha(capture / "source-template-capture.json"),
    "dry_validation_status": dry["status"],
    "plan_only_no_hardware_or_process_inventory": True,
    "authorization_present": False,
    "controller_directory_present": (RUN / "controller").exists(),
    "runtime_start_or_request_or_build": False,
    "artifact_hashes": inventory,
}
prep_raw = dump(RUN / "preparation.json", preparation)
receipt = {
    "schema_version": 1,
    "status": "prepared_and_static_validations_passed_no_lifecycle_execution",
    "run_id": RUN_ID,
    "preparation_sha256": hashlib.sha256(prep_raw).hexdigest(),
    "matrix_sha256": sha(MATRIX),
    "dry_validation_sha256": sha(receipt_dir / "dry-validation.json"),
    "source_template_capture_sha256": sha(capture / "source-template-capture.json"),
    "config_count": 10,
    "plan_step_count": len(plan["steps"]),
    "validated_default_scripts": ["launch", "watch", "quality"],
    "validated_first_phase": first["run_id"],
    "runtime_or_admission_evidence_created": False,
    "authorization_created": False,
}
dump(RUN / "preparation-receipt.json", receipt)
print(json.dumps(receipt, indent=2))
