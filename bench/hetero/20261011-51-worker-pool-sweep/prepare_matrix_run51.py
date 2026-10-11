"""Create the source-only run51 worker-pool matrix from immutable run47 inputs."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
BASE = REPO / "bench/hetero/20261009-47-worker-pool-sweep-preparation"
BASE_MATRIX_PATH = BASE / "matrix-tokenizer-rebased-03.json"
MATRIX_PATH = RUN / "matrix-tokenizer-rebased-04.json"
OLD_RUN_ROOT = str(BASE.resolve())
NEW_RUN_ROOT = str(RUN.resolve())
OLD_RUN_PREFIX = "20261009-47"
NEW_RUN_PREFIX = "20261011-51"
ARMS = (
    ("w10-all", 10, "all", "baseline"),
    ("w04-all", 4, "all", "candidate"),
    ("w06-all", 6, "all", "candidate"),
    ("w16-all", 16, "all", "candidate"),
    ("w06-auto", 6, "auto", "candidate"),
)
PHASES = (("quality-on", "quality_on", True), ("performance-off", "performance_off", False))
CURRENT_TOKENIZER = "tokenizer-source-01/strata_tokenizer-current-461b8afb.py"
LEGACY_TOKENIZER = "tokenizer-source-01/strata_tokenizer-legacy-b8af8e56.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes())


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def transform(value):
    if isinstance(value, dict):
        return {key: transform(item) for key, item in value.items()}
    if isinstance(value, list):
        return [transform(item) for item in value]
    if isinstance(value, str):
        return value.replace(OLD_RUN_ROOT, NEW_RUN_ROOT).replace(OLD_RUN_PREFIX, NEW_RUN_PREFIX)
    return value


def canonical_config(config: dict, keep_capture: bool) -> str:
    value = copy.deepcopy(config)
    value["exe"] = "@SAME_ENGINE@"
    value["log"] = "@RUN_LOG@"
    args = value["args"]
    args[args.index("--pool-workers") + 1] = "@WORKERS@"
    args[args.index("--pool-affinity") + 1] = "@AFFINITY@"
    if not keep_capture:
        value["hetero_capture_token_ids"] = "@CAPTURE_MODE@"
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def tree_receipt(root: Path) -> dict:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            rows.append({"path": path.relative_to(root).as_posix(), "kind": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            data = path.read_bytes()
            rows.append({"path": path.relative_to(root).as_posix(), "kind": "file", "bytes": len(data), "sha256": sha256(data)})
    packed = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"file_count": sum(row["kind"] == "file" for row in rows),
            "file_bytes": sum(row.get("bytes", 0) for row in rows),
            "tree_sha256": sha256(packed), "files": rows}


def main() -> None:
    if not BASE_MATRIX_PATH.is_file():
        raise SystemExit("immutable run47 matrix is missing")
    existing = {str(path.resolve()) for path in RUN.iterdir()}
    if existing != {str(Path(__file__).resolve())}:
        raise SystemExit("run51 destination must contain only this preparation source")

    original_tree_before = tree_receipt(BASE)
    base_matrix_bytes = BASE_MATRIX_PATH.read_bytes()
    base_matrix = json.loads(base_matrix_bytes.decode("utf-8"))
    if base_matrix.get("run_id") != "20261009-47-worker-pool-sweep-preparation":
        raise SystemExit("unexpected run47 baseline matrix identity")
    base_prep_path = BASE / "preparation.json"
    base_prep_sha = file_sha(base_prep_path)
    source_script_sha = file_sha(Path(__file__).resolve())
    created_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

    # The prompt manifest changes only its frozen implementation paths; raw prompts and
    # every local token count / quality expectation remain byte-for-byte bound.
    source_manifest_path = BASE / "quality-manifest/manifest.json"
    old_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    new_manifest = transform(old_manifest)
    manifest_path = RUN / "quality-manifest/manifest.json"
    new_manifest["tokenizer"]["module"] = str(RUN / CURRENT_TOKENIZER)
    new_manifest["tokenizer"]["implementation_source"]["path"] = str(RUN / CURRENT_TOKENIZER)
    manifest_raw = json_bytes(new_manifest)
    manifest_sha = sha256(manifest_raw)

    # Copy only small prompt/tokenizer source material and receipts; model/pack files
    # stay at their existing external paths.
    tokenizer_files = {}
    for relative in (CURRENT_TOKENIZER, LEGACY_TOKENIZER):
        source = BASE / relative
        raw = source.read_bytes()
        write_exclusive(RUN / relative, raw)
        tokenizer_files[relative] = {"bytes": len(raw), "sha256": sha256(raw)}
    equivalence = transform(json.loads((BASE / "tokenizer-equivalence.json").read_text(encoding="utf-8")))
    equivalence_raw = json_bytes(equivalence)
    write_exclusive(RUN / "tokenizer-equivalence.json", equivalence_raw)
    equivalence_sha = sha256(equivalence_raw)
    tokenizer_provenance = transform(json.loads((BASE / "tokenizer-manifest-provenance.json").read_text(encoding="utf-8")))
    tokenizer_provenance_raw = json_bytes(tokenizer_provenance)
    write_exclusive(RUN / "tokenizer-manifest-provenance.json", tokenizer_provenance_raw)
    tokenizer_provenance_sha = sha256(tokenizer_provenance_raw)

    prompt_assets = {}
    for prompt in old_manifest["prompts"]:
        relative = prompt["file"]
        source = BASE / "quality-manifest" / relative
        raw = source.read_bytes()
        if len(raw) != prompt["bytes"] or sha256(raw) != prompt["sha256"]:
            raise SystemExit(f"run47 prompt asset changed: {relative}")
        write_exclusive(RUN / "quality-manifest" / relative, raw)
        prompt_assets[f"quality-manifest/{relative}"] = {"bytes": len(raw), "sha256": sha256(raw)}
    for name in ("sustained_decode.prompt.txt", "sustained_decode.prompt.json"):
        source = BASE / "prompts" / name
        raw = source.read_bytes()
        if name.endswith(".json"):
            raw = json_bytes(transform(json.loads(raw.decode("utf-8"))))
        write_exclusive(RUN / "prompts" / name, raw)
        prompt_assets[f"prompts/{name}"] = {"bytes": len(raw), "sha256": sha256(raw)}
    write_exclusive(manifest_path, manifest_raw)

    base_phase_hashes = {}
    output_phase_hashes = {}
    rows_by_arm = {row["arm"]: row for row in base_matrix["arms"]}
    new_arms = []
    normalized_common, normalized_quality, normalized_performance = [], [], []
    runtime_dirs = []
    for arm, workers, affinity, role in ARMS:
        if arm not in rows_by_arm:
            raise SystemExit(f"run47 arm missing: {arm}")
        row = {"arm": arm, "workers": workers, "affinity": affinity, "role": role}
        new_arm = {"arm": arm, "role": role, "workers": workers, "affinity": affinity}
        for leaf_name, matrix_phase, capture in PHASES:
            base_leaf = BASE / "arms" / arm / leaf_name
            run_id = f"{NEW_RUN_PREFIX}-{arm}-{leaf_name}"
            leaf = RUN / "arms" / arm / leaf_name
            source_config = json.loads((base_leaf / "config.json").read_text(encoding="utf-8"))
            config = transform(source_config)
            config["log"] = str(leaf / "logs/engine.log")
            if (config["hetero_capture_token_ids"] is not capture or
                    config["args"][config["args"].index("--pool-workers") + 1] != str(workers) or
                    config["args"][config["args"].index("--pool-affinity") + 1] != affinity):
                raise SystemExit(f"run47 controls disagree with requested matrix arm: {run_id}")
            config_raw = json_bytes(config)
            config_sha = sha256(config_raw)
            write_exclusive(leaf / "config.json", config_raw)

            identity = transform(json.loads((base_leaf / "identity.json").read_text(encoding="utf-8")))
            identity["config"]["sha256"] = config_sha
            identity["config"]["pool_workers"] = workers
            identity["config"]["pool_affinity"] = affinity
            identity["config"]["token_capture"] = capture
            identity["config"]["arm_id"] = arm
            identity["config"]["only_runtime_difference"] = (
                "Pool workers and affinity differ by arm; token capture and the run-specific log differ by phase."
            )
            identity["quality_prompt_manifest"]["path"] = str(manifest_path)
            identity["quality_prompt_manifest"]["sha256"] = manifest_sha
            identity["tokenizer_implementation"]["path"] = str(RUN / CURRENT_TOKENIZER)
            identity["tokenizer_implementation"]["equivalence_receipt"] = str(RUN / "tokenizer-equivalence.json")
            identity_raw = json_bytes(identity)
            identity_sha = sha256(identity_raw)
            write_exclusive(leaf / "identity.json", identity_raw)

            provenance = transform(json.loads((base_leaf / "provenance.json").read_text(encoding="utf-8")))
            base_provenance_sha = file_sha(base_leaf / "provenance.json")
            prior_root_review = copy.deepcopy(provenance.get("root_review"))
            provenance["prepared_utc"] = created_utc
            provenance["run_id"] = run_id
            provenance["arm"] = arm
            provenance["acceptance_status"] = "prepared_not_launched"
            provenance["process_receipt"] = None
            provenance["root_review"] = {
                "status": "not_reviewed_for_run51",
                "source_run47_root_review": prior_root_review,
                "source_run47_provenance_sha256": base_provenance_sha,
            }
            shared = provenance["shared_controls"]
            shared["quality_prompt_manifest"] = str(manifest_path)
            shared["quality_prompt_manifest_sha256"] = manifest_sha
            shared["pool_workers"] = workers
            shared["pool_affinity"] = affinity
            provenance["worker_pool"].update({"workers": workers, "affinity_cli": affinity, "role": role})
            provenance["token_capture"].update({"enabled": capture, "phase": leaf_name})
            mode_hash = canonical_config(config, keep_capture=True)
            provenance["comparison_contract"].update({
                "normalized_config_sha256": mode_hash,
                "worker_pool_arm": arm,
                "pool_workers": workers,
                "pool_affinity": affinity,
                "token_capture_enabled": capture,
                "capture_on_off_pair_same_worker_and_affinity": True,
                "capture_on_off_pair_only_capture_flag_and_log_differ": True,
                "all_nonarm_runtime_controls_unchanged": True,
            })
            provenance["tokenizer_manifest_rebase"].update({
                "new_manifest_path": str(manifest_path),
                "new_manifest_sha256": manifest_sha,
                "equivalence_receipt_sha256": equivalence_sha,
                "original_prompt_files_and_quality_expectations_unchanged": True,
            })
            provenance_raw = json_bytes(provenance)
            provenance_sha = sha256(provenance_raw)
            write_exclusive(leaf / "provenance.json", provenance_raw)

            source_prep = json.loads((base_leaf / "preparation.json").read_text(encoding="utf-8"))
            source_fresh = base_leaf / "fresh-identities.json"
            old_fresh_sha = file_sha(source_fresh) if source_fresh.is_file() else None
            prep = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "prepared_not_launched",
                "config_sha256": config_sha,
                "identity_sha256": identity_sha,
                "provenance_sha256": provenance_sha,
                "fresh_identities_sha256": None,
                "fresh_identities_status": "NOT_CAPTURED; require fresh launch admission and file identities",
                "source_run47_fresh_identities_sha256": old_fresh_sha,
                "worker_count": workers,
                "affinity": affinity,
                "token_capture": capture,
                "normalized_config_sha256": mode_hash,
                "quality_prompt_manifest_sha256": manifest_sha,
                "tokenizer_implementation_sha256": tokenizer_files[CURRENT_TOKENIZER]["sha256"],
                "source_run47_phase": {
                    "path": str(base_leaf),
                    "config_sha256": source_prep["config_sha256"],
                    "identity_sha256": source_prep["identity_sha256"],
                    "provenance_sha256": source_prep["provenance_sha256"],
                    "preparation_sha256": file_sha(base_leaf / "preparation.json"),
                    "fresh_identities_path": str(source_fresh),
                },
            }
            prep_raw = json_bytes(prep)
            write_exclusive(leaf / "preparation.json", prep_raw)
            base_phase_hashes[run_id] = {
                "path": str(base_leaf),
                "config_sha256": file_sha(base_leaf / "config.json"),
                "identity_sha256": file_sha(base_leaf / "identity.json"),
                "provenance_sha256": base_provenance_sha,
                "preparation_sha256": file_sha(base_leaf / "preparation.json"),
            }
            output_phase_hashes[run_id] = {
                "path": str(leaf),
                "config_sha256": config_sha,
                "identity_sha256": identity_sha,
                "provenance_sha256": provenance_sha,
                "preparation_sha256": sha256(prep_raw),
            }
            phase_row = {"run_id": run_id, "path": str(leaf), "config_sha256": config_sha,
                         "identity_sha256": identity_sha, "provenance_sha256": provenance_sha}
            new_arm[matrix_phase] = phase_row
            normalized_common.append(canonical_config(config, keep_capture=False))
            (normalized_quality if capture else normalized_performance).append(mode_hash)
            for directory in ("controller", "resource", "logs", "quality", "performance"):
                (leaf / directory).mkdir(parents=True, exist_ok=False)
                runtime_dirs.append(str(leaf / directory))
        # Paired phases must differ only by capture flag and the run-specific log.
        q = json.loads((RUN / "arms" / arm / "quality-on/config.json").read_text(encoding="utf-8"))
        p = json.loads((RUN / "arms" / arm / "performance-off/config.json").read_text(encoding="utf-8"))
        q_normal, p_normal = copy.deepcopy(q), copy.deepcopy(p)
        q_normal["log"] = p_normal["log"] = "@RUN_LOG@"
        q_normal["hetero_capture_token_ids"] = p_normal["hetero_capture_token_ids"] = "@CAPTURE@"
        if q_normal != p_normal:
            raise SystemExit(f"unexpected quality/performance config delta for {arm}")
        new_arms.append(new_arm)

    if len(set(normalized_common)) != 1 or len(set(normalized_quality)) != 1 or len(set(normalized_performance)) != 1:
        raise SystemExit("normalized worker-pool controls differ across the five arms")
    baseline_hashes = base_matrix["canonical_config_hashes"]
    if (normalized_common[0] != baseline_hashes["all_modes_capture_placeholder"] or
            normalized_quality[0] != baseline_hashes["quality_capture_on"] or
            normalized_performance[0] != baseline_hashes["performance_capture_off"]):
        raise SystemExit("normalized controls no longer match the reviewed run47 baseline")

    history = {
        "schema_version": 1,
        "status": "historical_source_reference_only",
        "run_id": "20261009-47-worker-pool-sweep-preparation",
        "matrix_path": str(BASE_MATRIX_PATH.resolve()),
        "matrix_sha256": sha256(base_matrix_bytes),
        "preparation_path": str(base_prep_path.resolve()),
        "preparation_sha256": base_prep_sha,
        "tree_before_sha256": original_tree_before["tree_sha256"],
        "phase_artifacts": base_phase_hashes,
        "note": "Run47 is preserved as source history; this run51 preparation writes no files into it.",
    }
    write_exclusive(RUN / "source-run47-history.json", json_bytes(history))

    matrix = transform(base_matrix)
    matrix["run_id"] = "20261011-51-worker-pool-sweep"
    matrix["status"] = "prepared_for_root_review_not_launched"
    matrix["prepared_utc"] = created_utc
    matrix["arms"] = new_arms
    matrix["canonical_config_hashes"] = {
        "all_modes_capture_placeholder": normalized_common[0],
        "quality_capture_on": normalized_quality[0],
        "performance_capture_off": normalized_performance[0],
    }
    matrix["quality_prompt_manifest"].update({"path": str(manifest_path), "sha256": manifest_sha})
    matrix["tokenizer_implementation_snapshot"].update({
        "path": str(RUN / CURRENT_TOKENIZER),
        "sha256": tokenizer_files[CURRENT_TOKENIZER]["sha256"],
        "bytes": tokenizer_files[CURRENT_TOKENIZER]["bytes"],
    })
    matrix["tokenizer_manifest_provenance"].update({
        "path": str(RUN / "tokenizer-manifest-provenance.json"),
        "sha256": tokenizer_provenance_sha,
    })
    matrix["sustained_decode_prompt"].update({
        "path": str(RUN / "prompts/sustained_decode.prompt.txt"),
        "sha256": prompt_assets["prompts/sustained_decode.prompt.txt"]["sha256"],
        "bytes": prompt_assets["prompts/sustained_decode.prompt.txt"]["bytes"],
    })
    matrix["performance_phase"]["scenarios"] = transform(matrix["performance_phase"]["scenarios"])
    matrix["planned_arm_order"] = {
        "quality_capture_on": [arm for arm, _, _, _ in ARMS],
        "performance_capture_off": [arm for arm, _, _, _ in ARMS],
    }
    matrix["arm_execution_order"] = [arm for arm, _, _, _ in ARMS]
    matrix["source_preparation_history"] = {
        "path": str(RUN / "source-run47-history.json"),
        "sha256": file_sha(RUN / "source-run47-history.json"),
        "source_matrix_path": str(BASE_MATRIX_PATH.resolve()),
        "source_matrix_sha256": sha256(base_matrix_bytes),
    }
    matrix["preparation_source"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": source_script_sha,
    }

    # Create fresh empty runtime/evidence roots without fabricating evidence.
    for directory in ("controller", "resources", "resource", "logs", "quality", "performance", "receipts"):
        (RUN / directory).mkdir(parents=True, exist_ok=False)
        runtime_dirs.append(str(RUN / directory))

    matrix_raw = json_bytes(matrix)
    write_exclusive(MATRIX_PATH, matrix_raw)
    write_exclusive(RUN / "README.md", (
        "# Run51 worker-pool matrix preparation\n\n"
        "Prepared from the immutable run47 tokenizer-rebased matrix. Five arms, ten phase configs; "
        "quality capture-on runs first, followed by performance capture-off, with w10-all first in both.\n\n"
        "All model, pack, MTP, engine, bridge, sampling, cache, context, and CPU-share controls retain "
        "the reviewed run47 values. Only run IDs, run-local paths/logs, frozen tokenizer/manifest paths, "
        "and phase hashes were rebased. No runtime evidence, fresh model identities, or authorization is present.\n"
    ).encode("utf-8"))

    # Confirm every copied quality prompt record (including expectations and local
    # counts) is unchanged, and verify raw files at their new paths.
    reread_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for old_prompt, new_prompt in zip(old_manifest["prompts"], reread_manifest["prompts"]):
        if old_prompt != new_prompt:
            raise SystemExit(f"prompt expectation or local count changed: {old_prompt['id']}")
        path = manifest_path.parent / new_prompt["file"]
        if path.stat().st_size != new_prompt["bytes"] or file_sha(path) != new_prompt["sha256"]:
            raise SystemExit(f"copied prompt hash/size mismatch: {new_prompt['id']}")
    if (reread_manifest["tokenizer"]["implementation_source"]["path"] != str(RUN / CURRENT_TOKENIZER) or
            file_sha(RUN / CURRENT_TOKENIZER) != reread_manifest["tokenizer"]["implementation_source"]["sha256"]):
        raise SystemExit("new manifest tokenizer source binding failed")

    matrix_check = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if len(matrix_check["arms"]) != 5 or sum(len([phase for phase in ("quality_on", "performance_off") if phase in arm]) for arm in matrix_check["arms"]) != 10:
        raise SystemExit("matrix does not bind exactly five arms and ten phase configs")
    if matrix_check["arm_execution_order"] != [arm for arm, _, _, _ in ARMS]:
        raise SystemExit("matrix execution order does not put w10-all first")
    runtime_empty = all(not any(Path(directory).iterdir()) for directory in runtime_dirs)
    if not runtime_empty:
        raise SystemExit("a precreated runtime/evidence directory unexpectedly contains files")
    forbidden_model_files = [str(path.relative_to(RUN)) for path in RUN.rglob("*")
                             if path.is_file() and path.suffix.casefold() in {".gguf", ".safetensors", ".bin"}]
    if forbidden_model_files:
        raise SystemExit(f"model/pack artifacts were copied: {forbidden_model_files}")

    original_tree_after = tree_receipt(BASE)
    if original_tree_before != original_tree_after:
        raise SystemExit("run47 source tree changed during run51 preparation")

    artifact_hashes = {}
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.name not in ("preparation.json", "preparation-receipt.json"):
            raw = path.read_bytes()
            artifact_hashes[path.relative_to(RUN).as_posix()] = {"bytes": len(raw), "sha256": sha256(raw)}
    prep = {
        "schema_version": 1,
        "run_id": matrix["run_id"],
        "status": "prepared_for_root_review_not_launched",
        "prepared_utc": created_utc,
        "matrix_path": str(MATRIX_PATH.resolve()),
        "matrix_sha256": sha256(matrix_raw),
        "preparation_source_path": str(Path(__file__).resolve()),
        "preparation_source_sha256": source_script_sha,
        "source_run47_history_path": str(RUN / "source-run47-history.json"),
        "source_run47_history_sha256": file_sha(RUN / "source-run47-history.json"),
        "source_run47_tree_sha256_before_after": original_tree_before["tree_sha256"],
        "source_run47_file_count": original_tree_before["file_count"],
        "source_run47_file_bytes": original_tree_before["file_bytes"],
        "source_run47_unchanged": True,
        "five_arms_ten_phases": True,
        "phase_order": ["quality_on", "performance_off"],
        "w10_first_each_phase": True,
        "normalized_controls": {
            "all_modes_capture_placeholder_sha256": normalized_common[0],
            "quality_capture_on_sha256": normalized_quality[0],
            "performance_capture_off_sha256": normalized_performance[0],
            "same_as_run47": True,
            "paired_configs_differ_only_capture_and_new_log": True,
            "workers_and_affinity_match_requested_five_arms": True,
        },
        "quality_prompts_byte_identical": True,
        "quality_expectations_and_local_token_counts_identical": True,
        "quality_prompt_count": len(old_manifest["prompts"]),
        "quality_formals": matrix["quality_phase"]["formal_requests"],
        "performance_scenarios": len(matrix["performance_phase"]["scenarios"]),
        "prompt_and_tokenizer_files": {**prompt_assets, **tokenizer_files,
                                         "tokenizer-equivalence.json": {"bytes": len(equivalence_raw), "sha256": equivalence_sha},
                                         "tokenizer-manifest-provenance.json": {"bytes": len(tokenizer_provenance_raw), "sha256": tokenizer_provenance_sha}},
        "runtime_directories_precreated_empty": runtime_empty,
        "runtime_evidence_created": False,
        "fresh_model_identities_created": False,
        "models_or_pack_assets_copied": False,
        "authorization_created": False,
        "model_or_api_execution": False,
        "wsl_gpu_or_compile": False,
        "phase_artifacts": output_phase_hashes,
        "artifacts_excluding_receipts": artifact_hashes,
    }
    write_exclusive(RUN / "preparation.json", json_bytes(prep))

    receipt = {
        "schema_version": 1,
        "status": "prepared_metadata_hashes_validated_no_execution",
        "run_id": matrix["run_id"],
        "preparation_source_sha256": source_script_sha,
        "preparation_sha256": file_sha(RUN / "preparation.json"),
        "matrix_sha256": file_sha(MATRIX_PATH),
        "manifest_sha256": file_sha(manifest_path),
        "source_run47_tree_before_after": original_tree_before,
        "source_run47_unchanged": True,
        "phase_artifacts": output_phase_hashes,
        "models_or_pack_assets_copied": False,
        "runtime_evidence_created": False,
        "authorization_or_execution": False,
    }
    write_exclusive(RUN / "preparation-receipt.json", json_bytes(receipt))
    print(json.dumps({
        "status": receipt["status"],
        "matrix_path": str(MATRIX_PATH.resolve()),
        "matrix_sha256": receipt["matrix_sha256"],
        "preparation_source_sha256": source_script_sha,
        "preparation_sha256": receipt["preparation_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "run47_tree_sha256": original_tree_before["tree_sha256"],
        "run47_unchanged": original_tree_before == original_tree_after,
        "arms": len(new_arms),
        "phases": len(output_phase_hashes),
        "quality_manifest_prompts": len(old_manifest["prompts"]),
        "runtime_dirs_empty": runtime_empty,
    }, indent=2))


if __name__ == "__main__":
    main()
