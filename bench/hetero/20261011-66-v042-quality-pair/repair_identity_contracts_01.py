from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
SRC = {
    "upstream42": REPO / "bench/hetero/20261009-40-upstream41-direct-quality",
    "hetero42": REPO / "bench/hetero/20261009-41-hetero41-direct-quality",
}
BRIDGE = r"E:\Strata-Hetero-data\source\hetero-0-1-42\serve\server.py"
BRIDGE_SHA = "9bc794cae42317dd3283c5f41d98a5c42b4053fd5c8aae8504669ade44a63ded"
JSONL_SHA = "52e15e5fbe3f798efeeb91d0ad09d46fc0ac3ccd737ef7073e63f46b82984e1d"
MANIFEST = str(REPO / "bench/hetero/20261011-51-worker-pool-sweep/quality-manifest/manifest.json")
MANIFEST_SHA = "616c0685c55d8118ad32b24499a01c10021f02f408edfc570c95e7f6d646dde6"
PLE_PATH = r"F:\Strata-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
PLE_SHA = "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"
CONFIG_KEYS = ("exe", "model_name", "log")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> str:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return sha_bytes(raw)


matrix_path = RUN / "matrix.json"
bindings_path = RUN / "template-bindings.json"
audit_path = RUN / "controls-normalization.json"
receipt_path = RUN / "preparation-receipt.json"
matrix = read_json(matrix_path)
bindings = read_json(bindings_path)
audit = read_json(audit_path)
if matrix.get("root_authorization") is not None or matrix.get("execution_enabled") is not False:
    raise RuntimeError("run66 must remain unauthorized and non-executable")
if matrix.get("execution_mode") != "direct_powershell_controllers":
    raise RuntimeError("direct PowerShell controller mode is not frozen")
if sha(Path(BRIDGE)) != BRIDGE_SHA:
    raise RuntimeError("shared v0.1.42 bridge changed")
if sha(REPO / "tools/hetero_jsonl_tail.ps1") != JSONL_SHA:
    raise RuntimeError("JSONL evidence reader changed")
manifest_path = Path(MANIFEST)
if sha(manifest_path) != MANIFEST_SHA:
    raise RuntimeError("v05 nine-prompt manifest changed")

schema_sources = {}
for side in matrix["sides"]:
    key = side["id"]
    source = SRC[key]
    identity_template = source / "identity.json"
    provenance_template = source / "provenance.json"
    identity = read_json(identity_template)
    provenance = read_json(provenance_template)
    prior_placement = copy.deepcopy(provenance.get("placement_contrast"))
    if (not prior_placement or prior_placement.get("path") != PLE_PATH
            or prior_placement.get("sha256") != PLE_SHA):
        raise RuntimeError(f"{key}: prior full-hash/copy placement evidence does not bind original F shard 2")
    config_path = RUN / side["config_path"]
    config = read_json(config_path)

    source_pin = {
        "commit": matrix["upstream"]["commit"] if key == "upstream42" else matrix["hetero"]["commit"],
        "tree": matrix["upstream"]["source_tree"] if key == "upstream42" else matrix["hetero"]["source_tree"],
        "binary_path": config["exe"],
        "binary_sha256": side["binary_sha256"],
        "binary_bytes": side["binary_bytes"],
    }
    if not source_pin["binary_sha256"] or not source_pin["binary_bytes"]:
        raise RuntimeError(f"{key}: finalized binary pin is missing")

    dense_paths = []
    for index, arg in enumerate(config["args"][:-1]):
        if arg == "--native-dense-gguf":
            dense_paths.append(config["args"][index + 1])
    native_index = config["args"].index("--native")
    model_path = config["args"][native_index + 1]
    if len(dense_paths) != 4 or PLE_PATH not in dense_paths or config["env"].get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise RuntimeError(f"{key}: frozen model/config controls do not satisfy v42 recipe")

    identity["source"].update({"sha": source_pin["commit"], "tree": source_pin["tree"], "clean": True,
                               "binary_hash_finalized": True})
    identity["config"].update({
        "sha256": sha(config_path), "scope": "v0.1.42-matched-quality-pair",
        "native_dense_gguf_paths": dense_paths,
        "ple_gguf_path": PLE_PATH,
        "only_runtime_difference": "upstream42 versus hetero42 engine source/binary; matched request and placement controls",
    })
    identity["model"]["id"] = config["model_name"]
    identity["model"]["native_path"] = model_path
    identity["engine"].update({
        "version": "0.1.42", "path": source_pin["binary_path"],
        "sha256": source_pin["binary_sha256"], "size_bytes": source_pin["binary_bytes"],
        "source_sha": source_pin["commit"],
    })
    identity["server_python"].update({
        "checkout_sha": matrix["hetero"]["commit"], "server_file_path": BRIDGE,
        "server_file_sha256": BRIDGE_SHA, "quality_capture_opt_in": True,
        "scope": "identical v1.42 captured-token bridge for both binaries",
    })
    identity["acceptance_status"] = "prepared_not_launched"

    provenance.update({
        "run_id": side["run_id"], "arm": key, "status": "prepared_not_launched",
        "engine_cpp_sha": source_pin["commit"], "engine_version": "0.1.42",
        "engine_path": source_pin["binary_path"], "engine_sha256": source_pin["binary_sha256"],
        "server_python_checkout_sha": matrix["hetero"]["commit"],
        "server_file_sha256": BRIDGE_SHA, "jsonl_tail_reader_sha256": JSONL_SHA,
        "acceptance_status": "prepared_not_launched",
    })
    provenance["shared_controls"].update({
        "model_id": config["model_name"],
        "native_dense_gguf_paths": dense_paths,
        "quality_prompt_manifest": MANIFEST,
        "quality_prompt_manifest_sha256": MANIFEST_SHA,
        "ple_gguf_path": PLE_PATH,
        "environment_frozen": copy.deepcopy(config["env"]),
        "new_v0_1_42_controls": {
            "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_STAGE_PIN": "0", "STRATA_ROUTE_TAIL_SKIP": "0",
        },
    })
    provenance["shared_controls"].pop("new_v0_1_41_controls", None)
    provenance["placement_contrast"].update({
        "path": PLE_PATH, "sha256": PLE_SHA,
        "file_bytes": 49_859_583_136,
        "copy_receipt": r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-08-ple-copy\copy-receipt.json",
    })
    provenance["comparison_contract"].update({
        "normalized_controls_sha256": audit["canonical_normalized_config_sha256"],
        "only_runtime_setting_difference": "v0.1.42 engine source/binary identity; configs and token-ID bridge are matched",
        "new_0_1_42_defaults_disabled": {
            "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_STAGE_PIN": "0", "STRATA_ROUTE_TAIL_SKIP": "0",
        },
        "normalized_config_sha256": audit["canonical_normalized_config_sha256"],
        "server_bridge_identical_for_both_runs": True,
    })
    provenance["comparison_contract"].pop("new_0_1_41_defaults_disabled", None)
    provenance["server_bridge"].update({
        "path": BRIDGE, "sha256": BRIDGE_SHA, "source_checkout_sha": matrix["hetero"]["commit"],
    })
    provenance["build_receipt"].update({
        "pair_summary": str(REPO / "bench/hetero/20261011-64-upstream42-cuda-build/preparation-receipt.json"),
        "source_tree": source_pin["tree"], "gpu_executed": False, "engine_cli_executed": False,
        "binary_size_bytes": source_pin["binary_bytes"],
        "binary_sha256": source_pin["binary_sha256"],
        "build_receipt": str(REPO / ("bench/hetero/20261011-64-upstream42-cuda-build/execution/"
                                      + ("upstream" if key == "upstream42" else "hetero")
                                      + "/e-build-metadata/build-receipt.json")),
    })
    # Upstream v0.1.41 provenance had no reader pin; copy the field from the
    # adjacent Run41 schema because all Run66 watch/quality scripts require it.
    if key == "upstream42" and "jsonl_tail_reader_sha256" not in provenance:
        provenance["jsonl_tail_reader_sha256"] = JSONL_SHA

    identity_path = RUN / side["identity_path"]
    provenance_path = RUN / side["provenance_path"]
    identity_sha = write_json(identity_path, identity)
    provenance_sha = write_json(provenance_path, provenance)
    prep_path = RUN / side["preparation_path"]
    prep = read_json(prep_path)
    prep.update({"identity_sha256": identity_sha, "provenance_sha256": provenance_sha,
                 "status": "prepared_not_launched_full_v41_schema_rebased_to_v42"})
    prep_sha = write_json(prep_path, prep)
    side.update({"identity_sha256": identity_sha, "provenance_sha256": provenance_sha,
                 "preparation_sha256": prep_sha})
    schema_sources[key] = {
        "identity_template": str(identity_template.relative_to(REPO)),
        "identity_template_sha256": sha(identity_template),
        "provenance_template": str(provenance_template.relative_to(REPO)),
        "provenance_template_sha256": sha(provenance_template),
        "identity_sha256": identity_sha, "provenance_sha256": provenance_sha,
        "full_original_schema_copied": True,
        "placements_contrast_retained_from_prior_full_hash_and_copy_receipt": True,
    }

    quality_path = RUN / "rendered-scripts" / key / "quality.ps1"
    quality_text = quality_path.read_text(encoding="utf-8")
    old_model = "$expectedModel = 'qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero'"
    new_model = "$expectedModel = '" + config["model_name"] + "'"
    if quality_text.count(old_model) == 1:
        quality_text = quality_text.replace(old_model, new_model)
    elif new_model not in quality_text:
        raise RuntimeError(f"{key}: expected model anchor changed")
    quality_path.write_text(quality_text, encoding="utf-8", newline="\n")
    qmeta = bindings["rendered_scripts"][key]["quality"]
    qmeta.update({"rendered_sha256": sha(quality_path), "rendered_bytes": quality_path.stat().st_size,
                  "status": "rendered_model_name_contract_aligned_not_executed",
                  "expected_model_name": config["model_name"]})
    side["rendered_scripts"]["quality"].update({"rendered_sha256": sha(quality_path),
                                                 "rendered_bytes": quality_path.stat().st_size,
                                                 "status": "rendered_model_name_contract_aligned_not_executed"})

matrix["status"] = "prepared_contract_repaired_validation_only_pending_root_review"
matrix["root_authorization"] = None
matrix["execution_enabled"] = False
matrix["authorization_file_status"] = "root-authorization-01 is preserved but stale against repaired matrix; replacement required"
matrix["identity_provenance_schema_sources"] = schema_sources
matrix["preflight_repair_snapshot"] = {
    "path": "preflight-repair-01/pre-edit-snapshot/snapshot-manifest.json",
    "sha256": "8b75cbdb3c587d09d829e8be6f485d47a5971aab8bdb5f497a56a1093904b908",
    "failed_upstream_launch_evidence_preserved": True,
    "authorization_and_draft_receipts_snapshotted_before_edits": True,
}

bindings["identity_provenance_schema_sources"] = schema_sources
bindings["v42_contract_fields"] = {
    "all_sides_have_placement_contrast_path_and_prior_sha256": True,
    "all_sides_have_quality_manifest_path_and_sha256": True,
    "all_sides_have_jsonl_reader_sha256": True,
    "all_sides_have_engine_cpp_sha_version_path_sha256": True,
    "expected_model_names_match_each_config": True,
    "all_sides_have_route_tail_skip_zero": True,
    "snapshot_sha256": matrix["preflight_repair_snapshot"]["sha256"],
}
bindings["rendered_status"] = "full_v42_identity_contracts_rebound_pending_safe_no_start_validation"
bindings["scripts_rendered_or_executed"] = {"rendered": True, "executed": False}

audit["identity_provenance_schema_sources"] = schema_sources
audit["v42_contract_coverage"] = bindings["v42_contract_fields"]
audit["authorization_status"] = "none; prior auth is stale after matrix repair and preserved unchanged"
bindings_sha = write_json(RUN / "template-bindings.json", bindings)
audit_sha = write_json(audit_path, audit)
matrix["template_bindings_sha256"] = bindings_sha
matrix["controls_normalization_sha256"] = audit_sha
matrix_sha = write_json(matrix_path, matrix)

preparation_path = RUN / "preparation.json"
preparation = read_json(preparation_path)
preparation.update({"status": "prepared_contract_repaired_validation_only_pending_root_review",
                    "authorization_file_status": "prior auth preserved but stale after matrix update",
                    "root_authorization_present": False,
                    "full_identity_provenance_schema_rebased": True})
preparation_sha = write_json(preparation_path, preparation)
receipt = read_json(receipt_path)
receipt.update({"status": "prepared_contract_repaired_validation_only_pending_root_review",
                "matrix_sha256": matrix_sha, "preparation_sha256": preparation_sha,
                "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
                "authorization": None, "no_start_validation": "pending"})
receipt_sha = write_json(receipt_path, receipt)

print(json.dumps({"status": receipt["status"], "matrix_sha256": matrix_sha,
                  "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
                  "preparation_sha256": preparation_sha, "preparation_receipt_sha256": receipt_sha,
                  "contract_sides": list(schema_sources), "model_runtime_payload_read": False,
                  "authorization": None}, indent=2))
