from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
FINAL = RUN / "finalization-01"
SNAPSHOT = RUN / "fresh-input-identities-01.json"
BINARIES = {
    "upstream42": {
        "path": r"E:\Strata-Hetero-data\build\cuda-upstream42-run64-01\strata.exe",
        "sha256": "b4e8f28eebd012c015905c1ffae461ba22d069d1c997a0b3ef91f1edd8e6a786",
        "bytes": 52763648,
        "source_commit": "61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406",
        "build_receipt": "bench/hetero/20261011-64-upstream42-cuda-build/execution/upstream/e-build-metadata/build-receipt.json",
        "build_receipt_sha256": "4aa32aa084c832bdf01f0209bd8a735f780f5207e34d2737ad2426cfea592e81",
    },
    "hetero42": {
        "path": r"E:\Strata-Hetero-data\build\cuda-hetero42-run64-01\strata.exe",
        "sha256": "deba5179200e3d1a934d7fd800e482dcbde27c655c03ea66c217aa58e6792db0",
        "bytes": 52769792,
        "source_commit": "8cf0e89be5da8485cbeb6ac1b527089add25fbcf",
        "build_receipt": "bench/hetero/20261011-64-upstream42-cuda-build/execution/hetero/e-build-metadata/build-receipt.json",
        "build_receipt_sha256": "b700c40c0113f0419e04546170c3cf945d23b989fa5151b19add0d0e79ba47c6",
    },
}


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


snapshot = read_json(SNAPSHOT)
if snapshot.get("all_current_identities_match") is not True or snapshot.get("payload_bytes_read") != 0:
    raise RuntimeError("fresh input metadata snapshot is not clean / zero-read")
if snapshot.get("counts") != {"model_shards": 4, "pack_mtp_runtime_files": 14}:
    raise RuntimeError("fresh input snapshot counts do not match the four-shard plus fourteen-runtime contract")
if len(snapshot.get("input_files", [])) != 4 or len(snapshot.get("runtime_identity_files", [])) != 14:
    raise RuntimeError("fresh input snapshot record counts changed")
FINAL.mkdir(exist_ok=True)

# Preserve prior draft receipts before updating the canonical source-preparation files.
draft = RUN / "draft-01"
draft.mkdir(exist_ok=True)
for name in ("matrix.json", "preparation.json", "preparation-receipt.json",
             "template-bindings.json", "controls-normalization.json"):
    src = RUN / name
    dst = draft / name
    if not dst.exists():
        shutil.copy2(src, dst)
    elif sha(src) != sha(dst):
        raise RuntimeError(f"preserved draft differs on repeat finalization: {name}")

matrix_path = RUN / "matrix.json"
bindings_path = RUN / "template-bindings.json"
audit_path = RUN / "controls-normalization.json"
matrix = read_json(matrix_path)
bindings = read_json(bindings_path)
audit = read_json(audit_path)
if matrix.get("execution_enabled") is not False or matrix.get("root_authorization") is not None:
    raise RuntimeError("execution must remain unauthorized")

snapshot_bytes = SNAPSHOT.read_bytes()
snapshot_sha = sha_bytes(snapshot_bytes)
for side in matrix["sides"]:
    key = side["id"]
    binary = BINARIES[key]
    phase = RUN / side["arm_path"]
    config_path = phase / "config.json"
    identity_path = phase / "identity.json"
    provenance_path = phase / "provenance.json"
    prep_path = phase / "preparation.json"

    identity = read_json(identity_path)
    provenance = read_json(provenance_path)
    prep = read_json(prep_path)
    config = read_json(config_path)
    if config.get("exe") != binary["path"] or identity.get("engine", {}).get("path") != binary["path"]:
        raise RuntimeError(f"{key}: config/identity executable path differs from run64 receipt")
    build_receipt_path = REPO / binary["build_receipt"]
    build_receipt = read_json(build_receipt_path)
    if (sha(build_receipt_path) != binary["build_receipt_sha256"]
            or build_receipt.get("status") != "success"
            or build_receipt.get("binary", {}).get("sha256") != binary["sha256"]
            or build_receipt.get("binary", {}).get("size_bytes") != binary["bytes"]
            or build_receipt.get("source_after", {}).get("head") != binary["source_commit"]
            or build_receipt.get("source_after", {}).get("clean") is not True):
        raise RuntimeError(f"{key}: run64 build receipt does not verify against the independently captured identity")
    if identity["source"]["sha"] != binary["source_commit"] or provenance["engine_source_sha"] != binary["source_commit"]:
        raise RuntimeError(f"{key}: source commit differs from run64 build receipt")
    identity["source"]["binary_hash_finalized"] = True
    identity["engine"].update({
        "sha256": binary["sha256"], "size_bytes": binary["bytes"],
        "hash_status": "RUN64_BUILD_RECEIPT_AND_LOCAL_GETHASH_VERIFIED_FRESH_LAUNCH_REHASH_REQUIRED",
    })
    provenance["engine_sha256"] = binary["sha256"]
    provenance["acceptance_status"] = "prepared_not_launched"

    config_sha = sha(config_path)
    identity["config"]["sha256"] = config_sha
    identity_sha = write_json(identity_path, identity)
    provenance["config_sha256"] = config_sha
    provenance_sha = write_json(provenance_path, provenance)

    fresh_path = phase / "fresh-identities.json"
    fresh_path.write_bytes(snapshot_bytes)
    fresh_sha = sha(fresh_path)
    prep.update({
        "status": "prepared_not_launched_run64_binary_and_fresh_metadata_bound_launch_rehash_required",
        "config_sha256": config_sha,
        "identity_sha256": identity_sha,
        "provenance_sha256": provenance_sha,
        "engine_sha256": binary["sha256"],
        "engine_size_bytes": binary["bytes"],
        "fresh_identities_sha256": fresh_sha,
        "fresh_identities_status": "four model shards and fourteen pack/MTP files stat/fstat matched prior receipts; launch recheck still mandatory",
        "binary": {
            "path": binary["path"], "sha256": binary["sha256"], "bytes": binary["bytes"],
            "status": "run64 build receipt and local Get-FileHash agree; rehash required at launch",
        },
    })
    prep_sha = write_json(prep_path, prep)
    side.update({
        "binary_sha256": binary["sha256"], "binary_bytes": binary["bytes"],
        "binary_identity_status": "run64 receipt and local Get-FileHash verified; fresh launch rehash required",
        "config_sha256": config_sha, "identity_sha256": identity_sha,
        "provenance_sha256": provenance_sha, "preparation_sha256": prep_sha,
        "fresh_identities_sha256": fresh_sha,
        "status": "prepared_not_launched",
    })

matrix["upstream"]["binary_sha256"] = BINARIES["upstream42"]["sha256"]
matrix["upstream"]["binary_bytes"] = BINARIES["upstream42"]["bytes"]
matrix["upstream"]["binary_identity_status"] = "run64 receipt and local Get-FileHash verified; fresh launch rehash required"
matrix["hetero"]["binary_sha256"] = BINARIES["hetero42"]["sha256"]
matrix["hetero"]["binary_bytes"] = BINARIES["hetero42"]["bytes"]
matrix["hetero"]["binary_identity_status"] = "run64 receipt and local Get-FileHash verified; fresh launch rehash required"
matrix["status"] = "finalized_source_and_input_identity_preparation_pending_root_review_and_authorization"
matrix["fresh_input_identity_snapshot"] = {
    "path": "fresh-input-identities-01.json", "sha256": snapshot_sha,
    "model_shards": 4, "pack_mtp_runtime_files": 14,
    "all_current_identities_match_prior_receipts": True,
    "payload_bytes_read": 0,
    "fresh_launch_stat_recheck_required": True,
}
matrix["run64_build_receipts"] = {
    key: {"path": value["build_receipt"], "sha256": value["build_receipt_sha256"],
          "status": "success", "source_clean": True}
    for key, value in BINARIES.items()
}
matrix["execution_enabled"] = False
matrix["root_authorization"] = None

bindings["status"] = "finalized_source_only_both_run64_binary_hashes_and_script_hashes_bound_pending_root_authorization"
bindings["scripts_rendered_or_executed"] = {"rendered": True, "executed": False}
bindings["binary_bindings"] = copy.deepcopy(BINARIES)
bindings["fresh_input_identity_snapshot"] = {
    "path": "fresh-input-identities-01.json", "sha256": snapshot_sha,
    "model_shards": 4, "pack_mtp_runtime_files": 14, "payload_bytes_read": 0,
    "prior_full_hash_receipt": "bench/hetero/20261008-00-admission/model-integrity-02.json",
    "launch_time_recheck_still_required": True,
}

audit["status"] = "exact_normalized_json_pair_match_with_final_binary_and_asset_bindings"
audit["binary_bindings"] = {key: {"sha256": v["sha256"], "bytes": v["bytes"]} for key, v in BINARIES.items()}
audit["source_commits"] = {key: v["source_commit"] for key, v in BINARIES.items()}
audit["fresh_input_identity_snapshot"] = {
    "sha256": snapshot_sha, "model_shards": 4, "pack_mtp_runtime_files": 14,
    "all_current_identities_match_prior_receipts": True, "payload_bytes_read": 0,
    "launch_time_stat_recheck_required": True,
}

matrix["template_bindings_sha256"] = write_json(bindings_path, bindings)
matrix["controls_normalization_sha256"] = write_json(audit_path, audit)
matrix_sha = write_json(matrix_path, matrix)

prep_root_path = RUN / "preparation.json"
preparation = read_json(prep_root_path)
preparation.update({
    "status": "finalized_source_and_metadata_preparation_pending_root_review_and_authorization",
    "binaries": {key: {"path": value["path"], "sha256": value["sha256"], "bytes": value["bytes"],
                        "status": "run64 receipt and local Get-FileHash verified; launch rehash required"}
                  for key, value in BINARIES.items()},
    "fresh_input_identities_path": "fresh-input-identities-01.json",
    "fresh_input_identities_sha256": snapshot_sha,
    "fresh_model_shards": 4,
    "fresh_pack_mtp_runtime_files": 14,
    "fresh_input_all_match_prior_receipts": True,
    "fresh_input_payload_bytes_read": 0,
    "fresh_launch_stat_recheck_required": True,
    "root_authorization_present": False,
    "model_api_device_build_or_test_invoked": False,
})
preparation_sha = write_json(prep_root_path, preparation)

draft_receipt_path = RUN / "preparation-receipt.json"
draft_receipt = read_json(draft_receipt_path)
draft_receipt.update({
    "status": "finalized_source_and_metadata_preparation_pending_root_review_and_authorization",
    "matrix_sha256": matrix_sha,
    "preparation_sha256": preparation_sha,
    "template_bindings_sha256": sha(bindings_path),
    "controls_normalization_sha256": sha(audit_path),
    "binary_hashes": "BOTH_RUN64_RECEIPTS_AND_LOCAL_GETHASH_VERIFIED; LAUNCH REHASH REQUIRED",
    "binary_hashes_bound": {key: {"sha256": value["sha256"], "bytes": value["bytes"]} for key, value in BINARIES.items()},
    "fresh_input_stat_recheck": "snapshot_passed_against_prior_receipts; fresh launch stat/fstat recheck still required",
    "fresh_input_identity_snapshot_sha256": snapshot_sha,
    "template_render_hashes": "BOUND; all 8 scripts parser-checked only; none invoked",
    "authorization": None,
})
write_json(draft_receipt_path, draft_receipt)

binary_inventory = {
    "schema_version": 1,
    "method": "independent PowerShell Get-FileHash SHA-256 and Get-Item size after run64 builds terminal",
    "local_rehash": {"tool": "PowerShell Get-FileHash -Algorithm SHA256", "independent_from_run64_receipt": True,
                     "matches_run64_receipts": True},
    "observed_date": "2026-10-11",
    "binaries": BINARIES,
    "verified_against_run64_receipts": True,
    "launch_time_rehash_required": True,
}
write_json(FINAL / "binary-hash-inventory-01.json", binary_inventory)
shutil.copy2(SNAPSHOT, FINAL / "fresh-input-identities-01.json")
shutil.copy2(matrix_path, FINAL / "matrix.json")
shutil.copy2(bindings_path, FINAL / "contracts.json")
shutil.copy2(audit_path, FINAL / "controls-normalization.json")

receipt = {
    "schema_version": 1,
    "status": "finalized_source_and_metadata_preparation_pending_root_review_and_authorization",
    "run_id": matrix["run_id"],
    "matrix_sha256": matrix_sha,
    "preparation_sha256": preparation_sha,
    "template_bindings_sha256": sha(bindings_path),
    "controls_normalization_sha256": sha(audit_path),
    "fresh_input_identity_snapshot_sha256": snapshot_sha,
    "binary_hash_inventory_sha256": sha(FINAL / "binary-hash-inventory-01.json"),
    "both_binary_hashes_bound": True,
    "model_shards": 4,
    "pack_mtp_runtime_files": 14,
    "fresh_identity_match": True,
    "payload_bytes_read": 0,
    "rendered_scripts": 8,
    "parser_checks": "pass_parse_only from rendered preparation",
    "scripts_invoked": False,
    "api_model_device_build_test_or_admission_invoked": False,
    "authorization": None,
    "launch_time_asset_recheck_required": True,
}
write_json(FINAL / "receipt.json", receipt)

print(json.dumps(receipt, indent=2))
