from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
VALIDATION = RUN / "preflight-repair-01" / "no-start-validation"
FINAL = RUN / "preflight-repair-01" / "final-contracts"
SIDES = ("upstream42", "hetero42")
PHASES = ("launch", "watch", "quality", "cleanup")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> str:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return sha_bytes(raw)


matrix_path = RUN / "matrix.json"
bindings_path = RUN / "template-bindings.json"
audit_path = RUN / "controls-normalization.json"
prep_path = RUN / "preparation.json"
receipt_path = RUN / "preparation-receipt.json"
controller_path = RUN / "run-phase.ps1"
matrix = read_json(matrix_path)
bindings = read_json(bindings_path)
audit = read_json(audit_path)
snapshot_manifest = read_json(RUN / "preflight-repair-01/pre-edit-snapshot/snapshot-manifest.json")

validation_rows = {}
for side in SIDES:
    rows = {}
    for phase in ("launch", "watch", "quality"):
        receipt_path_v = VALIDATION / side / f"{phase}-validation-receipt.json"
        if not receipt_path_v.exists() and phase == "launch":
            receipt_path_v = VALIDATION / side / "validation-receipt.json"
        receipt = read_json(receipt_path_v)
        if receipt.get("exit_code") != 0 or receipt.get("status") != "prepared_validated":
            raise RuntimeError(f"{side}/{phase}: no-start/default validation did not pass")
        stdout = read_json(VALIDATION / side / f"{phase}.stdout.log")
        expected_flag = {"launch": "start_performed", "watch": "monitor_started", "quality": "run_started"}[phase]
        if stdout.get(expected_flag) is not False:
            raise RuntimeError(f"{side}/{phase}: action flag is not false")
        if receipt["script_sha256"] != bindings["rendered_scripts"][side][phase]["rendered_sha256"]:
            raise RuntimeError(f"{side}/{phase}: validated script hash differs from matrix binding")
        if (sha(VALIDATION / side / f"{phase}.stdout.log") != receipt["stdout_sha256"]
                or sha(VALIDATION / side / f"{phase}.stderr.log") != receipt["stderr_sha256"]):
            raise RuntimeError(f"{side}/{phase}: captured raw log hash mismatch")
        rows[phase] = {
            "status": stdout["status"], "script_sha256": receipt["script_sha256"],
            "stdout_sha256": receipt["stdout_sha256"], "stderr_sha256": receipt["stderr_sha256"],
            "exit_code": 0, "action_flag": {expected_flag: False},
            "receipt_path": str(receipt_path_v.relative_to(RUN)),
        }
    cleanup_receipt_path = VALIDATION / side / "cleanup-validation-receipt.json"
    cleanup = read_json(cleanup_receipt_path)
    if cleanup.get("status") != "pass_parse_only_not_invoked" or cleanup.get("parse_errors") != 0:
        raise RuntimeError(f"{side}/cleanup: parser-only check did not pass")
    if cleanup["script_sha256"] != bindings["rendered_scripts"][side]["cleanup"]["rendered_sha256"]:
        raise RuntimeError(f"{side}/cleanup: parser-checked script hash differs from binding")
    rows["cleanup"] = {
        "status": cleanup["status"], "script_sha256": cleanup["script_sha256"],
        "stdout_sha256": cleanup["stdout_sha256"], "stderr_sha256": cleanup["stderr_sha256"],
        "parse_errors": 0, "script_invoked": False,
        "receipt_path": str(cleanup_receipt_path.relative_to(RUN)),
    }
    validation_rows[side] = rows

for side in matrix["sides"]:
    key = side["id"]
    identity = read_json(RUN / side["identity_path"])
    provenance = read_json(RUN / side["provenance_path"])
    config = read_json(RUN / side["config_path"])
    expected_model = config["model_name"]
    if (identity["model"]["id"] != expected_model
            or provenance["shared_controls"]["model_id"] != expected_model
            or identity["engine"]["sha256"] != side["binary_sha256"]
            or provenance["engine_sha256"] != side["binary_sha256"]
            or provenance["shared_controls"]["quality_prompt_manifest_sha256"] != matrix["quality_recipe"]["manifest_sha256"]
            or provenance["placement_contrast"]["path"] != config["args"][config["args"].index("--ple-gguf") + 1]
            or provenance["placement_contrast"]["sha256"] != "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"
            or provenance.get("jsonl_tail_reader_sha256") != bindings["templates"]["jsonl_reader"]["sha256"]
            or config["env"].get("STRATA_ROUTE_TAIL_SKIP") != "0"):
        raise RuntimeError(f"{key}: post-validation v42 identity/provenance contract coverage failed")

all_scripts = {}
for side in SIDES:
    all_scripts[side] = {}
    for phase in PHASES:
        script_meta = bindings["rendered_scripts"][side][phase]
        script_path = Path(script_meta["rendered_path"])
        if sha(script_path) != script_meta["rendered_sha256"]:
            raise RuntimeError(f"{side}/{phase}: current rendered script hash changed")
        all_scripts[side][phase] = {
            "path": str(script_path), "sha256": script_meta["rendered_sha256"],
            "validation": validation_rows[side][phase]["status"],
            "exit_code": validation_rows[side][phase].get("exit_code"),
            "script_invoked": phase != "cleanup",
            "action_performed": False,
        }

if matrix.get("root_authorization") is not None or matrix.get("execution_enabled") is not False:
    raise RuntimeError("no-start validation must leave execution unauthorized")
wrapper_pin = {"path": str(controller_path.resolve()), "sha256": sha(controller_path),
               "owner": "root", "mutated_by_this_preparation": False}
matrix["controller_wrapper"] = wrapper_pin
bindings["controller_wrapper"] = wrapper_pin
audit["controller_wrapper"] = wrapper_pin

bindings["no_start_validation"] = {
    "status": "six_read_only_default_invocations_passed; two cleanup scripts parser-only",
    "results": validation_rows,
    "all_action_flags_false": True,
    "cleanup_invocations": 0,
    "no_model_api_or_start": True,
    "no_monitoring_or_unload": True,
}
renderer_path = RUN / "render_pair_v42.py"
repair_script_path = RUN / "repair_identity_contracts_01.py"
validation_script_path = RUN / "record_no_start_validation_01.py"
bindings["render_pair_renderer"] = {
    "path": str(renderer_path.resolve()), "sha256": sha(renderer_path),
    "direct_powershell_argument_policy": "does not add owned-controller-proof",
}
bindings["preflight_repair_source_pins"] = {
    "identity_provenance_repair": {"path": str(repair_script_path.resolve()), "sha256": sha(repair_script_path)},
    "no_start_validation_recorder": {"path": str(validation_script_path.resolve()), "sha256": sha(validation_script_path)},
}
bindings["rendered_status"] = "v42_full_contracts_passed_readonly_defaults_validated"
audit["no_start_validation"] = bindings["no_start_validation"]
audit["preflight_repair_source_pins"] = bindings["preflight_repair_source_pins"]
audit["v42_contract_coverage"].update({
    "placement_contrast_path_and_prior_sha": "pass_both_sides",
    "quality_manifest_sha256": "pass_both_sides",
    "jsonl_reader_sha256": "pass_both_sides",
    "engine_cpp_sha_version_path_binary_sha": "pass_both_sides",
    "model_name_matches_config": "pass_both_sides",
    "launch_watch_quality_readonly_defaults": "pass_both_sides",
    "cleanup_parser_only": "pass_both_sides",
})
matrix["status"] = "prepared_v42_contracts_validated_no_actions_pending_root_review_and_new_authorization"
matrix["no_start_validation"] = {
    "path": "preflight-repair-01/no-start-validation",
    "sides": validation_rows,
    "launch_start_performed": False, "watch_monitor_started": False,
    "quality_run_started": False, "cleanup_script_invocations": 0,
    "cleanup_parser_errors": 0,
}
matrix["execution_enabled"] = False
matrix["root_authorization"] = None
matrix["authorization_file_status"] = "root-authorization-01 remains unchanged but is stale against repaired matrix; root review and replacement required"

bindings_sha = write_json(bindings_path, bindings)
audit_sha = write_json(audit_path, audit)
matrix["template_bindings_sha256"] = bindings_sha
matrix["controls_normalization_sha256"] = audit_sha
matrix_sha = write_json(matrix_path, matrix)

preparation = read_json(prep_path)
preparation.update({"status": "v42_identity_contracts_passed_readonly_default_validation_pending_root_review",
                    "no_start_validation": "six safe read-only defaults pass; cleanup parser-only",
                    "root_authorization_present": False})
preparation_sha = write_json(prep_path, preparation)
receipt = read_json(receipt_path)
receipt.update({"status": "v42_identity_contracts_passed_readonly_default_validation_pending_root_review",
                "matrix_sha256": matrix_sha, "preparation_sha256": preparation_sha,
                "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
                "no_start_validation": bindings["no_start_validation"], "authorization": None})
receipt_sha = write_json(receipt_path, receipt)

FINAL.mkdir(exist_ok=True)
for src, name in ((matrix_path, "matrix.json"), (bindings_path, "contracts.json"),
                  (audit_path, "controls-normalization.json"),
                  (RUN / "fresh-input-identities-01.json", "fresh-input-identities-01.json"),
                  (RUN / "finalization-01" / "binary-hash-inventory-01.json", "binary-hash-inventory-01.json")):
    shutil.copy2(src, FINAL / name)

final_receipt = {
    "schema_version": 1,
    "status": matrix["status"], "run_id": matrix["run_id"],
    "matrix_sha256": matrix_sha, "preparation_sha256": preparation_sha,
    "preparation_receipt_sha256": receipt_sha,
    "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
    "no_start_validation": validation_rows,
    "rendered_script_count": 8, "launched_scripts": 6,
    "cleanup_parser_only_count": 2,
    "start_performed": False, "monitor_started": False, "quality_run_started": False,
    "cleanup_invoked": False, "model_api_called": False,
    "authorization": None,
    "failed_initial_launch_preserved": True,
    "root_authorization_01_preserved_but_stale": True,
    "root_run_phase_wrapper_untouched": True,
    "root_run_phase_wrapper_sha256": wrapper_pin["sha256"],
    "preflight_repair_source_pins": bindings["preflight_repair_source_pins"],
}
final_receipt_sha = write_json(FINAL / "receipt.json", final_receipt)

print(json.dumps({"status": final_receipt["status"], "matrix_sha256": matrix_sha,
                  "preparation_sha256": preparation_sha, "preparation_receipt_sha256": receipt_sha,
                  "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
                  "final_receipt_sha256": final_receipt_sha, "all_eight_script_contracts_covered": True,
                  "launches_prepared_validated": 2, "watch_defaults_prepared": 2,
                  "quality_defaults_prepared": 2, "cleanup_parser_only": 2,
                  "all_action_flags_false": True, "authorization": None}, indent=2))
