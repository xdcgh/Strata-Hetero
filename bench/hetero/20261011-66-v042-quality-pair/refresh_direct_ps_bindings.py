from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
FINAL = RUN / "finalization-02"
MODE = "direct_powershell_controllers"
CONTROLLER = RUN / "run-phase.ps1"
ARG = b" --owned-controller-proof $env:RUN47_CONTROLLER_PROOF"
ADMISSION = b"--required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current"


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
preparation_path = RUN / "preparation.json"
receipt_path = RUN / "preparation-receipt.json"
matrix = read_json(matrix_path)
bindings = read_json(bindings_path)
audit = read_json(audit_path)
if matrix.get("root_authorization") is not None or matrix.get("execution_enabled") is not False:
    raise RuntimeError("Run66 must remain unauthorized and non-executable")
if sha(CONTROLLER) != "aa0097217b57d195b7b3be1ca743ac18bbade2305c023b2fb8ff61d45dafeb68":
    raise RuntimeError("root-owned run-phase.ps1 hash differs from its reviewed pin")

old_hashes = {}
for side in matrix["sides"]:
    key = side["id"]
    entry = bindings["rendered_scripts"][key]["launch"]
    launch_path = Path(entry["rendered_path"])
    raw = launch_path.read_bytes()
    if raw.count(ARG) == 1:
        updated = raw.replace(ARG, b"")
    elif ARG not in raw and b"RUN47_CONTROLLER_PROOF" not in raw:
        # The one-line edit may already have been applied before this binder runs.
        updated = raw
    else:
        raise RuntimeError(f"{key}: proof argument state is ambiguous")
    if b"RUN47_CONTROLLER_PROOF" in updated or updated.count(ADMISSION) != 1:
        raise RuntimeError(f"{key}: launch proof removal or strict admission anchor is invalid")
    launch_path.write_bytes(updated)
    old_hashes[key] = {"sha256": entry["rendered_sha256"], "bytes": entry["rendered_bytes"]}
    new_sha = sha(launch_path)
    new_bytes = launch_path.stat().st_size
    entry["rendered_sha256"] = new_sha
    entry["rendered_bytes"] = new_bytes
    entry["status"] = "rendered_direct_powershell_controller_not_executed"
    side["rendered_scripts"]["launch"]["rendered_sha256"] = new_sha
    side["rendered_scripts"]["launch"]["rendered_bytes"] = new_bytes
    side["rendered_scripts"]["launch"]["status"] = "rendered_direct_powershell_controller_not_executed"

matrix["execution_mode"] = MODE
matrix["controller_wrapper"] = {
    "path": str(CONTROLLER.resolve()), "sha256": sha(CONTROLLER),
    "owner": "root", "mutated_by_this_preparation": False,
}
matrix["controller_contract"] = {
    "controllers": "direct PowerShell scripts; no Python pool coordinator",
    "admission_required": {"wsl_shutdown_before_observe": True, "minimum_ram_gib": 12,
                            "minimum_commit_gib": 4, "minimum_vram_mib": 46000},
    "foreign_python_and_gpu_workload_gates": "strict and unchanged",
    "gate_exemptions": [],
}
matrix["status"] = "finalized_direct_powershell_preparation_pending_root_authorization"
matrix["execution_enabled"] = False
matrix["root_authorization"] = None

bindings["status"] = "finalized_direct_powershell_rendered_scripts_pending_root_authorization"
bindings["execution_mode"] = MODE
bindings["controller_wrapper"] = matrix["controller_wrapper"]
bindings["direct_powershell_render_change"] = {
    "removed_argument": "--owned-controller-proof $env:RUN47_CONTROLLER_PROOF",
    "reason": "direct PowerShell controllers do not use the Python pool-coordinator proof",
    "strict_original_admission_arguments_retained": ADMISSION.decode("ascii"),
    "prior_launch_hashes": old_hashes,
    "rendered_launches_changed_only_by_removing_one_argument_each": True,
    "raw_templates_modified": False,
}
bindings["render_pair_renderer"] = {
    "path": str((RUN / "render_pair_v42.py").resolve()),
    "sha256": sha(RUN / "render_pair_v42.py"),
    "direct_powershell_argument_policy": "does not add owned-controller-proof",
}

audit["status"] = "exact_normalized_json_pair_match_final_direct_powershell_contract"
audit["execution_mode"] = MODE
audit["controller_wrapper"] = matrix["controller_wrapper"]
audit["admission_proof"] = {
    "strict_original_gates_retained": True,
    "required": ["WSL shutdown", "12 GiB available RAM", "4 GiB available commit", "46000 MiB VRAM"],
    "foreign_python_and_gpu_workload_checks": "retained without exemptions",
    "owned_python_controller_proof_argument": "removed from rendered launch commands; not a gate exemption",
}
audit["rendered_launch_hashes"] = {
    key: bindings["rendered_scripts"][key]["launch"]["rendered_sha256"]
    for key in ("upstream42", "hetero42")
}

bindings_sha = write_json(bindings_path, bindings)
audit_sha = write_json(audit_path, audit)
matrix["template_bindings_sha256"] = bindings_sha
matrix["controls_normalization_sha256"] = audit_sha
matrix_sha = write_json(matrix_path, matrix)

preparation = read_json(preparation_path)
preparation.update({"status": "finalized_direct_powershell_preparation_pending_root_authorization",
                    "execution_mode": MODE, "root_authorization_present": False})
preparation_sha = write_json(preparation_path, preparation)
receipt = read_json(receipt_path)
receipt.update({"status": "finalized_direct_powershell_preparation_pending_root_authorization",
                "execution_mode": MODE, "matrix_sha256": matrix_sha,
                "preparation_sha256": preparation_sha, "template_bindings_sha256": bindings_sha,
                "controls_normalization_sha256": audit_sha, "authorization": None,
                "scripts_invoked": False})
write_json(receipt_path, receipt)

FINAL.mkdir(exist_ok=True)
for src, name in ((matrix_path, "matrix.json"), (bindings_path, "contracts.json"),
                  (audit_path, "controls-normalization.json"),
                  (RUN / "fresh-input-identities-01.json", "fresh-input-identities-01.json"),
                  (RUN / "finalization-01" / "binary-hash-inventory-01.json", "binary-hash-inventory-01.json")):
    shutil.copy2(src, FINAL / name)

final_receipt = {
    "schema_version": 1,
    "status": "finalized_direct_powershell_preparation_pending_root_authorization",
    "run_id": matrix["run_id"], "execution_mode": MODE,
    "matrix_sha256": matrix_sha, "preparation_sha256": preparation_sha,
    "template_bindings_sha256": bindings_sha, "controls_normalization_sha256": audit_sha,
    "rendered_launch_hashes": {key: bindings["rendered_scripts"][key]["launch"]["rendered_sha256"]
                               for key in ("upstream42", "hetero42")},
    "rendered_script_count": 8, "parser_only_validation_required": True,
    "scripts_invoked": False, "authorization": None,
    "run_phase_sha256": sha(CONTROLLER), "prior_finalization_01_preserved": True,
}
write_json(FINAL / "receipt.json", final_receipt)
print(json.dumps(final_receipt, indent=2))
