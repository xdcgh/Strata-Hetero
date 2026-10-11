"""Restore run51 dry-validation inputs and run only default lifecycle validation modes."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
BASE = REPO / "bench/hetero/20261009-47-worker-pool-sweep-preparation"
MATRIX = RUN / "matrix-tokenizer-rebased-04.json"
RECEIPTS = RUN / "receipts/run51-dry-validation-03"
PRIOR_ATTEMPT_RECEIPT = RUN / "receipts/run51-dry-validation-01/supplement-attempt-failure.json"
PRIOR_ATTEMPT_SOURCE_SHA256 = "67f91da50f5df66f5c75bc092b4980eeaf65a88fc8efc0549d3abb22b2991f7b"
PRIOR_AUDIT_FAILURE_RECEIPT = RUN / "receipts/run51-dry-validation-02/supplement-attempt-failure.json"
PRIOR_AUDIT_SOURCE_SHA256 = "84a29bcffbb8cedbf0c3a518af00b7c61420532b0af9f7e8c08c1187bf51aa1f"
ARMS = ("w10-all", "w04-all", "w06-all", "w16-all", "w06-auto")
PHASES = (("quality-on", "quality_on"), ("performance-off", "performance_off"))

sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as sweep  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_exclusive(path: Path, value) -> None:
    write_exclusive(path, json_bytes(value))


def expected_files(config: dict, manifest: dict, prior_manifest: dict, matrix: dict) -> list[dict]:
    rows: dict[str, str] = {}

    def add(raw, role):
        if raw:
            path = Path(str(raw))
            rows[str(path)] = role

    add(config["exe"], "pinned engine binary; hash checked by default lifecycle validation")
    add(config.get("tokenizer"), "configured tokenizer directory")
    add(config.get("cwd"), "configured repository working directory")
    for index, arg in enumerate(config.get("args", [])[:-1]):
        if arg in ("--native", "--expert-profile", "--mtp", "--pack", "--native-dense-gguf", "--ple-gguf"):
            add(config["args"][index + 1], f"config input for {arg}")
    add(REPO / "tools/hetero_verify_existing.py", "stdlib file identity verifier")
    add(REPO / "bench/hetero/20261008-00-admission/model-integrity-02.json", "prior pack/MTP stat identity inventory")
    # The pinned quality validator's actual H4 path is authoritative.
    add(REPO / "bench/hetero/20261009-04-hetero04-direct-quality/shutdown-final.json", "H4 quality reference receipt")
    for rel in (
        "tools/hetero_quality.py", "tools/hetero_compare_quality.py", "tools/hetero_bench.py",
        "tools/hetero_jsonl_tail.ps1", "tools/hetero_pool_sweep.py",
    ):
        add(REPO / rel, "pinned/default validation helper")
    add(Path(r"E:\Strata-Hetero-data\source\hetero-0-1-41\serve\server.py"), "pinned server bridge")
    add(Path(r"E:\Strata-Hetero-data\venv\Scripts\python.exe"), "pinned Python runtime")
    add(Path(r"C:\Python314\python.exe"), "pinned standard-library Python runtime")
    add(sweep.PWsh7, "pinned PowerShell 7 executable")
    for name, (relative, _) in sweep.RUN41_TEMPLATES.items():
        add(REPO / relative, f"pinned run41 {name} source/helper")
    add(Path(manifest["tokenizer"]["implementation_source"]["path"]), "frozen tokenizer implementation")
    add(RUN / "quality-manifest/manifest.json", "run51 frozen quality prompt manifest")
    for asset in manifest["tokenizer"].get("assets", []):
        add(asset["path"], "manifest-bound pack tokenizer asset")
    for row in manifest.get("prompts", []):
        add(RUN / "quality-manifest" / row["file"], "frozen quality prompt")
    for row in prior_manifest.get("runtime_identity", {}).values():
        for item in row.get("files", []):
            add(item.get("path"), "prior pack/MTP runtime identity input")
    for row in manifest.get("prompts", []):
        prompt_id = row["id"]
        for formal in range(3):
            add(REPO / "bench/hetero/20261009-04-hetero04-direct-quality/requests" /
                prompt_id / f"formal-{formal:04d}.sse.txt", "H4 actual-token-ID reference")
    for scenario in json.loads(MATRIX.read_text(encoding="utf-8"))["performance_phase"]["scenarios"]:
        if scenario.get("prompt_file"):
            add(scenario["prompt_file"], "matrix performance prompt")
    for arm in matrix["arms"]:
        for _, phase in PHASES:
            leaf = Path(arm[phase]["path"])
            for name in ("config.json", "identity.json", "provenance.json", "fresh-identities.json"):
                add(leaf / name, f"run51 phase input {name}")
            old_controller = leaf / "precreated-controller"
            add(old_controller, "preserved empty precreated controller directory")
    return [{"path": path, "role": role} for path, role in sorted(rows.items())]


def main() -> int:
    if RECEIPTS.exists():
        raise SystemExit(f"refusing to overwrite dry-validation receipt directory: {RECEIPTS}")
    RECEIPTS.mkdir(parents=True, exist_ok=False)
    if not MATRIX.is_file():
        raise SystemExit("run51 matrix is missing")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_arm = {row["arm"]: row for row in matrix["arms"]}
    if set(by_arm) != set(ARMS):
        raise SystemExit("run51 matrix arm set changed")

    copied = {}
    for arm in ARMS:
        for leaf, phase in PHASES:
            source = BASE / "arms" / arm / leaf / "fresh-identities.json"
            target = RUN / "arms" / arm / leaf / "fresh-identities.json"
            if not source.is_file():
                raise SystemExit(f"run47 fresh identity source is missing: {source}")
            raw = source.read_bytes()
            if len(raw) > 32 * 1024:
                raise SystemExit(f"unexpectedly large fresh identity metadata: {source}")
            value = json.loads(raw.decode("utf-8"))
            if value.get("all_current_identities_match") is not True:
                raise SystemExit(f"run47 fresh identity receipt does not pass: {source}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != raw:
                    raise SystemExit(f"existing run51 fresh identity is not byte-identical to run47: {target}")
                copy_state = "previously_copied_exclusively_in_attempt01_byte_identical"
            else:
                write_exclusive(target, raw)
                copy_state = "copied_exclusively"
            copied[f"{arm}/{leaf}"] = {
                "run_id": by_arm[arm][phase]["run_id"],
                "source_path": str(source.resolve()),
                "source_sha256": sha(raw),
                "source_bytes": len(raw),
                "run51_path": str(target.resolve()),
                "run51_sha256": file_sha(target),
                "run51_bytes": target.stat().st_size,
                "byte_identical": target.read_bytes() == raw,
                "copy_state": copy_state,
            }

    # All target phase controller directories must remain precreated and empty;
    # dry-validation records live only in this new receipt folder.
    for arm in ARMS:
        for leaf, phase in PHASES:
            controller_dir = RUN / "arms" / arm / leaf / "controller"
            preserved_dir = RUN / "arms" / arm / leaf / "precreated-controller"
            if controller_dir.exists() or not preserved_dir.is_dir() or any(preserved_dir.iterdir()):
                raise SystemExit(f"phase controller state is not preserved correctly: {leaf}")

    manifest_path = Path(matrix["quality_prompt_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_manifest_path = REPO / "bench/hetero/20261008-00-admission/model-integrity-02.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    inventory = expected_files(
        json.loads((RUN / "arms/w10-all/quality-on/config.json").read_text(encoding="utf-8")),
        manifest, prior_manifest, matrix)
    required_inputs = []
    missing = []
    for item in inventory:
        path = Path(item["path"])
        exists = path.is_file() or path.is_dir()
        record = {**item, "exists": exists, "kind": "file" if path.is_file() else ("directory" if path.is_dir() else "missing")}
        if path.is_file():
            record["bytes"] = path.stat().st_size
        required_inputs.append(record)
        if not exists:
            missing.append(record)

    template_hashes = sweep.template_source_hashes(REPO)
    pinned_templates = sweep.pinned_template_hashes()
    if template_hashes != pinned_templates:
        raise SystemExit("current run41 lifecycle templates differ from the pinned source hashes")
    if missing:
        write_json_exclusive(RECEIPTS / "required-input-audit.json", {
            "status": "failed_missing_dry_inputs", "missing": missing,
            "required_input_count": len(required_inputs), "inputs": required_inputs,
        })
        raise SystemExit(f"required dry inputs missing: {len(missing)}")

    phase_results = []
    failures = []
    write_json_exclusive(RECEIPTS / "required-input-audit.json", {
        "status": "all_required_dry_inputs_present",
        "required_input_count": len(required_inputs),
        "missing_count": 0,
        "inputs": required_inputs,
        "run41_template_source_sha256s": template_hashes,
        "run41_template_pinned_sha256s": pinned_templates,
        "pwsh7_sha256": sweep.PWSH7_SHA256,
    })

    def run_validation(step: dict, action: str, template: str, rendered: str) -> dict:
        safe_id = step["run_id"]
        prefix = RECEIPTS / safe_id / action
        prefix.parent.mkdir(parents=True, exist_ok=True)
        script_path = prefix.with_suffix(".ps1.txt")
        owner_path = prefix.with_suffix(".owner.json")
        write_exclusive(script_path, rendered.encode("utf-8"))
        callback_rows = []

        def on_started(owner: dict, proc) -> None:
            owner_record = {
                "status": "normal_controller_process_identity_captured",
                "purpose": "default rendered script validation only",
                "action": action,
                "template": template,
                "run_id": safe_id,
                "owner": owner,
                "started_pid": int(proc.pid),
                "admission_proof_created": False,
                "model_start_switch_passed": False,
                "monitor_or_run_switch_passed": False,
            }
            write_json_exclusive(owner_path, owner_record)
            callback_rows.append(owner_record)

        env = os.environ.copy()
        env["RUN47_LAUNCHER_PID"] = "0"
        env["RUN47_SAMPLER_PID"] = "0"
        exit_code = None
        stdout = b""
        stderr = b""
        owner = None
        error = None
        try:
            exit_code, stdout, stderr, owner = sweep._invoke_pwsh_text(
                rendered, "validate", env=env, cwd=REPO, on_started=on_started, timeout_seconds=300)
        except BaseException as exc:  # retain controller output/identity if a bounded call fails
            error = {"type": type(exc).__name__, "message": str(exc)}
            if getattr(exc, "owner", None) is not None:
                owner = exc.owner
            if getattr(exc, "stdout", None) is not None:
                stdout = exc.stdout or b""
            if getattr(exc, "stderr", None) is not None:
                stderr = exc.stderr or b""
        write_exclusive(prefix.with_suffix(".stdout.bin"), stdout)
        write_exclusive(prefix.with_suffix(".stderr.bin"), stderr)
        output_text = stdout.decode("utf-8", "replace")
        parsed = None
        try:
            parsed = json.loads(output_text.strip()) if output_text.strip() else None
        except json.JSONDecodeError:
            pass
        expected_status = {
            "launch": "prepared_validated",
            "watch": "startup_watch_prepared",
            "quality": "prepared_quality_validated",
            "performance": "prepared_quality_validated",
        }[action]
        valid = (error is None and exit_code == 0 and isinstance(parsed, dict) and
                 parsed.get("status") == expected_status)
        if action == "launch":
            valid = valid and parsed.get("start_performed") is False
        elif action == "watch":
            valid = valid and parsed.get("monitor_started") is False
        else:
            valid = valid and parsed.get("run_started") is False
        invocation = {
            "schema_version": 1,
            "status": "pass" if valid else "failed",
            "run_id": safe_id,
            "action": action,
            "template": template,
            "invocation_mode": "default parameters only; no Start, Monitor, Run, CheckReady, or Execute switch",
            "script_path": str(script_path.resolve()),
            "script_sha256": file_sha(script_path),
            "template_source_sha256": template_hashes[template],
            "exit_code": exit_code,
            "stdout_path": str(prefix.with_suffix(".stdout.bin").resolve()),
            "stdout_sha256": sha(stdout),
            "stderr_path": str(prefix.with_suffix(".stderr.bin").resolve()),
            "stderr_sha256": sha(stderr),
            "owner_path": str(owner_path.resolve()) if callback_rows else None,
            "owner_pid": owner.get("pid") if isinstance(owner, dict) else None,
            "admission_proof_created": False,
            "parsed_default_response": parsed,
            "error": error,
        }
        write_json_exclusive(prefix.with_suffix(".exit.json"), invocation)
        if not valid:
            failures.append({"run_id": safe_id, "action": action, "exit_code": exit_code,
                             "error": error, "stdout": output_text[-2000:],
                             "stderr": stderr.decode("utf-8", "replace")[-2000:]})
        return invocation

    for arm in ARMS:
        arm_row = by_arm[arm]
        for leaf, phase in PHASES:
            row = arm_row[phase]
            config_path = Path(row["path"]) / "config.json"
            step = {
                "run_id": row["run_id"], "arm": arm, "phase": phase,
                "matrix_root": str(RUN), "config": str(config_path),
                "quality_prompt_manifest_path": str(manifest_path),
                "quality_prompt_manifest_sha256": matrix["quality_prompt_manifest"]["sha256"],
                "tokenizer_implementation_path": matrix["tokenizer_implementation_snapshot"]["path"],
                "tokenizer_implementation_sha256": matrix["tokenizer_implementation_snapshot"]["sha256"],
            }
            for action, template in (("launch", "launch"), ("watch", "watch")):
                rendered = sweep.render_run41_template(template, step, REPO)
                phase_results.append(run_validation(step, action, template, rendered))
            if phase == "quality_on":
                rendered = sweep.render_run41_template("quality", step, REPO)
                action, template = "quality", "quality"
            else:
                rendered = sweep.render_performance_controller(step, REPO)
                action, template = "performance", "quality"
            phase_results.append(run_validation(step, action, template, rendered))

    receipt = {
        "schema_version": 1,
        "status": "all_default_validations_passed" if not failures else "default_validation_failures_preserved",
        "run_id": "20261011-51-worker-pool-sweep",
        "matrix_path": str(MATRIX.resolve()),
        "matrix_sha256": file_sha(MATRIX),
        "supplement_source_path": str(Path(__file__).resolve()),
        "supplement_source_sha256": file_sha(Path(__file__).resolve()),
        "prior_attempt": {
            "receipt_path": str(PRIOR_ATTEMPT_RECEIPT.resolve()),
            "receipt_sha256": file_sha(PRIOR_ATTEMPT_RECEIPT),
            "source_sha256": PRIOR_ATTEMPT_SOURCE_SHA256,
            "status": "stopped_before_input_audit_or_default_validation; original 10 copy operations retained",
        },
        "prior_audit_attempt": {
            "receipt_path": str(PRIOR_AUDIT_FAILURE_RECEIPT.resolve()),
            "receipt_sha256": file_sha(PRIOR_AUDIT_FAILURE_RECEIPT),
            "source_sha256": PRIOR_AUDIT_SOURCE_SHA256,
            "status": "stopped_on_false_positive_optional_BUILD_json_inventory_item; no default validations run",
        },
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "fresh_identity_inputs_copied_byteexact": copied,
        "phase_fresh_identity_sha256_for_matrix_version05_binding": {
            row["run_id"]: copied[f"{arm}/{leaf}"]["run51_sha256"]
            for arm in ARMS for leaf, phase in PHASES
            for row in (by_arm[arm][phase],)
        },
        "required_dry_input_count": len(required_inputs),
        "required_dry_inputs_missing": missing,
        "run41_template_source_sha256s": template_hashes,
        "default_function_validations": phase_results,
        "validation_count": len(phase_results),
        "failure_count": len(failures),
        "failures": failures,
        "controller_process_identities_captured_via_normal_cim": True,
        "admission_proof_or_probe_created": False,
        "model_api_wsl_gpu_or_compile_started": False,
        "resource_or_listener_observation_performed": False,
        "run51_preparation_json_or_matrix_modified": False,
        "run47_sources_modified": False,
    }
    write_json_exclusive(RECEIPTS / "supplement-receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "fresh_identity_copies": len(copied),
        "required_dry_input_count": len(required_inputs),
        "default_validations": len(phase_results),
        "failures": len(failures),
        "receipt": str((RECEIPTS / "supplement-receipt.json").resolve()),
        "receipt_sha256": file_sha(RECEIPTS / "supplement-receipt.json"),
        "source_sha256": receipt["supplement_source_sha256"],
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
