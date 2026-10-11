from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
SRC = REPO / "bench/hetero/20261011-51-worker-pool-sweep"
MATRIX_PATH = RUN / "matrix.json"
RUN_ID = RUN.name

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    value["log"] = "@RUN_LOCAL_LOG@"
    return value

matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
source_matrix = json.loads((SRC / "matrix-tokenizer-rebased-05.json").read_text(encoding="utf-8"))
checks = []
def record(name: str, passed: bool, detail):
    checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})
    if not passed:
        raise SystemExit(f"contract audit failed: {name}")

record("run_id_and_source_pins",
       matrix.get("run_id") == RUN_ID and
       matrix.get("candidate_binary_sha256") == source_matrix.get("candidate_binary_sha256") ==
       "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb" and
       matrix.get("candidate_source_sha") == source_matrix.get("candidate_source_sha") ==
       "ffeb748d190ddeedff23587ce848596d4db0c752",
       {"binary_sha256": matrix.get("candidate_binary_sha256"), "source": matrix.get("candidate_source_sha")})

config_deltas = []
for arm in matrix["arms"]:
    for phase, phase_leaf in (("quality_on", "quality-on"), ("performance_off", "performance-off")):
        source_leaf = SRC / "arms" / arm["arm"] / phase_leaf
        leaf = RUN / "arms" / arm["arm"] / phase_leaf
        old = json.loads((source_leaf / "config.json").read_text(encoding="utf-8"))
        new = json.loads((leaf / "config.json").read_text(encoding="utf-8"))
        diagnostic_phase = arm["arm"] == "w10-all" and phase == "quality_on"
        env_old = old.get("env", {})
        env_new = new.get("env", {})
        expected_env = dict(env_old)
        if diagnostic_phase:
            expected_env["STRATA_DIAG_ZERO_LOGITS"] = "1"
        check_config = dict(new)
        check_config["env"] = expected_env
        allowed_config = dict(old)
        allowed_config["env"] = expected_env
        allowed_config["log"] = new["log"]
        equal = check_config == allowed_config
        record(f"config_only_runid_log_and_expected_diagnostic:{arm['arm']}:{phase}", equal,
               {"diagnostic_phase": diagnostic_phase, "environment_added": sorted(set(env_new) - set(env_old)),
                "log": new["log"]})
        config_deltas.append({"arm": arm["arm"], "phase": phase, "diagnostic_phase": diagnostic_phase,
                              "added_environment": {"STRATA_DIAG_ZERO_LOGITS": "1"} if diagnostic_phase else {},
                              "source_config_sha256": sha(source_leaf / "config.json"),
                              "run57_config_sha256": sha(leaf / "config.json")})
        record(f"config_hash_binding:{arm['arm']}:{phase}",
               sha(leaf / "config.json") == arm[phase]["config_sha256"],
               {"sha256": sha(leaf / "config.json")})
        record(f"identity_hash_binding:{arm['arm']}:{phase}",
               sha(leaf / "identity.json") == arm[phase]["identity_sha256"],
               {"sha256": sha(leaf / "identity.json")})
        record(f"provenance_hash_binding:{arm['arm']}:{phase}",
               sha(leaf / "provenance.json") == arm[phase]["provenance_sha256"],
               {"sha256": sha(leaf / "provenance.json")})
        record(f"fresh_identity_hash_binding:{arm['arm']}:{phase}",
               sha(leaf / "fresh-identities.json") == arm[phase]["fresh_identities_sha256"],
               {"sha256": sha(leaf / "fresh-identities.json")})
        for name in ("logs", "resource", "quality", "performance", "cleanup"):
            record(f"empty_runtime_dir:{arm['arm']}:{phase}:{name}",
                   not any((leaf / name).iterdir()), {"path": str(leaf / name)})
        record(f"empty_request_dirs:{arm['arm']}:{phase}",
               all(not any(p.iterdir()) for p in (leaf / "requests").iterdir()),
               {"prompt_dirs": len(list((leaf / "requests").iterdir()))})
        record(f"phase_controller_absent:{arm['arm']}:{phase}",
               not (leaf / "controller").exists(), {"controller_absent": True})

record("exactly_ten_phase_configs", len(config_deltas) == 10,
       {"phase_configs": len(config_deltas)})
record("diagnostic_in_one_phase_only",
       sum(bool(d["added_environment"]) for d in config_deltas) == 1,
       {"phase": "w10-all/quality_on", "env": "STRATA_DIAG_ZERO_LOGITS=1"})

source_manifest = json.loads((SRC / "quality-manifest/manifest.json").read_text(encoding="utf-8"))
run_manifest_path = RUN / "quality-manifest/manifest.json"
run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
record("nine_prompt_records_and_expectations_identical",
       source_manifest["prompts"] == run_manifest["prompts"] and len(run_manifest["prompts"]) == 9,
       {"prompt_ids": [p["id"] for p in run_manifest["prompts"]]})
prompt_bytes = []
for prompt in run_manifest["prompts"]:
    oldp = SRC / "quality-manifest" / prompt["file"]
    newp = RUN / "quality-manifest" / prompt["file"]
    same = oldp.read_bytes() == newp.read_bytes() and sha(oldp) == prompt["sha256"] == sha(newp)
    record(f"prompt_bytes_identical:{prompt['id']}", same, {"sha256": sha(newp), "bytes": newp.stat().st_size})
    prompt_bytes.append({"id": prompt["id"], "bytes": newp.stat().st_size, "sha256": sha(newp)})
record("manifest_and_tokenizer_bindings",
       sha(run_manifest_path) == matrix["quality_prompt_manifest"]["sha256"] and
       Path(run_manifest["tokenizer"]["implementation_source"]["path"]).resolve() ==
       (RUN / "tokenizer-source-01/strata_tokenizer-current-461b8afb.py").resolve() and
       sha(Path(run_manifest["tokenizer"]["implementation_source"]["path"])) ==
       matrix["tokenizer_implementation_snapshot"]["sha256"],
       {"manifest_sha256": sha(run_manifest_path), "tokenizer_sha256": matrix["tokenizer_implementation_snapshot"]["sha256"]})
record("manifest_provenance_binding",
       sha(RUN / "tokenizer-manifest-provenance.json") == matrix["tokenizer_manifest_provenance"]["sha256"],
       {"sha256": sha(RUN / "tokenizer-manifest-provenance.json")})
record("equivalence_binding",
       sha(RUN / "tokenizer-equivalence.json") == matrix["tokenizer_implementation_snapshot"]["equivalence_receipt_sha256"],
       {"sha256": sha(RUN / "tokenizer-equivalence.json")})

capture = json.loads((RUN / "source-capture/source-template-capture.json").read_text(encoding="utf-8"))
record("engine_source_diagnostic_order",
       capture["diagnostic_source"]["readback_after_sampling_selected_id_zero_and_window_sync"] and
       capture["diagnostic_source"]["adds_prefill_gemm_synchronization"] is False and
       capture["diagnostic_source"]["sha256"] == sha(RUN / "source-capture/engine-source/src/core/verify.cpp"),
       capture["diagnostic_source"])
dry = json.loads((RUN / "receipts/dry-validation-01/dry-validation.json").read_text(encoding="utf-8"))
record("default_scripts_parser_only",
       dry["status"] == "all_default_first_phase_scripts_parse_only_validated" and
       [x["template"] for x in dry["actions"]] == ["launch", "watch", "quality"] and
       all(x["status"] == "pass" and x["action"] == "validate_only_parse_no_script_evaluation" for x in dry["actions"]),
       {"actions": [{"template": x["template"], "pid": x["process"]["pid"],
                     "exit_code": x["process"]["exit_code"], "stdout_sha256": x["stdout_sha256"],
                     "stderr_sha256": x["stderr_sha256"]} for x in dry["actions"]]})
record("no_authorization_or_controller",
       not (RUN / "authorization.json").exists() and not (RUN / "controller").exists(),
       {"authorization_absent": True, "controller_absent": True})
record("diagnostic_not_launched",
       preparation_status := json.loads((RUN / "preparation.json").read_text(encoding="utf-8")).get("runtime_start_or_request_or_build") is False,
       {"runtime_start_request_build": False})

audit = {
    "schema_version": 1,
    "status": "all_run57_preparation_contract_checks_passed",
    "run_id": RUN_ID,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "matrix_sha256": sha(MATRIX_PATH),
    "source_run51_v05_sha256": sha(SRC / "matrix-tokenizer-rebased-05.json"),
    "config_deltas": config_deltas,
    "quality_prompts": prompt_bytes,
    "checks": checks,
    "scope": "file/config/hash/manifest verification only; no hardware, process-tree, admission, server, API, request, GPU, WSL, or build observation",
}
out = RUN / "contract-audit.json"
out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"status": audit["status"], "check_count": len(checks), "matrix_sha256": audit["matrix_sha256"],
                  "audit_path": str(out), "audit_sha256": sha(out)}, indent=2))
