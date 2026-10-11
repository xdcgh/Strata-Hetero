from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as sweep

RUN = Path(__file__).resolve().parent
SOURCE = REPO / "bench" / "hetero" / "20261011-57-direct-zero-logits-diagnostic"
DEST_MATRIX = RUN / "matrix.json"
SOURCE_MATRIX = SOURCE / "matrix.json"


def sha(path: Path) -> str:
    return sweep.sha256_file(path)


def save_exclusive(path: Path, obj: object) -> bytes:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


old, new = json.loads(SOURCE_MATRIX.read_text(encoding="utf-8")), json.loads(DEST_MATRIX.read_text(encoding="utf-8"))
if old["candidate_binary_sha256"] != new["candidate_binary_sha256"] or old["candidate_source_sha"] != new["candidate_source_sha"]:
    raise SystemExit("candidate binary/source identity changed")
if old["shared_bridge_sha256"] != new["shared_bridge_sha256"]:
    raise SystemExit("shared bridge identity changed")
if old["fixed_controls"] != new["fixed_controls"]:
    raise SystemExit("fixed benchmark controls changed")

source_manifest = json.loads((SOURCE / "quality-manifest" / "manifest.json").read_text(encoding="utf-8"))
new_manifest = json.loads((RUN / "quality-manifest" / "manifest.json").read_text(encoding="utf-8"))
prompt_fields = ("id", "bytes", "sha256", "body_tokens_local", "chat_prompt_tokens_local",
                 "chat_wrapper_delta_local", "max_output_tokens", "quality")
source_prompts = [{key: prompt.get(key) for key in prompt_fields} for prompt in source_manifest["prompts"]]
new_prompts = [{key: prompt.get(key) for key in prompt_fields} for prompt in new_manifest["prompts"]]
if source_prompts != new_prompts or len(new_prompts) != 9:
    raise SystemExit("quality prompt bytes/expectations/local tokenizer counts changed")
prompt_file_comparisons = []
for source_prompt, new_prompt in zip(source_manifest["prompts"], new_manifest["prompts"]):
    old_path = SOURCE / "quality-manifest" / source_prompt["file"]
    new_path = RUN / "quality-manifest" / new_prompt["file"]
    old_bytes, new_bytes = old_path.read_bytes(), new_path.read_bytes()
    if old_bytes != new_bytes:
        raise SystemExit(f"quality prompt bytes changed: {source_prompt['id']}")
    prompt_file_comparisons.append({"id": source_prompt["id"], "bytes": len(new_bytes),
                                    "sha256": hashlib.sha256(new_bytes).hexdigest()})
if sha(SOURCE / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py") != sha(
        RUN / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py"):
    raise SystemExit("tokenizer implementation source changed")

old_arms, new_arms = {arm["arm"]: arm for arm in old["arms"]}, {arm["arm"]: arm for arm in new["arms"]}
phase_controls = []
for arm_id in sweep.ARM_EXECUTION_ORDER:
    if (old_arms[arm_id]["workers"], old_arms[arm_id]["affinity"]) != (new_arms[arm_id]["workers"], new_arms[arm_id]["affinity"]):
        raise SystemExit(f"worker/affinity control changed: {arm_id}")
    for phase_key in ("quality_on", "performance_off"):
        source_leaf = Path(old_arms[arm_id][phase_key]["path"])
        new_leaf = Path(new_arms[arm_id][phase_key]["path"])
        source_config = json.loads((source_leaf / "config.json").read_text(encoding="utf-8"))
        new_config = json.loads((new_leaf / "config.json").read_text(encoding="utf-8"))
        source_log, new_log = source_config.pop("log"), new_config.pop("log")
        if source_config != new_config:
            raise SystemExit(f"normalized config differs beyond run-local log path: {arm_id}/{phase_key}")
        expected_diag = arm_id == "w10-all" and phase_key == "quality_on"
        if (new_config.get("env", {}).get("STRATA_DIAG_ZERO_LOGITS") == "1") is not expected_diag:
            raise SystemExit(f"diagnostic env scope mismatch: {arm_id}/{phase_key}")
        phase_controls.append({"arm": arm_id, "phase": phase_key, "workers": new_arms[arm_id]["workers"],
                               "affinity": new_arms[arm_id]["affinity"], "capture_token_ids": new_config["hetero_capture_token_ids"],
                               "only_config_difference": "run-local log path", "source_log": source_log, "new_log": new_log,
                               "diagnostic_env": "1" if expected_diag else None})

source_decode = next(x for x in old["performance_phase"]["scenarios"] if x["id"] == "decode-cap-256")
new_decode = next(x for x in new["performance_phase"]["scenarios"] if x["id"] == "decode-cap-256")
if (Path(source_decode["prompt_file"]).read_bytes() != Path(new_decode["prompt_file"]).read_bytes() or
        source_decode["prompt_sha256"] != new_decode["prompt_sha256"] or source_decode["prompt_bytes"] != new_decode["prompt_bytes"]):
    raise SystemExit("sustained decode prompt bytes changed")

receipt = {
    "schema_version": 1, "status": "run57_controls_and_prompt_assets_match_run63",
    "source_matrix": str(SOURCE_MATRIX.resolve()), "source_matrix_sha256": sha(SOURCE_MATRIX),
    "new_matrix": str(DEST_MATRIX.resolve()), "new_matrix_sha256": sha(DEST_MATRIX),
    "candidate_binary_sha256": new["candidate_binary_sha256"], "candidate_source_sha": new["candidate_source_sha"],
    "shared_bridge_sha256": new["shared_bridge_sha256"], "fixed_controls_equal": True,
    "quality_prompt_count": len(new_prompts), "quality_prompt_expectations_and_local_counts_equal": True,
    "quality_prompt_files": prompt_file_comparisons, "tokenizer_source_sha256": sha(RUN / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py"),
    "sustained_decode_prompt_sha256": new_decode["prompt_sha256"], "normalized_configs_equal_except_log_path": True,
    "phase_controls": phase_controls, "models_or_runtime_assets_copied": False,
    "request_output_leaves_created": 0, "authorization_or_execution": False,
    "audit_source_sha256": sha(Path(__file__)),
}
save_exclusive(RUN / "controls-audit.json", receipt)
print(json.dumps({"status": receipt["status"], "matrix_sha256": receipt["new_matrix_sha256"],
                  "audit_sha256": sha(RUN / "controls-audit.json"), "quality_prompts": len(new_prompts),
                  "phase_configs": len(phase_controls)}, sort_keys=True))
