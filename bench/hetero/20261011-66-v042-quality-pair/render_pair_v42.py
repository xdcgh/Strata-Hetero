from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as lifecycle

RUN = Path(__file__).resolve().parent
MATRIX_PATH = RUN / "matrix.json"
BINDINGS_PATH = RUN / "template-bindings.json"
AUDIT_PATH = RUN / "controls-normalization.json"
MANIFEST = Path(r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261011-51-worker-pool-sweep\quality-manifest\manifest.json")
OLD_SERVER_SHA = "0ca2dd70485e5e03b0a6fcb5dfc1ee1a8863d2dc44947257e342ae608326b026"
V42_SERVER_SHA = "9bc794cae42317dd3283c5f41d98a5c42b4053fd5c8aae8504669ade44a63ded"
V42_SERVER_ROOT = r"E:\Strata-Hetero-data\source\hetero-0-1-42"
OLD_SERVER_ROOT = r"E:\Strata-Hetero-data\source\hetero-0-1-41"
UPSTREAM_BINARY_SHA = "b4e8f28eebd012c015905c1ffae461ba22d069d1c997a0b3ef91f1edd8e6a786"
UPSTREAM_BINARY_BYTES = 52763648
SOURCE_DATE = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha(path: Path) -> str:
    return lifecycle.sha256_file(path)

def write_json(path: Path, value) -> bytes:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw

def canonical_normalized_config(config: dict) -> tuple[dict, str]:
    value = copy.deepcopy(config)
    value["exe"] = "@ENGINE_PATH@"
    value["model_name"] = "@SIDE_LABEL@"
    value["log"] = "@SIDE_LOG@"
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return value, sha_bytes(raw)

def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one pinned template anchor, found {count}")
    return source.replace(old, new)

def render(name: str, side: dict) -> tuple[str, dict]:
    if name not in ("launch", "watch", "quality", "cleanup"):
        raise ValueError(name)
    source_path, expected_sha = lifecycle.RUN41_TEMPLATES[name]
    raw_path = REPO / source_path
    raw = raw_path.read_text(encoding="utf-8-sig")
    raw_sha = sha(raw_path)
    if raw_sha != expected_sha:
        raise RuntimeError(f"raw {name} template hash changed: {raw_sha}")
    run_id = side["run_id"]
    phase = RUN / side["arm_path"]
    source_count = raw.count(lifecycle.RUN41_ID)
    expected_count = lifecycle.RUN41_ID_REPLACEMENT_COUNTS[name]
    if source_count != expected_count:
        raise RuntimeError(f"{name}: raw run41 ID replacement count changed")
    rendered = raw.replace(lifecycle.RUN41_ID, run_id)
    if lifecycle.RUN41_ID in rendered:
        raise RuntimeError(f"{name}: old run41 RunId remains")
    if name == "launch":
        rendered = replace_once(
            rendered,
            "$run = Join-Path $repo 'bench\\hetero\\" + run_id + "'",
            "$run = '" + str(phase.resolve()) + "'",
            "launch run directory",
        )
        admission = "--required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current"
        if rendered.count(admission) != 1:
            raise RuntimeError("launch admission anchor changed")
    elif name == "watch":
        rendered = replace_once(
            rendered,
            "$runId='" + run_id + "'; $run=Join-Path $repo ('bench\\hetero\\'+$runId)",
            "$runId='" + run_id + "'; $run='" + str(phase.resolve()) + "'",
            "watch run directory",
        )
    elif name == "quality":
        rendered = replace_once(
            rendered,
            "$runDir = Join-Path $repo 'bench\\hetero\\" + run_id + "'",
            "$runDir = '" + str(phase.resolve()) + "'",
            "quality run directory",
        )
    else:
        rendered = replace_once(
            rendered,
            "$repo='C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero';$run=Join-Path $repo 'bench\\hetero\\" + run_id + "';",
            "$repo='C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero';$run='" + str(phase.resolve()) + "';",
            "cleanup run directory",
        )
    if name in ("launch", "quality"):
        if name == "launch":
            old_manifest = "$manifestPath = Join-Path $repo 'bench\\hetero\\20261009-06-quality-prompts\\manifest.json'"
        else:
            old_manifest = "$manifestPath = 'C:\\Users\\DC\\Documents\\ChatGPT\\Strata-Hetero\\bench\\hetero\\20261009-06-quality-prompts\\manifest.json'"
        rendered = replace_once(rendered, old_manifest, "$manifestPath = '" + str(MANIFEST.resolve()) + "'",
                                f"{name} manifest path")
        expected_model = "$expectedModel = 'qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero'"
        rendered = replace_once(rendered, expected_model, "$expectedModel = '" + side["model_name"] + "'",
                                "quality expected model identity")
    root_count = rendered.count(OLD_SERVER_ROOT)
    if root_count:
        rendered = rendered.replace(OLD_SERVER_ROOT, V42_SERVER_ROOT)
    if name == "launch":
        rendered = replace_once(rendered, OLD_SERVER_SHA, V42_SERVER_SHA, "launch shared bridge SHA")
    version_count = rendered.count("0.1.41")
    if name in ("watch", "quality", "cleanup"):
        if version_count < 1:
            raise RuntimeError(f"{name}: expected v0.1.41 status/version checks")
        rendered = rendered.replace("0.1.41", "0.1.42")
    if name == "cleanup":
        rendered = lifecycle._render_sampler_first_cleanup(rendered)
    if lifecycle.RUN41_ID in rendered or "0.1.41" in rendered or OLD_SERVER_ROOT in rendered:
        raise RuntimeError(f"{name}: stale version/path identifier remains in v42 rendering")
    if name == "launch" and OLD_SERVER_SHA in rendered:
        raise RuntimeError("launch: old run41 bridge hash remains")
    return rendered, {
        "raw_path": str(raw_path),
        "raw_sha256": raw_sha,
        "raw_bytes": len(raw.encode("utf-8")),
        "run_id_replacements": source_count,
        "engine_version_replacements": version_count if name in ("watch", "quality", "cleanup") else 0,
        "server_root_replacements": root_count,
        "rendered_sha256": sha_bytes(rendered.encode("utf-8")),
        "rendered_bytes": len(rendered.encode("utf-8")),
        "status": "rendered_not_executed",
    }

matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
if sha(MANIFEST) != matrix["quality_recipe"]["manifest_sha256"]:
    raise RuntimeError("nine-prompt manifest hash differs from run66 binding")
bridge_path = Path(matrix["shared_server_bridge"]["path"])
if sha(bridge_path) != V42_SERVER_SHA:
    raise RuntimeError("shared E42 bridge does not match the bound hash")
if sha(Path(lifecycle.__file__).resolve()) != "1942736d217f376f82702f62027302de70f8ff222919d38a4916b57627b6bd8f":
    raise RuntimeError("shared sampler-first cleanup renderer source changed")
if bindings["cleanup_order_fix"]["required_function"] != "_render_sampler_first_cleanup":
    raise RuntimeError("cleanup renderer binding is not the reviewed function")
if matrix.get("authorization") is not None or matrix.get("execution_enabled") is not False:
    raise RuntimeError("run66 must remain un-authorized and non-executable")

# Root supplied an immutable build receipt for the upstream binary. The Hetero
# binary is still building and must remain pending until root rehashes it.
matrix["upstream"]["binary_sha256"] = UPSTREAM_BINARY_SHA
matrix["upstream"]["binary_bytes"] = UPSTREAM_BINARY_BYTES
matrix["upstream"]["binary_identity_status"] = "ROOT_BUILD_RECEIPT_BOUND_FRESH_LAUNCH_REHASH_REQUIRED"
matrix["hetero"]["binary_identity_status"] = "PENDING_BUILD_COMPLETION_AND_ROOT_REHASH"
matrix["sides"][0]["binary_sha256"] = UPSTREAM_BINARY_SHA
matrix["sides"][0]["binary_bytes"] = UPSTREAM_BINARY_BYTES
matrix["sides"][1]["binary_sha256"] = None
matrix["sides"][1]["binary_bytes"] = None

rendered_index = {}
normalized_configs = {}
for side in matrix["sides"]:
    phase = RUN / side["arm_path"]
    if not phase.is_dir():
        raise RuntimeError(f"phase files missing for {side['id']}")
    config_path = phase / "config.json"
    identity_path = phase / "identity.json"
    provenance_path = phase / "provenance.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized_configs[side["id"]], normalized_sha = canonical_normalized_config(config)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if side["id"] == "upstream42":
        identity["source"]["binary_hash_finalized"] = True
        identity["engine"].update({"sha256": UPSTREAM_BINARY_SHA, "size_bytes": UPSTREAM_BINARY_BYTES,
                                   "hash_status": "ROOT_BUILD_RECEIPT_BOUND_FRESH_LAUNCH_REHASH_REQUIRED"})
        provenance["engine_sha256"] = UPSTREAM_BINARY_SHA
    else:
        identity["source"]["binary_hash_finalized"] = False
        identity["engine"].update({"sha256": None, "size_bytes": None,
                                   "hash_status": "PENDING_BUILD_COMPLETION_AND_ROOT_REHASH"})
        provenance["engine_sha256"] = None
    config_sha = sha(config_path)
    identity["config"]["sha256"] = config_sha
    identity_sha = sha_bytes(write_json(identity_path, identity))
    provenance["config_sha256"] = config_sha
    provenance_sha = sha_bytes(write_json(provenance_path, provenance))
    prep_path = phase / "preparation.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    prep.update({"config_sha256": config_sha, "identity_sha256": identity_sha,
                 "provenance_sha256": provenance_sha})
    prep["binary"] = {"sha256": side.get("binary_sha256"), "bytes": side.get("binary_bytes"),
                      "status": "root_receipt_bound_fresh_launch_rehash_required" if side["id"] == "upstream42"
                      else "pending_build_completion_and_root_rehash"}
    prep["engine_sha256"] = side.get("binary_sha256")
    prep["engine_size_bytes"] = side.get("binary_bytes")
    prep["status"] = "prepared_not_launched_root_receipt_bound_fresh_launch_rehash_required" if side["id"] == "upstream42" else "prepared_not_launched_binary_identity_pending"
    prep_sha = sha_bytes(write_json(prep_path, prep))
    side.update({"config_sha256": config_sha, "identity_sha256": identity_sha,
                 "provenance_sha256": provenance_sha, "preparation_sha256": prep_sha})
    scripts = {}
    out_dir = RUN / "rendered-scripts" / side["id"]
    raw_dir = RUN / "source-templates" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (rel, expected) in lifecycle.RUN41_TEMPLATES.items():
        source_path = REPO / rel
        raw_target = raw_dir / source_path.name
        if raw_target.exists():
            if sha(raw_target) != expected:
                raise RuntimeError(f"captured {name} raw source hash mismatch")
        else:
            shutil.copy2(source_path, raw_target)
        if name not in ("launch", "watch", "quality", "cleanup"):
            continue
        rendered, meta = render(name, side)
        rendered_path = out_dir / ({"watch": "startup-watch"}.get(name, name) + ".ps1")
        rendered_path.write_text(rendered, encoding="utf-8", newline="\n")
        meta.update({"rendered_path": str(rendered_path.resolve()),
                     "rendered_sha256": sha(rendered_path)})
        lifecycle.validate_powershell_syntax(rendered)
        meta["parser_status"] = "pass_parse_only"
        scripts[name] = meta
    rendered_index[side["id"]] = scripts
    side["rendered_scripts"] = scripts

if normalized_configs["upstream42"] != normalized_configs["hetero42"]:
    raise RuntimeError("v42 pair config mismatch after only exe/model_name/log normalization")
canonical_raw = json.dumps(normalized_configs["upstream42"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
canonical_sha = sha_bytes(canonical_raw)
if canonical_sha != normalized_sha:
    raise RuntimeError("canonical normalized config hash changed during pair comparison")

bindings["rendered_scripts"] = rendered_index
bindings["rendered_status"] = "rendered_and_parser_checked_no_script_invocation"
bindings["status"] = "rendered_source_only_pending_hetero_binary_hash_and_root_authorization"
bindings["scripts_rendered_or_executed"] = {"rendered": True, "executed": False}
bindings["expected_v42_render_parameters"]["script_hashes"] = "bound below by rendered_scripts entries"
bindings["generated_cleanup_renderer"] = {
    "function": "_render_sampler_first_cleanup",
    "source_path": str(Path(lifecycle.__file__).resolve()),
    "source_sha256": sha(Path(lifecycle.__file__).resolve()),
    "source_receipt": "bench/hetero/20261011-67-lifecycle-sampler-order/hostvalidation-03/result.json",
    "result": "applied to cleanup output after v42 run/path substitutions",
}
audit["normalization_status"] = "structural_pair_match_asserted"
audit["status"] = "exact_normalized_json_pair_match"
audit["config_hashes"] = {s["id"]: s["config_sha256"] for s in matrix["sides"]}
audit["identity_hashes"] = {s["id"]: s["identity_sha256"] for s in matrix["sides"]}
audit["provenance_hashes"] = {s["id"]: s["provenance_sha256"] for s in matrix["sides"]}
audit["canonical_normalized_config_sha256"] = canonical_sha
audit["normalization_status"] = "exact_normalized_json_pair_match"
audit["normalization_verification"] = {
    "status": "pass",
    "normalized_fields": ["exe", "model_name", "log"],
    "all_other_config_fields_equal": True,
    "canonical_normalized_config_sha256": canonical_sha,
    "route_tail_skip": "0 on both",
    "strata_diag_zero_logits": "absent on both",
}
audit["normalized_config_equality"] = "pass_exact_json_equality_after_only_exe_model_name_log_replacement"
audit["normalization_verification"] = {
    "status": "pass_exact_json_equality",
    "normalized_fields": ["exe", "model_name", "log"],
    "all_other_config_fields_equal": True,
    "canonical_normalized_config_sha256": canonical_sha,
    "route_tail_skip": "0 on both",
    "strata_diag_zero_logits": "absent on both",
}
bindings["canonical_normalized_config_sha256"] = canonical_sha
bindings["normalized_config_equality"] = "pass_exact_json_equality_after_only_exe_model_name_log_replacement"
bindings["upstream_binary_binding"] = {
    "sha256": UPSTREAM_BINARY_SHA,
    "bytes": UPSTREAM_BINARY_BYTES,
    "basis": "root-provided run64 build receipt; requires fresh launch-time file identity recheck",
}
bindings["hetero_binary_binding"] = {
    "sha256": None,
    "bytes": None,
    "status": "pending root confirmation after build completion",
}
bindings["runtime_and_quality_source_pins"] = {
    role: {"path": path, "sha256": sha(REPO / path)}
    for role, path in {
        "admission": "tools/hetero_admission.py",
        "resource_monitor": "tools/hetero_resources.py",
        "quality_runner": "tools/hetero_quality.py",
        "asset_stat_verifier": "tools/hetero_verify_existing.py",
        "jsonl_reader": "tools/hetero_jsonl_tail.ps1",
    }.items()
}
required_role_hashes = {
    "admission": "b9dfad2e25840ea27afbbb39f08555607448a85db3bf6f10eb41699f73f1ca3e",
    "resource_monitor": "5d0fc03363304f1fa2d7e6a8325775467b0a240122d78466078b4340395aa285",
    "jsonl_reader": "52e15e5fbe3f798efeeb91d0ad09d46fc0ac3ccd737ef7073e63f46b82984e1d",
}
for role, expected in required_role_hashes.items():
    actual = bindings["runtime_and_quality_source_pins"][role]["sha256"]
    if actual != expected:
        raise RuntimeError(f"frozen {role} source hash changed: {actual}")
bindings["runtime_and_quality_source_pins_status"] = "frozen_admission_resource_jsonl_hashes_match"
bindings["shared_input_pins"] = {
    "quality_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST)},
    "tokenizer_snapshot": {"path": str(REPO / "bench/hetero/20261011-51-worker-pool-sweep/tokenizer-source-01/strata_tokenizer-current-461b8afb.py"),
                           "sha256": "461b8afb2139f5efee58ee8f725039634bb38c3f6040d3004555ddb912b1cf00"},
    "tokenizer_equivalence_receipt_sha256": "aa0921ffc49be8e9c54212fc3e59ed113607ff568d7b2f89880d0912a3c1271d",
}
write_json(BINDINGS_PATH, bindings)
write_json(AUDIT_PATH, audit)
matrix["template_bindings_sha256"] = sha(BINDINGS_PATH)
matrix["controls_normalization_sha256"] = sha(AUDIT_PATH)
matrix["status"] = "rendered_source_review_pending_hetero_binary_and_root_authorization"
matrix["prepared_at_utc"] = SOURCE_DATE
write_json(MATRIX_PATH, matrix)
preparation = json.loads((RUN / "preparation.json").read_text(encoding="utf-8"))
preparation["status"] = "rendered_source_review_pending_hetero_binary_hash_root_authorization"
preparation["binaries"]["upstream42"]["sha256"] = UPSTREAM_BINARY_SHA
preparation["binaries"]["upstream42"]["bytes"] = UPSTREAM_BINARY_BYTES
preparation["binaries"]["upstream42"]["status"] = "root_receipt_bound_fresh_launch_rehash_required"
preparation["binaries"]["hetero42"]["sha256"] = None
preparation["binaries"]["hetero42"]["bytes"] = None
preparation["binaries"]["hetero42"]["status"] = "pending_build_completion_and_root_rehash"
preparation["binaries"]["upstream42"]["identity_file_sha256"] = matrix["sides"][0]["identity_sha256"]
preparation["binaries"]["hetero42"]["identity_file_sha256"] = matrix["sides"][1]["identity_sha256"]
preparation["rendered_script_count"] = 8
preparation["rendered_scripts_executed"] = False
preparation["template_bindings_sha256"] = sha(BINDINGS_PATH)
preparation["controls_normalization_sha256"] = sha(AUDIT_PATH)
preparation["rendered_scripts"] = rendered_index
prep_path = RUN / "preparation.json"
prep_raw = (json.dumps(preparation, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
prep_path.write_bytes(prep_raw)
receipt = json.loads((RUN / "preparation-receipt.json").read_text(encoding="utf-8"))
receipt["status"] = "rendered_and_parser_checked_source_only_hetero_binary_hash_pending"
receipt["matrix_sha256"] = sha(MATRIX_PATH)
receipt["preparation_sha256"] = sha_bytes(prep_raw)
receipt["template_bindings_sha256"] = sha(BINDINGS_PATH)
receipt["controls_normalization_sha256"] = sha(AUDIT_PATH)
receipt["rendered_script_count"] = sum(len(x) for x in rendered_index.values())
receipt["rendered_script_execution"] = False
receipt["binary_hashes"] = "UPSTREAM_ROOT_RECEIPT_BOUND; HETERO_PENDING_BUILD_COMPLETION_AND_ROOT_REHASH"
receipt["fresh_input_stat_recheck"] = "PENDING_ROOT_LAUNCH_ADMISSION"
receipt["template_render_hashes"] = "BOUND_IN_TEMPLATE_BINDINGS_JSON; ROOT_REVIEW_PENDING"
receipt["upstream_binary_sha256"] = UPSTREAM_BINARY_SHA
receipt["upstream_binary_bytes"] = UPSTREAM_BINARY_BYTES
receipt["hetero_binary_sha256"] = None
receipt["hetero_binary_bytes"] = None
write_json(RUN / "preparation-receipt.json", receipt)
print(json.dumps({
    "status": receipt["status"],
    "matrix_sha256": receipt["matrix_sha256"],
    "template_bindings_sha256": receipt["template_bindings_sha256"],
    "controls_normalization_sha256": receipt["controls_normalization_sha256"],
    "rendered_script_count": receipt["rendered_script_count"],
    "binary_hashes_pending": True,
    "authorization_present": False,
    "model_api_device_or_build_action": False,
}, indent=2))
