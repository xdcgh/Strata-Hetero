from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO = Path(r"C:\Users\DC\Documents\ChatGPT\Strata-Hetero")
RUN = REPO / "bench/hetero/20261025-73-upstream-control"
RUN_ID = "20261025-73-upstream-control"
SIDE_ID = "upstream73"
OLD_RUN = REPO / "bench/hetero/20261011-66-v042-quality-pair"
OLD_RUN_ID = "20261011-66-upstream42-quality"
OLD_ARM = OLD_RUN / "arms/upstream42/quality-on"
OLD_SCRIPTS = OLD_RUN / "rendered-scripts/upstream42"
SOURCE = Path(r"E:\Strata-Hetero-data\source\upstream-61b3fb5d-build-01")
SOURCE_HEAD = "61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406"
BINARY = Path(r"E:\Strata-Hetero-data\build\cuda-upstream42-run64-01\strata.exe")
BINARY_SHA = "b4e8f28eebd012c015905c1ffae461ba22d069d1c997a0b3ef91f1edd8e6a786"
BINARY_BYTES = 52_763_648
BRIDGE = Path(r"E:\Strata-Hetero-data\source\hetero-0-1-42\serve\server.py")
BRIDGE_SHA = "9bc794cae42317dd3283c5f41d98a5c42b4053fd5c8aae8504669ade44a63ded"
PWSH = Path(r"C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe")
PWSH_SHA = "362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139"
PROMPT_IDS = (
    "smoke_arithmetic", "smoke_unicode", "smoke_json_arithmetic", "smoke_python_function",
    "smoke_three_key_retrieval", "long_1k",
)
MANIFEST_IDS = PROMPT_IDS + ("long_4k", "long_16k", "long_30k7")
OLD_RUN_DIR = str(OLD_ARM).replace("/", "\\")
NEW_RUN_DIR = str((RUN / "arms/upstream73/quality-on").resolve()).replace("/", "\\")
DIAGNOSTIC_ENV_KEYS = ("STRATA_PREFILL_FINITE_PROBE", "STRATA_DIAG_ZERO_LOGITS", "STRATA_PREFILL_DETAIL_LAYER")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one source anchor, found {count}")
    return text.replace(old, new, 1)


def pinned_source_identity() -> dict:
    head = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    tree = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD^{tree}"],
                          capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(SOURCE), "status", "--porcelain=v1", "--untracked-files=all"],
                            capture_output=True, text=True, check=False)
    if head.returncode or tree.returncode or status.returncode:
        raise RuntimeError("upstream source git identity could not be read")
    if head.stdout.strip() != SOURCE_HEAD or status.stdout.strip():
        raise ValueError("upstream source does not match the required clean pinned HEAD")
    if not BINARY.is_file() or BINARY.stat().st_size != BINARY_BYTES or sha_file(BINARY) != BINARY_SHA:
        raise ValueError("upstream binary path, size, or SHA differs from the authorized pin")
    if not BRIDGE.is_file() or sha_file(BRIDGE) != BRIDGE_SHA:
        raise ValueError("shared v0.1.42 bridge hash differs from Run66")
    return {"head": SOURCE_HEAD, "tree": tree.stdout.strip(), "clean": True,
            "binary_path": str(BINARY), "binary_sha256": BINARY_SHA, "binary_bytes": BINARY_BYTES,
            "bridge_path": str(BRIDGE), "bridge_sha256": BRIDGE_SHA}


def render_quality(source: str, new_run_dir: str) -> str:
    result = source.replace(OLD_RUN_ID, RUN_ID)
    if OLD_RUN_ID in result:
        raise ValueError("quality template retained the old RunId")
    result = result.replace(OLD_RUN_DIR, new_run_dir)
    old_ids = "$expectedIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k','long_4k','long_16k','long_30k7')"
    new_ids = "$expectedIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k')\n$expectedManifestIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k','long_4k','long_16k','long_30k7')"
    result = replace_once(result, old_ids, new_ids, "six-prompt prefix")
    start = "    $manifestCheck = @'\n"
    end = "\n'@\n    $idsJson = ConvertTo-Json -InputObject $expectedIds -Compress"
    if result.count(start) != 1 or result.count(end) != 1:
        raise ValueError("quality manifest checker anchors changed")
    checker = r'''    $manifestCheck = @'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest_path=pathlib.Path(sys.argv[2]); expected_ids=json.loads(__import__("base64").b64decode(sys.argv[3]))
all_ids=["smoke_arithmetic","smoke_unicode","smoke_json_arithmetic","smoke_python_function","smoke_three_key_retrieval","long_1k","long_4k","long_16k","long_30k7"]
sys.path.insert(0,str(root/"tools"))
from hetero_quality import load_manifest
from hetero_compare_quality import verify_tokenizer_manifest
m,by_id=load_manifest(manifest_path)
prompts=m.get("prompts",[])
if [x.get("id") for x in prompts]!=all_ids: raise SystemExit("pinned nine-prompt manifest order changed")
if [x.get("id") for x in prompts[:len(expected_ids)]]!=expected_ids: raise SystemExit("six-prompt baseline prefix changed")
caps={x:(256 if x=="smoke_python_function" else 128) for x in all_ids}
checks=[]
for p in prompts:
    path=(manifest_path.parent/p["file"]).resolve(strict=True); data=path.read_bytes()
    if len(data)!=p["bytes"] or hashlib.sha256(data).hexdigest()!=p["sha256"]: raise SystemExit("prompt bytes/SHA mismatch: "+p["id"])
    if p.get("chat_prompt_tokens_local") is None or p.get("max_output_tokens",0)<1: raise SystemExit("prompt counts missing: "+p["id"])
    checks.append({"id":p["id"],"body_tokens":p["body_tokens_local"],"chat_tokens":p["chat_prompt_tokens_local"],"max_tokens":caps[p["id"]],"sha256":p["sha256"]})
tok=verify_tokenizer_manifest(m)
if tok.get("status")!="pass": raise SystemExit("tokenizer asset/source verification failed: "+json.dumps(tok))
selected=[p for p in checks if p["id"] in expected_ids]
print(json.dumps({"status":"pass","manifest_prompt_count":len(checks),"prompt_count":len(selected),"prompts":selected,"tokenizer":tok},separators=(",",":")))
'@
    $idsJson = ConvertTo-Json -InputObject $expectedIds -Compress'''
    before = result.split(start, 1)[0]
    after = result.split(end, 1)[1]
    result = before + checker + after
    result = replace_once(result,
        "if ($manifestInfo.status -ne 'pass' -or $manifestInfo.prompt_count -ne 9) { throw 'manifest validation result was incomplete' }",
        "if ($manifestInfo.status -ne 'pass' -or $manifestInfo.manifest_prompt_count -ne 9 -or $manifestInfo.prompt_count -ne $expectedIds.Count) { throw 'manifest/prefix validation result was incomplete' }",
        "manifest and prefix counts")
    result = replace_once(result,
        "[ordered]@{status='prepared_quality_validated';run_started=$false;prompt_count=9;formal_per_prompt=3;warmups_per_prompt=1;\n        caps='128 for eight non-code tasks, 256 for smoke_python_function';manifest_sha256=$contract.manifest_sha256;",
        "[ordered]@{status='prepared_quality_validated';run_started=$false;prompt_count=6;manifest_prompt_count=9;formal_per_prompt=3;warmups_per_prompt=1;\n        caps='128 for five prompts, 256 for smoke_python_function';manifest_sha256=$contract.manifest_sha256;",
        "six-prompt default quality validation")
    result = replace_once(result,
        "$preflight = [ordered]@{status='live_quality_preflight_pass';chat_requests_sent=0;prompt_count=$contract.manifest.prompts.Count;",
        "$preflight = [ordered]@{status='live_quality_preflight_pass';chat_requests_sent=0;prompt_count=$expectedIds.Count;manifest_prompt_count=$contract.manifest.prompts.Count;",
        "six-prompt ready preflight")
    selected = "$selectedPrompts = @()\nforeach ($expectedId in $expectedIds) {\n    $match = @($contract.manifest.prompts | Where-Object { $_.id -eq $expectedId })\n    if ($match.Count -ne 1) { throw \"$expectedId is not unique in the validated manifest\" }\n    $selectedPrompts += $match[0]\n}\n"
    result = replace_once(result, "$config = $contract.config\n", "$config = $contract.config\n" + selected,
                          "selected six prompt list")
    result = replace_once(result, "foreach ($prompt in $contract.manifest.prompts) {",
                          "foreach ($prompt in $selectedPrompts) {", "selected prompt execution loop")
    final_start = f"    $final = [ordered]@{{schema_version=1;run_id='{RUN_ID}';status='nine_prompt_matrix_complete';"
    final_end = "sampler_remains_owned_for_root_cleanup=$true} | ConvertTo-Json -Compress"
    begin = result.find(final_start)
    finish = result.find(final_end, begin)
    if begin < 0 or finish < 0:
        raise ValueError("nine-prompt final receipt anchors changed")
    finish += len(final_end)
    final_block = f'''    $remaining=@('long_4k','long_16k','long_30k7')
    $final = [ordered]@{{schema_version=1;run_id='{RUN_ID}';status='six_prompt_baseline_replay_complete';
        completed_utc=[DateTime]::UtcNow.ToString('o');prompt_count=6;manifest_prompt_count=9;warmups_per_prompt=1;formal_per_prompt=3;
        formal_requests=18;warmup_requests=6;caps='128 for five prompts; 256 for smoke_python_function';
        engine_sha256=$identity.engine.sha256;server_file_sha256=$provenance.server_file_sha256;
        long_1k_completed=(@($promptResults|Where-Object{{$_.prompt_id -eq 'long_1k'}}).Count -eq 1);
        remaining_long_prompts_not_run=$remaining;results=$promptResults;acceptance_scope='current-environment baseline replay; no performance acceptance'}}
    Write-ExclusiveJson (Join-Path (Join-Path $runDir 'quality') 'F-six-prompt-stage.json') $final
    Write-Progress @{{phase='complete';status='six_prompt_baseline_replay_complete';sampler_stop_requested=$false;sampler_remains_owned_for_root_cleanup=$true}}
    [ordered]@{{run=$runDir;status='six_prompt_baseline_replay_complete';prompt_count=6;formal_requests=18;warmup_requests=6;
        long_1k_completed=$final.long_1k_completed;remaining_long_prompts_not_run=$remaining;
        sampler_stop_requested=$false;sampler_remains_owned_for_root_cleanup=$true}} | ConvertTo-Json -Compress'''
    return result[:begin] + final_block + result[finish:]


def render_scripts() -> tuple[dict[str, str], dict[str, dict]]:
    outputs: dict[str, str] = {}
    bindings: dict[str, dict] = {}
    target_dir = RUN / "rendered-scripts" / SIDE_ID
    for key, file_name in (("launch", "launch.ps1"), ("watch", "startup-watch.ps1"),
                           ("quality", "quality.ps1"), ("cleanup", "cleanup.ps1")):
        source_path = OLD_SCRIPTS / file_name
        source = source_path.read_text(encoding="utf-8-sig")
        if key == "quality":
            rendered = render_quality(source, NEW_RUN_DIR)
        else:
            rendered = source.replace(OLD_RUN_ID, RUN_ID).replace(OLD_RUN_DIR, NEW_RUN_DIR)
        if key == "launch":
            shutdown = "# This shutdown is only reached after the explicit -Start flag; it is the authorized owned pre-admission WSL shutdown.\n& wsl.exe --shutdown\nif ($LASTEXITCODE -ne 0) { throw 'Authorized WSL shutdown failed; server was not started.' }\n"
            rendered = replace_once(rendered, shutdown, "", "remove stale WSL shutdown action")
            rendered = replace_once(rendered,
                "--required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current",
                "--required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current --inspect-running-wsl",
                "launch-only running WSL inspection opt-in")
        if key == "cleanup" and (rendered.count("RUN41-SAMPLER-FIRST-FINALIZATION-BEGIN") != 1 or "finally {" not in rendered):
            raise ValueError("Run66 cleanup lost sampler-first finalization")
        if OLD_RUN_ID in rendered or OLD_RUN_DIR in rendered:
            raise ValueError(f"{file_name} retained Run66 run binding")
        target = target_dir / file_name
        outputs[file_name] = rendered
        bindings[key] = {"source_run66_path": str(source_path.resolve()), "source_run66_sha256": sha_file(source_path),
                         "rendered_path": str(target.resolve()), "rendered_sha256": sha_bytes(rendered.encode()),
                         "rendered_bytes": len(rendered.encode()), "status": "prepared_not_invoked"}

    wrapper_source = (OLD_RUN / "run-phase.ps1").read_text(encoding="utf-8-sig")
    wrapper = replace_once(wrapper_source,
        "[ValidateSet('upstream42','hetero42')][string]$Side='upstream42'",
        "[ValidateSet('upstream73')][string]$Side='upstream73'", "single-side wrapper selector")
    wrapper_path = RUN / "run-phase.ps1"
    outputs["run-phase.ps1"] = wrapper
    bindings["run_phase"] = {"source_run66_path": str((OLD_RUN / "run-phase.ps1").resolve()),
                             "source_run66_sha256": sha_file(OLD_RUN / "run-phase.ps1"),
                             "rendered_path": str(wrapper_path.resolve()),
                             "rendered_sha256": sha_bytes(wrapper.encode()), "rendered_bytes": len(wrapper.encode()),
                             "status": "prepared_not_invoked"}
    return outputs, bindings


def parse_powershell_only(files: dict[str, str]) -> dict[str, dict]:
    if not PWSH.is_file() or sha_file(PWSH) != PWSH_SHA:
        raise ValueError("pinned PowerShell 7 identity changed")
    with tempfile.TemporaryDirectory(prefix="upstream73-ps-ast-") as temp_dir:
        root = Path(temp_dir)
        file_paths = []
        for name, source in files.items():
            path = root / name
            path.write_text(source, encoding="utf-8")
            file_paths.append(path)
        parser = root / "parse-only.ps1"
        parser.write_text(r'''param([string]$PathsJson)
$ErrorActionPreference='Stop'
$Paths=ConvertFrom-Json -InputObject $PathsJson -AsHashtable
foreach($path in $Paths){
  $tokens=$null;$errors=$null
  [void][System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
  if($errors.Count){$errors|ForEach-Object{[Console]::Error.WriteLine($_.ToString())};exit 2}
  [Console]::Out.WriteLine((Split-Path -Leaf $path)+' parse-only pass')
}
''', encoding="utf-8")
        paths_json = json.dumps([str(path) for path in file_paths], separators=(",", ":"))
        result = subprocess.run([str(PWSH), "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(parser),
                                 paths_json], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise ValueError("generated PowerShell parser reported errors: " + result.stderr.strip())
        rows = {}
        for line in result.stdout.splitlines():
            name, status = line.split(" ", 1)
            rows[name] = {"status": status, "stdout": line}
        if len(rows) != len(files) or any(row["status"] != "parse-only pass" for row in rows.values()):
            raise ValueError(f"PowerShell parse-only output was incomplete: expected={len(files)} got={len(rows)} stdout={result.stdout!r} stderr={result.stderr!r}")
        return rows


def build_preparation() -> dict:
    phase_dir = RUN / "arms" / SIDE_ID / "quality-on"
    rendered_dir = RUN / "rendered-scripts" / SIDE_ID
    expected_outputs = [RUN / "matrix.json", RUN / "template-bindings.json", RUN / "preparation.json",
                        RUN / "preparation-receipt.json", RUN / "run-phase.ps1",
                        RUN / "controls-diff.json", RUN / "no-start-review.json",
                        phase_dir / "config.json", phase_dir / "identity.json", phase_dir / "provenance.json",
                        phase_dir / "fresh-identities.json", phase_dir / "preparation.json",
                        *(rendered_dir / n for n in ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1"))]
    preexisting = [str(p) for p in expected_outputs if p.exists()]
    if preexisting:
        raise FileExistsError("Run73 output exists; refusing overwrite: " + ", ".join(preexisting))
    source_identity = pinned_source_identity()
    old_matrix = load_json(OLD_RUN / "matrix.json")
    base_side = next(x for x in old_matrix["sides"] if x.get("id") == "upstream42")
    if old_matrix.get("execution_mode") != "direct_powershell_controllers":
        raise ValueError("Run66 is not the direct PowerShell template source")
    config = load_json(OLD_RUN / base_side["config_path"])
    identity = load_json(OLD_RUN / base_side["identity_path"])
    provenance = load_json(OLD_RUN / base_side["provenance_path"])
    if any(key in config.get("env", {}) for key in DIAGNOSTIC_ENV_KEYS):
        raise ValueError("Run66 base config unexpectedly contains E71 diagnostic environment variables")
    manifest_path = Path(old_matrix["quality_recipe"]["manifest_path"])
    manifest = load_json(manifest_path)
    manifest_ids = tuple(item["id"] for item in manifest.get("prompts", []))
    if manifest_ids != MANIFEST_IDS or sha_file(manifest_path) != old_matrix["quality_recipe"]["manifest_sha256"]:
        raise ValueError("pinned nine-prompt manifest identity/order differs from Run66")

    config = copy.deepcopy(config)
    config["exe"] = str(BINARY)
    config["model_name"] = identity["model"]["id"]
    config["log"] = str((phase_dir / "logs" / "engine.log").resolve())
    if any(key.startswith("STRATA_PREFILL_") and "PROBE" in key or key.startswith("STRATA_DIAG_") for key in config["env"]):
        raise ValueError("upstream config contains a prefill probe/diagnostic environment variable")
    if config["env"].get("STRATA_PREFILL_CPU_SHARE") != "0" or config["env"].get("STRATA_STAGE_PIN") != "0" or config["env"].get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise ValueError("upstream CPU-share/stage-pin/route-tail settings are not all explicitly zero")
    config_raw = json_bytes(config)
    config_sha = sha_bytes(config_raw)

    identity = copy.deepcopy(identity)
    identity["source"].update({"sha": SOURCE_HEAD, "tree": source_identity["tree"], "clean": True,
                               "binary_hash_finalized": True})
    identity["config"].update({"sha256": config_sha,
                               "scope": "v0.1.42-current-environment-upstream-control-replay",
                               "only_runtime_difference": "upstream v0.1.42 source/binary identity; Run66 quality controls retained"})
    identity["engine"].update({"version": "0.1.42", "path": str(BINARY), "sha256": BINARY_SHA,
                               "size_bytes": BINARY_BYTES, "source_sha": SOURCE_HEAD})
    identity["acceptance_status"] = "prepared_not_launched"
    identity_raw = json_bytes(identity)
    identity_sha = sha_bytes(identity_raw)

    provenance = copy.deepcopy(provenance)
    provenance.update({"prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "run_id": RUN_ID,
                       "arm": SIDE_ID, "status": "prepared_not_launched_current_environment_baseline_replay",
                       "engine_cpp_sha": SOURCE_HEAD, "engine_version": "0.1.42", "engine_path": str(BINARY),
                       "engine_sha256": BINARY_SHA, "acceptance_status": "prepared_not_launched",
                       "current_source_path": str(SOURCE), "current_source_tree": source_identity["tree"],
                       "current_source_clean": True, "current_binary_size_bytes": BINARY_BYTES,
                       "comparison_scope": "replay of the same first-six-prompt prefix in the current host environment; no performance or model acceptance claim"})
    provenance["shared_controls"]["model_id"] = config["model_name"]
    provenance["shared_controls"]["environment_frozen"] = copy.deepcopy(config["env"])
    provenance["shared_controls"]["new_v0_1_42_controls"] = {
        "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_STAGE_PIN": "0", "STRATA_ROUTE_TAIL_SKIP": "0"}
    provenance["launch_admission"] = {"status": "not_run_prepared_only", "must_be_fresh_at_launch": True,
                                       "inspect_running_wsl": True, "minimum_vram_mib": 46000,
                                       "minimum_ram_gib": 12, "minimum_commit_gib": 4}
    provenance["process_receipt"] = None
    provenance_raw = json_bytes(provenance)
    provenance_sha = sha_bytes(provenance_raw)

    fresh = load_json(OLD_RUN / base_side["arm_path"] / "fresh-identities.json")
    fresh["snapshot_scope"] = "Run66 upstream42 prior stat/file-ID reference only; fresh launch-time identity recheck required"
    fresh["launch_time_recheck_required"] = True
    fresh_raw = json_bytes(fresh)

    scripts, rendered = render_scripts()
    ps_parse = parse_powershell_only(scripts)
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    wrapper_sha = rendered["run_phase"]["rendered_sha256"]
    side = copy.deepcopy(base_side)
    side.update({"id": SIDE_ID, "run_id": RUN_ID, "arm_path": f"arms/{SIDE_ID}/quality-on",
                 "config_path": f"arms/{SIDE_ID}/quality-on/config.json",
                 "identity_path": f"arms/{SIDE_ID}/quality-on/identity.json",
                 "provenance_path": f"arms/{SIDE_ID}/quality-on/provenance.json",
                 "preparation_path": f"arms/{SIDE_ID}/quality-on/preparation.json",
                 "source_commit": SOURCE_HEAD, "binary_sha256": BINARY_SHA, "binary_bytes": BINARY_BYTES,
                 "config_sha256": config_sha, "identity_sha256": identity_sha, "provenance_sha256": provenance_sha,
                 "status": "prepared_not_launched_root_review_required", "rendered_scripts": {
                     key: {"rendered_path": value["rendered_path"], "rendered_sha256": value["rendered_sha256"],
                           "rendered_bytes": value["rendered_bytes"], "status": value["status"]}
                     for key, value in rendered.items() if key != "run_phase"}})
    arm_preparation = {"schema_version": 1, "run_id": RUN_ID, "status": "prepared_not_launched",
                       "config_sha256": config_sha, "identity_sha256": identity_sha,
                       "provenance_sha256": provenance_sha, "engine_sha256": BINARY_SHA,
                       "engine_size_bytes": BINARY_BYTES, "server_bridge_sha256": BRIDGE_SHA,
                       "manifest_sha256": old_matrix["quality_recipe"]["manifest_sha256"],
                       "fresh_identity_source": "Run66 prior reference; launch-time stat/file-ID recheck required",
                       "model_or_api_started": False, "device_observed": False,
                       "request_output_leaves_created": 0, "runtime_environment_delta": {}}
    arm_prep_raw = json_bytes(arm_preparation)
    side["preparation_sha256"] = sha_bytes(arm_prep_raw)

    matrix = {
        "schema_version": 1, "schema": "strata-v042-upstream-control-preparation-v1", "run_id": RUN_ID,
        "status": "prepared_upstream_control_review_pending_no_execution",
        "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "shared_server_bridge": {"path": str(BRIDGE), "sha256": BRIDGE_SHA, "bytes": BRIDGE.stat().st_size,
                                 "used_by_both_sides": False, "capture_actual_engine_token_ids": True},
        "quality_recipe": {"source_run": old_matrix["quality_recipe"]["source_run"],
                           "base_h4_run": old_matrix["quality_recipe"]["base_h4_run"],
                           "manifest_path": str(manifest_path),
                           "manifest_sha256": old_matrix["quality_recipe"]["manifest_sha256"],
                           "prompt_ids": list(PROMPT_IDS), "manifest_prompt_ids": list(MANIFEST_IDS),
                           "warmups_per_prompt": 1, "formals_per_prompt": 3,
                           "quality_capture": True, "capture_actual_engine_token_ids": True,
                           "strict_quality_stop_first": True, "performance_phase": False},
        "quality_pair_order": [SIDE_ID], "sides": [side],
        "fixed_controls": {"max_context": 32768, "kv": "int8", "resident_budget_gib": 20,
                           "vram_reserve_mib": 4096, "spec": 4, "spec_min_p": 0.5, "pool_workers": 10,
                           "pool_affinity": "all", "prompt_cache": 0, "adapt_swaps": 0, "pcie_frac": 0,
                           "ple_io": "direct", "prefill_cpu_share": "0", "stage_pin": "0",
                           "route_tail_skip": "0", "sampling": config["sampling"],
                           "capture_actual_engine_token_ids": True, "quality_prompts": 6,
                           "manifest_prompts": 9, "performance": False},
        "fresh_launch_gates": {"minimum_ram_gib": 12, "minimum_commit_gib": 4,
                               "minimum_vram_mib": 46000, "inspect_running_wsl": True,
                               "exact_current_binary_hash_and_size": True,
                               "fresh_asset_stat_recheck": True, "authorization": None},
        "template_bindings": "template-bindings.json", "template_bindings_sha256": None,
        "execution_enabled": False, "no_model_api_device_or_build_action": True,
        "execution_mode": "direct_powershell_controllers", "root_authorization": None,
        "authorization_file_status": "not created; fresh root review and authorization required",
        "controller_wrapper": {"path": str((RUN / "run-phase.ps1").resolve()),
                               "sha256": wrapper_sha, "status": "prepared_not_invoked"},
        "controller_contract": {"controllers": "direct PowerShell scripts; no pool coordinator",
                                "admission_required": {"inspect_running_wsl": True, "minimum_ram_gib": 12,
                                                       "minimum_commit_gib": 4, "minimum_vram_mib": 46000},
                                "foreign_python_and_gpu_workload_gates": "strict and unchanged",
                                "gate_exemptions": []},
        "upstream_source_identity": source_identity,
        "current_environment_replay": {"source_history": str((OLD_RUN / "arms/upstream42/quality-on/quality/F-quality-failure.json").resolve()),
                                        "purpose": "same short-prompt history replay under current host/driver conditions",
                                        "performance_or_model_acceptance": False},
        "no_start_review": {"wrapper_path": str((RUN / "run-phase.ps1").resolve()),
                            "command": f"& '{RUN / 'run-phase.ps1'}' -Side {SIDE_ID} -Phase launch",
                            "execute_switch_required_for_lifecycle": True, "command_invoked_by_preparation": False},
    }
    template_bindings = {"schema_version": 1, "run_id": RUN_ID, "side_id": SIDE_ID,
                         "source_run66_root": str(OLD_RUN.resolve()), "rendered_scripts": rendered,
                         "powershell_parse_only": ps_parse, "script_invocations": False}
    template_raw = json_bytes(template_bindings)
    matrix["template_bindings_sha256"] = sha_bytes(template_raw)
    matrix_raw = json_bytes(matrix)
    source_sha = sha_file(Path(__file__))
    prep_summary = {"schema_version": 1, "run_id": RUN_ID,
                    "status": "prepared_not_launched_root_review_required",
                    "prepared_at_utc": matrix["prepared_at_utc"], "side_id": SIDE_ID,
                    "preparation_source": {"path": str(Path(__file__).resolve()), "sha256": source_sha},
                    "source_head": SOURCE_HEAD, "source_tree": source_identity["tree"],
                    "binary_path": str(BINARY), "binary_sha256": BINARY_SHA, "binary_bytes": BINARY_BYTES,
                    "bridge_sha256": BRIDGE_SHA, "manifest_sha256": old_matrix["quality_recipe"]["manifest_sha256"],
                    "prompt_ids": list(PROMPT_IDS), "manifest_prompt_ids": list(MANIFEST_IDS),
                    "warmups_per_prompt": 1, "formals_per_prompt": 3,
                    "runtime_environment_delta": {}, "request_output_leaves_created": 0,
                    "model_or_api_started": False, "device_observed": False,
                    "scripts_invoked": False, "root_authorization_present": False,
                    "matrix_sha256": sha_bytes(matrix_raw), "template_bindings_sha256": sha_bytes(template_raw),
                    "powershell_ast_parse_only": True}
    prep_raw = json_bytes(prep_summary)
    receipt = {"schema_version": 1, "run_id": RUN_ID, "status": "prepared_not_executed",
               "matrix_sha256": sha_bytes(matrix_raw), "preparation_sha256": sha_bytes(prep_raw),
               "preparation_source_sha256": source_sha,
               "template_bindings_sha256": sha_bytes(template_raw), "rendered_scripts": len(scripts) - 1,
               "request_output_leaves_created": 0, "authorization_present": False,
               "script_or_lifecycle_execution": False, "model_payload_read_or_hashed": False}

    phase_dirs = (phase_dir / "logs", phase_dir / "resource", phase_dir / "requests",
                  phase_dir / "quality", phase_dir / "cleanup")
    phase_dir.mkdir(parents=True, exist_ok=False)
    for path in phase_dirs:
        path.mkdir()
    rendered_dir.mkdir(parents=True, exist_ok=False)
    output_map = {
        RUN / "matrix.json": matrix_raw,
        RUN / "template-bindings.json": template_raw,
        RUN / "preparation.json": prep_raw,
        RUN / "preparation-receipt.json": json_bytes(receipt),
        RUN / "run-phase.ps1": scripts["run-phase.ps1"].encode("utf-8"),
        RUN / "controls-diff.json": json_bytes({
            "schema_version": 1, "run_id": RUN_ID,
            "unchanged_from_run66_upstream42": ["model/input paths", "all model args", "env", "sampling", "port", "shared captured-token bridge", "quality checks", "sampler-first cleanup"],
            "changed": {"run_id": [OLD_RUN_ID, RUN_ID], "side_id": ["upstream42", SIDE_ID],
                        "engine_path": [load_json(OLD_ARM / "config.json")["exe"], str(BINARY)],
                        "engine_source_commit": [load_json(OLD_ARM / "identity.json")["source"]["sha"], SOURCE_HEAD],
                        "run_local_log": [load_json(OLD_ARM / "config.json")["log"], config["log"]],
                        "quality_prompt_ids": {"from": list(MANIFEST_IDS), "to": list(PROMPT_IDS)},
                        "manifest_validation_ids": list(MANIFEST_IDS),
                        "launch_admission": {"removed": "pre-admission wsl.exe --shutdown", "added": "--inspect-running-wsl opt-in on launch admission only"}},
            "runtime_environment_delta": {}, "probe_environment_added": [], "build_option_claims_added": [],
            "request_output_leaves_created": 0, "model_or_api_started": False, "device_observed": False}),
        RUN / "no-start-review.json": json_bytes({
            "schema_version": 1, "run_id": RUN_ID, "status": "prepared_no_start_review_not_invoked",
            "wrapper": str((RUN / "run-phase.ps1").resolve()), "wrapper_sha256": wrapper_sha,
            "command": matrix["no_start_review"]["command"], "expected_status": "phase_wrapper_validated_no_start",
            "lifecycle_scripts_invoked": False, "authorization_required_for_lifecycle": True}),
        phase_dir / "config.json": config_raw,
        phase_dir / "identity.json": identity_raw,
        phase_dir / "provenance.json": provenance_raw,
        phase_dir / "fresh-identities.json": fresh_raw,
        phase_dir / "preparation.json": arm_prep_raw,
    }
    for name in ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1"):
        output_map[rendered_dir / name] = scripts[name].encode("utf-8")
    if any(path.exists() for path in output_map):
        raise FileExistsError("Run73 exclusive output unexpectedly exists")
    for path, data in output_map.items():
        exclusive_write(path, data)
    if list((phase_dir / "requests").iterdir()):
        raise RuntimeError("Run73 preparation unexpectedly created request output leaves")
    return {"status": "prepared", "run_id": RUN_ID, "matrix_sha256": sha_file(RUN / "matrix.json"),
            "side_id": SIDE_ID, "request_output_leaves_created": 0, "scripts_invoked": False}


def main() -> int:
    try:
        result = build_preparation()
    except Exception as exc:
        failure = RUN / "preparation" / "preparation-failure.json"
        if not failure.exists():
            exclusive_write(failure, json_bytes({"schema_version": 1, "run_id": RUN_ID,
                                                  "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                                  "status": "preparation_failed_preserved_no_retry",
                                                  "error_type": type(exc).__name__, "error": str(exc),
                                                  "source_sha256": sha_file(Path(__file__))}))
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
