from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as sweep

RUN_ID = "20261011-63-direct-zero-logits-diagnostic"
SOURCE_ID = "20261011-57-direct-zero-logits-diagnostic"
SOURCE = REPO / "bench" / "hetero" / SOURCE_ID
RUN = REPO / "bench" / "hetero" / RUN_ID
SOURCE_MATRIX = SOURCE / "matrix.json"
MATRIX_PATH = RUN / "matrix.json"
STAMP = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return sweep.sha256_file(path)


def dump(path: Path, obj: object) -> bytes:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def copy_small_tree(source: Path, target: Path) -> dict[str, dict[str, object]]:
    copied = {}
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            (target / item.relative_to(source)).mkdir(parents=True, exist_ok=True)
            continue
        if item.stat().st_size > 1_000_000:
            raise SystemExit(f"refusing to copy non-small preparation asset: {item}")
        dest = target / item.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with item.open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        copied[dest.relative_to(RUN).as_posix()] = {"bytes": dest.stat().st_size, "sha256": sha(dest)}
    return copied


def rebase(value):
    if isinstance(value, str):
        return value.replace(str(SOURCE), str(RUN)).replace(SOURCE_ID, RUN_ID)
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item) for key, item in value.items()}
    return value


if RUN.exists() and {item.name for item in RUN.iterdir()} - {Path(__file__).name}:
    raise SystemExit(f"refusing to modify existing run63 preparation: {RUN}")
if not SOURCE_MATRIX.is_file():
    raise SystemExit(f"missing immutable run57 matrix: {SOURCE_MATRIX}")
RUN.mkdir(parents=True, exist_ok=True)
source_raw = SOURCE_MATRIX.read_bytes()
source_matrix = json.loads(source_raw)
if (source_matrix.get("candidate_binary_sha256") != "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb" or
        source_matrix.get("candidate_source_sha") != "ffeb748d190ddeedff23587ce848596d4db0c752"):
    raise SystemExit("run57 binary/source pin differs from the reviewed run41 candidate")

asset_inventory = {}
asset_inventory.update(copy_small_tree(SOURCE / "quality-manifest", RUN / "quality-manifest"))
asset_inventory.update(copy_small_tree(SOURCE / "tokenizer-source-01", RUN / "tokenizer-source-01"))
for name in ("tokenizer-equivalence.json", "tokenizer-manifest-provenance.json", "source-run51-recipe.json"):
    src, dst = SOURCE / name, RUN / name
    with src.open("rb") as stream, dst.open("xb") as out:
        shutil.copyfileobj(stream, out)
        out.flush()
        os.fsync(out.fileno())
    asset_inventory[name] = {"bytes": dst.stat().st_size, "sha256": sha(dst)}
(RUN / "prompts").mkdir()
for name in ("sustained_decode.prompt.txt", "sustained_decode.prompt.json"):
    src, dst = SOURCE / "prompts" / name, RUN / "prompts" / name
    with src.open("rb") as stream, dst.open("xb") as out:
        shutil.copyfileobj(stream, out)
        out.flush()
        os.fsync(out.fileno())
    asset_inventory["prompts/" + name] = {"bytes": dst.stat().st_size, "sha256": sha(dst)}

manifest_path = RUN / "quality-manifest" / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
tokenizer_path = RUN / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py"
tokenizer_sha, tokenizer_bytes = sha(tokenizer_path), tokenizer_path.stat().st_size
manifest["tokenizer"]["implementation_source"].update(
    {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha, "bytes": tokenizer_bytes})
manifest_raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
tmp_manifest = manifest_path.with_suffix(".json.tmp")
with tmp_manifest.open("xb") as stream:
    stream.write(manifest_raw)
    stream.flush()
    os.fsync(stream.fileno())
tmp_manifest.replace(manifest_path)
manifest_sha = sha(manifest_path)

equiv_path = RUN / "tokenizer-equivalence.json"
equiv = rebase(json.loads(equiv_path.read_text(encoding="utf-8")))
equiv["current_implementation_snapshot"].update(
    {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha, "bytes": tokenizer_bytes})
equiv_raw = (json.dumps(equiv, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
equiv_path.write_bytes(equiv_raw)
equiv_sha = sha(equiv_path)

manifest_prov_path = RUN / "tokenizer-manifest-provenance.json"
manifest_prov = rebase(json.loads(manifest_prov_path.read_text(encoding="utf-8")))
manifest_prov["manifest_path"] = str(manifest_path.resolve())
manifest_prov["manifest_sha256"] = manifest_sha
manifest_prov["tokenizer_equivalence_receipt"] = str(equiv_path.resolve())
manifest_prov["tokenizer_equivalence_receipt_sha256"] = equiv_sha
manifest_prov["implementation_snapshot"].update(
    {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha, "bytes": tokenizer_bytes})
legacy_path = RUN / "tokenizer-source-01" / "strata_tokenizer-legacy-b8af8e56.py"
manifest_prov["legacy_implementation"].update(
    {"path": str(legacy_path.resolve()), "sha256": sha(legacy_path), "bytes": legacy_path.stat().st_size})
manifest_prov_raw = (json.dumps(manifest_prov, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
manifest_prov_path.write_bytes(manifest_prov_raw)
manifest_prov_sha = sha(manifest_prov_path)

matrix = rebase(source_matrix)
matrix["run_id"] = RUN_ID
matrix["status"] = "prepared_for_root_review_not_launched"
matrix["prepared_utc"] = STAMP
matrix["quality_prompt_manifest"].update({"path": str(manifest_path.resolve()), "sha256": manifest_sha})
matrix["tokenizer_implementation_snapshot"].update(
    {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha, "bytes": tokenizer_bytes,
     "equivalence_receipt_path": str(equiv_path.resolve()), "equivalence_receipt_sha256": equiv_sha})
matrix["tokenizer_manifest_provenance"].update(
    {"path": str(manifest_prov_path.resolve()), "sha256": manifest_prov_sha})
matrix["diagnostic_plan"] = {
    "only_extra_environment": {"STRATA_DIAG_ZERO_LOGITS": "1"},
    "applies_to": "first quality phase only: w10-all",
    "other_arms_and_phases": "UNRUN and diagnostic environment absent",
    "maximum_authorized_phase_intent": 1,
    "diagnostic_is_not_a_quality_acceptance_or_cause_claim": True,
}
matrix["source_run57_preparation"] = {
    "matrix_path": str(SOURCE_MATRIX.resolve()), "matrix_sha256": hashlib.sha256(source_raw).hexdigest(),
    "source_run51_recipe_reference_preserved": str(SOURCE / "source-run51-recipe.json"),
    "source_run51_recipe_sha256": sha(RUN / "source-run51-recipe.json"),
    "runtime_evidence_copied": False,
}
matrix["source_template_capture"] = source_matrix.get("source_template_capture")
matrix["preparation_source"] = {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}

phase_inventory = {}
for arm in matrix["arms"]:
    for phase_key, phase_leaf, phase_name in (
        ("quality_on", "quality-on", "quality-on"),
        ("performance_off", "performance-off", "performance-off"),
    ):
        row = arm[phase_key]
        leaf = RUN / "arms" / arm["arm"] / phase_leaf
        leaf.mkdir(parents=True, exist_ok=False)
        source_leaf = SOURCE / "arms" / arm["arm"] / phase_leaf
        config = rebase(json.loads((source_leaf / "config.json").read_text(encoding="utf-8")))
        config["log"] = str((leaf / "logs" / "engine.log").resolve())
        config.setdefault("env", {}).pop("STRATA_DIAG_ZERO_LOGITS", None)
        if arm["arm"] == "w10-all" and phase_key == "quality_on":
            config["env"]["STRATA_DIAG_ZERO_LOGITS"] = "1"
        config_raw = dump(leaf / "config.json", config)

        identity = rebase(json.loads((source_leaf / "identity.json").read_text(encoding="utf-8")))
        identity["config"]["sha256"] = hashlib.sha256(config_raw).hexdigest()
        identity["quality_prompt_manifest"].update(
            {"path": str(manifest_path.resolve()), "sha256": manifest_sha})
        identity["tokenizer_implementation"].update(
            {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_sha, "bytes": tokenizer_bytes})
        identity_raw = dump(leaf / "identity.json", identity)

        provenance = rebase(json.loads((source_leaf / "provenance.json").read_text(encoding="utf-8")))
        run_id = f"20261011-63-{arm['arm']}-{phase_name}"
        provenance["run_id"] = run_id
        provenance["prepared_utc"] = STAMP
        provenance["process_receipt"] = None
        provenance["acceptance_status"] = "prepared_not_launched"
        provenance["quality_prompt_manifest"] = str(manifest_path.resolve())
        provenance["quality_prompt_manifest_sha256"] = manifest_sha
        provenance.setdefault("shared_controls", {}).update(
            {"quality_prompt_manifest": str(manifest_path.resolve()),
             "quality_prompt_manifest_sha256": manifest_sha})
        delta = ({"STRATA_DIAG_ZERO_LOGITS": "1"}
                 if arm["arm"] == "w10-all" and phase_key == "quality_on" else {})
        provenance["diagnostic_environment_delta"] = delta
        provenance["diagnostic_scope"] = "first quality phase only; no output-quality or causal claim"
        if isinstance(provenance.get("comparison_contract"), dict):
            provenance["comparison_contract"]["diagnostic_environment_delta"] = delta
        if isinstance(provenance.get("tokenizer_manifest_rebase"), dict):
            provenance["tokenizer_manifest_rebase"].update(
                {"new_manifest_path": str(manifest_path.resolve()), "new_manifest_sha256": manifest_sha,
                 "equivalence_receipt_sha256": equiv_sha})
        provenance_raw = dump(leaf / "provenance.json", provenance)

        fresh_path = leaf / "fresh-identities.json"
        fresh_source = source_leaf / "fresh-identities.json"
        with fresh_source.open("rb") as src, fresh_path.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        fresh_sha = sha(fresh_path)
        row.update({"run_id": run_id, "path": str(leaf.resolve()),
                    "config_sha256": hashlib.sha256(config_raw).hexdigest(),
                    "identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
                    "provenance_sha256": hashlib.sha256(provenance_raw).hexdigest(),
                    "fresh_identities_sha256": fresh_sha})
        for dirname in ("logs", "resource", "requests", "quality", "performance", "cleanup"):
            (leaf / dirname).mkdir()
        phase_inventory[run_id] = {"config_sha256": row["config_sha256"], "identity_sha256": row["identity_sha256"],
                                   "provenance_sha256": row["provenance_sha256"], "fresh_identities_sha256": fresh_sha,
                                   "request_id_leaves_created": 0}

matrix_raw = dump(MATRIX_PATH, matrix)
plan = sweep.build_plan(MATRIX_PATH)
if len(plan["steps"]) != 10 or plan["maximum_authorized_phases"] != 1:
    raise SystemExit("run63 plan did not validate as ten planned phases with a one-phase cap")
if plan["steps"][0]["run_id"] != "20261011-63-w10-all-quality-on":
    raise SystemExit("run63 first phase is not the w10-all quality diagnostic")
if list((RUN / "arms" / "w10-all" / "quality-on" / "requests").iterdir()):
    raise SystemExit("run63 request output directory must remain empty before authorization")

prep = {
    "schema_version": 1, "run_id": RUN_ID, "status": "prepared_for_root_review_not_launched",
    "prepared_utc": STAMP, "matrix_sha256": hashlib.sha256(matrix_raw).hexdigest(),
    "source_run57_matrix_sha256": hashlib.sha256(source_raw).hexdigest(),
    "binary_sha256": matrix["candidate_binary_sha256"], "candidate_source_sha": matrix["candidate_source_sha"],
    "bridge_sha256": matrix["shared_bridge_sha256"], "quality_prompt_count": 9,
    "prompt_bytes_and_expectations_inherited_unchanged": True,
    "diagnostic_delta": {"first_phase_only": "STRATA_DIAG_ZERO_LOGITS=1", "max_authorized_phases": 1},
    "phase_inventory": phase_inventory, "copied_small_input_assets": asset_inventory,
    "models_or_packs_copied": False, "request_output_leaves_created": 0,
    "controller_or_authorization_created": False, "runtime_admission_or_execution": False,
}
prep_raw = dump(RUN / "preparation.json", prep)
dump(RUN / "preparation-receipt.json", {
    "schema_version": 1, "status": "prepared_plan_validated_no_lifecycle_execution",
    "run_id": RUN_ID, "preparation_sha256": hashlib.sha256(prep_raw).hexdigest(),
    "matrix_sha256": sha(MATRIX_PATH), "plan_step_count": len(plan["steps"]),
    "plan_status": plan["status"], "maximum_authorized_phases": plan["maximum_authorized_phases"],
    "runtime_admission_or_execution": False,
})
dump(RUN / "README.md", {"run_id": RUN_ID, "status": "prepared_for_root_review_not_launched",
                           "scope": "first-phase-only direct zero-logits diagnostic continuation",
                           "authorization_present": False, "request_leaves_precreated": False})
print(json.dumps({"status": "prepared", "run_id": RUN_ID, "matrix": str(MATRIX_PATH),
                  "matrix_sha256": sha(MATRIX_PATH), "plan_steps": len(plan["steps"]),
                  "maximum_authorized_phases": plan["maximum_authorized_phases"]}, sort_keys=True))
