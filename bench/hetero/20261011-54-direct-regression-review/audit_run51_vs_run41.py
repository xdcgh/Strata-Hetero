#!/usr/bin/env python3
"""Read-only comparison of the preserved run51 w10 failure with run41 H4 quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HET = ROOT / "bench" / "hetero"
R41 = HET / "20261009-41-hetero41-direct-quality"
R51 = HET / "20261011-51-worker-pool-sweep"
ARM = R51 / "arms" / "w10-all" / "quality-on"
OUT = HET / "20261011-54-direct-regression-review"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(path: Path):
    events = read(path)
    final = events[-1]
    d = final["strata_diagnostics"]
    vals = d["actual_generated_token_ids"]
    canon = json.dumps(vals, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sse_path = path.with_name(path.name.replace(".events.json", ".sse.txt"))
    return {"ids": vals, "count": len(vals), "sha256": hashlib.sha256(canon).hexdigest(),
            "source": d.get("source"), "include_stop": d.get("include_stop"),
            "finish_reason": final["choices"][0].get("finish_reason"),
            "usage": final.get("usage"), "sse_file_sha256": sha(sse_path)}


def artifact_hashes(root: Path, prompt: str, kind: str):
    stem = "warmup-0000" if kind == "warmup" else kind
    return ids(root / "requests" / prompt / f"{stem}.events.json")


def main():
    m51 = read(R51 / "quality-manifest" / "manifest.json")
    m06 = read(HET / "20261009-06-quality-prompts" / "manifest.json")
    manifest51 = {x["id"]: {k: x.get(k) for k in ("bytes", "sha256", "body_tokens_local",
                   "chat_prompt_tokens_local", "max_output_tokens", "quality")} for x in m51["prompts"]}
    manifest06 = {x["id"]: {k: x.get(k) for k in ("bytes", "sha256", "body_tokens_local",
                   "chat_prompt_tokens_local", "max_output_tokens", "quality")} for x in m06["prompts"]}

    c41, c51 = read(R41 / "config.json"), read(ARM / "config.json")
    a41, a51 = list(c41["args"]), list(c51["args"])
    normalized_a51 = []
    index = 0
    while index < len(a51):
        if a51[index:index+2] == ["--pool-affinity", "all"]:
            index += 2
        else:
            normalized_a51.append(a51[index])
            index += 1
    arg_delta = {"run41": a41, "run51": a51, "run51_without_explicit_default_affinity": normalized_a51,
                 "same_after_default_affinity_removed": a41 == normalized_a51}
    env_equal = c41["env"] == c51["env"]
    config_changes = {k: {"run41": c41.get(k), "run51": c51.get(k)} for k in c41
                      if c41.get(k) != c51.get(k)}

    id41, id51 = read(R41 / "identity.json"), read(ARM / "identity.json")
    stable_identity = {
        "source_sha_and_tree_equal": id41["source"] == id51["source"],
        "engine_equal": id41["engine"] == id51["engine"],
        "model_id_equal": id41["model"]["id"] == id51["model"]["id"],
        "native_model_path_equal": id41["model"]["native_path"] == id51["model"]["native_path"],
        "shared_model_identity_fields_equal": all(id41["model"].get(k) == id51["model"].get(k)
                                                   for k in ("integrity_manifest", "integrity_status", "historical_manifest")),
        "server_python_bridge_equal": id41["server_python"] == id51["server_python"],
        "config_identity_differences": {k: {"run41": id41["config"].get(k), "run51": id51["config"].get(k)}
                                         for k in id41["config"] | id51["config"]
                                         if id41["config"].get(k) != id51["config"].get(k)},
    }

    stage41 = read(R41 / "quality" / "F-nine-prompt-stage.json")
    base_short = {x["prompt_id"]: x for x in stage41["results"]}
    fail51 = read(ARM / "quality" / "F-quality-failure.json")
    cand_short = {x["prompt_id"]: x for x in fail51["completed_prompts"]}
    req_compare = {}
    for prompt in base_short:
        if prompt not in cand_short:
            continue
        b, c = base_short[prompt], cand_short[prompt]
        breq = read(R41 / "requests" / prompt / "requests.json")
        creq = read(ARM / "requests" / prompt / "requests.json")
        bf = [artifact_hashes(R41, prompt, f"formal-{i:04d}") for i in range(3)]
        cf = [artifact_hashes(ARM, prompt, f"formal-{i:04d}") for i in range(3)]
        bw = artifact_hashes(R41, prompt, "warmup")
        cw = artifact_hashes(ARM, prompt, "warmup")
        req_compare[prompt] = {
            "prompt_sha256_equal": b["prompt_sha256"] == c["prompt_sha256"],
            "local_chat_tokens": {"run41": b["local_chat_tokens"], "run51": c["local_chat_tokens"]},
            "max_tokens": {"run41": b["max_tokens"], "run51": c["max_tokens"]},
            "formal_prompt_tokens": {"run41": b["formal_prompt_tokens"], "run51": c["formal_prompt_tokens"]},
            "cache_n": {"run41": b["cache_n"], "run51": c["cache_n"]},
            "warmup_ids": {"run41": bw, "run51": cw, "match": bw["sha256"] == cw["sha256"]},
            "formal_ids": [{"run41": bf[i], "run51": cf[i], "match": bf[i]["sha256"] == cf[i]["sha256"]}
                            for i in range(3)],
            "quality": {"run41": b["quality_statuses"], "run51": c["quality_statuses"]},
            "request_history": {"run41": [{"status": x["status"], "finish_reason": x["finish_reason"],
                                              "usage": x["usage"], "timings": x.get("timings"),
                                              "client_e2e_s": x.get("client_e2e_s"),
                                              "request_start_monotonic_ns": x.get("request_start_monotonic_ns"),
                                              "request_end_monotonic_ns": x.get("request_end_monotonic_ns"),
                                              "reasoning_effort": x.get("reasoning_effort")}
                                             for x in breq],
                                "run51": [{"status": x["status"], "finish_reason": x["finish_reason"],
                                           "usage": x["usage"], "timings": x.get("timings"),
                                           "client_e2e_s": x.get("client_e2e_s"),
                                           "request_start_monotonic_ns": x.get("request_start_monotonic_ns"),
                                           "request_end_monotonic_ns": x.get("request_end_monotonic_ns"),
                                           "reasoning_effort": x.get("reasoning_effort")}
                                          for x in creq]},
        }

    long41 = R41 / "requests" / "long_1k"
    long51 = ARM / "requests" / "long_1k"
    baseline_long = read(R41 / "quality" / "F-long_1k-result.json")
    failure_long = read(ARM / "quality" / "F-long_1k-quality.json")
    long_rows = {"prompt_sha256_equal": baseline_long["prompt_sha256"] == read(long51 / "metadata.json")["prompt_sha256"],
                 "token_count": {"run41": baseline_long["local_chat_tokens"], "run51": manifest51["long_1k"]["chat_prompt_tokens_local"]},
                 "max_tokens": {"run41": baseline_long["max_tokens"], "run51": read(long51 / "metadata.json")["max_tokens"]},
                 "run41_h4_actual_id_hashes": baseline_long["actual_token_id_hashes"],
                 "run41_h4_reference_hashes": baseline_long["h4_token_id_hashes"],
                 "run51_quality_records": failure_long["results"],
                 "run41_warmup_ids": artifact_hashes(R41, "long_1k", "warmup"),
                 "run51_warmup_ids": artifact_hashes(ARM, "long_1k", "warmup"),
                 "run41_formals": [artifact_hashes(R41, "long_1k", f"formal-{i:04d}") for i in range(3)],
                 "run51_formals": [artifact_hashes(ARM, "long_1k", f"formal-{i:04d}") for i in range(3)],
                 "request_history": {"run41": read(long41 / "requests.json"), "run51": read(long51 / "requests.json")}}

    logs = {}
    for name, p in (("run41", R41 / "logs" / "engine.log"), ("run51", ARM / "logs" / "engine.log")):
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        logs[name] = {"sha256": sha(p), "startup_lines": [line for line in lines[:45]
                    if any(s in line.lower() for s in ("engine started:", "cpu pool tasks", "pool workers on",
                       "expert cache ", "resident ram mode", "pinned cache complement ready", "draft layer loaded", "session is up"))]}

    run41f = read(R41 / "fresh-identities.json")
    run51f = read(ARM / "fresh-identities.json")
    input_identity = []
    by41 = {x["path"]: x for x in run41f["input_files"]}
    by51 = {x["path"]: x for x in run51f["input_files"]}
    for path in sorted(set(by41) | set(by51)):
        x, y = by41.get(path), by51.get(path)
        input_identity.append({"path": path, "present_both": x is not None and y is not None,
            "size_equal": bool(x and y and x["current_identity"]["size_bytes"] == y["current_identity"]["size_bytes"]),
            "mtime_equal": bool(x and y and x["current_identity"]["mtime_ns"] == y["current_identity"]["mtime_ns"]),
            "device_equal": bool(x and y and x["current_identity"]["device"] == y["current_identity"]["device"]),
            "file_index_equal": bool(x and y and x["current_identity"]["file_index"] == y["current_identity"]["file_index"]),
            "fstat_equal": bool(x and y and x["current_identity"]["fstat"] == y["current_identity"]["fstat"]),
            "sha256_prior_receipt_equal": bool(x and y and x.get("prior_sha256") == y.get("prior_sha256")),
            "run41_prior_sha256": x.get("prior_sha256") if x else None,
            "run51_prior_sha256": y.get("prior_sha256") if y else None})

    pyversions = {}
    for path in (Path(r"C:\Python314\python.exe"), Path(r"E:\Strata-Hetero-data\venv\Scripts\python.exe")):
        pyversions[str(path)] = "Windows PE ProductVersion/FileVersion 3.14.8 (read-only file metadata)"

    monotonic = {}
    for name, path in (("run41", R41 / "resource" / "samples.jsonl"), ("run51", ARM / "resource" / "samples.jsonl")):
        first = json.loads(path.open(encoding="utf-8").readline())
        monotonic[name] = {"first_observed_at_utc": first["observed_at_utc"],
                           "first_monotonic_seconds": first["monotonic_seconds"],
                           "boot_id": read(R41 / "admission.json" if name == "run41" else ARM / "admission.json").get("windows", {}).get("boot_id")}

    result = {
        "schema": "strata-run51-vs-run41-direct-regression-review-v1",
        "scope": "read-only artifact comparison; no model/API/device calls",
        "claims": {"run41_is_complete_nine_prompt_baseline": stage41["status"],
                   "run51_is_stopped_after_first_failure": fail51["status"],
                   "failing_prompt": fail51["completed_prompts"][-1]["prompt_id"]},
        "prompt_manifest": {"run51_equals_legacy_prompt_and_expectation_fields": manifest51 == manifest06,
                            "run51_prompt_rows": manifest51,
                            "run51_manifest_sha256": sha(R51 / "quality-manifest" / "manifest.json"),
                            "legacy_manifest_sha256": sha(HET / "20261009-06-quality-prompts" / "manifest.json")},
        "controls": {"config_differences": config_changes, "args": arg_delta, "environment_exact_equal": env_equal,
                      "identity": stable_identity,
                      "root_control_comparison": read(R51 / "root-control-comparison-01.json"),
                      "inputs": input_identity,
                      "engine_startup_runtime": logs, "python_executables": pyversions},
        "successful_first_five_against_run41_h4": req_compare,
        "long_1k_regression": long_rows,
        "monotonic_clock_evidence": monotonic,
        "evidence_sha256": {
            "run41_config": sha(R41 / "config.json"), "run51_config": sha(ARM / "config.json"),
            "run41_identity": sha(R41 / "identity.json"), "run51_identity": sha(ARM / "identity.json"),
            "run41_provenance": sha(R41 / "provenance.json"), "run51_provenance": sha(ARM / "provenance.json"),
            "run41_engine_log": sha(R41 / "logs" / "engine.log"), "run51_engine_log": sha(ARM / "logs" / "engine.log"),
            "run41_h4_short_stage": sha(R41 / "quality" / "F-nine-prompt-stage.json"),
            "run41_h4_long": sha(R41 / "quality" / "F-long_1k-result.json"),
            "run51_failure": sha(ARM / "quality" / "F-quality-failure.json"),
            "run51_long_failure": sha(ARM / "quality" / "F-long_1k-quality.json")}}

    out_json = OUT / "run51-vs-run41-evidence.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_json), "sha256": sha(out_json), "short_prompts": list(req_compare),
                      "long51_formal_hashes": [x["sha256"] for x in long_rows["run51_formals"]],
                      "all_short_formal_matches": all(row["match"] for p in req_compare.values() for row in p["formal_ids"]),
                      "all_short_warmup_matches": all(p["warmup_ids"]["match"] for p in req_compare.values())}, indent=2))


if __name__ == "__main__":
    main()
