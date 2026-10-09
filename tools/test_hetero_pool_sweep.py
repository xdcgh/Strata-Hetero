from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import hetero_pool_sweep as sweep


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _fixture_matrix(root: Path) -> Path:
    run = root / "matrix-fixture"
    run.mkdir()
    manifest_dir = run / "quality-manifest"
    prompt_dir = manifest_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    tokenizer_snapshot = run / "tokenizer-source-01" / "strata_tokenizer-current-461b8afb.py"
    tokenizer_snapshot.parent.mkdir()
    tokenizer_bytes = b"synthetic tokenizer source for no-execution fixtures\n"
    _write(tokenizer_snapshot, tokenizer_bytes)
    tokenizer_digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    prompt_specs = [
        ("smoke_arithmetic", 15, 27, 128), ("smoke_unicode", 20, 32, 128),
        ("smoke_json_arithmetic", 38, 50, 128), ("smoke_python_function", 27, 39, 256),
        ("smoke_three_key_retrieval", 102, 114, 128), ("long_1k", 1024, 1036, 1024),
        ("long_4k", 4097, 4109, 1024), ("long_16k", 16384, 16396, 1024),
        ("long_30k7", 30700, 30712, 1024),
    ]
    manifest_prompts = []
    for prompt_id, body_tokens, chat_tokens, max_tokens in prompt_specs:
        rel = f"prompts/{prompt_id}.prompt.txt"
        body = f"Host-only fixture prompt {prompt_id}.\n".encode()
        _write(prompt_dir / f"{prompt_id}.prompt.txt", body)
        manifest_prompts.append({"id": prompt_id, "file": rel, "bytes": len(body),
                                 "sha256": hashlib.sha256(body).hexdigest(),
                                 "body_tokens_local": body_tokens, "chat_prompt_tokens_local": chat_tokens,
                                 "chat_wrapper_delta_local": chat_tokens - body_tokens,
                                 "max_output_tokens": max_tokens,
                                 "quality": {"kind": "fixture_exact_text", "expected": prompt_id}})
    manifest = {"schema": "strata-hetero-quality-prompts-v1", "prompts": manifest_prompts,
                "tokenizer": {"module": str(tokenizer_snapshot),
                              "implementation_source": {"path": str(tokenizer_snapshot),
                                                         "bytes": len(tokenizer_bytes),
                                                         "sha256": tokenizer_digest},
                              "assets": []}}
    manifest_path = manifest_dir / "manifest.json"
    _write(manifest_path, (json.dumps(manifest, sort_keys=True) + "\n").encode())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    arms = []
    choices = (("w10-all", 10, "all"), ("w04-all", 4, "all"), ("w06-all", 6, "all"),
               ("w16-all", 16, "all"), ("w06-auto", 6, "auto"))
    for arm, workers, affinity in choices:
        rows = {"arm": arm, "workers": workers, "affinity": affinity}
        for phase, capture, key in (("quality-on", True, "quality_on"),
                                    ("performance-off", False, "performance_off")):
            run_id = f"20261009-47-{arm}-{phase}"
            leaf = run / "arms" / arm / phase
            leaf.mkdir(parents=True)
            config = {"exe": "fake-engine.exe", "hetero_capture_token_ids": capture,
                      "args": ["--pool-workers", str(workers), "--pool-affinity", affinity]}
            identity = {"run_id": run_id, "engine": {"path": "fake-engine.exe", "sha256": "a" * 64},
                        "config": {"capture_actual_engine_token_ids": capture},
                        "quality_prompt_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
                        "tokenizer_implementation": {"path": str(tokenizer_snapshot), "sha256": tokenizer_digest}}
            provenance = {"run_id": run_id, "engine_sha256": "a" * 64, "server_file_sha256": "b" * 64,
                          "shared_controls": {"quality_prompt_manifest": str(manifest_path),
                                              "quality_prompt_manifest_sha256": manifest_sha}}
            for name, obj in (("config.json", config), ("identity.json", identity),
                              ("provenance.json", provenance)):
                _write(leaf / name, (json.dumps(obj, sort_keys=True) + "\n").encode())
            rows[key] = {"run_id": run_id, "path": str(leaf),
                         "config_sha256": hashlib.sha256((leaf / "config.json").read_bytes()).hexdigest(),
                         "identity_sha256": hashlib.sha256((leaf / "identity.json").read_bytes()).hexdigest(),
                         "provenance_sha256": hashlib.sha256((leaf / "provenance.json").read_bytes()).hexdigest()}
        arms.append(rows)
    prompt = run / "sustained_decode.prompt.txt"
    prompt_bytes = b"A deterministic sustained-output fixture prompt with several sections.\n"
    _write(prompt, prompt_bytes)
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    scenarios = [{"id": f"decode-cap-{cap}", "prompt_id": "sustained_decode",
                  "max_tokens": cap, "prompt_file": str(prompt),
                  "prompt_sha256": prompt_sha, "prompt_bytes": len(prompt_bytes)}
                 for cap in (256, 512, 1024)]
    scenarios.extend([
        {"id": "context-4k", "prompt_id": "long_4k", "prompt_tokens": 4109, "max_tokens": 128},
        {"id": "context-16k", "prompt_id": "long_16k", "prompt_tokens": 16396, "max_tokens": 128},
        {"id": "context-32k", "prompt_id": "long_30k7", "prompt_tokens": 30712, "max_tokens": 128},
    ])
    matrix = {"status": "prepared_for_root_review_not_launched", "arms": arms,
              "arm_execution_order": list(sweep.ARM_EXECUTION_ORDER),
              "quality_prompt_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
              "tokenizer_implementation_snapshot": {"path": str(tokenizer_snapshot),
                                                     "sha256": tokenizer_digest, "bytes": len(tokenizer_bytes)},
              "performance_phase": {"scenarios": scenarios}}
    path = run / "matrix.json"
    _write(path, (json.dumps(matrix, sort_keys=True) + "\n").encode())
    return path


def _auth(matrix: Path, nonce: str = "a" * 32) -> dict:
    repo = Path(sweep.__file__).resolve().parents[1]
    return {"status": "root_reviewed_authorized", "matrix_sha256": sweep.sha256_file(matrix),
            "arms": [x["arm"] for x in json.loads(matrix.read_text())["arms"]], "nonce": nonce,
            "output_root": str(matrix.parent.resolve()), "adapter_sha256": sweep.sha256_file(Path(sweep.__file__)),
            "template_sha256s": sweep.template_source_hashes(repo)}


def _valid_performance_run(root: Path, *, text: str = "Detailed output with several useful words.",
                           count: int = 8, finish: str = "length", status: str = "truncated_by_length",
                           cache: int = 0, done: bool = True, cap: int = 8,
                           timings: dict | None = None, extra_event: dict | None = None) -> Path:
    root.mkdir(parents=True)
    content = text.encode("utf-8")
    event1 = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    event2 = {"choices": [{"delta": {}, "finish_reason": finish}],
              "usage": {"prompt_tokens": 12, "completion_tokens": count, "total_tokens": 12 + count}}
    events = [event1, event2]
    if extra_event is not None:
        events[0]["nonfinite"] = extra_event
    sse = ["data: " + json.dumps(event1), "data: " + json.dumps(event2)]
    if done:
        sse.append("data: [DONE]")
    record = {
        "status": status, "finish_reason": finish,
        "usage": {"prompt_tokens": 12, "completion_tokens": count, "total_tokens": 12 + count},
        "timings": {"cache_n": cache, "predicted_per_second": 20.0} if timings is None else timings,
        "client_e2e_s": 0.5, "first_generated_s": 0.1, "first_visible_content_s": 0.1,
        "content_sha256": hashlib.sha256(content).hexdigest(), "content_chars": len(text),
    }
    (root / "metadata.json").write_text(json.dumps({"max_tokens": cap, "repeats": 3, "warmup": 1,
        "prompt_sha256": "fixture-prompt-sha", "prompt_bytes": 19}), encoding="utf-8")
    (root / "warmups.json").write_text(json.dumps([{"status": "completed"}]), encoding="utf-8")
    (root / "requests.json").write_text(json.dumps([record, record, record]), encoding="utf-8")
    (root / "formal-0000.events.json").write_text(json.dumps(events), encoding="utf-8")
    (root / "formal-0000.sse.txt").write_text("\n".join(sse) + "\n", encoding="utf-8")
    (root / "formal-0000.text.txt").write_bytes(content)
    return root


class PoolSweepPlanTests(unittest.TestCase):
    def test_plan_mode_has_all_arms_and_never_launches_or_reads_payload(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-plan-") as temp:
            matrix = _fixture_matrix(Path(temp))
            out = io.StringIO()
            with mock.patch.object(sweep, "launch_model", side_effect=AssertionError("launch called")), \
                 mock.patch.object(sweep, "load_model_payload", side_effect=AssertionError("payload read")), \
                 mock.patch.object(sweep, "initialize_core", side_effect=AssertionError("Core initialized")), \
                 mock.patch.object(sweep.subprocess, "run", side_effect=AssertionError("subprocess called")), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(sweep.main(["--matrix", str(matrix)]), 0)
            plan = json.loads(out.getvalue())
            self.assertEqual(plan["status"], "plan_only_no_hardware_or_process_access")
            self.assertFalse(plan["execution_enabled"])
            self.assertEqual(len(plan["steps"]), 10)
            self.assertEqual([x["phase"] for x in plan["steps"][:5]], ["quality_on"] * 5)
            self.assertEqual([x["phase"] for x in plan["steps"][5:]], ["performance_off"] * 5)
            self.assertEqual([x["arm"] for x in plan["steps"][:5]], list(sweep.ARM_EXECUTION_ORDER))
            self.assertEqual([x["arm"] for x in plan["steps"][5:]], list(sweep.ARM_EXECUTION_ORDER))
            context = [x for x in plan["performance_scenarios"] if x["id"].startswith("context-")]
            self.assertTrue(all(Path(x["prompt_file"]).is_file() and len(x["prompt_sha256"]) == 64 and x["prompt_bytes"] > 0 for x in context))

    def test_execute_without_reviewed_auth_and_adapter_is_rejected_without_spawn(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-no-auth-") as temp:
            matrix = _fixture_matrix(Path(temp))
            out = io.StringIO()
            with mock.patch.object(sweep.subprocess, "run", side_effect=AssertionError("process spawned")), \
                 mock.patch.object(sweep.subprocess, "Popen", side_effect=AssertionError("process spawned")), \
                 contextlib.redirect_stdout(out):
                code = sweep.main(["--matrix", str(matrix), "--execute"])
            self.assertEqual(code, 2)
            self.assertIn("root authorization", json.loads(out.getvalue())["error"])

    def test_execution_failure_is_reported_as_preserved_after_controller_start(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-exec-failure-") as temp:
            matrix = _fixture_matrix(Path(temp))
            auth_path = Path(temp) / "root-authorization.json"
            auth_path.write_text(json.dumps(_auth(matrix)), encoding="utf-8")
            failure_path = matrix.parent / "fake-live-owner-state.json"
            with failure_path.open("x", encoding="utf-8") as stream:
                stream.write('{"status":"fixture-preserved"}')

            class FakeAdapter:
                def __init__(self, matrix_path, auth):
                    self.execution_state = {"execution_attempted": True, "lifecycle_controller_started": True,
                                            "model_launch_reported": False, "stage": "launch_controller_running",
                                            "controller_dir": str(matrix.parent)}
                def write_failure_receipt(self, exc): return str(failure_path)

            out = io.StringIO()
            with mock.patch.object(sweep, "Run41LifecycleAdapter", FakeAdapter), \
                 mock.patch.object(sweep, "execute_controlled_coordinator", side_effect=sweep.SweepError("fake preserved failure")), \
                 contextlib.redirect_stdout(out):
                code = sweep.main(["--matrix", str(matrix), "--execute", "--adapter", "run41", "--authorization", str(auth_path)])
            result = json.loads(out.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "failed_preserved_after_execution_attempt")
            self.assertTrue(result["execution_attempted"])
            self.assertFalse(result["model_launch_reported"])
            self.assertEqual(result["failure_receipt"], str(failure_path))
            self.assertEqual(result["live_state_path"], str(matrix.parent))

    def test_matrix_refuses_config_tampering(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-plan-") as temp:
            matrix = _fixture_matrix(Path(temp))
            config = Path(json.loads(matrix.read_text())["arms"][0]["quality_on"]["path"]) / "config.json"
            config.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.build_plan(matrix)

    def test_matrix_rejects_tokenizer_snapshot_sha_version_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tokenizer-version-") as temp:
            matrix = _fixture_matrix(Path(temp))
            doc = json.loads(matrix.read_text(encoding="utf-8"))
            doc["tokenizer_implementation_snapshot"]["sha256"] = "0" * 64
            matrix.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(sweep.SweepError, "tokenizer implementation snapshot"):
                sweep.build_plan(matrix)

    def test_manifest_implementation_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tokenizer-manifest-mismatch-") as temp:
            matrix = _fixture_matrix(Path(temp))
            doc = json.loads(matrix.read_text(encoding="utf-8"))
            manifest_path = Path(doc["quality_prompt_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tokenizer"]["implementation_source"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_sha = sweep.sha256_file(manifest_path)
            doc["quality_prompt_manifest"]["sha256"] = manifest_sha
            for arm in doc["arms"]:
                for phase in ("quality_on", "performance_off"):
                    row = arm[phase]
                    provenance_path = Path(row["path"]) / "provenance.json"
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    provenance["shared_controls"]["quality_prompt_manifest_sha256"] = manifest_sha
                    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
                    row["provenance_sha256"] = sweep.sha256_file(provenance_path)
            matrix.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(sweep.SweepError, "manifest tokenizer source identity"):
                sweep.build_plan(matrix)

    def test_run_id_is_allowlisted(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-plan-") as temp:
            matrix = _fixture_matrix(Path(temp))
            doc = json.loads(matrix.read_text())
            doc["arms"][0]["quality_on"]["run_id"] = "../../other"
            matrix.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.build_plan(matrix)

    def test_execution_requires_authorization_and_adapter_before_callbacks(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-auth-") as temp:
            matrix = _fixture_matrix(Path(temp))
            plan = sweep.build_plan(matrix)
            calls = []

            class Adapter:
                def run_phase(self, phase):
                    calls.append(phase)
                    return {"status": "pass"}
                def cleanup_previous(self, previous):
                    return {"status": "terminal_owned_cleanup_pass"}

            with self.assertRaises(sweep.SweepError):
                sweep.execute_controlled_coordinator(plan, {}, Adapter())
            self.assertEqual(calls, [])
            with self.assertRaises(sweep.SweepError):
                sweep.execute_controlled_coordinator(plan, _auth(matrix), object())

    def test_root_authorization_binds_exact_matrix_hash_arms_and_nonce(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-auth-") as temp:
            matrix = _fixture_matrix(Path(temp))
            matrix_doc = json.loads(matrix.read_text())
            arms = [x["arm"] for x in matrix_doc["arms"]]
            auth = Path(temp) / "authorization.json"
            valid = _auth(matrix)
            auth.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(sweep.validate_root_authorization(matrix, auth)["arms"], arms)
            valid["matrix_sha256"] = "0" * 64
            auth.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.validate_root_authorization(matrix, auth)
            valid["matrix_sha256"] = sweep.sha256_file(matrix)
            valid["adapter_sha256"] = "0" * 64
            auth.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.validate_root_authorization(matrix, auth)

    def test_fake_cli_runs_on_host_inside_temp_root_and_preserves_argv(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-fakecli-") as temp:
            root = Path(temp).resolve()
            fixture_root = root / "allowed"
            fixture_root.mkdir()
            fake = fixture_root / "fake_cli.py"
            fake.write_text("import json,sys;print(json.dumps({'argv':sys.argv[1:]}))\n", encoding="utf-8")
            args = ["--prompt-file", str(fixture_root / "prompt with spaces.txt"), "--max-tokens", "512"]
            result = sweep.invoke_host_fake_cli(fake, args, fixture_root)
            self.assertEqual(result["argv"], args)
            self.assertEqual(sweep.serialize_windows_argv(args), subprocess.list2cmdline(args))
            outside_dir = root / "untrusted"
            outside_dir.mkdir()
            outside = outside_dir / "outside-fake-cli.py"
            outside.write_text("print('{}')\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.invoke_host_fake_cli(outside, [], fixture_root)

    def test_temp_fixture_and_fake_authorized_execution_are_isolated(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-fixture-") as temp:
            root = Path(temp).resolve()
            matrix = _fixture_matrix(root)
            auth = _auth(matrix, "b" * 32)
            # The fake adapter writes only beneath this temporary output root and invokes a host fake CLI.
            fixture_root = root / "adapter-fixtures"
            fixture_root.mkdir()
            fake = fixture_root / "fake_cli.py"
            fake.write_text("import json,sys;print(json.dumps({'run_id':sys.argv[1]}))\n", encoding="utf-8")
            output_root = matrix.parent / "fake-outputs"
            output_root.mkdir()

            class FakeAdapter:
                def run_phase(self, phase):
                    result = sweep.invoke_host_fake_cli(fake, [phase["run_id"]], fixture_root)
                    target = output_root / (phase["run_id"] + ".json")
                    with target.open("x", encoding="utf-8") as stream:
                        stream.write(json.dumps(result))
                    return {"status": "pass", "run_id": phase["run_id"], "path": str(target)}
                def cleanup_previous(self, previous):
                    return {"status": "terminal_owned_cleanup_pass", "run_id": previous["run_id"]}

            auth_path = fixture_root / "authorization.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            checked = sweep.validate_root_authorization(matrix, auth_path)
            plan = sweep.build_plan(matrix)
            result = sweep.execute_controlled_coordinator(plan, checked, FakeAdapter())
            self.assertEqual(len(result), 10)
            self.assertTrue(all(root in Path(x["path"]).resolve().parents for x in result))
            self.assertEqual(len(list(output_root.iterdir())), 10)

    def test_stop_first_adapter_does_not_launch_later_phases(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-sweep-stopfirst-") as temp:
            matrix = _fixture_matrix(Path(temp))
            plan = sweep.build_plan(matrix)
            output_root = matrix.parent / "fake-outputs"
            output_root.mkdir()
            auth = _auth(matrix)
            auth["_resolved_output_root"] = str(matrix.parent.resolve())
            calls = []
            run_count = 0

            class Adapter:
                def run_phase(self, phase):
                    nonlocal run_count
                    run_count += 1
                    calls.append(phase["run_id"])
                    target = output_root / f"{len(calls)}.json"
                    with target.open("x", encoding="utf-8") as stream: stream.write("{}")
                    return {"status": "failed" if run_count == 2 else "pass", "run_id": phase["run_id"], "path": str(target)}
                def cleanup_previous(self, previous):
                    calls.append("cleanup:" + previous["run_id"])
                    return {"status": "terminal_owned_cleanup_pass"}

            with self.assertRaises(sweep.SweepError):
                sweep.execute_controlled_coordinator(plan, auth, Adapter())
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[-1].startswith("20261009-47-"))

    def test_controlled_coordinator_uses_fake_lifecycle_and_cleans_only_prior_pass(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-controlled-") as temp:
            matrix = _fixture_matrix(Path(temp))
            plan = sweep.build_plan(matrix)
            auth = _auth(matrix)
            auth["_resolved_output_root"] = str(matrix.parent.resolve())
            events = []

            class FakeLifecycle:
                def __init__(self): self.index = 0
                def run_phase(self, step):
                    self.index += 1
                    events.append(("run", step["run_id"]))
                    folder = matrix.parent / "fake-lifecycle-output"
                    folder.mkdir(exist_ok=True)
                    target = folder / f"{self.index}.json"
                    with target.open("x", encoding="utf-8") as stream:
                        stream.write(json.dumps({"run_id": step["run_id"], "launch": "fake", "watch": "fake", "task": "fake"}))
                    return {"status": "pass", "run_id": step["run_id"], "path": str(target)}
                def cleanup_previous(self, prior):
                    events.append(("cleanup", prior["run_id"]))
                    return {"status": "terminal_owned_cleanup_pass", "run_id": prior["run_id"]}

            with mock.patch.object(sweep, "launch_model", side_effect=AssertionError("real model launcher called")), \
                 mock.patch.object(sweep, "load_model_payload", side_effect=AssertionError("payload read")), \
                 mock.patch.object(sweep, "initialize_core", side_effect=AssertionError("Core initialized")):
                results = sweep.execute_controlled_coordinator(plan, auth, FakeLifecycle())
            self.assertEqual(len(results), 10)
            self.assertEqual(sum(kind == "cleanup" for kind, _ in events), 9)
            self.assertEqual(events[-1][0], "run")
            self.assertEqual(events[-2][0], "cleanup")

    def test_real_run41_adapter_cleanup_schema_crosses_coordinator(self):
        with tempfile.TemporaryDirectory(prefix="strata-run41-adapter-mock-") as temp:
            matrix = _fixture_matrix(Path(temp))
            auth_path = Path(temp) / "adapter-authorization.json"
            auth_path.write_text(json.dumps(_auth(matrix)), encoding="utf-8")
            auth = sweep.validate_root_authorization(matrix, auth_path)
            plan = sweep.build_plan(matrix)
            adapter = sweep.Run41LifecycleAdapter(matrix, auth)
            phases = plan["steps"][:2]
            plan["steps"] = phases
            calls = []
            created = "2026-10-09T12:34:56.1234567Z"
            launch_pid, sampler_pid, sampler_child_pid = 30101, 30102, 30103

            def fake_call(step, template, action, controller_dir, **kwargs):
                calls.append((action, step["run_id"]))
                if action == "launch":
                    config_path = Path(step["config"])
                    identity = json.loads(Path(step["identity"]).read_text(encoding="utf-8"))
                    provenance = json.loads(Path(step["provenance"]).read_text(encoding="utf-8"))
                    with (config_path.parent / "process.json").open("x", encoding="utf-8") as stream:
                        json.dump({"run_id": step["run_id"], "launcher_pid": launch_pid,
                                   "engine_sha256": identity["engine"]["sha256"],
                                   "config_sha256": sweep.sha256_file(config_path),
                                   "server_file_sha256": provenance["server_file_sha256"]}, stream)
                    with (config_path.parent / "sampler-process.json").open("x", encoding="utf-8") as stream:
                        json.dump({"sampler_launcher_pid": sampler_pid, "actual_child_pid": sampler_child_pid,
                                   "actual_child_create_utc": created}, stream)
                    return {"json": {"status": "launched_not_yet_ready", "start_performed": True,
                                     "launcher_pid": launch_pid, "sampler_pid": sampler_pid}}
                if action == "watch":
                    identity = json.loads(Path(step["identity"]).read_text(encoding="utf-8"))
                    return {"json": {"status": "startup_ready", "launcher_pid": launch_pid,
                                     "engine_exe": identity["engine"]["path"],
                                     "sampler_child_pid": sampler_child_pid,
                                     "sampler_child_create_utc": created}}
                if action == "check-ready":
                    return {"json": {"status": "live_quality_preflight_pass", "chat_requests_sent": 0,
                                     "model_requests": 0, "manifest_has_prompt_paths": True,
                                     "identity_bound": True, "physical_gib": 80.0, "commit_gib": 40.0}}
                if action == "quality":
                    return {"json": {"status": "nine_prompt_matrix_complete", "formal_requests": 27}}
                if action == "cleanup":
                    return {"json": {"status": "terminal_owned_server_and_sampler_released",
                                     "engine_server_sampler_terminal": True, "listener8081_released": True}}
                raise AssertionError(f"unexpected action {action}")

            with mock.patch.object(adapter, "_call", side_effect=fake_call):
                results = sweep.execute_controlled_coordinator(plan, auth, adapter)
            self.assertEqual(len(results), 2)
            self.assertEqual([x[0] for x in calls], ["launch", "watch", "check-ready", "quality", "cleanup",
                                                       "launch", "watch", "check-ready", "quality"])
            self.assertEqual(adapter.execution_state["last_cleanup_receipt"]["status"],
                             "terminal_owned_server_and_sampler_released")
            self.assertEqual(adapter.live_result["run_dir"], str(Path(phases[1]["config"]).parent))

    def test_controlled_coordinator_stops_first_failure_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-controlled-fail-") as temp:
            matrix = _fixture_matrix(Path(temp))
            plan = sweep.build_plan(matrix)
            auth = _auth(matrix)
            auth["_resolved_output_root"] = str(matrix.parent.resolve())
            calls = []
            runs = 0

            class FakeLifecycle:
                def run_phase(self, step):
                    nonlocal runs
                    runs += 1
                    calls.append(("run", step["run_id"]))
                    folder = matrix.parent / "fake-lifecycle-output"; folder.mkdir(exist_ok=True)
                    target = folder / f"{len(calls)}.json"
                    with target.open("x", encoding="utf-8") as stream: stream.write("{}")
                    return {"status": "failed" if runs == 2 else "pass", "run_id": step["run_id"], "path": str(target)}
                def cleanup_previous(self, prior): calls.append(("cleanup", prior["run_id"])); return {"status": "terminal_owned_cleanup_pass"}

            with self.assertRaises(sweep.SweepError):
                sweep.execute_controlled_coordinator(plan, auth, FakeLifecycle())
            self.assertEqual([kind for kind, _ in calls], ["run", "cleanup", "run"])

    def test_run41_template_parameterization_is_parse_only_and_binds_fake_recipe(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-template-") as temp:
            matrix = _fixture_matrix(Path(temp))
            plan = sweep.build_plan(matrix)
            repo = Path(sweep.__file__).resolve().parents[1]
            quality = next(step for step in plan["steps"] if step["phase"] == "quality_on")
            perf = next(step for step in plan["steps"] if step["phase"] == "performance_off")
            for kind, step in (("launch", quality), ("watch", quality), ("quality", quality), ("cleanup", quality)):
                text = sweep.render_run41_template(kind, step, repo)
                self.assertNotIn(sweep.RUN41_ID, text)
                sweep.validate_powershell_syntax(text)
            perf_text = sweep.render_performance_controller(perf, repo)
            self.assertIn("performance-stage.json", perf_text)
            self.assertIn("Get-LatestResourceGate", perf_text)
            self.assertIn("--allow-capped-performance", perf_text)
            sweep.validate_powershell_syntax(perf_text)


class ProcessAndTailFixtureTests(unittest.TestCase):
    def test_controller_owner_receipt_callback_precedes_single_script_send_and_timeout_keeps_identity(self):
        owner = {"pid": 987, "parent_pid": 654, "exe": str(sweep.PWsh7),
                 "create_utc": "2026-10-09T12:34:56.1234567Z",
                 "command_line": 'pwsh.exe -NoProfile -Command "$source=[Console]::In.ReadToEnd()"',
                 "argv": [str(sweep.PWsh7), "-NoProfile", "-Command", "stdin"],
                 "argv_windows": 'pwsh.exe -NoProfile -Command stdin'}

        class FakePopen:
            pid = 987
            returncode = 0
            calls = 0
            def __init__(self, timeout=False): self.timeout = timeout
            def communicate(self, payload, timeout):
                self.calls += 1
                self.assert_callback()
                if self.timeout:
                    raise subprocess.TimeoutExpired("fake-pwsh", timeout, output=b"partial-out", stderr=b"partial-err")
                return b"done", b"warn"
            def assert_callback(self):
                if not callback_seen[0]: raise AssertionError("script bytes sent before exact owner receipt callback")

        callback_seen = [False]
        fake = FakePopen()
        def callback(actual_owner, process):
            self.assertEqual(actual_owner["create_utc"], owner["create_utc"])
            self.assertEqual(actual_owner["argv_windows"], owner["argv_windows"])
            self.assertIs(process, fake)
            callback_seen[0] = True

        with mock.patch.object(type(sweep.PWsh7), "is_file", return_value=True), \
             mock.patch.object(sweep, "sha256_file", return_value=sweep.PWSH7_SHA256), \
             mock.patch.object(sweep.subprocess, "Popen", return_value=fake), \
             mock.patch.object(sweep, "_query_pwsh_process_identity", return_value=owner):
            code, stdout, stderr, got_owner = sweep._invoke_pwsh_text("fake source", "launch", env={}, cwd=Path.cwd(), on_started=callback)
        self.assertEqual((code, stdout, stderr), (0, b"done", b"warn"))
        self.assertEqual(got_owner["create_utc"], owner["create_utc"])
        self.assertEqual(fake.calls, 1)

        callback_seen[0] = False
        timed = FakePopen(timeout=True)
        def timeout_callback(actual_owner, process):
            self.assertEqual(actual_owner["argv"], owner["argv"])
            self.assertIs(process, timed)
            callback_seen[0] = True
        with mock.patch.object(type(sweep.PWsh7), "is_file", return_value=True), \
             mock.patch.object(sweep, "sha256_file", return_value=sweep.PWSH7_SHA256), \
             mock.patch.object(sweep.subprocess, "Popen", return_value=timed), \
             mock.patch.object(sweep, "_query_pwsh_process_identity", return_value=owner):
            with self.assertRaises(sweep.LifecycleTimeout) as caught:
                sweep._invoke_pwsh_text("fake source", "launch", env={}, cwd=Path.cwd(), on_started=timeout_callback)
        self.assertEqual(caught.exception.owner["create_utc"], owner["create_utc"])
        self.assertEqual(caught.exception.stdout, b"partial-out")
        self.assertEqual(caught.exception.stderr, b"partial-err")
        self.assertIs(caught.exception.process, timed)
        self.assertEqual(timed.calls, 1)

    def test_full_precision_create_time_is_preserved(self):
        value = "2026-10-09T11:27:55.5209790Z"
        self.assertEqual(sweep.validate_create_time(value), value)
        with self.assertRaises(sweep.SweepError):
            sweep.validate_create_time("2026-10-09T11:27:55.5209790+08:00")

    def test_cleanup_order_is_engine_then_server_then_launcher(self):
        owners = {
            "launcher": {"pid": 10, "exe": "launcher.exe", "create_utc": "2026-10-09T00:00:00.0000000Z", "parent_pid": 1, "command_contains": "server.py"},
            "server": {"pid": 20, "exe": "server.exe", "create_utc": "2026-10-09T00:00:01.0000000Z", "parent_pid": 10, "command_contains": "config.json"},
            "engine": {"pid": 30, "exe": "strata.exe", "create_utc": "2026-10-09T00:00:02.0000000Z", "parent_pid": 20, "command_contains": "--serve"},
        }
        snapshot = [{"pid": value["pid"], "exe": value["exe"], "create_utc": value["create_utc"],
                     "parent_pid": value["parent_pid"], "command_line": value["command_contains"]}
                    for value in owners.values()]
        order = sweep.owned_cleanup_order(snapshot, owners)
        self.assertEqual([x["role"] for x in order], ["engine", "server", "launcher"])
        self.assertTrue(all(x["status"] == "exact_match_stop_child_first" for x in order))

    def test_cleanup_rejects_reused_pid_with_changed_create_time(self):
        owners = {
            "launcher": {"pid": 10, "exe": "launcher.exe", "create_utc": "2026-10-09T00:00:00.0000000Z", "parent_pid": 1, "command_contains": "server.py"},
            "server": {"pid": 20, "exe": "server.exe", "create_utc": "2026-10-09T00:00:01.0000000Z", "parent_pid": 10, "command_contains": "config.json"},
            "engine": {"pid": 30, "exe": "strata.exe", "create_utc": "2026-10-09T00:00:02.0000000Z", "parent_pid": 20, "command_contains": "--serve"},
        }
        snapshot = [{"pid": value["pid"], "exe": value["exe"], "create_utc": value["create_utc"],
                     "parent_pid": value["parent_pid"], "command_line": value["command_contains"]}
                    for value in owners.values()]
        snapshot[1]["create_utc"] = "2026-10-09T00:00:09.0000000Z"
        with self.assertRaises(sweep.SweepError):
            sweep.owned_cleanup_order(snapshot, owners)

    def test_naturally_exited_child_is_not_stopped_again(self):
        owners = {
            "launcher": {"pid": 10, "exe": "launcher.exe", "create_utc": "2026-10-09T00:00:00.0000000Z", "parent_pid": 1, "command_contains": "server.py"},
            "server": {"pid": 20, "exe": "server.exe", "create_utc": "2026-10-09T00:00:01.0000000Z", "parent_pid": 10, "command_contains": "config.json"},
            "engine": {"pid": 30, "exe": "strata.exe", "create_utc": "2026-10-09T00:00:02.0000000Z", "parent_pid": 20, "command_contains": "--serve"},
        }
        snapshot = [{"pid": 10, "exe": "launcher.exe", "create_utc": owners["launcher"]["create_utc"], "parent_pid": 1, "command_line": "server.py"},
                    {"pid": 20, "exe": "server.exe", "create_utc": owners["server"]["create_utc"], "parent_pid": 10, "command_line": "config.json"}]
        order = sweep.owned_cleanup_order(snapshot, owners)
        self.assertEqual(order[0]["status"], "already_absent")

    def test_jsonl_tail_crlf_multiple_records_and_partial_preservation(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tail-") as temp:
            root = Path(temp).resolve()
            path = root / "samples.jsonl"
            partials = root / "partial-evidence"
            now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            rows = [{"record_type": "resource_sample", "observed_at_utc": now, "owner_pid": 77, "n": 1},
                    {"record_type": "resource_sample", "observed_at_utc": now, "owner_pid": 77, "n": 2}]
            fragment = b'{"broken":'
            path.write_bytes((json.dumps(rows[0]) + "\r\n" + json.dumps(rows[1]) + "\r\n").encode() + fragment)
            result = sweep.read_last_complete_jsonl_record(path, expected_owner_pid=77,
                expected_record_type="resource_sample", partial_evidence_dir=partials)
            self.assertEqual(result["record"]["n"], 2)
            self.assertEqual(Path(result["trailing_partial_path"]).read_bytes(), fragment)

    def test_jsonl_tail_malformed_latest_is_not_skipped(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tail-") as temp:
            path = Path(temp) / "samples.jsonl"
            now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            valid = json.dumps({"observed_at_utc": now, "owner_pid": 1})
            path.write_text(valid + "\n{" + "bad: true}" + "\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(path, owner_field="owner_pid", expected_owner_pid=1)

    def test_jsonl_tail_stale_owner_oversize_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tail-") as temp:
            root = Path(temp)
            stale = root / "stale.jsonl"
            old = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=1)).isoformat().replace("+00:00", "Z")
            stale.write_text(json.dumps({"observed_at_utc": old, "owner_pid": 1}) + "\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(stale, expected_owner_pid=1)
            now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            bad_owner = root / "owner.jsonl"
            bad_owner.write_text(json.dumps({"observed_at_utc": now, "owner_pid": 2}) + "\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(bad_owner, expected_owner_pid=1)
            huge = root / "huge.jsonl"
            huge.write_text(json.dumps({"observed_at_utc": now, "owner_pid": 1, "payload": "x"*500}) + "\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(huge, max_record_bytes=128, max_tail_bytes=1024, expected_owner_pid=1)
            nonfinite = root / "nonfinite.jsonl"
            nonfinite.write_text('{"observed_at_utc":"'+now+'","owner_pid":1,"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(nonfinite, expected_owner_pid=1)

    def test_jsonl_tail_rejects_two_json_values_on_one_line(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-tail-") as temp:
            path = Path(temp) / "two.jsonl"
            now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            path.write_text(json.dumps({"observed_at_utc": now}) + json.dumps({"observed_at_utc": now}) + "\n", encoding="utf-8")
            with self.assertRaises(sweep.SweepError):
                sweep.read_last_complete_jsonl_record(path, owner_field=None, record_type_field=None)


class PerformanceGuardTests(unittest.TestCase):
    def test_length_capped_performance_keeps_actual_counts_and_is_not_quality_acceptance(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-perf-") as temp:
            run = _valid_performance_run(Path(temp) / "run")
            result = sweep.validate_performance_request(run, 0, 8, allow_capped=True,
                expected_prompt_sha256="fixture-prompt-sha", expected_prompt_bytes=19)
            self.assertEqual(result["status"], "timing_eligible_quality_not_assessed")
            self.assertEqual(result["completion_tokens"], 8)
            self.assertEqual(result["cache_n"], 0)

    def test_clean_eos_records_actual_count_without_reprompt(self):
        with tempfile.TemporaryDirectory(prefix="strata-pool-perf-") as temp:
            run = _valid_performance_run(Path(temp) / "run", count=4, finish="stop", status="completed", cap=8)
            result = sweep.validate_performance_request(run, 0, 8, allow_capped=True)
            self.assertEqual(result["completion_tokens"], 4)
            self.assertEqual(result["finish_reason"], "stop")

    def test_single_token_bang_loop_nan_missing_done_and_cache_fail_closed(self):
        cases = [
            (dict(text="one", count=1, finish="stop", status="completed", cap=8), "single-token"),
            (dict(text="!"*32, count=8, finish="length", status="truncated_by_length", cap=8), "bang-loop"),
            (dict(extra_event={"x": float("nan")}), "nonfinite"),
            (dict(done=False), "DONE"),
            (dict(cache=1), "cache"),
        ]
        for kwargs, _label in cases:
            with self.subTest(case=_label), tempfile.TemporaryDirectory(prefix="strata-pool-perf-") as temp:
                run = _valid_performance_run(Path(temp) / "run", **kwargs)
                with self.assertRaises(sweep.SweepError):
                    sweep.validate_performance_request(run, 0, kwargs.get("cap", 8), allow_capped=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
