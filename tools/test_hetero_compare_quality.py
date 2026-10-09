import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from hetero_compare_quality import (
    _load_request_entries,
    _buffer_override_error,
    _prompt_identity_check,
    _resources_for_run,
    _verify_admission_evidence,
    _verified_file_hash,
    compare_runs,
    analyze_run,
    classify_file_tier_log_line,
    evaluate_formal,
    extract_final_sse,
    load_artifact_writer_receipt,
    main,
    resolve_run,
    verify_text_artifact,
    LEGACY_WRITER_API,
    LEGACY_WRITER_BLOB,
    LEGACY_WRITER_COMMIT,
    LEGACY_WRITER_SOURCE_SHA256,
    LEGACY_TEXT_NEWLINE_CONVERSION,
    LEGACY_MODEL_HASH_SCOPE,
)


def final_sse(text, ids=(73, 2), finish="stop", include_stop=True, copies=1):
    parts = []
    for _ in range(copies):
        first = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": len(ids), "total_tokens": 5 + len(ids)},
                 "strata_diagnostics": {"actual_generated_token_ids": list(ids), "source": "engine.generate emitted IDs",
                                        "include_stop": include_stop}}
        parts.extend(["data: " + json.dumps(first), "data: " + json.dumps(final)])
    parts.append("data: [DONE]")
    return ("\n".join(parts) + "\n").encode("utf-8")


def request_row(text, ids=(73, 2), finish="stop"):
    raw = text.encode("utf-8")
    return {"status": "completed", "finish_reason": finish,
            "usage": {"prompt_tokens": 5, "completion_tokens": len(ids), "total_tokens": 5 + len(ids)},
            "content_sha256": hashlib.sha256(raw).hexdigest(), "request_start_monotonic_ns": 100,
            "request_end_monotonic_ns": 200, "batch_index": 0}


class CompareQualityFixtureTests(unittest.TestCase):
    @staticmethod
    def _write_launch_admission(root: Path, receipt_name="admission-02.json"):
        import hashlib
        admission = {"status": "pass", "pass": True, "reasons": []}
        admission_path = root / "admission-02.json"
        admission_path.write_text(json.dumps(admission), encoding="utf-8")
        process = {"schema_version": 1, "launcher_pid": 1234,
                   "run": str(root.resolve()), "config": str((root / "config.json").resolve()),
                   "argv": ["fixture"], "role": "fixture process"}
        process_path = root / "process.json"
        process_path.write_text(json.dumps(process), encoding="utf-8")
        provenance = {"launch_admission": {"receipt": receipt_name,
                                            "receipt_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
                                            "status": "pass", "launch_process_receipt": "process.json",
                                            "process_receipt_sha256": hashlib.sha256(process_path.read_bytes()).hexdigest()}}
        (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    def test_explicit_provenance_selects_checked_retry_admission_not_initial_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "admission.json").write_text(json.dumps({"status": "blocked", "pass": False}), encoding="utf-8")
            self._write_launch_admission(root)
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["selected_receipt"], "admission-02.json")
            self.assertEqual(result["mode"], "launch_admission_provenance")
            self.assertTrue(result["process_receipt_structurally_valid"])
            self.assertEqual(result["sha256_scope"], "each digest covers exact stored JSON file bytes")

    def test_provenance_with_bad_process_hash_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_launch_admission(root)
            (root / "process.json").write_text("{}", encoding="utf-8")
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "unknown")
            self.assertIn("SHA-256 does not match provenance", result["errors"][0])

    def test_tampered_retry_admission_invalidates_launch_proof(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_launch_admission(root)
            (root / "admission-02.json").write_text(json.dumps({"status": "blocked", "pass": False}), encoding="utf-8")
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "unknown")
            self.assertIn("SHA-256 does not match provenance", result["errors"][0])

    def test_provenance_path_escape_is_rejected_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            root.mkdir()
            outside = Path(td) / "admission-02.json"
            outside.write_text(json.dumps({"status": "pass", "pass": True}), encoding="utf-8")
            (root / "admission.json").write_text(json.dumps({"status": "pass", "pass": True}), encoding="utf-8")
            self._write_launch_admission(root, receipt_name="../admission-02.json")
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "unknown")
            self.assertIn("relative path within the run", result["errors"][0])

    def test_legacy_run_without_launch_admission_uses_legacy_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "provenance.json").write_text(json.dumps({"prepared": True}), encoding="utf-8")
            (root / "admission.json").write_text(json.dumps({"status": "pass", "pass": True}), encoding="utf-8")
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["mode"], "legacy_admission")
            self.assertEqual(result["selected_receipt"], "admission.json")

    def test_present_but_null_launch_provenance_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "provenance.json").write_text(json.dumps({"launch_admission": None}), encoding="utf-8")
            (root / "admission.json").write_text(json.dumps({"status": "pass", "pass": True}), encoding="utf-8")
            result = _verify_admission_evidence(root)
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result["mode"], "launch_admission_provenance")

    def test_buffer_override_gate_is_scoped_to_distinct_ab_candidate(self):
        base = {"root": r"C:\runs\05", "allowed_env_override": {"STRATA_UNBUFFERED_LOAD": None}}
        self.assertIsNone(_buffer_override_error(base, dict(base)))
        self.assertIn("approved", _buffer_override_error(base, {"root": r"C:\runs\07", "allowed_env_override": {}}))
        self.assertIsNone(_buffer_override_error(base, {"root": r"C:\runs\07", "allowed_env_override": {"STRATA_UNBUFFERED_LOAD": "0"}}))

    def test_numeric_run_id_resolution_is_scoped_to_manifest_date(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "20261007-01-ple").mkdir()
            (root / "20261007-01-other").mkdir()
            expected = root / "20261008-07-ram20-quality"
            expected.mkdir()
            self.assertEqual(resolve_run("07", root, "2026-10-06T00:00:00Z"), expected.resolve())

    def test_buffered_prefix_is_not_misclassified_by_unbuffered_zero_reason(self):
        self.assertEqual(classify_file_tier_log_line(
            "strata generate: the file tier reads through the file cache (STRATA_UNBUFFERED_LOAD=0)"), "buffered")
        self.assertEqual(classify_file_tier_log_line(
            "strata generate: the file tier reads unbuffered (read path selected)"), "unbuffered")
        self.assertEqual(classify_file_tier_log_line(
            "diagnostic says unbuffered was explicitly disabled"), "unknown")

    def test_actual_raw_ids_are_retained_not_reencoded_from_visible_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sse_path = root / "formal-0000.sse.txt"
            sse_path.write_bytes(final_sse("4", ids=(9001, 2)))
            got = extract_final_sse(sse_path)
            self.assertTrue(got["valid"], got["errors"])
            self.assertEqual(got["actual_generated_token_ids"], [9001, 2])
            self.assertNotEqual(len(got["visible_text_from_sse"]), len(got["actual_generated_token_ids"]))
            self.assertTrue(got["include_stop"])

    def test_duplicate_or_missing_final_diagnostics_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "formal.sse.txt"
            path.write_bytes(final_sse("x", copies=2))
            duplicate = extract_final_sse(path)
            self.assertFalse(duplicate["valid"])
            self.assertTrue(any("exactly one" in e for e in duplicate["errors"]))
            first_only = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
            path.write_text("data: " + json.dumps(first_only) + "\ndata: [DONE]\n", encoding="utf-8")
            missing = extract_final_sse(path)
            self.assertFalse(missing["valid"])
            self.assertTrue(any("diagnostics" in e for e in missing["errors"]))

    def test_bool_negative_or_missing_actual_token_ids_are_not_valid_integers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "formal.sse.txt"
            for invalid in ([True, 2], [-1, 2], [], None):
                final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                         "strata_diagnostics": {"actual_generated_token_ids": invalid,
                                                "source": "engine.generate", "include_stop": True}}
                path.write_text("data: " + json.dumps(final) + "\ndata: [DONE]\n", encoding="utf-8")
                self.assertFalse(extract_final_sse(path)["valid"], repr(invalid))

    def test_manifest_prompt_hash_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompt = root / "p.txt"
            prompt.write_bytes(b"expected prompt")
            result = _prompt_identity_check(root / "manifest.json", {"file": "p.txt", "sha256": "0" * 64})
            self.assertEqual(result["status"], "failed")
            self.assertNotEqual(result["sha256"], result["expected_sha256"])

    def test_binary_or_identity_hash_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "strata.exe"
            binary.write_bytes(b"fixture binary identity")
            result = _verified_file_hash(str(binary), "0" * 64, "engine_binary")
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["matches_record"])

    def test_requests_repeat_count_shortfall_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "requests.json").write_text(json.dumps([request_row("4")]), encoding="utf-8")
            rows, errors = _load_request_entries(root, repeats=3)
            self.assertEqual(len(rows), 1)
            self.assertTrue(any("expected 3" in e for e in errors))

    def test_formal_evaluator_fails_invalid_quality_and_checks_saved_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "formal-0000.sse.txt").write_bytes(final_sse("5", ids=(73, 2)))
            (root / "formal-0000.text.txt").write_bytes(b"5")
            metadata = {"_request_dir": str(root), "max_tokens": 128}
            metadata["artifact_writer"] = {"client_os_name": "nt", "text_encoding": "utf-8 bytes without newline translation",
                                           "content_sha256_scope": "exact stored model-output bytes"}
            request = request_row("5")
            request["output_artifact_encoding"] = "utf-8 bytes without newline translation"
            result = evaluate_formal(root, {"id": "p", "quality": {"kind": "exact_text", "expected": "4"}},
                                     request, 0, metadata)
            self.assertEqual(result["status"], "quality_failed")
            self.assertEqual(result["quality"]["quality_status"], "exact_text_mismatch")
            self.assertEqual(result["token_ids"], [73, 2])
            self.assertEqual(result["text_sha256"], hashlib.sha256(b"5").hexdigest())

    def test_crlf_artifact_requires_receipt_and_content_hash_uses_sse_model_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            request_dir = root / "requests" / "smoke-code-run"; request_dir.mkdir(parents=True)
            model_text = "line one\nline two"
            model_bytes = model_text.encode("utf-8")
            artifact = model_bytes.replace(b"\n", b"\r\n")
            text_path = request_dir / "formal-0000.text.txt"
            text_path.write_bytes(artifact)
            sse_path = request_dir / "formal-0000.sse.txt"
            sse_path.write_bytes(final_sse(model_text, ids=(11, 12)))
            metadata_bytes = b'{"prompt_file":"p.prompt.txt"}'
            (request_dir / "metadata.json").write_bytes(metadata_bytes)
            prompt_id = "p"
            meta_sha = hashlib.sha256(metadata_bytes).hexdigest()
            model_sha = hashlib.sha256(model_bytes).hexdigest()
            artifact_sha = hashlib.sha256(artifact).hexdigest()
            row = {"relative_path": "requests/smoke-code-run/formal-0000.text.txt", "prompt_id": prompt_id,
                   "repeat_index": 0, "metadata_sha256": meta_sha, "model_text_sha256": model_sha,
                   "artifact_sha256": artifact_sha, "conversion": "windows_textio_lf_to_crlf"}
            receipt = {"schema_version": 1, "status": "verified_each_formal_against_raw_sse_and_legacy_writer",
                       "client_os_name": "nt", "writer_git_commit": LEGACY_WRITER_COMMIT,
                       "writer_git_blob_sha1": LEGACY_WRITER_BLOB,
                       "writer_source_bytes_sha256": LEGACY_WRITER_SOURCE_SHA256,
                       "writer_api": LEGACY_WRITER_API, "model_content_hash_scope": LEGACY_MODEL_HASH_SCOPE,
                       "files": [row]}
            (root / "artifact_writer_provenance.json").write_text(json.dumps(receipt), encoding="utf-8")
            loaded = load_artifact_writer_receipt(root)
            self.assertEqual(loaded["status"], "pass")
            result = evaluate_formal(root, {"id": "p", "quality": {"kind": "exact_text", "expected": model_text}},
                                     request_row(model_text, ids=(11, 12)), 0,
                                     {"_request_dir": str(request_dir), "_metadata_raw_sha256": meta_sha,
                                      "max_tokens": 10}, loaded)
            self.assertEqual(result["status"], "pass")
            self.assertNotEqual(result["artifact_text_sha256"], result["model_text_sha256"])
            self.assertEqual(result["artifact_conversion"]["status"], "verified_legacy_conversion")

    def test_exact_artifact_without_new_metadata_flag_or_legacy_receipt_is_unproven(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); req_dir = root / "requests" / "run"; req_dir.mkdir(parents=True)
            (req_dir / "formal-0000.text.txt").write_bytes(b"4")
            (req_dir / "formal-0000.sse.txt").write_bytes(final_sse("4", ids=(4, 2)))
            result = evaluate_formal(root, {"id": "p", "quality": {"kind": "exact_text", "expected": "4"}},
                                     request_row("4", ids=(4, 2)), 0,
                                     {"_request_dir": str(req_dir), "max_tokens": 10})
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(any("writer flag" in err for err in result["errors"]))

    def test_missing_run_prompt_is_incomplete_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run = base / "run-short"
            (run / "requests").mkdir(parents=True)
            prompts_dir = base / "prompts"
            prompts_dir.mkdir()
            prompt_file = prompts_dir / "p.prompt.txt"
            prompt_file.write_bytes(b"p")
            prompt = {"p": {"id": "p", "file": "prompts/p.prompt.txt",
                            "sha256": hashlib.sha256(b"p").hexdigest(), "quality": {"kind": "exact_text", "expected": "x"}}}
            result = analyze_run(run, base / "manifest.json", prompt)
            self.assertEqual(result["prompts"]["p"]["status"], "incomplete")
            self.assertEqual(result["missing_prompt_ids"], ["p"])

    def test_resource_window_is_streamed_and_marks_commit_scope_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resource = root / "resource"
            resource.mkdir()
            rows = [
                {"record_type": "resource_sample", "monotonic_seconds": 1.0,
                 "memory": {"physical_available_bytes": 100, "commit_available_bytes": 40,
                            "source": "GlobalMemoryStatusEx"},
                 "disk_io": {"PhysicalDrive0": {"read_bytes": 10, "write_bytes": 2, "read_count": 1, "write_count": 1}}},
                {"record_type": "resource_sample", "monotonic_seconds": 2.0,
                 "memory": {"physical_available_bytes": 80, "commit_available_bytes": None,
                            "source": "GlobalMemoryStatusEx"},
                 "disk_io": {"PhysicalDrive0": {"read_bytes": 30, "write_bytes": 5, "read_count": 3, "write_count": 2}}},
            ]
            (resource / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            result = _resources_for_run(root, [(1.0, 2.0)])
            self.assertEqual(result["physical_available_min"]["bytes"], 80)
            self.assertEqual(result["commit_unknown_samples"], 1)
            self.assertFalse(result["commit_systemwide_proven"])
            self.assertEqual(result["disk_io_delta"]["scope"], "systemwide per-device counters; not attributable only to this model")
            self.assertEqual(result["disk_io_delta"]["devices"]["PhysicalDrive0"]["read_bytes"], 20)

    def test_validate_only_writes_nothing_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "compare_fixture_newdir"
            root.mkdir()
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            prompts = []
            for i in range(9):
                name = f"p{i}.prompt.txt"
                (prompt_dir / name).write_bytes(f"prompt-{i}".encode())
                prompts.append({"id": f"p{i}", "file": f"prompts/{name}",
                               "sha256": hashlib.sha256(f"prompt-{i}".encode()).hexdigest(),
                               "quality": {"kind": "exact_text", "expected": "ok"}})
            manifest_dir = root / "quality_manifest"
            manifest_dir.mkdir()
            manifest = manifest_dir / "manifest.json"
            manifest.write_text(json.dumps({"schema": "strata-hetero-quality-prompts-v1", "prompts": prompts,
                                            "tokenizer": {"implementation_source": {}, "assets": []}}), encoding="utf-8")
            (root / "run-baseline" / "requests").mkdir(parents=True)
            (root / "run-candidate" / "requests").mkdir(parents=True)
            output = root / "result.json"
            args = ["--baseline-run", "baseline", "--candidate-run", "candidate",
                    "--manifest", str(manifest), "--output", str(output), "--validate-only"]
            self.assertEqual(main(args), 0)
            self.assertFalse(output.exists())
            partial = compare_runs(root / "run-baseline", root / "run-candidate", manifest)
            self.assertEqual(partial["prompt_comparisons"]["p0"]["request_controls"], "not_observed")
            self.assertFalse(any("request controls differ" in x for x in partial["controls"]["unexpected_differences"]))
            self.assertFalse(partial["accepted"])
            self.assertEqual(partial["overall"]["expected_pairs"], 27)
            self.assertEqual(partial["overall"]["observed_pairs"], 0)
            self.assertFalse(partial["overall"]["all_actual_token_ids_equal"])
            self.assertFalse(partial["overall"]["all_model_text_from_sse_equal"])
            self.assertFalse(partial["overall"]["all_artifact_byte_hashes_equal"])
            self.assertFalse(partial["overall"]["all_quality_pass"])
            result = analyze_run(root / "run-baseline", manifest, {x["id"]: x for x in prompts})
            self.assertEqual(len(result["missing_prompt_ids"]), 9)
            output.write_text("keep", encoding="utf-8")
            args.remove("--validate-only")
            self.assertEqual(main(args), 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def _make_compare_manifest(self, root: Path) -> Path:
        prompt_dir = root / "prompts"
        prompt_dir.mkdir(exist_ok=True)
        prompts = []
        for i in range(9):
            raw = f"prompt-{i}".encode()
            path = prompt_dir / f"p{i}.txt"
            path.write_bytes(raw)
            prompts.append({"id": f"p{i}", "file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "quality": {"kind": "exact_text", "expected": "ok"}})
        implementation = root / "tokenizer.py"
        asset = root / "tokenizer.model"
        implementation.write_bytes(b"fixture tokenizer implementation")
        asset.write_bytes(b"fixture tokenizer asset")
        tokenizer = {"implementation_source": {"path": str(implementation),
                                                "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
                                                "bytes": implementation.stat().st_size},
                     "assets": [{"path": str(asset), "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                                 "bytes": asset.stat().st_size}]}
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "strata-hetero-quality-prompts-v1",
                                        "prompts": prompts, "tokenizer": tokenizer}), encoding="utf-8")
        return manifest

    @staticmethod
    def _comparison_control(root: Path, candidate: bool = False):
        normalized = {"fixture": "same-controls"}
        return {"root": str(root), "status": "pass", "errors": [],
                "normalized_controls": normalized, "normalized_controls_sha256": "same",
                "model_identity": {"id": "fixture-model"},
                "server_python": {"checkout_sha": "fixture"},
                "effective_file_tier_mode": "buffered",
                "allowed_env_override": {"STRATA_UNBUFFERED_LOAD": "0" if candidate else None},
                "strict_env_value_sha256": {}}

    @staticmethod
    def _comparison_run(root: Path, available_prompts: int):
        prompts = {}
        metadata = {"prompt_file": "p", "prompt_sha256": "sha", "prompt_bytes": 1,
                    "repeats": 3, "warmup": 1, "max_tokens": 16, "concurrency": 1,
                    "seed": 17, "reasoning_effort": "medium"}
        for prompt_index in range(available_prompts):
            prompt_id = f"p{prompt_index}"
            formals = [{"status": "pass", "token_ids": [prompt_index, repeat, 2], "include_stop": True,
                        "model_text_sha256": f"text-{prompt_index}-{repeat}",
                        "artifact_text_sha256": f"artifact-{prompt_index}-{repeat}",
                        "quality": {"quality_status": "pass"}}
                       for repeat in range(3)]
            prompts[prompt_id] = {"metadata": metadata, "formals": formals}
        return {"root": str(root), "prompts": prompts,
                "missing_prompt_ids": [f"p{i}" for i in range(available_prompts, 9)],
                "unexpected_prompt_ids": []}

    def test_three_of_nine_is_not_vacuous_and_full_27_pair_fixture_stays_positive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self._make_compare_manifest(root)
            base_root, cand_root = root / "baseline", root / "candidate"
            base_root.mkdir(); cand_root.mkdir()
            for available, expected_observed, expected_accept in ((3, 9, False), (9, 27, True)):
                base_result = self._comparison_run(base_root, available)
                cand_result = self._comparison_run(cand_root, available)
                with mock.patch("hetero_compare_quality.analyze_run", side_effect=[base_result, cand_result]), \
                        mock.patch("hetero_compare_quality._load_run_controls",
                                   side_effect=[self._comparison_control(base_root),
                                                self._comparison_control(cand_root, candidate=True)]):
                    result = compare_runs(base_root, cand_root, manifest)
                self.assertEqual(result["overall"]["expected_pairs"], 27)
                self.assertEqual(result["overall"]["observed_pairs"], expected_observed)
                self.assertEqual(result["accepted"], expected_accept)
                if available == 3:
                    self.assertFalse(result["overall"]["all_actual_token_ids_equal"])
                    self.assertFalse(result["overall"]["all_model_text_from_sse_equal"])
                    self.assertFalse(result["overall"]["all_quality_pass"])
                else:
                    self.assertTrue(result["overall"]["required_pairs_complete"])
                    self.assertTrue(result["overall"]["all_actual_token_ids_equal"])
                    self.assertTrue(result["overall"]["all_model_text_from_sse_equal"])
                    self.assertTrue(result["overall"]["all_quality_pass"])


if __name__ == "__main__":
    unittest.main()
