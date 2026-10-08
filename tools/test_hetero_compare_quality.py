import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hetero_compare_quality import (
    _load_request_entries,
    _buffer_override_error,
    _prompt_identity_check,
    _resources_for_run,
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
            result = analyze_run(root / "run-baseline", manifest, {x["id"]: x for x in prompts})
            self.assertEqual(len(result["missing_prompt_ids"]), 9)
            output.write_text("keep", encoding="utf-8")
            args.remove("--validate-only")
            self.assertEqual(main(args), 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
