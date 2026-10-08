from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import hetero_bench as hb


def sse(*events: bytes) -> io.BytesIO:
    return io.BytesIO(b"".join(events))


def event(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class BrokenStream:
    def __iter__(self):
        yield event({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]})
        raise TimeoutError("mock read timed out")


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class HeteroBenchTests(unittest.TestCase):
    def test_sse_usage_reasoning_and_role_empty_delta_ttft(self):
        import time
        start = time.perf_counter()
        stream = sse(b"\xef\xbb\xbf: ping\n\n",
                     event({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
                     event({"choices": [{"delta": {}, "finish_reason": None}]}),
                     event({"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]}),
                     event({"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]}),
                     event({"choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                            "timings": {"predicted_per_second": 44.2}}),
                     b"data: [DONE]\n\n")
        got = hb._read_sse(stream, "", start)
        self.assertEqual(got["reasoning"], "think")
        self.assertEqual(got["content"], "answer")
        self.assertIsNotNone(got["first_generated_s"])
        self.assertIsNotNone(got["first_visible_content_s"])
        self.assertEqual(got["finish_reason"], "stop")
        self.assertEqual(got["usage"]["completion_tokens"], 2)
        self.assertEqual(got["timings"]["predicted_per_second"], 44.2)
        self.assertIsNone(got["status"])

    def test_server_error_event_preserved_as_failure(self):
        got = hb._read_sse(sse(event({"error": {"message": "bad request"}}), b"data: [DONE]\n\n"), "", 0)
        self.assertEqual(got["status"], "server_error")
        self.assertEqual(len(got["events"]), 1)

    def test_truncated_and_timeout_keep_partial_event(self):
        got = hb._read_sse(sse(event({"choices": [{"delta": {"content": "partial"}}]})), "", 0)
        self.assertEqual(got["status"], "truncated")
        self.assertEqual(got["content"], "partial")
        timed = hb._read_sse(BrokenStream(), "", 0)
        self.assertEqual(timed["status"], "timeout")
        self.assertEqual(timed["content"], "partial")

    def test_redaction_and_response_artifacts_never_store_api_key(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = argparse.Namespace(model="", max_tokens=8, seed=42, timeout=1, reasoning_effort="none")
            secret = "test-secret-value"
            response = JsonResponse(b"".join(sse(
                event({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}),
                event({"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 1, "prompt_tokens": 4, "total_tokens": 5}}),
                b"data: [DONE]\n\n")))
            reqs = []
            with mock.patch.object(hb._LOOPBACK_OPENER, "open", side_effect=lambda req, timeout: (reqs.append(req) or response)):
                rec = hb.request_once("http://127.0.0.1:9", "prompt", a, d, "formal", 0, secret)
            self.assertEqual(rec["status"], "completed")
            self.assertEqual(reqs[0].get_header("Authorization"), f"Bearer {secret}")
            event_rows = json.loads((d / "formal-0000.events.json").read_text(encoding="utf-8"))
            self.assertEqual(event_rows[-1]["usage"], {"completion_tokens": 1, "prompt_tokens": 4, "total_tokens": 5})
            for p in d.iterdir():
                self.assertNotIn(secret, p.read_text(encoding="utf-8"))

    def test_generated_token_equals_seven_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = argparse.Namespace(model="", max_tokens=8, seed=42, timeout=1, reasoning_effort="none")
            payload = sse(event({"choices": [{"delta": {"content": "token=7"}, "finish_reason": None}]}),
                          event({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                 "usage": {"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5}}),
                          b"data: [DONE]\n\n")
            with mock.patch.object(hb._LOOPBACK_OPENER, "open", return_value=payload):
                row = hb.request_once("http://127.0.0.1:9", "p", a, d, "formal", 0, "")
            self.assertEqual((d / "formal-0000.text.txt").read_text(encoding="utf-8"), "token=7")
            evs = json.loads((d / "formal-0000.events.json").read_text(encoding="utf-8"))
            self.assertEqual(evs[0]["choices"][0]["delta"]["content"], "token=7")
            self.assertEqual(evs[1]["usage"]["completion_tokens"], 2)
            self.assertEqual(row["status"], "completed")

    def test_output_bytes_preserve_unicode_and_intentional_line_endings(self):
        import hashlib
        content = "one\ntwo\r\n中文"
        reasoning = "first\r\nsecond\n"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(model="", max_tokens=32, seed=42, timeout=1, reasoning_effort="none")
            response = sse(event({"choices": [{"delta": {"content": content, "reasoning_content": reasoning}}]}),
                           event({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                           b"data: [DONE]\n\n")
            with mock.patch.object(hb._LOOPBACK_OPENER, "open", return_value=response):
                row = hb.request_once("http://127.0.0.1:9", "p", args, root, "formal", 0, "")
            self.assertEqual((root / "formal-0000.text.txt").read_bytes(), content.encode("utf-8"))
            self.assertEqual((root / "formal-0000.reasoning.txt").read_bytes(), reasoning.encode("utf-8"))
            self.assertEqual(row["content_sha256"], hashlib.sha256(content.encode("utf-8")).hexdigest())

    def test_validate_only_has_no_network_or_output_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            prompt = d / "p.txt"
            prompt.write_text("hello", encoding="utf-8")
            argv = ["--prompt-file", str(prompt), "--output", str(d / "runs"), "--max-tokens", "8",
                    "--validate-only", "--run-id", "dry-run"]
            with mock.patch.object(hb._LOOPBACK_OPENER, "open", side_effect=AssertionError("network used")):
                self.assertEqual(hb.main(argv), 0)
            self.assertFalse((d / "runs").exists())

    def test_existing_run_path_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            prompt = d / "p.txt"
            prompt.write_text("hello", encoding="utf-8")
            root = d / "runs"
            existing = root / "same"
            existing.mkdir(parents=True)
            marker = existing / "keep"
            marker.write_text("unchanged", encoding="utf-8")
            argv = ["--prompt-file", str(prompt), "--output", str(root), "--max-tokens", "8", "--run-id", "same"]
            with mock.patch.object(hb._LOOPBACK_OPENER, "open", side_effect=AssertionError("network used")):
                self.assertEqual(hb.main(argv), 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_remote_base_url_rejected(self):
        with self.assertRaises(ValueError):
            hb.validate_base_url("http://0.0.0.0:8081")
        with self.assertRaises(ValueError):
            hb.validate_base_url("https://127.0.0.1:8081")

    def test_identity_shape_and_live_comparison_do_not_auto_accept(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "identity.json"
            p.write_text(json.dumps({"schema": "v1", "source": "new-build", "config": "cfg-hash",
                                     "model": "qwen", "engine": "0.1.39", "api_key": "secret-value"}),
                         encoding="utf-8")
            identity, supplied = hb.load_identity(str(p))
            self.assertTrue(supplied)
            self.assertEqual(identity["api_key"], "[redacted]")
            check = hb.compare_identity(identity, {"health": {"model": "qwen"},
                                                    "models": {"data": [{"id": "qwen"}]},
                                                    "props": {"build_info": "Strata 0.1.39"}}, supplied)
            self.assertEqual(check["matches"], {"model": True, "engine": True})
            self.assertEqual(check["unverifiable"], ["schema", "source", "config"])
            self.assertNotEqual(check["status"], "accepted")

    def test_proxy_is_disabled_and_redirects_are_rejected(self):
        self.assertEqual(hb._NO_PROXY_HANDLER.proxies, {})
        req = urllib.request.Request("http://127.0.0.1:9/", method="POST")
        self.assertIsNone(hb._NO_REDIRECT_HANDLER.redirect_request(req, None, 302, "Found", {}, "http://example.com/"))
        with mock.patch.object(hb._LOOPBACK_OPENER, "open", return_value=io.BytesIO(b"ok")) as opened:
            hb.safe_open(req, timeout=1)
            self.assertIs(opened.call_args.args[0], req)

    def test_summary_excludes_failures_and_uses_formal_batch_wall_time(self):
        rows = [
            {"status": "completed", "client_e2e_s": 1.0, "first_generated_s": .2,
             "first_visible_content_s": .3, "usage": {"completion_tokens": 5},
             "timings": {"predicted_per_second": 8.0, "cache_n": 0}, "batch_index": 0},
            {"status": "completed", "client_e2e_s": 2.0, "usage": None,
             "timings": None, "batch_index": 0},
            {"status": "error", "client_e2e_s": 100.0, "usage": None,
             "timings": None, "batch_index": 0},
        ]
        batches = [{"batch_index": 0, "elapsed_s": 2.0}]
        got = hb.summarize(rows, 3, batches)
        self.assertEqual(got["successful_requests"], 2)
        self.assertEqual(got["completion_tokens_total_matched_subset"], 5)
        self.assertEqual(got["completion_token_count_missing_or_invalid"], 1)
        self.assertEqual(got["concurrent_batch_output_tok_s"], 2.5)
        self.assertEqual(got["client_e2e_s_mean_success_only"], 1.5)
        self.assertIsNone(got["all_successful_requests_cache_clean"])

    def test_live_identity_mismatch_writes_receipt_without_benchmark_requests(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            prompt = d / "prompt.txt"
            prompt.write_text("hi", encoding="utf-8")
            identity = d / "identity.json"
            identity.write_text(json.dumps({"schema": "v1", "source": "s", "config": "c",
                                            "model": "expected-model", "engine": "1.0"}), encoding="utf-8")
            live = {"health": {"model": "wrong-model"},
                    "models": {"data": [{"id": "wrong-model"}]},
                    "props": {"build_info": "Strata 2.0"}}
            argv = ["--prompt-file", str(prompt), "--output", str(d / "runs"), "--max-tokens", "8",
                    "--identity-json", str(identity), "--run-id", "mismatch"]
            with mock.patch.object(hb, "fetch_live_identity", return_value=live), \
                 mock.patch.object(hb, "request_once", side_effect=AssertionError("benchmark request must not run")):
                self.assertEqual(hb.main(argv), 3)
            receipt = json.loads(((d / "runs" / "mismatch") / "preflight-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "identity_mismatch_preflight_failed")
            self.assertFalse(receipt["benchmark_started"])

    def test_clean_length_cap_can_time_output_without_claiming_quality(self):
        rows = [{"status": "truncated_by_length", "client_e2e_s": 2.0,
                 "usage": {"completion_tokens": 256}, "batch_index": 0,
                 "timings": {"cache_n": 0, "prompt_per_second": 900.0}}]
        batches = [{"batch_index": 0, "elapsed_s": 2.0}]
        strict = hb.summarize(rows, 1, batches)
        timed = hb.summarize(rows, 1, batches, allow_capped=True)
        self.assertEqual(strict["successful_requests"], 0)
        self.assertEqual(timed["concurrent_batch_output_tok_s"], 128.0)
        self.assertEqual(timed["length_capped_requests_in_timing"], 1)
        self.assertFalse(timed["quality_pass_inferred_from_finish"])
        self.assertEqual(timed["uncached_prompt_per_second_mean"], 900.0)

    def test_cached_or_unknown_prefill_does_not_become_uncached_speed(self):
        rows = [{"status": "completed", "client_e2e_s": 1.0,
                 "usage": None, "timings": {"cache_n": 500, "prompt_per_second": 9000.0}},
                {"status": "completed", "client_e2e_s": 1.0,
                 "usage": None, "timings": {"prompt_per_second": 9000.0}}]
        got = hb.summarize(rows, 1)
        self.assertIsNone(got["uncached_prompt_per_second_mean"])
        self.assertIsNone(got["all_successful_requests_cache_clean"])


if __name__ == "__main__":
    unittest.main()
