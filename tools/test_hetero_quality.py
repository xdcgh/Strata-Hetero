from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import hetero_quality as quality


class FakeTokenizer:
    def encode(self, text: str, parse_special: bool = False):
        # Deterministic fixture tokenizer; tests prompt/evaluator logic, never substitutes for production counts.
        import re
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class HeteroQualityTests(unittest.TestCase):
    def test_marker_generation_is_seeded_and_avoids_source_literals(self):
        a = quality.make_markers("ordinary prose only", 42)
        b = quality.make_markers("ordinary prose only", 42)
        self.assertEqual(a, b)
        self.assertEqual(len({m["key"] for m in a}), 3)
        self.assertTrue(all(m["key"] not in "ordinary prose only" and m["value"] not in "ordinary prose only" for m in a))

    def test_long_body_inserts_each_marker_once_and_no_source_repetition(self):
        tokenizer = FakeTokenizer()
        paragraphs = [f"Unique natural paragraph number {i:04d} has prose about a harbor and a book." for i in range(100)]
        markers = quality.make_markers(" ".join(paragraphs), 42)
        body, positions = quality.build_long_body(paragraphs, 40, len(paragraphs[39]), markers, tokenizer,
                                                  "Read once.", "Return JSON once.")
        for marker in markers:
            self.assertEqual(body.count(marker["marker"]), 1)
        for p in paragraphs[:40]:
            self.assertEqual(body.count(p), 1)
        self.assertEqual([p["target_relative_depth"] for p in positions], [0.1, 0.5, 0.9])
        self.assertEqual(sorted(p["token_position_body_prefix"] for p in positions),
                         [p["token_position_body_prefix"] for p in positions])

    def test_long_prompt_sizer_respects_tolerance_and_returns_count(self):
        tokenizer = FakeTokenizer()
        paragraphs = [f"Paragraph {i} includes varied natural language about history, weather, and people." for i in range(100)]
        markers = quality.make_markers(" ".join(paragraphs), 42)
        body, positions, count, selected = quality.size_long_body(paragraphs, 180, markers, tokenizer,
                                                                  "Read once and remember markers.",
                                                                  "Return three values as JSON.")
        self.assertLessEqual(abs(count - 180), 64)
        self.assertEqual(len(tokenizer.encode(body)), count)
        self.assertTrue(0 < selected <= len(paragraphs))
        self.assertEqual(len(positions), 3)

    def test_json_numeric_nonfinite_is_separate_from_retrieval_and_arithmetic(self):
        json_prompt = {"id": "json", "quality": {"kind": "json_arithmetic", "expected_result": 2,
                                                       "expected_expression": "1+1"}}
        result = quality.evaluate_text(json_prompt, '{"result": NaN, "expression":"1+1"}')
        self.assertEqual(result["quality_status"], "nonfinite_json_number")
        self.assertEqual(result["nonfinite_json_numeric_check"], "fail")
        retrieval_prompt = {"id": "ret", "quality": {"kind": "json_retrieval", "expected": {"K": "V"}}}
        result = quality.evaluate_text(retrieval_prompt, '{"K":"NaN"}')
        self.assertEqual(result["quality_status"], "retrieval_mismatch")
        self.assertEqual(result["exact_values_found"], {"K": False})
        exact = quality.evaluate_text(retrieval_prompt, '{"K":"V"}')
        self.assertEqual(exact["quality_status"], "pass")
        extra = quality.evaluate_text(retrieval_prompt, '{"K":"V","other":"x"}')
        self.assertEqual(extra["quality_status"], "retrieval_mismatch")
        infinite = quality.evaluate_text(retrieval_prompt, '{"K":"V","other":1e999}')
        self.assertEqual(infinite["nonfinite_json_numeric_check"], "fail")

    def test_truncation_bang_loop_and_exact_text_statuses(self):
        p = {"id": "a", "quality": {"kind": "exact_text", "expected": "4"}}
        self.assertEqual(quality.evaluate_text(p, "!" * 128)["generation_stability"], "fail_single_character_bang_loop")
        capped_loop = quality.evaluate_text(p, "!" * 128, run_status="truncated_by_length")
        self.assertEqual(capped_loop["quality_status"], "generation_degenerate_bang_loop")
        self.assertEqual(capped_loop["generation_stability"], "fail_single_character_bang_loop")
        self.assertEqual(quality.evaluate_text(p, "4", run_status="truncated_by_length")["quality_status"],
                         "not_assessed_incomplete_run")
        self.assertEqual(quality.evaluate_text(p, "4")["quality_status"], "pass")

    def test_python_is_parsed_but_never_executed(self):
        p = {"id": "code", "quality": {"kind": "python_ast", "function": "add_two", "arguments": ["x"],
                                           "return_expression": {"left_name": "x", "operator": "Add", "right_constant": 2}}}
        result = quality.evaluate_text(p, "def add_two(x):\n    return x + 2\n")
        self.assertEqual(result["quality_status"], "pass")
        self.assertEqual(result["execution"], "not_run; generated code is parsed only")
        bad = quality.evaluate_text(p, "def add_two(x)\n return x+2")
        self.assertEqual(bad["quality_status"], "invalid_python_syntax")
        wrong = quality.evaluate_text(p, "def add_two(x):\n    return x + 3\n")
        self.assertEqual(wrong["quality_status"], "function_return_expression_mismatch")
        reencoded = quality.evaluate_text(p, "def add_two(x):\n    return x + 2\n", tokenizer=FakeTokenizer())
        self.assertIsInstance(reencoded["canonical_visible_text_reencoded_tokens"], int)
        self.assertIsNone(reencoded["actual_emitted_token_ids"])

    def test_python_rejects_coroutines_overrides_and_signature_side_effects(self):
        p = {"id": "code", "quality": {"kind": "python_ast", "function": "add_two", "arguments": ["x"],
                                           "return_expression": {"left_name": "x", "operator": "Add", "right_constant": 2}}}
        cases = [
            "async def add_two(x):\n    return x + 2\n",
            "def add_two(x):\n    return x + 2\nadd_two = None\n",
            "@alter\ndef add_two(x):\n    return x + 2\n",
            "def add_two(x=external_call()):\n    return x + 2\n",
            "def add_two(x, *, extra):\n    return x + 2\n",
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                self.assertNotEqual(quality.evaluate_text(p, candidate)["quality_status"], "pass")

    def test_collect_hetero_bench_artifacts_maps_prompt_and_separates_failed_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run-01"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({"prompt_file": "smoke_arithmetic.prompt.txt"}), encoding="utf-8")
            (run / "requests.json").write_text(json.dumps([{"status": "completed"},
                                                             {"status": "truncated_by_length"}]), encoding="utf-8")
            (run / "formal-0000.text.txt").write_text("4", encoding="utf-8")
            (run / "formal-0001.text.txt").write_text("4", encoding="utf-8")
            prompts = {"smoke_arithmetic": {"id": "smoke_arithmetic",
                                            "quality": {"kind": "exact_text", "expected": "4"}}}
            got = quality.collect_responses(root, prompts, 100)
            self.assertEqual([r["quality_status"] for r in got], ["pass", "not_assessed_incomplete_run"])

    def test_response_file_size_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.txt"
            p.write_text("12345", encoding="utf-8")
            with self.assertRaises(ValueError):
                quality._read_bounded(p, 4)


if __name__ == "__main__":
    unittest.main()
