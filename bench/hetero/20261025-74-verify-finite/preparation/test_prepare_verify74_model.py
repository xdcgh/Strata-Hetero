from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "prepare_verify74_model", Path(__file__).with_name("prepare_verify74_model.py"))
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

PINNED_PWSH = Path(r"C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe")
PINNED_PWSH_SHA256 = "362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139"


def pinned_powershell(testcase: unittest.TestCase) -> str:
    testcase.assertTrue(PINNED_PWSH.is_file(), f"pinned PowerShell 7 is missing: {PINNED_PWSH}")
    testcase.assertEqual(hashlib.sha256(PINNED_PWSH.read_bytes()).hexdigest(), PINNED_PWSH_SHA256)
    return str(PINNED_PWSH)


def record(*, bad: bool = False, **changes) -> dict:
    value = {
        "schema": 1, "owner": 7, "window": 2, "device": 0, "pos0": 100, "rows": 3,
        "position_kind": "consecutive", "diagnostic_only": True, "timing_may_perturb": True,
        "bad": False, "checkpoints": 5, "ordinal": 0, "stage": "none", "format": "f32",
        "layer": -1, "half": -1, "scan_rows": 0, "width": 0, "row_offset": 0,
        "local_row": -1, "row": -1, "token_row": -1, "rows_per_token": 1,
        "position": -1, "column": -1, "nan": 0, "posinf": 0, "neginf": 0,
    }
    value["bad"] = bad
    if bad:
        value.update({"ordinal": 3, "stage": "routed_parts", "format": "bf16", "layer": 45, "half": 1,
                      "scan_rows": 24, "width": 8, "row_offset": 8, "local_row": 13,
                      "row": 21, "token_row": 2, "rows_per_token": 8,
                      "position": 102, "column": 5, "nan": 1, "posinf": 0, "neginf": 0})
    value.update(changes)
    return value


def line(value: dict) -> str:
    return "strata verify_nonfinite: " + json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def bytes_line(value: dict) -> bytes:
    return (line(value) + "\n").encode("utf-8")


def ps_classify(testcase: unittest.TestCase, inputs: list[tuple[str, str]]) -> dict[str, str]:
    ps = pinned_powershell(testcase)
    with tempfile.TemporaryDirectory(prefix="verify74-classifier-") as temp:
        path = Path(temp) / "classify.ps1"
        cases = json.dumps([{"name": name, "line": raw} for name, raw in inputs], separators=(",", ":"), ensure_ascii=False)
        path.write_text(
            "$ErrorActionPreference='Stop'\n" + prep._powershell_diagnostic_functions() +
            "\n$cases=@'\n" + cases + "\n'@|ConvertFrom-Json -AsHashtable\n" +
            "$out=foreach($case in $cases){$r=Convert-DiagnosticLine $case.line;"
            "[ordered]@{name=$case.name;classification=$r.classification}}\n" +
            "$out|ConvertTo-Json -Compress -Depth 8\n",
            encoding="utf-8",
        )
        result = subprocess.run([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(path)],
                                capture_output=True, text=True, timeout=30, check=False)
    testcase.assertEqual(result.returncode, 0, result.stderr)
    decoded = json.loads(result.stdout)
    if isinstance(decoded, dict):
        decoded = [decoded]
    return {item["name"]: item["classification"] for item in decoded}


def parse_powershell_files(testcase: unittest.TestCase, paths: list[Path]) -> None:
    ps = pinned_powershell(testcase)
    with tempfile.TemporaryDirectory(prefix="verify74-five-script-ast-") as temp:
        parser = Path(temp) / "parse-only.ps1"
        parser.write_text(
            "param([string]$PathsJson)\n$ErrorActionPreference='Stop'\n"
            "$paths=ConvertFrom-Json -InputObject $PathsJson -AsHashtable\n"
            "foreach($path in $paths){$tokens=$null;$errors=$null;"
            "[void][System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors);"
            "if($errors.Count){$errors|ForEach-Object{[Console]::Error.WriteLine($_.ToString())};exit 1};"
            "[Console]::Out.WriteLine((Split-Path -Leaf $path)+' parse-only pass')}\n",
            encoding="utf-8",
        )
        path_json = json.dumps([str(path) for path in paths], separators=(",", ":"))
        result = subprocess.run([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(parser), path_json],
                                capture_output=True, text=True, timeout=30, check=False)
    testcase.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    testcase.assertEqual(len(result.stdout.splitlines()), len(paths))
    testcase.assertTrue(all(line.endswith(" parse-only pass") for line in result.stdout.splitlines()))


class VerifyDiagnosticContractTests(unittest.TestCase):
    def test_bad_false_defaults_are_normal_and_two_owner_windows_stay_distinct(self):
        first = record(owner=100, window=1)
        second = record(owner=101, window=2, checkpoints=0)
        parsed = [prep.classify_diagnostic_line(line(item)) for item in (first, second)]
        self.assertEqual([item["classification"] for item in parsed],
                         ["normal_verify_window", "normal_verify_window"])
        self.assertEqual([item["parsed"]["owner"] for item in parsed], [100, 101])
        self.assertEqual([item["parsed"]["window"] for item in parsed], [1, 2])
        self.assertFalse(prep.diagnostic_is_fatal(parsed))

    def test_bad_true_consecutive_coordinates_and_per_row_runtime_positions(self):
        consecutive = record(bad=True)
        per_row = record(bad=True, position_kind="per_row", pos0=0, position=991)
        records = [prep.classify_diagnostic_line(line(item)) for item in (consecutive, per_row)]
        self.assertEqual([item["classification"] for item in records], ["fatal_nonfinite", "fatal_nonfinite"])
        self.assertEqual(records[0]["parsed"]["position"], 102)
        self.assertEqual(records[1]["parsed"]["position"], 991)

    def test_bad_true_rejects_invalid_coords_kind_ordinal_counts_and_owner_types(self):
        base = record(bad=True)
        invalid = (
            {"local_row": 24}, {"row": 22}, {"token_row": 1}, {"column": 8},
            {"rows_per_token": 0}, {"ordinal": 6}, {"checkpoints": 2},
            {"nan": 0, "posinf": 0, "neginf": 0}, {"position": 103},
            {"owner": True}, {"format": "f64"}, {"stage": "new_stage"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                parsed = prep.classify_diagnostic_line(line({**base, **changes}))
                self.assertEqual(parsed["classification"], "fatal_invalid")

    def test_bad_false_rejects_nondefault_coords(self):
        for changes in ({"stage": "input_residual"}, {"column": 0}, {"nan": 1}, {"ordinal": 1}):
            with self.subTest(changes=changes):
                self.assertEqual(prep.classify_diagnostic_line(line(record(**changes)))["classification"],
                                 "fatal_invalid")

    def test_unknown_schema_format_stage_and_zero_logits_fail_closed(self):
        bad = record(bad=True)
        records = [
            line({**record(), "schema": True}),
            line({**bad, "schema": 2}),
            line({**bad, "format": "f64"}),
            line({**bad, "stage": "new_stage"}),
            "strata verify_nonfinite: {broken",
            "strata zero-logits diag: position 1 selected 0",
            "strata zero-logits unexpected format",
        ]
        parsed = [prep.classify_diagnostic_line(item) for item in records]
        self.assertEqual([item["classification"] for item in parsed], [
            "fatal_invalid", "fatal_invalid", "fatal_invalid", "fatal_invalid", "fatal_invalid",
            "fatal_zero_logits", "fatal_invalid",
        ])

    def test_partial_utf8_record_does_not_advance_until_newline_and_bad_utf8_is_fatal(self):
        normal = bytes_line(record())
        split_utf8 = b"strata verify_nonfinite: \xff"
        initial = b"ordinary\n" + normal + split_utf8
        complete, cursor, partial = prep.split_complete_log_bytes(initial, 0)
        self.assertEqual(complete, [b"ordinary", normal[:-1]])
        self.assertEqual(cursor, len(b"ordinary\n") + len(normal))
        self.assertEqual(partial, split_utf8)
        continuation = b"\xfe\n"
        complete2, cursor2, partial2 = prep.split_complete_log_bytes(initial + continuation, cursor)
        self.assertEqual(cursor2, len(initial + continuation))
        self.assertEqual(partial2, b"")
        bad = prep.classify_complete_lines(complete2)
        self.assertEqual(bad[0]["classification"], "fatal_invalid")
        self.assertIn("raw_base64", bad[0])

    def test_stage_enum_header_cases_and_formatter_strings_match(self):
        header = prep.SOURCE / "include/strata/core/verify_probe.hpp"
        formatter = prep.SOURCE / "src/core/verify_probe.cu"
        self.assertEqual(prep.verify_stage_contract(header, formatter), prep.PROBE_STAGES)

    def test_python_and_generated_powershell_classifiers_have_same_protocol_results(self):
        bad = record(bad=True)
        cases = [
            ("normal_false", line(record())),
            ("owner_window_b", line(record(owner=99, window=7))),
            ("bad_consecutive", line(bad)),
            ("bad_per_row", line(record(bad=True, position_kind="per_row", pos0=0, position=4500))),
            ("input_layer_minus_one", line(record(bad=True, stage="input_residual", layer=-1, half=-1,
                                                   row_offset=0, local_row=0, row=0, token_row=0,
                                                   rows_per_token=1, position=100, column=0))),
            ("schema_unknown", line({**record(), "schema": 3})),
            ("format_unknown", line({**bad, "format": "f64"})),
            ("stage_unknown", line({**bad, "stage": "stage_future"})),
            ("coordinate_invalid", line({**bad, "row": 999})),
            ("zero_logits", "strata zero-logits diag: chosen token is zero"),
        ]
        expected = {}
        for name, raw in cases:
            parsed = prep.classify_diagnostic_line(raw)
            expected[name] = None if parsed is None else parsed["classification"]
        self.assertEqual(ps_classify(self, cases), expected)

    def test_quality_contract_keeps_six_prefix_full_nine_manifest_and_diagnostic_before_evaluator(self):
        source = prep.RUN66_QUALITY.read_text(encoding="utf-8-sig")
        rendered = prep.rewrite_quality_script(source, r"C:\Run74\arms\verify74\quality-on")
        self.assertIn("manifest_prompt_count -ne 9", rendered)
        self.assertIn("prompt_count -ne $expectedIds.Count", rendered)
        self.assertIn("foreach ($prompt in $selectedPrompts)", rendered)
        self.assertIn("STRATA_VERIFY_FINITE_PROBE -ne '1'", rendered)
        self.assertNotIn("STRATA_PREFILL_FINITE_PROBE -ne '1'", rendered)
        self.assertNotIn("STRATA_PREFILL_DETAIL_LAYER", rendered)
        saved = rendered.index("prompt_diagnostics_saved")
        evaluator = rendered.index("$rawIdArgs =")
        self.assertLess(saved, evaluator)
        self.assertIn("fatal_partial_line", rendered)
        self.assertIn("fatal_reader_error", rendered)
        self.assertIn("classification='normal_verify_window'", rendered)
        self.assertIn("actual_token_ids_match_h4", rendered)
        self.assertIn("six_prompt_verify_probe_diagnostic_complete", rendered)

    def test_generated_quality_script_has_valid_powershell_ast_only(self):
        source = prep.RUN66_QUALITY.read_text(encoding="utf-8-sig")
        rendered = prep.rewrite_quality_script(source, r"C:\Run74\arms\verify74\quality-on")
        ps = pinned_powershell(self)
        with tempfile.TemporaryDirectory(prefix="verify74-quality-ast-") as temp:
            quality = Path(temp) / "quality.ps1"
            parser = Path(temp) / "parse-only.ps1"
            quality.write_text(rendered, encoding="utf-8")
            parser.write_text(
                "param([string]$Path)\n$tokens=$null;$errors=$null\n"
                "[void][System.Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors)\n"
                "if($errors.Count){$errors|ForEach-Object{[Console]::Error.WriteLine($_.ToString())};exit 1}\nexit 0\n",
                encoding="utf-8")
            result = subprocess.run([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(parser), str(quality)],
                                    capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_build_receipt_must_prove_verify_on_prefill_off_and_global_tests_off(self):
        good = {"cmake_cache_controls": {"STRATA_VERIFY_FINITE_PROBE": "ON",
                                          "STRATA_PREFILL_FINITE_PROBE": "OFF", "STRATA_BUILD_TESTS": "OFF"},
                "configure_argv": ["-DSTRATA_VERIFY_FINITE_PROBE=ON", "-DSTRATA_PREFILL_FINITE_PROBE=OFF",
                                   "-DSTRATA_BUILD_TESTS=OFF"]}
        prep.validate_probe_build_controls(good)
        for change in (
            {"cmake_cache_controls": {**good["cmake_cache_controls"], "STRATA_VERIFY_FINITE_PROBE": "OFF"}},
            {"cmake_cache_controls": {**good["cmake_cache_controls"], "STRATA_PREFILL_FINITE_PROBE": "ON"}},
            {"cmake_cache_controls": {**good["cmake_cache_controls"], "STRATA_BUILD_TESTS": "ON"}},
            {"configure_argv": good["configure_argv"][:-1]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prep.validate_probe_build_controls({**good, **change})


class PreparationBindingFixtureTests(unittest.TestCase):
    def test_complete_temporary_preparation_writes_no_request_leaf_and_binds_only_actual_controls(self):
        real_run66, real_quality = prep.RUN66, prep.RUN66_QUALITY
        with tempfile.TemporaryDirectory(prefix="verify74-full-preparation-fixture-") as temp:
            root = Path(temp)
            run, run66 = root / "run74", root / "run66"
            build_dir = root / "mock-build"
            build_dir.mkdir()
            receipt = build_dir / "build-receipt.json"
            receipt.write_text('{"fixture":"metadata only"}\n', encoding="utf-8")
            review = run / "preparation/root-source-review.json"
            review.parent.mkdir(parents=True)
            review.write_text('{"status":"fixture"}\n', encoding="utf-8")

            def copy_rel(relative: str) -> None:
                source, target = real_run66 / relative, run66 / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

            for relative in ("matrix.json", "controls-normalization.json", "template-bindings.json", "run-phase.ps1"):
                copy_rel(relative)
            for side_id in ("upstream42", "hetero42"):
                phase = f"arms/{side_id}/quality-on"
                for name in ("config.json", "identity.json", "provenance.json", "fresh-identities.json", "preparation.json"):
                    copy_rel(f"{phase}/{name}")
            copy_rel("arms/upstream42/quality-on/quality/F-quality-failure.json")
            for name in ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1"):
                rel = f"rendered-scripts/hetero42/{name}"
                src, dst = real_run66 / rel, run66 / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                text = src.read_text(encoding="utf-8-sig")
                old_arm = str(real_run66 / "arms/hetero42/quality-on").replace("/", "\\")
                new_arm = str(run66 / "arms/hetero42/quality-on").replace("/", "\\")
                dst.write_text(text.replace(old_arm, new_arm), encoding="utf-8")
            fake = {
                "source_before": {"head": prep.SOURCE_HEAD, "tree": "tree-fixture", "clean": True},
                "source_after": {"head": prep.SOURCE_HEAD, "tree": "tree-fixture", "clean": True},
                "binary": {"path": str(build_dir / "strata.exe"), "sha256": "a" * 64, "size_bytes": 123},
                "test_binary": {"path": str(build_dir / "verify_finite_probe_graph_test.exe"),
                                "sha256": "b" * 64, "size_bytes": 45},
                "build_directory": str(build_dir), "tests_built": True, "tests_run": False,
                "test_binary_executed": False, "cmake_cache_controls": {"STRATA_VERIFY_FINITE_PROBE": "ON",
                    "STRATA_PREFILL_FINITE_PROBE": "OFF", "STRATA_BUILD_TESTS": "OFF"},
            }
            with patch.object(prep, "RUN", run), patch.object(prep, "RUN66", run66), \
                    patch.object(prep, "RUN66_QUALITY", run66 / "rendered-scripts/hetero42/quality.ps1"), \
                    patch.object(prep, "_source_tree_and_build_receipt", return_value=({}, fake, receipt)):
                result = prep._write_preparation()
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["request_output_leaves_created"], 0)
            matrix = prep.load_json(run / "matrix.json")
            self.assertEqual([side["id"] for side in matrix["sides"]], ["verify74"])
            self.assertIsNone(matrix["root_authorization"])
            self.assertFalse(matrix["execution_enabled"])
            self.assertEqual(matrix["quality_recipe"]["prompt_ids"], list(prep.PROMPT_IDS))
            self.assertEqual(matrix["quality_recipe"]["manifest_prompt_ids"], list(prep.MANIFEST_IDS))
            self.assertEqual(matrix["quality_recipe"]["warmups_per_prompt"], 1)
            self.assertEqual(matrix["quality_recipe"]["formals_per_prompt"], 3)
            self.assertTrue(matrix["quality_recipe"]["strict_quality_stop_first"])
            self.assertEqual(matrix["fresh_launch_gates"]["minimum_vram_mib"], 46000)
            self.assertEqual(matrix["fresh_launch_gates"]["minimum_ram_gib"], 12)
            self.assertTrue(matrix["fresh_launch_gates"]["inspect_running_wsl"])
            self.assertFalse(matrix["fresh_launch_gates"]["wsl_shutdown_before_observation"])
            config = prep.load_json(run / "arms/verify74/quality-on/config.json")
            self.assertEqual(config["env"]["STRATA_VERIFY_FINITE_PROBE"], "1")
            self.assertEqual(config["env"]["STRATA_DIAG_ZERO_LOGITS"], "1")
            self.assertNotIn("STRATA_PREFILL_FINITE_PROBE", config["env"])
            self.assertNotIn("STRATA_PREFILL_DETAIL_LAYER", config["env"])
            provenance = prep.load_json(run / "arms/verify74/quality-on/provenance.json")
            self.assertEqual(provenance["build_receipt"]["cmake_cache_controls"]["STRATA_VERIFY_FINITE_PROBE"], "ON")
            self.assertEqual(provenance["build_receipt"]["cmake_cache_controls"]["STRATA_PREFILL_FINITE_PROBE"], "OFF")
            self.assertEqual(provenance["build_receipt"]["cmake_cache_controls"]["STRATA_BUILD_TESTS"], "OFF")
            self.assertTrue(provenance["build_receipt"]["tests_built"])
            self.assertFalse(provenance["build_receipt"]["tests_run"])
            self.assertEqual(list((run / "arms/verify74/quality-on/requests").iterdir()), [])
            self.assertIn("--inspect-running-wsl", (run / "rendered-scripts/verify74/launch.ps1").read_text())
            self.assertNotIn("wsl.exe --shutdown", (run / "rendered-scripts/verify74/launch.ps1").read_text())
            parse_powershell_files(self, [run / "run-phase.ps1"] + [
                run / "rendered-scripts/verify74" / name for name in
                ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1")])


if __name__ == "__main__":
    unittest.main()
