from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "prepare_probe69_model", Path(__file__).with_name("prepare_probe69_model.py"))
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

PINNED_PWSH = Path(r"C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe")
PINNED_PWSH_SHA256 = "362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139"


def pinned_powershell(testcase: unittest.TestCase) -> str:
    testcase.assertTrue(PINNED_PWSH.is_file(), f"pinned PowerShell 7.6.5 is missing: {PINNED_PWSH}")
    actual = hashlib.sha256(PINNED_PWSH.read_bytes()).hexdigest()
    testcase.assertEqual(actual, PINNED_PWSH_SHA256, "pinned PowerShell 7.6.5 SHA256 changed")
    return str(PINNED_PWSH)


def probe_line(record: dict) -> bytes:
    return b"strata pf_nonfinite: " + json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"


class DiagnosticFixtureTests(unittest.TestCase):
    def test_partial_record_does_not_advance_cursor_until_lf(self):
        begin = probe_line({"schema": 1, "event": "begin", "run": 1, "device": 0,
                            "pos0": 0, "tokens": 32, "diagnostic_only": True})
        snapshot = b"unrelated\n" + begin + b"strata pf_nonfinite: {\"schema\":1,\"event\":\"checkpoint\""
        lines, cursor, partial = prep.split_complete_log_bytes(snapshot, 0)
        self.assertEqual(lines, [b"unrelated", begin[:-1]])
        self.assertEqual(cursor, len(b"unrelated\n") + len(begin))
        self.assertTrue(partial.startswith(b"strata pf_nonfinite:"))

        completed = snapshot + b",\"run\":1,\"device\":0,\"completed_chunk\":0,\"completed_pos0\":0,\"completed_rows\":32,\"scans\":1,\"first_bad\":null}\n"
        lines2, cursor2, partial2 = prep.split_complete_log_bytes(completed, cursor)
        self.assertEqual(len(lines2), 1)
        self.assertTrue(lines2[0].endswith(b'"first_bad":null}'))
        self.assertEqual(cursor2, len(completed))
        self.assertEqual(partial2, b"")

    def test_probe_begin_and_null_checkpoint_are_normal(self):
        begin = {"schema": 1, "event": "begin", "run": 1, "device": 0,
                 "pos0": 0, "tokens": 1024, "diagnostic_only": True}
        checkpoint = {"schema": 1, "event": "checkpoint", "run": 1, "device": 0,
                      "completed_chunk": 0, "completed_pos0": 0, "completed_rows": 1024,
                      "scans": 8, "first_bad": None}
        records = prep.classify_complete_lines([
            probe_line(begin).rstrip(b"\n"), probe_line(checkpoint).rstrip(b"\n")])
        self.assertEqual([item["classification"] for item in records], ["normal_begin", "normal_checkpoint"])
        self.assertFalse(prep.diagnostic_is_fatal(records))

    def test_zero_nonfinite_bad_schema_and_unknown_event_fail_closed(self):
        bad = {"ordinal": 1, "stage": "input_residual", "chunk": 0, "pos0": 0,
               "layer": -1, "half": -1, "rows": 1, "width": 4, "index": 0,
               "row": 0, "column": 0, "nan": 1, "positive_inf": 0, "negative_inf": 0}
        lines = [
            b"strata zero-logits diag: position 1 selected 0",
            probe_line({"schema": 1, "event": "checkpoint", "run": 1, "device": 0,
                        "completed_chunk": 0, "completed_pos0": 0, "completed_rows": 1,
                        "scans": 1, "first_bad": bad}).rstrip(b"\n"),
            b"strata pf_nonfinite: {broken json",
            probe_line({"schema": 9, "event": "begin", "run": 1, "device": 0,
                        "pos0": 0, "tokens": 1, "diagnostic_only": True}).rstrip(b"\n"),
            probe_line({"schema": True, "event": "begin", "run": 1, "device": 0,
                        "pos0": 0, "tokens": 1, "diagnostic_only": True}).rstrip(b"\n"),
            probe_line({"schema": 1, "event": "unknown", "run": 1}).rstrip(b"\n"),
        ]
        records = prep.classify_complete_lines(lines)
        self.assertEqual([record["classification"] for record in records], [
            "fatal_zero_logits", "fatal_nonfinite", "fatal_invalid", "fatal_invalid", "fatal_invalid", "fatal_invalid"])
        self.assertTrue(prep.diagnostic_is_fatal(records))

    def test_quality_scope_is_exact_six_prompt_prefix(self):
        self.assertEqual(prep.require_prompt_prefix(list(prep.MANIFEST_IDS)), prep.PROMPT_IDS)
        self.assertEqual(len(prep.PROMPT_IDS), 6)
        with self.assertRaises(ValueError):
            prep.require_prompt_prefix(list(prep.MANIFEST_IDS[1:]))

    def test_diagnostic_receipt_write_is_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="probe69-exclusive-diagnostics-") as temp:
            path = Path(temp) / "F-long_1k-diagnostics.json"
            body = {"schema_version": 1, "prompt_id": "long_1k", "records": []}
            prep.exclusive_json(path, body)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), body)
            with self.assertRaises(FileExistsError):
                prep.exclusive_json(path, body)

    def test_quality_renderer_keeps_full_manifest_validation_and_six_prompt_loop(self):
        source = prep.RUN66_QUALITY.read_text(encoding="utf-8-sig")
        rendered = prep.rewrite_quality_script(source, r"C:\Run69\arms\probe69\quality-on")
        self.assertIn("manifest_prompt_count -ne 9", rendered)
        self.assertIn("prompt_count=$expectedIds.Count", rendered)
        self.assertIn("foreach ($prompt in $selectedPrompts)", rendered)
        self.assertIn("F-six-prompt-stage.json", rendered)
        self.assertIn('"F-$id-diagnostics.json"', rendered)
        self.assertIn("classification='normal_checkpoint'", rendered)
        self.assertIn("classification='fatal_nonfinite'", rendered)
        self.assertNotIn("$afterDiag.lines", rendered)


class PreparationBindingFixtureTests(unittest.TestCase):
    def test_cleanup_without_text_run_id_requires_explicit_path_binding(self):
        source = "$run = 'C:\\old-arm'\n"
        with self.assertRaises(ValueError):
            prep._replace_run_bindings(source, r"C:\old-arm", r"C:\new-arm")
        rendered = prep._replace_run_bindings(source, r"C:\old-arm", r"C:\new-arm", allow_zero_run_id=True)
        self.assertEqual(rendered, "$run = 'C:\\new-arm'\n")
        with self.assertRaises(ValueError):
            prep._replace_run_bindings(source + source, r"C:\old-arm", r"C:\new-arm", allow_zero_run_id=True)

    def test_probe_build_requires_both_exact_cache_and_configure_argv(self):
        valid = {
            "cmake_cache_controls": {"STRATA_PREFILL_FINITE_PROBE": "ON"},
            "configure_argv": ["-DSTRATA_PREFILL_FINITE_PROBE=ON"],
        }
        prep.validate_probe_build_controls(valid)
        for bad in (
            {"cmake_cache_controls": {}, "configure_argv": ["-DSTRATA_PREFILL_FINITE_PROBE=ON"]},
            {"cmake_cache_controls": {"STRATA_PREFILL_FINITE_PROBE": "OFF"},
             "configure_argv": ["-DSTRATA_PREFILL_FINITE_PROBE=ON"]},
            {"cmake_cache_controls": {"STRATA_PREFILL_FINITE_PROBE": "ON"},
             "configure_argv": ["-DSTRATA_PREFILL_FINITE_PROBE=OFF"]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prep.validate_probe_build_controls(bad)

    def test_run66_failure_history_uses_actual_upstream42_arm(self):
        matrix = prep.load_json(prep.RUN66 / "matrix.json")
        history_path = prep.run66_failure_history_path(matrix)
        self.assertEqual(history_path, prep.RUN66 / "arms/upstream42/quality-on/quality/F-quality-failure.json")
        history = prep.load_json(history_path)
        prep.validate_run66_failure_history(history)
        self.assertEqual(len(history["completed_prompts"]), 5)

    def test_emitted_powershell_classifier_accepts_real_normal_and_nonfinite_records(self):
        stderr = (prep.RUN / "fixture-execution-01/synthetic.stderr.log").read_text(encoding="utf-8").splitlines()
        normal = next(line for line in stderr if '"event":"checkpoint"' in line and '"first_bad":null' in line)
        nonfinite = next(line for line in stderr if '"event":"checkpoint"' in line and '"first_bad":{' in line)
        ps = pinned_powershell(self)
        with tempfile.TemporaryDirectory(prefix="probe69-ps-classifier-") as temp:
            script = Path(temp) / "classifier-fixture.ps1"
            script.write_text(
                "$ErrorActionPreference='Stop'\n" + prep._powershell_diagnostic_functions() +
                "\n$normal=Convert-DiagnosticLine @'\n" + normal + "\n'@\n" +
                "$bad=Convert-DiagnosticLine @'\n" + nonfinite + "\n'@\n" +
                "if($normal.classification -ne 'normal_checkpoint'){throw 'normal checkpoint rejected'}\n" +
                "if($bad.classification -ne 'fatal_nonfinite'){throw 'nonfinite checkpoint rejected'}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rendered_quality_script_has_valid_powershell_ast(self):
        ps = pinned_powershell(self)
        source = prep.RUN66_QUALITY.read_text(encoding="utf-8-sig")
        rendered = prep.rewrite_quality_script(source, r"C:\Probe69\arms\probe69\quality-on")
        with tempfile.TemporaryDirectory(prefix="probe69-ps-ast-") as temp:
            quality = Path(temp) / "quality.ps1"
            parser = Path(temp) / "parse-only.ps1"
            quality.write_text(rendered, encoding="utf-8")
            parser.write_text(
                "param([Parameter(Mandatory=$true)][string]$Path)\n" +
                "$tokens=$null;$errors=$null\n" +
                "[System.Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors) | Out-Null\n" +
                "if($errors.Count){$errors|ForEach-Object{Write-Error $_.Message};exit 1}\nexit 0\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(parser), str(quality)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_preparation_writes_only_inside_temporary_fixture(self):
        real_run66 = prep.RUN66
        real_run = prep.RUN
        with tempfile.TemporaryDirectory(prefix="probe69-full-preparation-fixture-") as temp:
            root = Path(temp)
            run = root / "run69"
            run66 = root / "run66"
            build_dir = root / "mock-build"
            build_dir.mkdir()
            receipt = build_dir / "build-receipt.json"
            receipt.write_text('{"fixture":"metadata only"}\n', encoding="utf-8")
            review = run / "preparation/root-source-review.json"
            review.parent.mkdir(parents=True)
            review.write_text('{"status":"fixture"}\n', encoding="utf-8")

            def copy_relative(relative: str) -> None:
                source = real_run66 / relative
                target = run66 / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

            for relative in ("matrix.json", "controls-normalization.json", "template-bindings.json", "run-phase.ps1"):
                copy_relative(relative)
            for side_id in ("upstream42", "hetero42"):
                phase = f"arms/{side_id}/quality-on"
                copy_relative(f"{phase}/config.json")
                copy_relative(f"{phase}/identity.json")
                copy_relative(f"{phase}/provenance.json")
                copy_relative(f"{phase}/fresh-identities.json")
                copy_relative(f"{phase}/preparation.json")
            copy_relative("arms/upstream42/quality-on/quality/F-quality-failure.json")
            for name in ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1"):
                relative = f"rendered-scripts/hetero42/{name}"
                source = real_run66 / relative
                target = run66 / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                text = source.read_text(encoding="utf-8-sig")
                old_arm = str(real_run66 / "arms/hetero42/quality-on").replace("/", "\\")
                new_arm = str(run66 / "arms/hetero42/quality-on").replace("/", "\\")
                target.write_text(text.replace(old_arm, new_arm), encoding="utf-8")

            fake_build = {
                "source_before": {"head": prep.SOURCE_HEAD, "tree": "tree-fixture", "clean": True},
                "source_after": {"head": prep.SOURCE_HEAD, "tree": "tree-fixture", "clean": True},
                "binary": {"path": str(build_dir / "strata.exe"), "sha256": "a" * 64, "size_bytes": 123},
                "test_binary": {"path": str(build_dir / "fixture-test.exe"), "sha256": "b" * 64, "size_bytes": 45},
                "build_directory": str(build_dir),
            }
            with patch.object(prep, "RUN", run), patch.object(prep, "RUN66", run66), \
                    patch.object(prep, "_source_tree_and_build_receipt",
                                 return_value=({}, fake_build, receipt)):
                result = prep._write_preparation()
            self.assertEqual(result["status"], "prepared")
            self.assertFalse(result["authorization_created"])
            self.assertEqual(result["request_output_leaves_created"], 0)
            matrix = prep.load_json(run / "matrix.json")
            self.assertEqual([side["id"] for side in matrix["sides"]], ["probe69"])
            self.assertFalse(matrix["execution_enabled"])
            self.assertTrue((run / "rendered-scripts/probe69/quality.ps1").is_file())
            self.assertEqual(list((run / "arms/probe69/quality-on/requests").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
