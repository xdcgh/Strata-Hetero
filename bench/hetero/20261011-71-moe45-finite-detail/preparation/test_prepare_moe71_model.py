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
    "prepare_moe71_model", Path(__file__).with_name("prepare_moe71_model.py"))
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

        bad = {"ordinal": 1, "stage": "routed_activation_f16", "format": "f16", "chunk": 2, "pos0": 8,
               "layer": 45, "half": 0, "rows": 2, "width": 8, "index": 9,
               "row": 1, "column": 1, "nan": 0, "positive_inf": 1, "negative_inf": 0,
               "row_offset": 50, "source_row": 51, "expert": 4}
        typed_line = probe_line({"schema": 1, "event": "checkpoint", "run": 2, "device": 0,
                                 "completed_chunk": 3, "completed_pos0": 9, "completed_rows": 2,
                                 "scans": 4, "first_bad": bad})
        partial_snapshot = completed + typed_line[:-3]
        lines3, cursor3, partial3 = prep.split_complete_log_bytes(partial_snapshot, cursor2)
        self.assertEqual(lines3, [])
        self.assertEqual(cursor3, cursor2)
        self.assertTrue(partial3)
        typed_complete = partial_snapshot + typed_line[-3:]
        lines4, cursor4, partial4 = prep.split_complete_log_bytes(typed_complete, cursor3)
        self.assertEqual(cursor4, len(typed_complete))
        self.assertEqual(partial4, b"")
        parsed = prep.classify_complete_lines(lines4)
        self.assertEqual(parsed[0]["classification"], "fatal_nonfinite")
        self.assertEqual(parsed[0]["parsed"]["first_bad"]["source_row"], 51)

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
        bad = {"ordinal": 1, "stage": "input_residual", "format": "f32", "chunk": 0, "pos0": 0,
               "layer": -1, "half": -1, "rows": 1, "width": 4, "index": 0,
               "row": 0, "column": 0, "nan": 1, "positive_inf": 0, "negative_inf": 0,
               "row_offset": 7, "source_row": 7, "expert": -1}
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

    def test_every_header_stage_and_format_has_typed_first_bad_coordinates(self):
        header = prep.SOURCE / "include/strata/prefill/finite_probe.hpp"
        self.assertEqual(prep.verify_stage_contract(header), prep.PROBE_STAGES)
        for stage in prep.PROBE_STAGES:
            for fmt in prep.PROBE_FORMATS:
                bad = {"ordinal": 1, "stage": stage, "format": fmt, "chunk": 2, "pos0": 10,
                       "layer": 45, "half": 0, "rows": 2, "width": 8, "index": 9,
                       "row": 1, "column": 1, "nan": 1, "positive_inf": 0, "negative_inf": 0,
                       "row_offset": 100, "source_row": 101, "expert": 12}
                line = probe_line({"schema": 1, "event": "checkpoint", "run": 1, "device": 0,
                                   "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                   "scans": 1, "first_bad": bad}).decode().rstrip("\n")
                parsed = prep.classify_diagnostic_line(line)
                self.assertEqual(parsed["classification"], "fatal_nonfinite", (stage, fmt))
                for key, value in (("format", "fp16"), ("source_row", 100), ("row_offset", -1),
                                   ("expert", -2), ("index", 16)):
                    corrupt = dict(bad)
                    corrupt[key] = value
                    bad_line = probe_line({"schema": 1, "event": "checkpoint", "run": 1, "device": 0,
                                           "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                           "scans": 1, "first_bad": corrupt}).decode().rstrip("\n")
                    self.assertEqual(prep.classify_diagnostic_line(bad_line)["classification"],
                                     "fatal_invalid", (stage, fmt, key))

    def test_moe_path_is_normal_metadata_and_validates_flags_and_bounds(self):
        record = {"schema": 1, "event": "moe_path", "run": 2, "device": 0, "chunk": 4,
                  "pos0": 100, "layer": 45, "rows": 8, "experts": 45,
                  "native": 1, "mmq": 1, "fused": 1, "gu_type": 1, "down_type": 2,
                  "stream_all": 0, "peer": 1, "peer_sums": 1, "cpu_rows": 3}
        parsed = prep.classify_diagnostic_line(probe_line(record).decode().rstrip("\n"))
        self.assertEqual(parsed["classification"], "normal_moe_path")
        self.assertFalse(prep.diagnostic_is_fatal([parsed]))
        invalid_cases = (
            {**record, "peer": 0},
            {**record, "fused": 2},
            {**record, "cpu_rows": 81},
            {**record, "experts": True},
        )
        for invalid in invalid_cases:
            line = probe_line(invalid).decode().rstrip("\n")
            self.assertEqual(prep.classify_diagnostic_line(line)["classification"], "fatal_invalid")

    def test_python_and_pinned_powershell_moe_and_typed_failure_classification_match(self):
        moe = {"schema": 1, "event": "moe_path", "run": 2, "device": 0, "chunk": 4,
               "pos0": 100, "layer": 0, "rows": 8, "experts": 45,
               "native": 1, "mmq": 1, "fused": 1, "gu_type": 1, "down_type": 2,
               "stream_all": 0, "peer": 1, "peer_sums": 1, "cpu_rows": 3}
        cases = [("valid_orthogonal_moe_flags", probe_line(moe), "normal_moe_path")]
        for name, changes in (
            ("layer_minus_one", {"layer": -1}),
            ("layer_below_minimum", {"layer": -2}),
            ("peer_sums_without_peer", {"peer": 0}),
        ):
            cases.append((name, probe_line({**moe, **changes}), "fatal_invalid"))

        bad = {"ordinal": 1, "stage": "routed_activation_f16", "format": "f16", "chunk": 2, "pos0": 8,
               "layer": 45, "half": 0, "rows": 2, "width": 8, "index": 9,
               "row": 1, "column": 1, "nan": 1, "positive_inf": 0, "negative_inf": 0,
               "row_offset": 50, "source_row": 51, "expert": 4}
        checkpoint = {"schema": 1, "event": "checkpoint", "run": 2, "device": 0,
                      "completed_chunk": 3, "completed_pos0": 9, "completed_rows": 2,
                      "scans": 4}
        cases.append(("valid_typed_coordinates", probe_line({**checkpoint, "first_bad": bad}), "fatal_nonfinite"))
        cases.append(("input_embedding_layer_minus_one",
                      probe_line({**checkpoint, "first_bad": {**bad, "stage": "input_residual", "format": "f32",
                                                                 "layer": -1, "half": -1, "expert": -1}}),
                      "fatal_nonfinite"))
        cases.append(("unknown_format", probe_line({**checkpoint, "first_bad": {**bad, "format": "f64"}}), "fatal_invalid"))
        cases.append(("unknown_stage", probe_line({**checkpoint, "first_bad": {**bad, "stage": "new_stage"}}), "fatal_invalid"))
        cases.append(("typed_row_offset_mismatch",
                      probe_line({**checkpoint, "first_bad": {**bad, "source_row": 52}}), "fatal_invalid"))
        cases.append(("typed_expert_below_sentinel",
                      probe_line({**checkpoint, "first_bad": {**bad, "expert": -2}}), "fatal_invalid"))

        python_results = []
        powershell_cases = []
        for name, raw, expected in cases:
            line = raw.decode("utf-8").rstrip("\n")
            parsed = prep.classify_diagnostic_line(line)
            actual = None if parsed is None else parsed["classification"]
            self.assertEqual(actual, expected, name)
            python_results.append(actual)
            powershell_cases.append({"name": name, "line": line, "expected": expected})

        ps = pinned_powershell(self)
        with tempfile.TemporaryDirectory(prefix="moe71-classifier-parity-") as temp:
            script = Path(temp) / "classifier-parity.ps1"
            payload = json.dumps(powershell_cases, ensure_ascii=False, separators=(",", ":"))
            script.write_text(
                "$ErrorActionPreference='Stop'\n" + prep._powershell_diagnostic_functions() +
                "\n$cases = @'\n" + payload + "\n'@ | ConvertFrom-Json -AsHashtable\n" +
                "$results = foreach($case in $cases){$parsed=Convert-DiagnosticLine $case.line; "
                "[ordered]@{name=$case.name;classification=$parsed.classification;expected=$case.expected}}\n" +
                "$results | ConvertTo-Json -Compress -Depth 10\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script)],
                capture_output=True, text=True, timeout=30, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        ps_results = json.loads(result.stdout)
        if isinstance(ps_results, dict):
            ps_results = [ps_results]
        self.assertEqual([item["name"] for item in ps_results], [case[0] for case in cases])
        self.assertEqual([item["classification"] for item in ps_results], python_results)
        self.assertEqual([item["classification"] for item in ps_results],
                         [item["expected"] for item in powershell_cases])

    def test_quality_scope_is_exact_six_prompt_prefix(self):
        self.assertEqual(prep.require_prompt_prefix(list(prep.MANIFEST_IDS)), prep.PROMPT_IDS)
        self.assertEqual(len(prep.PROMPT_IDS), 6)
        with self.assertRaises(ValueError):
            prep.require_prompt_prefix(list(prep.MANIFEST_IDS[1:]))

    def test_diagnostic_receipt_write_is_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="moe71-exclusive-diagnostics-") as temp:
            path = Path(temp) / "F-long_1k-diagnostics.json"
            body = {"schema_version": 1, "prompt_id": "long_1k", "records": []}
            prep.exclusive_json(path, body)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), body)
            with self.assertRaises(FileExistsError):
                prep.exclusive_json(path, body)

    def test_quality_renderer_keeps_full_manifest_validation_and_six_prompt_loop(self):
        source = prep.RUN66_QUALITY.read_text(encoding="utf-8-sig")
        rendered = prep.rewrite_quality_script(source, r"C:\Run71\arms\moe71\quality-on")
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

    def test_emitted_powershell_classifier_accepts_protocol_shaped_records(self):
        normal = probe_line({"event": "checkpoint", "schema": 1, "run": 1, "device": 0,
                             "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                             "scans": 1, "first_bad": None}).decode().rstrip("\n")
        bad = {"ordinal": 1, "stage": "routed_activation_f16", "format": "f16", "chunk": 2, "pos0": 10,
               "layer": 45, "half": 0, "rows": 2, "width": 8, "index": 9,
               "row": 1, "column": 1, "nan": 1, "positive_inf": 0, "negative_inf": 0,
               "row_offset": 100, "source_row": 101, "expert": 12}
        nonfinite = probe_line({"event": "checkpoint", "schema": 1, "run": 1, "device": 0,
                                "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                "scans": 1, "first_bad": bad}).decode().rstrip("\n")
        moe = probe_line({"event": "moe_path", "schema": 1, "run": 2, "device": 0, "chunk": 4,
                          "pos0": 100, "layer": 45, "rows": 8, "experts": 45,
                          "native": 1, "mmq": 1, "fused": 1, "gu_type": 1, "down_type": 2,
                          "stream_all": 0, "peer": 1, "peer_sums": 1, "cpu_rows": 3}).decode().rstrip("\n")
        bad_format = dict(bad)
        bad_format["format"] = "f64"
        bad_format_line = probe_line({"event": "checkpoint", "schema": 1, "run": 1, "device": 0,
                                      "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                      "scans": 1, "first_bad": bad_format}).decode().rstrip("\n")
        bad_stage = dict(bad)
        bad_stage["stage"] = "unknown_stage"
        bad_stage_line = probe_line({"event": "checkpoint", "schema": 1, "run": 1, "device": 0,
                                     "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                     "scans": 1, "first_bad": bad_stage}).decode().rstrip("\n")
        bad_peer = probe_line({"event": "moe_path", "schema": 1, "run": 2, "device": 0, "chunk": 4,
                               "pos0": 100, "layer": 45, "rows": 8, "experts": 45,
                               "native": 1, "mmq": 1, "fused": 1, "gu_type": 1, "down_type": 2,
                               "stream_all": 0, "peer": 0, "peer_sums": 1, "cpu_rows": 3}).decode().rstrip("\n")
        stage_cases = ""
        for index, stage in enumerate(sorted(prep.PROBE_STAGES)):
            stage_bad = dict(bad)
            stage_bad["stage"] = stage
            stage_line = probe_line({"event": "checkpoint", "schema": 1, "run": 1, "device": 0,
                                     "completed_chunk": 2, "completed_pos0": 12, "completed_rows": 2,
                                     "scans": 1, "first_bad": stage_bad}).decode().rstrip("\n")
            variable = f"$stage{index}"
            stage_cases += (
                f"{variable}=Convert-DiagnosticLine @'\n{stage_line}\n'@\n" +
                f"if({variable}.classification -ne 'fatal_nonfinite'){{throw 'stage not allowlisted: {stage}'}}\n"
            )
        ps = pinned_powershell(self)
        with tempfile.TemporaryDirectory(prefix="moe71-ps-classifier-") as temp:
            script = Path(temp) / "classifier-fixture.ps1"
            script.write_text(
                "$ErrorActionPreference='Stop'\n" + prep._powershell_diagnostic_functions() +
                "\n$normal=Convert-DiagnosticLine @'\n" + normal + "\n'@\n" +
                "$bad=Convert-DiagnosticLine @'\n" + nonfinite + "\n'@\n" +
                "$moe=Convert-DiagnosticLine @'\n" + moe + "\n'@\n" +
                "$badFormat=Convert-DiagnosticLine @'\n" + bad_format_line + "\n'@\n" +
                "$badStage=Convert-DiagnosticLine @'\n" + bad_stage_line + "\n'@\n" +
                "$badPeer=Convert-DiagnosticLine @'\n" + bad_peer + "\n'@\n" +
                "if($normal.classification -ne 'normal_checkpoint'){throw 'normal checkpoint rejected'}\n" +
                "if($bad.classification -ne 'fatal_nonfinite'){throw 'nonfinite checkpoint rejected'}\n" +
                "if($moe.classification -ne 'normal_moe_path'){throw 'MoE metadata incorrectly rejected'}\n" +
                "if($badFormat.classification -ne 'fatal_invalid'){throw 'unknown format accepted'}\n" +
                "if($badStage.classification -ne 'fatal_invalid'){throw 'unknown stage accepted'}\n" +
                "if($badPeer.classification -ne 'fatal_invalid'){throw 'peer_sums without peer accepted'}\n" +
                stage_cases,
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
        rendered = prep.rewrite_quality_script(source, r"C:\Probe71\arms\moe71\quality-on")
        with tempfile.TemporaryDirectory(prefix="moe71-ps-ast-") as temp:
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
        with tempfile.TemporaryDirectory(prefix="moe71-full-preparation-fixture-") as temp:
            root = Path(temp)
            run = root / "run71"
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
            self.assertEqual([side["id"] for side in matrix["sides"]], ["moe71"])
            self.assertFalse(matrix["execution_enabled"])
            self.assertTrue((run / "rendered-scripts/moe71/quality.ps1").is_file())
            config = prep.load_json(run / "arms/moe71/quality-on/config.json")
            self.assertEqual(config["env"]["STRATA_PREFILL_DETAIL_LAYER"], "45")
            self.assertEqual(config["env"]["STRATA_PREFILL_FINITE_PROBE"], "1")
            self.assertEqual(config["env"]["STRATA_DIAG_ZERO_LOGITS"], "1")
            base_config = prep.load_json(run66 / "arms/hetero42/quality-on/config.json")
            normalized = dict(config)
            normalized["exe"] = base_config["exe"]
            normalized["log"] = base_config["log"]
            normalized["env"] = dict(config["env"])
            for key in prep.DIAGNOSTIC_ENV:
                normalized["env"].pop(key)
            self.assertEqual(normalized, base_config)
            self.assertEqual(matrix["quality_recipe"]["prompt_ids"], list(prep.PROMPT_IDS))
            self.assertEqual(matrix["quality_recipe"]["manifest_prompt_ids"], list(prep.MANIFEST_IDS))
            self.assertEqual(list((run / "arms/moe71/quality-on/requests").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
