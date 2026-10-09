import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUNNER = Path(__file__).with_name("run_pool_parity.py")


class RunnerFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = runpy.run_path(str(RUNNER), run_name="pool_parity_fixture")

    def test_default_plan_does_not_spawn(self):
        stdout = io.StringIO()
        with patch.object(self.ns["sys"], "argv", [str(RUNNER)]), \
             patch.object(self.ns["subprocess"], "Popen", side_effect=AssertionError("spawned")), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(self.ns["main"](), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(len(plan["cases"]), 12)
        self.assertFalse(plan["kernel_execution"])
        self.assertFalse(plan["payload_read"])

    def test_fake_bad_output_is_not_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.f32", Path(tmp) / "b.f32"
            a.write_bytes(b"\x00\x00\x80\x3f")
            b.write_bytes(b"\x00\x00\x00\x40")
            self.assertFalse(self.ns["equal_bytes"](a, b))

    def test_spawn_failure_keeps_original_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); exe = root / "fake.exe"; exe.write_bytes(b"fixture")
            folder = root / "case"
            def allowed_sample(stream, label, enforce=True):
                self.assertTrue(enforce)
                return {"gate_ok": True, "label": label}
            with patch.dict(self.ns, {"sample": allowed_sample}), \
                 patch.object(self.ns["subprocess"], "Popen", side_effect=RuntimeError("fixture spawn failure")), \
                 patch.object(self.ns["subprocess"], "CREATE_NO_WINDOW", 0, create=True):
                with self.assertRaisesRegex(RuntimeError, "fixture spawn failure"):
                    self.ns["child"](exe, [], folder, io.StringIO(), "fixture", self.ns["file_stat"](exe))
            failure = json.loads((folder / "failure.json").read_text())
            self.assertIn("fixture spawn failure", failure["original_exception"])
            self.assertTrue((folder / "attempt.json").exists())
            self.assertTrue((folder / "stage.json").exists())


if __name__ == "__main__":
    unittest.main()
