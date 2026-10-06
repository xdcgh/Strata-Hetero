import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hetero_admission import _probe_wsl, _risky_windows_processes, decode_wsl_list, evaluate, main, parse_nvidia_csv


def fixture(**changes):
    raw = {
        "observed_at_utc": "2026-10-06T00:00:00+00:00",
        "windows_memory": {"total_bytes": 64 * 1024**3, "available_bytes": 32 * 1024**3},
        "nvidia": {"gpus": [{"name": "RTX 4090 D", "vram_total_mib": 49140,
                              "vram_free_mib": 48000, "vram_used_mib": 1140,
                              "utilization_percent": 0, "driver_version": "x"}],
                   "compute_pids": [], "processes": [], "process_telemetry_unknown": False},
        "risky_windows_processes": [],
        "wsl": {"available": True, "running_list_raw": [], "ubuntu_running": []},
    }
    raw.update(changes)
    return raw


class AdmissionTests(unittest.TestCase):
    def test_zero_gpu_utilization_with_persistent_strata_process_is_blocked(self):
        raw = fixture(nvidia={"gpus": [{"name": "RTX 4090 D", "vram_free_mib": 48000,
                                        "utilization_percent": 0}], "processes": [
            {"pid": 77, "name": "strata.exe", "path": "C:\\Strata\\engine\\strata.exe"}],
            "process_telemetry_unknown": False})
        result = evaluate(raw, 46000, 12)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["gates"]["vram"]["pass"])
        self.assertFalse(result["gates"]["gpu_processes"]["pass"])

    def test_active_wsl_comfyui_blocks_even_without_dxg_users(self):
        raw = fixture(wsl={"available": True, "ubuntu_running": [
            {"name": "Ubuntu-24.04", "comfy_active": True, "dxg_exists": True, "dxg_users": []}]})
        result = evaluate(raw, 46000, 12)
        self.assertFalse(result["gates"]["wsl"]["pass"])
        self.assertIn("running Linux distro", result["gates"]["wsl"]["blocked"][0]["reason"])

    def test_missing_gpu_telemetry_fails_closed(self):
        result = evaluate(fixture(nvidia={"telemetry_error": "denied", "gpus": []}), 46000, 12)
        self.assertFalse(result["gates"]["vram"]["pass"])
        self.assertIn("NVIDIA GPU telemetry unknown", result["reasons"])

    def test_unknown_compute_process_identity_fails_closed(self):
        raw = fixture(nvidia={"gpus": [{"name": "GPU", "vram_free_mib": 48000}],
                              "compute_pids": [99], "processes": [{"pid": 99, "name": None, "path": None}],
                              "process_telemetry_unknown": False})
        result = evaluate(raw, 46000, 12)
        self.assertFalse(result["gates"]["gpu_processes"]["pass"])

    def test_known_os_display_process_with_null_path_is_allowed(self):
        raw = fixture(nvidia={"gpus": [{"name": "GPU", "vram_free_mib": 48000}],
                              "compute_pids": [99], "processes": [{"pid": 99, "name": "dwm.exe", "path": None}],
                              "process_telemetry_unknown": False})
        result = evaluate(raw, 46000, 12, check_wsl=False)
        self.assertTrue(result["gates"]["gpu_processes"]["pass"])

    def test_windows_display_path_basename_is_platform_independent(self):
        raw = fixture(nvidia={"gpus": [{"name": "GPU", "vram_free_mib": 48000}],
                              "compute_pids": [99], "processes": [{"pid": 99, "name": "dwm.exe",
                                                                        "path": r"C:\Windows\System32\dwm.exe"}],
                              "process_telemetry_unknown": False})
        result = evaluate(raw, 46000, 12, check_wsl=False)
        self.assertTrue(result["gates"]["gpu_processes"]["pass"])

    def test_process_scan_excludes_only_exact_observer_pid(self):
        self_pid = os.getpid()
        payload = json.dumps([
            {"ProcessId": self_pid, "Name": "python.exe", "ExecutablePath": "C:\\observer.exe"},
            {"ProcessId": self_pid + 1000, "Name": "python.exe", "ExecutablePath": "C:\\test\\python.exe"},
        ])
        runner = Mock(return_value=Mock(returncode=0, stdout=payload, stderr=""))
        with patch("hetero_admission.shutil.which", return_value="powershell.exe"):
            found = _risky_windows_processes(runner)
        self.assertEqual([row["pid"] for row in found], [self_pid + 1000])
        script = runner.call_args.args[0][-1]
        self.assertIn(f"ProcessId -ne {self_pid}", script)
        decision = evaluate(fixture(risky_windows_processes=found), 46000, 12, check_wsl=False)
        self.assertFalse(decision["gates"]["windows_processes"]["pass"])

    def test_wsl_observer_only_enumerates_running_distros_never_executes_inside(self):
        called = []

        def fake_which(_name):
            return "C:\\Windows\\System32\\wsl.exe"

        def fake_runner(args, **kwargs):
            called.append(args)
            return Mock(returncode=0, stdout="Ubuntu-24.04\r\n".encode("utf-16le"), stderr=b"")

        with patch("hetero_admission.shutil.which", side_effect=fake_which):
            result = _probe_wsl(fake_runner)
        self.assertEqual(result["running_list_raw"], ["Ubuntu-24.04"])
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][1:], ["--list", "--running", "--quiet"])
        self.assertFalse(any("-d" in arg or "--distribution" in arg for call in called for arg in call))

    def test_strict_nvidia_csv_and_utf16_wsl_list(self):
        parsed = parse_nvidia_csv("GPU,49140,48000,1140,0,610\n", "12\n")
        self.assertEqual(parsed["gpus"][0]["utilization_percent"], 0.0)
        self.assertEqual(parsed["compute_pids"], [12])
        with self.assertRaises(ValueError):
            parse_nvidia_csv("GPU,49140,unknown,1140,0,610\n", "")
        self.assertEqual(decode_wsl_list("\ufeffUbuntu-22.04\r\n".encode("utf-16le")), ["Ubuntu-22.04"])

    def test_validate_only_does_not_observe_or_write_output(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "admission.json"
            observer = Mock(side_effect=AssertionError("observer must not run"))
            self.assertEqual(main(["--output", str(output), "--validate-only"], observer=observer), 0)
            observer.assert_not_called()
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "admission.json"
            output.write_text("keep", encoding="utf-8")
            observer = Mock(side_effect=AssertionError("output check must precede observe"))
            self.assertEqual(main(["--output", str(output), "--observe"], observer=observer), 1)
            observer.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
