from __future__ import annotations

import datetime as dt
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from hetero_admission import WSL_GPU_PROBE, _probe_wsl, evaluate, verified_wsl_idle


def admission_fixture(wsl: dict) -> dict:
    return {
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "windows_memory": {"available_bytes": 32 * 1024**3,
                           "available_commit_bytes": 32 * 1024**3},
        "nvidia": {
            "gpus": [{"name": "fixture GPU", "vram_total_mib": 49140,
                      "vram_free_mib": 48000, "vram_used_mib": 1140,
                      "utilization_percent": 0, "driver_version": "fixture"}],
            "compute_pids": [], "processes": [], "process_telemetry_unknown": False,
        },
        "risky_windows_processes": [],
        "wsl": wsl,
    }


def idle_record(*, inspected_at: dt.datetime | None = None, **proof_changes) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    proof = {
        "schema": "strata-wsl-gpu-idle-v1", "uid": 0,
        "clients": [], "errors": [], "scanned_pids": 3, "scanned_fds": 12,
        "complete": True, "boot_id": "01234567-89ab-cdef-0123-456789abcdef",
        "device": {"path": "/dev/dxg", "rdev": 12345, "character": True},
        "comfy": {"active": "inactive", "main_pid": 0, "load": "loaded", "exit_code": 0},
    }
    proof.update(proof_changes)
    stamp = inspected_at or now
    return {
        "name": "Ubuntu-24.04", "gpu_inspection": proof,
        "inspection_script_sha256": hashlib.sha256(WSL_GPU_PROBE.encode()).hexdigest(),
        "inspected_at_utc": stamp.isoformat(),
    }


def listed(names: list[str]) -> Mock:
    raw = ("\r\n".join(names) + ("\r\n" if names else "")).encode("utf-16le")
    return Mock(returncode=0, stdout=raw, stderr=b"")


class VerifiedIdleProofTests(unittest.TestCase):
    def decision(self, distro: dict, *, inspect: bool) -> dict:
        raw = admission_fixture({"available": True, "running_distros": [distro]})
        return evaluate(raw, 46000, 12, check_wsl=True, inspect_running_wsl=inspect)

    def test_strict_default_blocks_even_with_valid_idle_proof(self):
        distro = idle_record()
        self.assertTrue(verified_wsl_idle(distro))
        result = self.decision(distro, inspect=False)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["gates"]["wsl"]["pass"])

    def test_explicit_opt_in_accepts_complete_fresh_root_idle_inspection(self):
        result = self.decision(idle_record(), inspect=True)
        self.assertEqual(result["status"], "pass", result["reasons"])
        self.assertTrue(result["gates"]["wsl"]["pass"])

    def test_any_gpu_fd_client_blocks_even_for_idle_or_unrecognized_command(self):
        for client in ({"pid": 12, "fd": "5", "command": "idle-looking-service"},
                       {"pid": 13, "fd": "8", "command": None}):
            with self.subTest(client=client):
                distro = idle_record(clients=[client])
                self.assertFalse(verified_wsl_idle(distro))
                self.assertEqual(self.decision(distro, inspect=True)["status"], "blocked")

    def test_incomplete_or_unsafe_proof_fields_fail_closed(self):
        cases = {
            "missing_uid": {"uid": None},
            "nonroot_uid": {"uid": 1000},
            "missing_key": {"scanned_fds": None},
            "scan_error": {"errors": ["pid 17 fd list: denied"]},
            "not_complete": {"complete": False},
            "device_not_character": {"device": {"path": "/dev/dxg", "rdev": 12345, "character": False}},
            "device_wrong_path": {"device": {"path": "/dev/other", "rdev": 12345, "character": True}},
            "device_zero_rdev": {"device": {"path": "/dev/dxg", "rdev": 0, "character": True}},
            "comfy_active": {"comfy": {"active": "active", "main_pid": 42, "load": "loaded", "exit_code": 0}},
            "comfy_nonzero_main_pid": {"comfy": {"active": "inactive", "main_pid": 1, "load": "loaded", "exit_code": 0}},
            "pids_zero": {"scanned_pids": 0},
            "fds_zero": {"scanned_fds": 0},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                proof = idle_record(**changes)
                self.assertFalse(verified_wsl_idle(proof))
                self.assertEqual(self.decision(proof, inspect=True)["status"], "blocked")

    def test_expired_future_and_invalid_script_hash_proofs_fail_closed(self):
        now = dt.datetime.now(dt.timezone.utc)
        cases = (
            ("expired", idle_record(inspected_at=now - dt.timedelta(seconds=11))),
            ("future", idle_record(inspected_at=now + dt.timedelta(seconds=1))),
            ("script_sha", {**idle_record(), "inspection_script_sha256": "0" * 64}),
        )
        for name, distro in cases:
            with self.subTest(name=name):
                self.assertFalse(verified_wsl_idle(distro))
                self.assertEqual(self.decision(distro, inspect=True)["status"], "blocked")


class WslProbeSequenceTests(unittest.TestCase):
    def setUp(self):
        self.which = patch("hetero_admission.shutil.which", return_value=r"C:\Windows\System32\wsl.exe")
        self.which.start()
        self.addCleanup(self.which.stop)

    def test_opt_in_orders_enum_recheck_python_probe_and_final_enum(self):
        expected_proof = idle_record()["gpu_inspection"]
        calls = []
        replies = [listed(["Ubuntu-24.04"]), listed(["Ubuntu-24.04"]),
                   Mock(returncode=0, stdout=json.dumps(expected_proof), stderr=""),
                   listed(["Ubuntu-24.04"])]

        def runner(args, **kwargs):
            calls.append((args, kwargs))
            return replies.pop(0)

        result = _probe_wsl(runner, inspect_running=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual([call[0][1:4] for call in calls if call[0][1:2] == ["--list"]],
                         [["--list", "--running", "--quiet"]] * 3)
        self.assertEqual(calls[2][0][1:10], ["--distribution", "Ubuntu-24.04", "--user", "root", "--exec",
                                                 "python3", "-I", "-S", "-B"])
        self.assertIn("-c", calls[2][0])
        self.assertTrue(verified_wsl_idle(result["running_distros"][0]))
        self.assertEqual(result["running_distros"][0]["reason"], None)

    def test_distro_not_running_at_recheck_is_never_probed(self):
        calls = []
        replies = [listed(["Ubuntu-24.04"]), listed([]), listed(["Ubuntu-24.04"])]

        def runner(args, **kwargs):
            calls.append(args)
            return replies.pop(0)

        result = _probe_wsl(runner, inspect_running=True)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("--distribution" not in call for call in calls))
        self.assertIn("state changed", result["running_distros"][0]["error"])

    def test_running_set_change_after_probe_fails_closed(self):
        proof = idle_record()["gpu_inspection"]
        replies = [listed(["Ubuntu-24.04"]), listed(["Ubuntu-24.04"]),
                   Mock(returncode=0, stdout=json.dumps(proof), stderr=""),
                   listed(["Ubuntu-24.04", "Debian"])]

        def runner(_args, **_kwargs):
            return replies.pop(0)

        result = _probe_wsl(runner, inspect_running=True)
        self.assertIn("set changed", result["error"])
        raw = admission_fixture({"available": True, **result})
        self.assertEqual(evaluate(raw, 46000, 12, inspect_running_wsl=True)["status"], "blocked")

    def test_probe_exception_is_retained_and_blocks(self):
        replies = [listed(["Ubuntu-24.04"]), listed(["Ubuntu-24.04"]),
                   OSError("mock probe failure"), listed(["Ubuntu-24.04"])]

        def runner(_args, **_kwargs):
            value = replies.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        result = _probe_wsl(runner, inspect_running=True)
        distro = result["running_distros"][0]
        self.assertIn("RuntimeError", distro["error"])
        self.assertFalse(verified_wsl_idle(distro))
        raw = admission_fixture({"available": True, **result})
        self.assertEqual(evaluate(raw, 46000, 12, inspect_running_wsl=True)["status"], "blocked")

    def test_proof_generation_exception_is_captured_by_observer_as_blocking(self):
        replies = [listed(["Ubuntu-24.04"]), listed(["Ubuntu-24.04"]),
                   Mock(returncode=1, stdout="", stderr="failed"), listed(["Ubuntu-24.04"])]

        def runner(_args, **_kwargs):
            return replies.pop(0)

        result = _probe_wsl(runner, inspect_running=True)
        raw = admission_fixture({"available": True, **result})
        self.assertEqual(evaluate(raw, 46000, 12, inspect_running_wsl=True)["status"], "blocked")
        self.assertTrue(result["running_distros"][0].get("error"))


if __name__ == "__main__":
    unittest.main()
