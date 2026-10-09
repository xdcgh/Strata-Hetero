import contextlib
import importlib.util
import io
import json
import io
import os
from pathlib import Path
import unittest
from unittest import mock
import tempfile


RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_ipc_client", RUN / "client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class FakeNoSuchProcess(Exception):
    pass


class FakeProc:
    def __init__(self, pid, ppid, created, exe, argv):
        self.pid, self._ppid, self._created, self._exe, self._argv = pid, ppid, created, exe, argv
        self.alive = True
        self.terminated = False

    def ppid(self): return self._ppid
    def create_time(self): return self._created
    def exe(self): return self._exe
    def cmdline(self): return self._argv
    def children(self, recursive=True): return []
    def terminate(self): self.terminated = True; self.alive = False
    def kill(self): self.alive = False
    def wait(self, timeout=None): return 0


class FakePsutil:
    NoSuchProcess = FakeNoSuchProcess
    TimeoutExpired = TimeoutError

    def __init__(self, by_pid): self.by_pid = by_pid
    def Process(self, pid):
        proc = self.by_pid.get(pid)
        if proc is None or not proc.alive: raise FakeNoSuchProcess(pid)
        return proc


class PlanOnlyTests(unittest.TestCase):
    def test_default_cli_is_plan_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = client.main([])
        plan = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan["status"], "plan_only")
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["kernel_execution"])
        self.assertFalse(plan["core_created"])
        self.assertFalse(plan["child_started"])
        self.assertFalse(plan["payload_read"])

    def test_contract_rows_and_no_runtime_side_effects(self):
        contract = json.loads(client.CONTRACT.read_text(encoding="utf-8"))
        plan = client.make_plan(contract)
        self.assertEqual(plan["rows"], [1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.assertEqual(plan["output_root"], contract["output_root"])
        self.assertEqual(plan["argv"], contract["service_argv_template"])
        self.assertNotEqual(contract["runner_interpreter"], contract["interpreter"]["service_launcher"])
        self.assertEqual(contract["service_argv_template"][0], contract["interpreter"]["service_launcher"])
        self.assertFalse(Path(contract["output_root"]).exists())

    def test_run_without_two_confirmation_flags_stops_before_execute(self):
        with self.assertRaises(SystemExit) as caught:
            client.main(["--run"])
        self.assertEqual(caught.exception.code, 2)

    def test_windows_venv_argv0_with_c_python_executable_is_bound_exactly(self):
        guard = client.OwnedServiceTree.__new__(client.OwnedServiceTree)
        guard.worker = type("W", (), {"pid": 100})()
        guard.wrapper_path = os.path.normcase(os.path.realpath(r"E:\venv\Scripts\python.exe"))
        guard.child_path = os.path.normcase(os.path.realpath(r"C:\Python314\python.exe"))
        script = str(Path(r"C:\repo\tools\service.py"))
        guard.service_script = os.path.normcase(os.path.realpath(script))
        expected = ["-u", "-B", script, "--serve", "--operator", "native-q4k-q5_1"]
        guard.expected_args = expected
        child = {"executable": guard.child_path, "parent_pid": 100,
                 "command_line": [r"E:\venv\Scripts\python.exe", *expected]}
        self.assertTrue(guard._is_actual_child(child))
        self.assertFalse(guard._is_actual_child({**child, "parent_pid": 101}))
        self.assertFalse(guard._is_actual_child({**child, "command_line": [child["command_line"][0], "-u", "-B", script, "--serve", "--foreign"]}))

    def test_cleanup_reaps_bound_child_after_launcher_exit_but_not_reused_pid(self):
        wrapper = os.path.normcase(os.path.realpath(r"E:\venv\Scripts\python.exe"))
        childexe = os.path.normcase(os.path.realpath(r"C:\Python314\python.exe"))
        argv = [r"E:\venv\Scripts\python.exe", r"C:\repo\service.py", "--serve"]
        root_record = {"pid": 100, "parent_pid": 1, "create_time": 1.0,
                       "executable": wrapper, "command_line": [r"E:\venv\Scripts\python.exe", "-m", "shim"]}
        child_record = {"pid": 200, "parent_pid": 100, "create_time": 2.0,
                        "executable": childexe, "command_line": argv}
        worker = type("W", (), {"pid": 100, "returncode": 0})()
        # Parent is already gone; the known child's OS-reaper parent differs, but its full identity still matches.
        child = FakeProc(200, 1, 2.0, childexe, argv)
        guard = client.OwnedServiceTree.__new__(client.OwnedServiceTree)
        guard.worker, guard.psutil = worker, FakePsutil({200: child})
        guard.records, guard.lock = {(100, 1.0): root_record, (200, 2.0): child_record}, __import__("threading").RLock()
        guard.root_record = root_record
        guard.wrapper_path, guard.child_path = wrapper, childexe
        guard.service_script = os.path.normcase(os.path.realpath(argv[1]))
        guard.expected_args = argv[1:]
        result = guard.cleanup_descendants()
        self.assertTrue(child.terminated)
        self.assertTrue(result["terminal"])
        reused = FakeProc(200, 1, 9.0, childexe, argv)
        guard.psutil = FakePsutil({200: reused})
        result = guard.cleanup_descendants()
        self.assertFalse(reused.terminated)
        self.assertTrue(result["terminal"])

    def test_resource_gate_cleans_descendants_before_closing_launcher(self):
        events = []
        tree = type("T", (), {"cleanup_descendants": lambda self: events.append("child_cleanup") or {"terminal": True}})()
        worker = type("W", (), {"pid": 100, "returncode": 0,
                                 "close": lambda self: events.append("launcher_close")})()
        monitor = client.ResourceMonitor.__new__(client.ResourceMonitor)
        monitor.stream = io.StringIO()
        monitor.failed = {"gate_ok": False}
        monitor.last = None
        monitor.startup_error = None
        monitor.lock = __import__("threading").RLock()
        monitor._stop_owned_after_gate({}, worker, tree)
        self.assertEqual(events, ["child_cleanup", "launcher_close"])
        saved = json.loads(monitor.stream.getvalue())
        self.assertEqual(saved["cleanup"]["owned_descendant_gate_cleanup"], {"terminal": True})
        with self.assertRaises(client.BenchError):
            monitor.require()

    def test_resource_monitor_waits_for_first_written_sample(self):
        good = ({"physical_available_bytes": 20 * 1024**3,
                 "commit_available_bytes": 10 * 1024**3, "source": "fixture"}, [])
        with tempfile.TemporaryDirectory() as td, mock.patch("tools.hetero_resources.memory_snapshot", return_value=good):
            monitor = client.ResourceMonitor(Path(td) / "samples.jsonl", object(),
                                             minimum_physical=12 * 1024**3, minimum_commit=4 * 1024**3)
            monitor.start(timeout=2)
            self.assertTrue(monitor.require()["gate_ok"])
            monitor.close()
            self.assertEqual(len((Path(td) / "samples.jsonl").read_text().splitlines()), 1)

    def test_init_ready_requires_current_rows_exactly(self):
        contract = json.loads(client.CONTRACT.read_text(encoding="utf-8"))
        identity = {"weights_sha256": contract["identity_weights_sha256"],
                    "source": {"raw_selected_payload_sha256": contract["native_blob_binding"]["raw_selected_payload_sha256"]}}
        ready = {"schema": "strata-xpu-ready-v1", "status": "ready", "nonce": "a" * 32,
                 "identity_sha256": contract["identity_sha256"], "rows": 8, "operator": contract["operator"],
                 "operator_signature": contract["operator_signature"], "weights_ready": True,
                 "weights_loaded_and_hash_verified": True, "weights_sha256": identity["weights_sha256"],
                 "native_blob_binding": {"blob_sha256": contract["blob_sha256"],
                    "segment_sha256_actual": identity["source"]["raw_selected_payload_sha256"],
                    "down_offset_bytes": contract["native_blob_binding"]["down_offset_bytes"],
                    "q5_1_minimums_sha256_le_f32": contract["native_blob_binding"]["minimums_sha256_le_f32"]},
                 "device": "GPU.0", "execution_devices": ["GPU.0"], "device_full_name": "Intel Arc fixture",
                 "precision_requested": "f32", "reported_precision": "float32",
                 "precision_policy": {"precision_policy_matches": True},
                 "compile_property_policy": {"submitted": {"INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}},
                 "native_operator_build": {"operator": contract["operator_signature"], "compiled": False, "inference_run": False}}
        client.validate_ready(ready, contract, identity, "a" * 32, 8)
        with self.assertRaises(client.BenchError):
            client.validate_ready({**ready, "rows": 4}, contract, identity, "a" * 32, 8)


if __name__ == "__main__":
    unittest.main()
