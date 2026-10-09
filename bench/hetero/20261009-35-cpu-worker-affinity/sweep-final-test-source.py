import contextlib
import importlib.util
import io
import json
import math
import subprocess
from pathlib import Path
import tempfile
import struct
import unittest
from unittest import mock

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cpu_affinity_sweep", RUN / "run_cpu_affinity_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
CONTRACT = json.loads((RUN / "sweep-execution-contract-v2.json").read_text(encoding="utf-8"))


def disabled_contract(td):
    root = Path(td) / "unused-data-root"
    disabled = {**CONTRACT, "input_preparation_authorized": False, "execution_authorized": False,
                "data_root": str(root), "inputs_root": str(root / "inputs"),
                "preparation_receipt": str(root / "inputs-preparation.json"),
                "output_root": str(root / "outputs")}
    path = Path(td) / "disabled-contract.json"
    path.write_text(json.dumps(disabled), encoding="utf-8")
    return path, disabled, root


class FakeOwnedPopen:
    pid = 12345
    returncode = None
    def __init__(self): self.kills = 0; self.live = True
    def communicate(self, timeout=None):
        if self.live:
            raise subprocess.TimeoutExpired(["owned.exe"], timeout, output=b"partial-out", stderr=b"partial-err")
        return b"partial-out", b"partial-err"
    def poll(self): return None if self.live else self.returncode
    def kill(self): self.kills += 1; self.live = False; self.returncode = -9


class SweepPlanTests(unittest.TestCase):
    def test_import_root_resolves_to_strata_hetero_repo(self):
        self.assertEqual(sweep.ROOT, Path(__file__).resolve().parents[3])
        self.assertEqual(sweep.ROOT.name, "Strata-Hetero")

    def test_disk_reserve_comes_from_nested_resource_gate(self):
        self.assertEqual(sweep.minimum_free_disk_bytes(CONTRACT), CONTRACT["resource_gate"]["minimum_free_disk_bytes"])

    def test_plan_contains_two_gates_126_runnable_and_18_strict_exclusions(self):
        plan = sweep.make_plan(CONTRACT)
        cases = plan["cases"]
        self.assertEqual(plan["default_reference_gates"], 2)
        self.assertEqual(plan["planned_pool_cases"], 126)
        self.assertEqual(plan["strict_pcore_overflow_not_run"], 18)
        self.assertEqual(plan["all_auto_all_worker16_overflow_cases"], 12)
        self.assertEqual(len(cases), 146)
        excluded = [x for x in cases if x["status"] == "not_run_strict_candidate_overflow"]
        self.assertTrue(all(x["affinity"] == "p-cores" and x["workers"] > 5 for x in excluded))
        self.assertTrue(all(x["worker_pin_success_evidence"] is False for x in cases if x["phase"] == "pool_matrix"))
        self.assertFalse(plan["affinity_mutation"])
        self.assertFalse(plan["kernel_execution"])

    def test_default_cli_is_plan_only_and_does_not_create_e_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            path, disabled, root = disabled_contract(td)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = sweep.main(["--contract", str(path)])
            plan = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(plan["status"], "plan_only")
            self.assertFalse(plan["execution_authorized"])
            self.assertFalse(plan["payload_read"])
            self.assertFalse(Path(disabled["output_root"]).exists())
            self.assertFalse(root.exists())

    def test_prepare_and_run_reject_without_contract_authorization(self):
        with tempfile.TemporaryDirectory() as td:
            disabled_path, _, root = disabled_contract(td)
            for argv in (("--prepare", "--root-prepare-confirmed"),
                         ("--run", "--root-start-confirmed", "--models-terminal-confirmed")):
                with mock.patch.object(sweep, "prepare_inputs", side_effect=AssertionError("prepare called")), \
                     mock.patch.object(sweep, "run_sweep", side_effect=AssertionError("run called")), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    sweep.main(["--contract", str(disabled_path), *argv])
                self.assertEqual(caught.exception.code, 2)
            self.assertFalse(root.exists())

    def test_seeded_missing_inputs_are_deterministic_and_finite(self):
        import numpy as np
        for rows in (6, 10):
            a, recipe = sweep.generate_input_bytes(rows, np)
            b, _ = sweep.generate_input_bytes(rows, np)
            self.assertEqual(a, b)
            self.assertEqual(len(a), rows * 2560 * 4)
            self.assertEqual(recipe["seed"], 42 + rows)
            self.assertTrue(sweep.finite_f32le(a, rows * 2560))
        self.assertFalse(sweep.finite_f32le(struct.pack("<ff", 1.0, math.nan), 2))

    def test_pool_receipt_requires_exact_workers_affinity_and_reference_controls(self):
        case = {"phase": "pool_matrix", "rows": 4, "workers": 4, "affinity": "all"}
        c = {"binary_source_sha256": "s", "blob_sha256": "b",
             "reference_controls_expected": {"STRATA_KQ256": "0", "q8k_avx2_selected": True}}
        native = {"status": "success", "source": {"native_harness_source_sha256": "s"},
                  "blob": {"sha256": "b"}, "input": {"sha256": "i"},
                  "output": {"sha256": "o", "bytes": 40960},
                  "parameters": {"rows": 4, "hidden": 2560, "intermediate": 640, "gu_type": 12, "down_type": 7,
                                 "warmup": 1, "repeat": 5},
                  "execution": {"native_kernel_called": True, "gpu_executed": False,
                                "model_cli_executed": False, "engine_pool_called": True},
                  "reference_controls": {"STRATA_KQ256": "0", "q8k_avx2_selected": True},
                  "pool": {"requested_background_workers": 4, "actual_background_workers": 4,
                           "effective_compute_participants": 5, "host_works": True, "affinity": "all",
                           "pin": True, "host_pin_applied": True, "worker_pin_success_available": False}}
        self.assertTrue(sweep.validate_native_receipt(native, case, c, "i", "o", 40960))
        self.assertFalse(sweep.validate_native_receipt({**native, "pool": {**native["pool"], "actual_background_workers": 3}}, case, c, "i", "o", 40960))
        self.assertFalse(sweep.validate_native_receipt({**native, "pool": {**native["pool"], "worker_pin_success_available": True}}, case, c, "i", "o", 40960))

    def test_bounded_timeout_kills_only_owned_popen_and_preserves_captured_logs(self):
        proc = FakeOwnedPopen()
        watched = sweep.wait_owned_child(proc, None, {}, lambda: {"gate_ok": True},
                                         deadline_seconds=0, poll_seconds=0.001)
        self.assertTrue(watched["timed_out"])
        self.assertEqual(proc.kills, 1)
        self.assertEqual(watched["stdout"], b"partial-out")
        self.assertEqual(watched["stderr"], b"partial-err")


if __name__ == "__main__":
    unittest.main()
