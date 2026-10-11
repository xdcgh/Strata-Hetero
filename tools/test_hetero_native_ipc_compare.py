"""Pure protocol/READY fixtures for hetero_native_ipc_compare; no child or runtime is started."""
from __future__ import annotations

from dataclasses import dataclass
import json
import struct
import threading
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tools import hetero_native_ipc_compare as compare


@dataclass
class Frame:
    kind: int
    request_id: int
    payload: bytes


class FakeProtocol:
    INIT_KIND = 1
    INFER_KIND = 2


class FakeMonitor:
    def __init__(self, fail_at=None): self.require_calls = 0; self.fail_at = fail_at
    def require(self):
        self.require_calls += 1
        if self.require_calls == self.fail_at: raise compare.CompareError("synthetic PSAPI gate floor")


class FakeInput:
    def __init__(self, raw): self.raw = raw
    def tobytes(self, order="C"): return self.raw


class FakeOwnedArray:
    flags = type("Flags", (), {"c_contiguous": True, "owndata": True})()
    def reshape(self, rows, width): return self
    def copy(self, order="C"): return self


class FakeFinite:
    def all(self): return True


class FakeNumpy:
    @staticmethod
    def asarray(value, dtype=None, order=None): return value
    @staticmethod
    def frombuffer(raw, dtype=None): return FakeOwnedArray()
    @staticmethod
    def isfinite(value): return FakeFinite()


class FakeWorker:
    def __init__(self, protocol, ready_json, output_bytes):
        self.protocol = protocol; self.ready_json = ready_json; self.output_bytes = output_bytes
        self.kinds = []; self.init_bytes = None
    def request(self, kind, request_id, payload, timeout):
        self.kinds.append(kind)
        if kind == self.protocol.INIT_KIND:
            self.init_bytes = payload
            body = json.dumps(self.ready_json, separators=(",", ":")).encode("utf-8")
        elif kind == self.protocol.INFER_KIND:
            body = self.output_bytes
        else:
            raise AssertionError(f"unexpected request kind: {kind}")
        return Frame(kind | 0x8000, request_id, struct.pack("<IQ", 0, 1234) + body)


class TimedFakeWorker:
    def __init__(self, name, delay, body, fail=False):
        self.name = name; self.delay = delay; self.body = body; self.fail = fail; self.finished = threading.Event()
    def request(self, kind, request_id, payload, timeout):
        time.sleep(self.delay)
        self.finished.set()
        if self.fail: raise RuntimeError("synthetic transfer failure")
        return Frame(kind | 0x8000, request_id, struct.pack("<IQ", 0, 4321) + self.body)


class FakeQuality:
    DEFAULT_TOLERANCES = {"max_abs": 0.01, "relative_rmse": 0.001, "row_norm_relative": 0.001}
    def __init__(self, finished): self.finished = finished; self.calls = []
    def quality_report(self, actual, reference, tolerances):
        if not all(worker.finished.is_set() for worker in self.finished):
            raise AssertionError("quality ran before both paired transfers ended")
        self.calls.append("quality")
        return {"shape_ok": True, "finite": True, "pass": True, "tolerances": tolerances}


class CompareProtocolFixtures(unittest.TestCase):
    def test_contract_loader_accepts_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "contract.json"
            expected = {"run_id": "fixture-run-60", "backends": []}
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(compare.load_contract(path), expected)

    def test_authorization_uses_selected_contract_and_source_freeze_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            contract_path = directory / "compare-contract.json"
            freeze_path = directory / "source-freeze.json"
            auth_path = directory / "root-authorization.json"
            contract_path.write_text('{"fixture":true}', encoding="utf-8")
            freeze_path.write_text('{"fixture_freeze":true}', encoding="utf-8")
            contract = {"run_id": "fixture-run-60", "source_freeze_path": str(freeze_path)}
            auth = {"schema": "strata-native-ipc-compare-root-authorization-v1",
                "run_id": contract["run_id"], "execution_authorized": True,
                "model_matrix_stopped": True, "root_review_complete": True,
                "client_sha256": compare._stable_sha(Path(compare.__file__)),
                "contract_sha256": compare._stable_sha(contract_path),
                "source_freeze_sha256": compare._stable_sha(freeze_path),
                "cpu16_none_authorized": True}
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            args = SimpleNamespace(execute=True, root_start_confirmed=True,
                model_matrix_stopped_confirmed=True, root_authorization=auth_path,
                include_cpu16_none=True)
            result = compare._authorization(args, contract, contract_path)
            self.assertEqual(result["authorization"], auth)
            self.assertEqual(result["path"], str(auth_path))

    def test_cpu_binary_verification_binds_explicit_run59_and_legacy_run52(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            binary = directory / "service.exe"
            binary.write_bytes(b"synthetic fixture bytes; never executed")
            pins = {"cpu_service_source_sha256": "2" * 64,
                "cpu_kernel_source_sha256": "1" * 64, "cpu_harness_cmake_sha256": "3" * 64,
                "cpu_pool_cpp_sha256": "4" * 64, "cpu_pool_hpp_sha256": "5" * 64,
                "cpu_service_fixture_sha256": "6" * 64}
            source_map = {"src/hetero/native_cpu_service.cpp": pins["cpu_service_source_sha256"],
                "src/hetero/native_expert_bench.cpp": pins["cpu_kernel_source_sha256"],
                "tools/hetero_native_cpu/CMakeLists.txt": pins["cpu_harness_cmake_sha256"],
                "src/kernels/cpu/pool.cpp": pins["cpu_pool_cpp_sha256"],
                "include/strata/kernels/cpu/pool.hpp": pins["cpu_pool_hpp_sha256"],
                "src/hetero/native_cpu_service_fixture.cpp": pins["cpu_service_fixture_sha256"]}
            for run_id, explicit in (("native-cpu-service-20261011-59", True),
                                     ("native-cpu-service-20261011-52", False)):
                receipt_path = directory / (run_id + ".json")
                receipt = {"status": "success", "run_id": run_id,
                    "binaries": {"hetero_native_cpu_service": {"sha256": compare.sha(binary.read_bytes()),
                        "path": str(binary)}}, "source_sha256_before": source_map}
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                backend = {"exe": str(binary), "exe_sha256": compare.sha(binary.read_bytes()),
                    "build_receipt_path": str(receipt_path),
                    "build_receipt_sha256": compare._stable_sha(receipt_path)}
                if explicit:
                    backend["build_run_id"] = run_id
                self.assertEqual(compare._verify_cpu_binary({"source_pins": pins}, backend)["sha256"],
                                 backend["exe_sha256"])

    def test_init_schema_and_rows(self):
        raw = compare.init_payload("0123456789abcdef0123456789abcdef", 256, "a" * 64)
        self.assertEqual(json.loads(raw), {
            "schema": "strata-xpu-init-v1", "nonce": "0123456789abcdef0123456789abcdef",
            "rows": 256, "identity_sha256": "a" * 64,
        })
        with self.assertRaises(compare.CompareError):
            compare.init_payload("0123456789abcdef0123456789abcdef", True, "a" * 64)

    def test_result_prefix_is_little_endian_iq_and_body_is_owned_bytes(self):
        prefix = struct.pack("<IQ", 0, 0x0102030405060708)
        body, worker_ns = compare.parse_result_payload(2, 0x0102030405060708,
            Frame(0x8002, 0x0102030405060708, prefix + b"F32LE"))
        self.assertEqual(worker_ns, 0x0102030405060708)
        self.assertEqual(body, b"F32LE")
        self.assertIs(type(body), bytes)

    def test_result_rejects_frame_identity_and_surfaces_bounded_json_message(self):
        prefix = struct.pack("<IQ", 0, 5)
        with self.assertRaises(compare.CompareError):
            compare.parse_result_payload(2, 9, Frame(0x8002, 8, prefix))
        error = struct.pack("<IQ", 1, 7) + b'{"message":"shape mismatch"}'
        with self.assertRaisesRegex(compare.CompareError, "shape mismatch"):
            compare.parse_result_payload(2, 9, Frame(0x8002, 9, error))

    def test_cpu_ready_binds_pool_and_all_compiled_sources(self):
        contract = {"arc": {"identity_sha256": "a" * 64, "blob_sha256": "b" * 64},
                    "source_pins": {"cpu_kernel_source_sha256": "1" * 64,
                        "cpu_service_source_sha256": "2" * 64, "cpu_harness_cmake_sha256": "3" * 64,
                        "cpu_pool_cpp_sha256": "4" * 64, "cpu_pool_hpp_sha256": "5" * 64}}
        backend = {"pool_workers": 2, "pool_affinity": "all"}
        ready = {"schema": "strata-xpu-ready-v1", "status": "ready", "nonce": "0" * 32,
            "identity_sha256": "a" * 64, "rows": 4, "hidden": 2560, "intermediate": 640,
            "operator": "native-q4k-q5_1", "operator_signature": "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
            "native_blob_loaded_and_hash_verified": True, "blob_bytes": 3072000, "blob_sha256": "b" * 64,
            "identity_bytes_sha256": "a" * 64, "identity_bytes": 512,
            "native_source_sha256": "1" * 64, "service_source_sha256": "2" * 64,
            "service_cmake_sha256": "3" * 64, "pool_cpp_sha256": "4" * 64, "pool_hpp_sha256": "5" * 64,
            "pool": {"requested_workers": 2, "actual_workers": 2, "affinity": "all",
                     "host_pinned": True,
                     "worker_affinity_readback": [
                         {"worker": i, "requested_core": i + 1, "ready": True, "pin_requested": True,
                          "pin_applied": True, "mask_observed": True, "mask_matches_request": True,
                          "observed_group": 0, "observed_mask": 1 << (i + 1), "startup_processor": i + 1}
                         for i in range(2)]}}
        compare.validate_cpu_ready(ready, contract, backend, "0" * 32, 4)
        ready["pool"]["actual_workers"] = 1
        with self.assertRaises(compare.CompareError):
            compare.validate_cpu_ready(ready, contract, backend, "0" * 32, 4)

    def test_real_init_and_trial_loops_use_service_protocol_constants(self):
        protocol = FakeProtocol()
        identity_sha = "a" * 64
        source_pins = {"cpu_kernel_source_sha256": "1" * 64, "cpu_service_source_sha256": "2" * 64,
            "cpu_harness_cmake_sha256": "3" * 64, "cpu_pool_cpp_sha256": "4" * 64,
            "cpu_pool_hpp_sha256": "5" * 64}
        contract = {"arc": {"identity_sha256": identity_sha, "blob_sha256": "b" * 64},
                    "source_pins": source_pins, "request_timeout_seconds": 120,
                    "output": {"private_output_root": "fixture-output"}}
        backend = {"id": "cpu-test", "pool_workers": 1, "pool_affinity": "none"}
        ready = {"schema": "strata-xpu-ready-v1", "status": "ready", "nonce": "0" * 32,
            "identity_sha256": identity_sha, "rows": 1, "hidden": 2560, "intermediate": 640,
            "operator": "native-q4k-q5_1", "operator_signature": "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
            "native_blob_loaded_and_hash_verified": True, "blob_bytes": 3072000, "blob_sha256": "b" * 64,
            "identity_bytes_sha256": identity_sha, "identity_bytes": 512,
            "native_source_sha256": "1" * 64, "service_source_sha256": "2" * 64,
            "service_cmake_sha256": "3" * 64, "pool_cpp_sha256": "4" * 64, "pool_hpp_sha256": "5" * 64,
            "pool": {"requested_workers": 1, "actual_workers": 1, "affinity": "none", "host_pinned": False,
                "worker_affinity_readback": [{"worker": 0, "requested_core": -1, "ready": True,
                    "pin_requested": False, "pin_applied": False, "mask_observed": True,
                    "mask_matches_request": False, "observed_group": 0, "observed_mask": 1,
                    "startup_processor": 0}]}}
        raw = bytes(2560 * 4)
        worker = FakeWorker(protocol, ready, raw)
        handle = {"worker": worker, "request_id": 0, "launch_perf_counter_ns": 1}
        monitor = FakeMonitor(); init_rows = []
        compare._init(handle, 1, contract, {}, None, backend, "0" * 32, monitor,
                      protocol, "cpu", None, init_rows)
        input_record = {"array": FakeInput(raw), "bytes": raw, "sha256": compare.sha(raw)}
        gold_record = {"bytes": raw, "array": FakeOwnedArray(), "sha256": compare.sha(raw)}
        fake_writer = lambda path, body: {"path": str(path), "bytes": len(body),
                                          "sha256": compare.sha(body), "exclusive": True}
        result = compare._trial(handle, "cpu", backend, "cpu-solo", 1, input_record, gold_record,
            contract, monitor, protocol, FakeNumpy(), None, "formal-01", output_writer=fake_writer)
        self.assertEqual(worker.kinds, [protocol.INIT_KIND, protocol.INFER_KIND])
        self.assertEqual(json.loads(worker.init_bytes)["schema"], "strata-xpu-init-v1")
        self.assertTrue(result["comparison_pass"])
        self.assertEqual(result["worker_wall_ns"], 1234)
        self.assertGreaterEqual(monitor.require_calls, 4)

    def test_pin_readback_requires_complete_typed_fields(self):
        ready = self._ready_for_one_pinned_worker()
        contract = self._contract_for_ready(ready)
        backend = {"pool_workers": 1, "pool_affinity": "all"}
        compare.validate_cpu_ready(ready, contract, backend, "0" * 32, 1)
        ready["pool"]["worker_affinity_readback"][0].pop("mask_observed")
        with self.assertRaises(compare.CompareError):
            compare.validate_cpu_ready(ready, contract, backend, "0" * 32, 1)

    def test_arc_warmup_quality_failure_is_a_run53_gate(self):
        failing = {"rows": 1, "timed_input_hash_match": True, "output_shape_ok": True,
            "finite": True, "owned_output_copy": True, "output_storage_complete": True,
            "resource_gate_pass_after_transfer": True, "paired_transfer_complete": True,
            "comparison_pass": False, "numerical_quality_vs_native_cpu": {"pass": False}}
        with self.assertRaises(compare.CompareError):
            compare._require_trial_gate(failing, "arc", "warmup")

    def test_paired_end_barrier_precedes_output_save_and_quality(self):
        raw = bytes(2560 * 4); done = [TimedFakeWorker("cpu", 0.01, raw), TimedFakeWorker("arc", 0.05, raw)]
        worker_api = FakeQuality(done); callback_events = []; monitor = FakeMonitor()
        barrier = threading.Barrier(2)

        def after_transfer():
            if not all(worker.finished.is_set() for worker in done):
                raise AssertionError("post-transfer hook ran before both transfers ended")
            callback_events.append("after-transfer")

        def writer(path, body):
            if not all(worker.finished.is_set() for worker in done):
                raise AssertionError("raw output save ran before both transfers ended")
            callback_events.append("save")
            return {"path": str(path), "bytes": len(body), "sha256": compare.sha(body), "exclusive": True}

        def invoke(worker):
            return compare._timed_infer(worker, 2, 1, 1, FakeInput(raw), raw, compare.sha(raw), raw,
                FakeOwnedArray(), worker.name, "paired-fixture", monitor, FakeNumpy(), worker_api,
                barrier=barrier, output_base=Path("fixture-output") / worker.name,
                output_writer=writer, post_transfer_callback=after_transfer)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(invoke, worker) for worker in done]
            results = [future.result(timeout=2) for future in futures]
        self.assertTrue(all(result["comparison_pass"] for result in results))
        self.assertTrue(all(result["output_shape_ok"] and result["raw_output_storage"]["bytes"] == len(raw)
                            for result in results))
        self.assertEqual(callback_events.count("after-transfer"), 2)
        self.assertEqual(callback_events.count("save"), 2)
        self.assertEqual(worker_api.calls, ["quality", "quality"])
        self.assertTrue(all(worker.finished.is_set() for worker in done))

    def test_transfer_failure_aborts_peer_barrier_and_preserves_completed_response(self):
        raw = bytes(2560 * 4)
        failing = TimedFakeWorker("cpu", 0.005, raw, fail=True)
        completed = TimedFakeWorker("arc", 0.025, raw)
        workers = [failing, completed]; worker_api = FakeQuality(workers); monitor = FakeMonitor()
        barrier = threading.Barrier(2); saved = []

        def writer(path, body):
            if not all(worker.finished.is_set() for worker in workers):
                raise AssertionError("response was saved before peer transfer terminated")
            saved.append(str(path))
            return {"path": str(path), "bytes": len(body), "sha256": compare.sha(body), "exclusive": True}

        def invoke(worker):
            try:
                return compare._timed_infer(worker, 2, 1, 1, FakeInput(raw), raw, compare.sha(raw), raw,
                    FakeOwnedArray(), worker.name, "paired-failure-fixture", monitor, FakeNumpy(), worker_api,
                    barrier=barrier, output_base=Path("fixture-output") / worker.name, output_writer=writer)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            failed_future = pool.submit(invoke, failing)
            peer_future = pool.submit(invoke, completed)
            failed = failed_future.result(timeout=1)
            peer = peer_future.result(timeout=1)
        self.assertIsInstance(failed, RuntimeError)
        self.assertIsInstance(peer, dict)
        self.assertFalse(peer["paired_transfer_complete"])
        self.assertFalse(peer["comparison_pass"])
        self.assertEqual(len(saved), 1)
        self.assertEqual(worker_api.calls, [])

    def test_resource_gate_failure_keeps_raw_output_and_skips_quality(self):
        raw = bytes(2560 * 4); worker = TimedFakeWorker("arc", 0, raw)
        monitor = FakeMonitor(fail_at=3); worker_api = FakeQuality([worker]); saved = []
        def writer(path, body):
            saved.append((str(path), bytes(body)))
            return {"path": str(path), "bytes": len(body), "sha256": compare.sha(body), "exclusive": True}
        result = compare._timed_infer(worker, 2, 1, 1, FakeInput(raw), raw, compare.sha(raw), raw,
            FakeOwnedArray(), "arc", "gate-failure-fixture", monitor, FakeNumpy(), worker_api,
            output_base=Path("fixture-output") / "arc", output_writer=writer)
        self.assertFalse(result["resource_gate_pass_after_transfer"])
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][1], raw)
        self.assertEqual(worker_api.calls, [])

    @staticmethod
    def _contract_for_ready(ready):
        p = ready
        return {"arc": {"identity_sha256": p["identity_sha256"], "blob_sha256": p["blob_sha256"]},
            "source_pins": {"cpu_kernel_source_sha256": p["native_source_sha256"],
                "cpu_service_source_sha256": p["service_source_sha256"],
                "cpu_harness_cmake_sha256": p["service_cmake_sha256"],
                "cpu_pool_cpp_sha256": p["pool_cpp_sha256"], "cpu_pool_hpp_sha256": p["pool_hpp_sha256"]}}

    @classmethod
    def _ready_for_one_pinned_worker(cls):
        ready = {"schema": "strata-xpu-ready-v1", "status": "ready", "nonce": "0" * 32,
            "identity_sha256": "a" * 64, "rows": 1, "hidden": 2560, "intermediate": 640,
            "operator": "native-q4k-q5_1", "operator_signature": "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
            "native_blob_loaded_and_hash_verified": True, "blob_bytes": 3072000, "blob_sha256": "b" * 64,
            "identity_bytes_sha256": "a" * 64, "identity_bytes": 512,
            "native_source_sha256": "1" * 64, "service_source_sha256": "2" * 64,
            "service_cmake_sha256": "3" * 64, "pool_cpp_sha256": "4" * 64, "pool_hpp_sha256": "5" * 64,
            "pool": {"requested_workers": 1, "actual_workers": 1, "affinity": "all", "host_pinned": True,
                "worker_affinity_readback": [{"worker": 0, "requested_core": 2, "ready": True, "pin_requested": True,
                    "pin_applied": True, "mask_observed": True, "mask_matches_request": True,
                    "observed_group": 0, "observed_mask": 4, "startup_processor": 2}]}}
        return ready


if __name__ == "__main__":
    unittest.main()
