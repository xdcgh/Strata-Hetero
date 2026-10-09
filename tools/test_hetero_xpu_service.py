"""Pure fake-Core/fake-infer tests for the standalone Arc service state machine."""
from __future__ import annotations

import contextlib
import builtins
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import hetero_xpu_service as service_mod
from tools import hetero_xpu_transport as transport
from tools import hetero_xpu_worker as xpu_worker


class FakeInferRequest:
    def __init__(self):
        self.calls = 0
        self.last_input = None
        self.output = None

    def infer(self, inputs):
        self.calls += 1
        self.last_input = inputs["input"]
        self.output = np.array(self.last_input + np.float32(1), dtype=np.float32, copy=True, order="C")

    def get_output_tensor(self, _index):
        return SimpleNamespace(data=self.output)


class FakeCompiled:
    def __init__(self, rows, execution_devices=("GPU.0",), precision="f32"):
        self.rows = rows
        self.execution_devices = list(execution_devices)
        self.precision = precision
        self.request = FakeInferRequest()

    def get_property(self, key):
        if key == "execution_devices":
            return self.execution_devices
        if key == "inference_precision":
            return self.precision
        raise AssertionError(f"unexpected compiled property: {key}")

    def create_infer_request(self):
        return self.request

    def input(self, _index):
        return "input"

    def output(self, _index):
        return "output"


class FakeCore:
    def __init__(self, *, full_name="Intel(R) Arc(TM) A770", execution_devices=("GPU.0",), precision="f32"):
        self.available_devices = ["GPU.0"]
        self.full_name = full_name
        self.execution_devices = execution_devices
        self.precision = precision
        self.compile_calls = []
        self.compiled = []

    def get_property(self, device, prop):
        if device != "GPU.0":
            raise AssertionError(f"unexpected device query: {device}")
        if prop == "full_name":
            return self.full_name
        if prop == "supported_properties":
            return {"INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT"}
        raise AssertionError(f"unexpected Core property: {prop}")

    def compile_model(self, model, device, properties):
        if device != "GPU.0":
            raise AssertionError(f"unexpected compile device: {device}")
        if properties != {"inference_precision": "f32", "execution_mode": "ACCURACY"}:
            raise AssertionError(f"unexpected compile properties: {properties}")
        compiled = FakeCompiled(model["rows"], self.execution_devices, self.precision)
        self.compile_calls.append((model, device, properties))
        self.compiled.append(compiled)
        return compiled


class FakeWorkerApi:
    np = np
    _safe_identity = staticmethod(xpu_worker._safe_identity)
    compile_property_policy = staticmethod(xpu_worker.compile_property_policy)
    assess_precision_policy = staticmethod(xpu_worker.assess_precision_policy)

    def __init__(self):
        self.load_calls = 0
        self.model_calls = []

    def load_verified_weights(self, _path, _identity, _header):
        self.load_calls += 1
        return {"gate": "fake-gate", "up": "fake-up", "down": "fake-down"}

    def make_ffn_model(self, _weights, rows, precision):
        self.model_calls.append((rows, precision))
        return {"rows": rows, "precision": precision}


class HeteroXpuServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-xpu-service-")
        self.root = Path(self.temp.name)
        self.weights_path = self.root / "fake-header-only.npz"
        self.weights_path.write_bytes(b"not a real NPZ; fake service API owns this fixture")
        self.identity_path = self.root / "identity.json"
        self.identity = {
            "schema_version": 1,
            "model": "FAKE_ARC_SERVICE_FIXTURE",
            "shard": "no-model-data",
            "layer": 28,
            "expert": 288,
            "weights_sha256": {name: f"{i:064x}" for i, name in enumerate(("gate", "up", "down"), 1)},
        }
        self.identity_raw = json.dumps(self.identity, separators=(",", ":")).encode("utf-8")
        self.identity_path.write_bytes(self.identity_raw)
        stat = self.weights_path.stat()
        self.header = {
            "hidden": 3,
            "intermediate": 2,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "file_id": {"volume": stat.st_dev, "index": stat.st_ino},
            "payload_hashes_verified": False,
            "tensor_arrays_materialized": False,
        }
        self.worker_api = FakeWorkerApi()
        self.core = FakeCore()
        self.ov = SimpleNamespace(
            __version__="fake-openvino-version",
            Type=SimpleNamespace(f32="f32"),
            properties=SimpleNamespace(
                device=SimpleNamespace(full_name="full_name"),
                supported_properties="supported_properties",
                execution_devices="execution_devices",
                hint=SimpleNamespace(
                    inference_precision="inference_precision",
                    execution_mode="execution_mode",
                    ExecutionMode=SimpleNamespace(ACCURACY="ACCURACY"),
                ),
            ),
        )
        self.service = self.make_service()

    def tearDown(self):
        self.temp.cleanup()

    def make_service(self, *, core=None, identity_sha=None):
        return service_mod.ArcWorkerService(
            weights_path=self.weights_path,
            identity_path=self.identity_path,
            identity=self.identity,
            identity_sha256=identity_sha or hashlib.sha256(self.identity_raw).hexdigest(),
            header_info=self.header,
            core=self.core if core is None else core,
            ov_api=self.ov,
            worker_api=self.worker_api,
        )

    @staticmethod
    def init_frame(rows=2, nonce="0123456789abcdef0123456789abcdef", identity_sha="0" * 64, request_id=8, **extra):
        value = {"schema": service_mod.SERVICE_INIT_SCHEMA, "nonce": nonce,
                 "rows": rows, "identity_sha256": identity_sha, **extra}
        return transport.Frame(service_mod.INIT_KIND, request_id,
                              json.dumps(value, separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def decode_result(raw: bytes):
        frame = transport.read_frame(io.BytesIO(raw))
        status, worker_ns = service_mod.RESULT_PREFIX.unpack(frame.payload[:service_mod.RESULT_PREFIX.size])
        body = frame.payload[service_mod.RESULT_PREFIX.size:]
        return frame, status, worker_ns, body

    def valid_init(self, rows=2, nonce="0123456789abcdef0123456789abcdef", request_id=8):
        return self.init_frame(rows=rows, nonce=nonce,
                               identity_sha=hashlib.sha256(self.identity_raw).hexdigest(), request_id=request_id)

    def test_module_and_default_cli_validate_only_without_openvino_or_core(self):
        stdout = io.StringIO()
        real_import = builtins.__import__

        def deny_openvino(name, *args, **kwargs):
            if name == "openvino" or name.startswith("openvino."):
                raise AssertionError("validate-only imported OpenVINO")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(service_mod.xpu_worker, "validate_npz_metadata", return_value=self.header), \
             mock.patch.object(service_mod, "_create_openvino_runtime", side_effect=AssertionError("Core forbidden")), \
             mock.patch("builtins.__import__", side_effect=deny_openvino), \
             contextlib.redirect_stdout(stdout):
            rc = service_mod.main(["--validate-only", "--weights", str(self.weights_path),
                                   "--identity", str(self.identity_path)])
        self.assertEqual(rc, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "validated_only")
        self.assertFalse(result["openvino_imported"])
        self.assertFalse(result["core_created"])
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["file_written"])

    def test_init_returns_ready_nonce_identity_and_reuses_same_static_graph(self):
        nonce = "abcdef0123456789abcdef0123456789"
        frame, status, worker_ns, body = self.decode_result(self.service.handle_frame(self.valid_init(nonce=nonce)))
        ready = json.loads(body)
        self.assertEqual((frame.kind, frame.request_id), (0x8001, 8))
        self.assertEqual(status, 0)
        self.assertGreaterEqual(worker_ns, 0)
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["nonce"], nonce)
        self.assertEqual(ready["identity_sha256"], hashlib.sha256(self.identity_raw).hexdigest())
        self.assertEqual(ready["execution_devices"], ["GPU.0"])
        self.assertEqual(ready["reported_precision"], "f32")
        self.assertEqual(self.worker_api.load_calls, 1)
        self.assertEqual(len(self.core.compile_calls), 1)

        next_nonce = "fedcba9876543210fedcba9876543210"
        _, status2, _, body2 = self.decode_result(self.service.handle_frame(
            self.valid_init(nonce=next_nonce, request_id=9)))
        cached = json.loads(body2)
        self.assertEqual(status2, 0)
        self.assertEqual(cached["nonce"], next_nonce)
        self.assertTrue(cached["cached_graph_reused"])
        self.assertEqual(cached["compile_ns"], 0)
        self.assertEqual(len(self.core.compile_calls), 1)
        self.assertEqual(self.worker_api.load_calls, 1)

    def test_rows_change_releases_and_recompiles_only_one_static_shape(self):
        self.decode_result(self.service.handle_frame(self.valid_init(rows=2)))
        first_compiled = self.service._compiled
        self.decode_result(self.service.handle_frame(self.valid_init(rows=3, request_id=2)))
        self.assertEqual(self.service.active_rows, 3)
        self.assertIsNot(self.service._compiled, first_compiled)
        self.assertEqual([call[0] for call in self.worker_api.model_calls], [2, 3])
        self.assertEqual(self.worker_api.load_calls, 1)

    def test_nonce_identity_and_exact_init_schema_errors_are_framed(self):
        bad_identity = self.init_frame(identity_sha="f" * 64)
        _, status, _, body = self.decode_result(self.service.handle_frame(bad_identity))
        self.assertEqual(status, 1)
        self.assertIn("identity_sha256", json.loads(body)["message"])
        bad_nonce = self.valid_init(nonce="short", request_id=10)
        _, status, _, body = self.decode_result(self.service.handle_frame(bad_nonce))
        self.assertEqual(status, 1)
        self.assertIn("nonce", json.loads(body)["message"])
        extra_key = self.init_frame(identity_sha=hashlib.sha256(self.identity_raw).hexdigest(),
                                    request_id=11, unexpected="path")
        _, status, _, body = self.decode_result(self.service.handle_frame(extra_key))
        self.assertEqual(status, 1)
        self.assertIn("keys", json.loads(body)["message"])
        self.assertEqual(self.worker_api.load_calls, 0)
        self.assertEqual(len(self.core.compile_calls), 0)

    def test_failed_reinit_invalidates_previous_nonce(self):
        self.decode_result(self.service.handle_frame(self.valid_init()))
        self.assertTrue(self.service.initialized)
        _, status, _, body = self.decode_result(self.service.handle_frame(
            self.valid_init(nonce="bad", request_id=12)))
        self.assertEqual(status, 1)
        self.assertFalse(self.service.initialized)
        _, infer_status, _, infer_body = self.decode_result(self.service.handle_frame(
            transport.Frame(service_mod.INFER_KIND, 13, b"")))
        self.assertEqual(infer_status, 1)
        self.assertIn("before a successful INIT", json.loads(infer_body)["message"])

    def test_error_json_unicode_truncation_stays_valid_and_bounded(self):
        encoded = service_mod._error_json("失败🙂" * 1000)
        self.assertLessEqual(len(encoded), service_mod.MAX_ERROR_JSON_BYTES)
        parsed = json.loads(encoded.decode("ascii"))
        self.assertEqual(len(parsed["message"]), 256)
        self.assertTrue(parsed["message"].startswith("失败🙂"))

    def test_identity_growth_is_rejected_with_bounded_read(self):
        self.identity_path.write_bytes(b"x" * (service_mod.MAX_INIT_JSON_BYTES + 1))
        _, status, _, body = self.decode_result(self.service.handle_frame(self.valid_init()))
        self.assertEqual(status, 1)
        self.assertIn("bounded metadata size", json.loads(body)["message"])
        self.assertEqual(self.worker_api.load_calls, 0)

    def test_init_rejects_response_prefix_boundary_without_allocating_tensor(self):
        huge_hidden = transport.MAX_PAYLOAD_BYTES // 4
        header = dict(self.header, hidden=huge_hidden)
        service = service_mod.ArcWorkerService(
            weights_path=self.weights_path, identity_path=self.identity_path,
            identity=self.identity,
            identity_sha256=hashlib.sha256(self.identity_raw).hexdigest(),
            header_info=header, core=self.core, ov_api=self.ov, worker_api=self.worker_api)
        _, status, _, body = self.decode_result(service.handle_frame(self.valid_init(rows=1)))
        self.assertEqual(status, 1)
        self.assertIn("frame limit", json.loads(body)["message"])
        self.assertEqual(self.worker_api.load_calls, 0)
        self.assertEqual(len(self.core.compile_calls), 0)

    def test_infer_requires_init_exact_shape_finite_input_and_returns_le_f32(self):
        before, status, _, body = self.decode_result(self.service.handle_frame(
            transport.Frame(service_mod.INFER_KIND, 20, b"")))
        self.assertEqual(status, 1)
        self.assertIn("before a successful INIT", json.loads(body)["message"])
        self.decode_result(self.service.handle_frame(self.valid_init(rows=2)))

        wrong_size = transport.Frame(service_mod.INFER_KIND, 21, b"short")
        _, status, _, body = self.decode_result(self.service.handle_frame(wrong_size))
        self.assertEqual(status, 1)
        self.assertIn("size mismatch", json.loads(body)["message"])

        bad = np.zeros((2, 3), dtype="<f4")
        bad[0, 1] = np.nan
        _, status, _, body = self.decode_result(self.service.handle_frame(
            transport.Frame(service_mod.INFER_KIND, 22, bad.tobytes())))
        self.assertEqual(status, 1)
        self.assertIn("NaN", json.loads(body)["message"])
        self.assertEqual(self.core.compiled[-1].request.calls, 0)

        x = np.arange(6, dtype="<f4").reshape(2, 3)
        frame, status, worker_ns, body = self.decode_result(self.service.handle_frame(
            transport.Frame(service_mod.INFER_KIND, 23, x.tobytes(order="C"))))
        self.assertEqual((frame.kind, frame.request_id), (0x8002, 23))
        self.assertEqual(status, 0)
        self.assertGreaterEqual(worker_ns, 0)
        got = np.frombuffer(body, dtype="<f4").reshape(2, 3)
        np.testing.assert_array_equal(got, x + 1)
        self.assertTrue(self.core.compiled[-1].request.last_input.flags.c_contiguous)
        self.assertTrue(self.core.compiled[-1].request.last_input.flags.owndata)

    def test_unknown_operation_and_init_size_limit_use_error_frames(self):
        _, status, _, body = self.decode_result(self.service.handle_frame(
            transport.Frame(17, 99, b"")))
        self.assertEqual(status, 1)
        self.assertIn("unknown operation", json.loads(body)["message"])
        oversized_init = transport.Frame(service_mod.INIT_KIND, 100,
                                         b" " * (service_mod.MAX_INIT_JSON_BYTES + 1))
        frame, status, _, body = self.decode_result(self.service.handle_frame(oversized_init))
        self.assertEqual(frame.request_id, 100)
        self.assertEqual(status, 1)
        self.assertIn("exceeds", json.loads(body)["message"])
        self.assertEqual(self.worker_api.load_calls, 0)

    def test_non_arc_full_name_and_cpu_fallback_are_refused(self):
        wrong_name_core = FakeCore(full_name="Intel(R) UHD Graphics")
        wrong_name_service = self.make_service(core=wrong_name_core)
        _, status, _, body = self.decode_result(wrong_name_service.handle_frame(self.valid_init()))
        self.assertEqual(status, 1)
        self.assertIn("not Intel Arc", json.loads(body)["message"])
        self.assertEqual(len(wrong_name_core.compile_calls), 0)

        fallback_core = FakeCore(execution_devices=("CPU",))
        fallback_service = self.make_service(core=fallback_core)
        _, status, _, body = self.decode_result(fallback_service.handle_frame(self.valid_init()))
        self.assertEqual(status, 1)
        self.assertIn("exactly ['GPU.0']", json.loads(body)["message"])
        self.assertFalse(fallback_service.initialized)

    def test_stdio_loop_uses_only_binary_writer_for_frames(self):
        service = self.make_service()
        inbound = transport.encode_frame(77, 201, b"") + transport.encode_frame(3, 202, b"")
        reader, writer = io.BytesIO(inbound), io.BytesIO()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = service_mod.serve_stdio(service, reader, writer)
        self.assertEqual(rc, 0)
        written = io.BytesIO(writer.getvalue())
        one = transport.read_frame(written)
        two = transport.read_frame(written)
        self.assertEqual((one.kind, one.request_id), (0x804D, 201))
        self.assertEqual((two.kind, two.request_id), (0x8003, 202))
        self.assertEqual(service_mod.RESULT_PREFIX.unpack(one.payload[:12])[0], 1)
        self.assertEqual(service_mod.RESULT_PREFIX.unpack(two.payload[:12])[0], 1)
        self.assertEqual(written.read(), b"")
        self.assertEqual(self.worker_api.load_calls, 0)


if __name__ == "__main__":
    unittest.main()
