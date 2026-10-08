"""CPU-only parser and numerical fixtures for hetero_xpu_worker."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from tools import hetero_xpu_worker as worker


class HeteroXpuWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-xpu-worker-test-")
        self.root = Path(self.temp.name)
        self.weights_path = self.root / "fixture.npz"
        self.identity_path = self.root / "identity.json"
        self.weights = {
            "gate": np.arange(12, dtype=np.float32).reshape(4, 3) / 13,
            "up": np.arange(12, dtype=np.float32).reshape(4, 3) / 17,
            "down": np.arange(12, dtype=np.float32).reshape(3, 4) / 19,
        }
        np.savez(self.weights_path, **self.weights)
        self.identity = {
            "schema_version": 1,
            "model": "TEST_ONLY_FIXTURE",
            "shard": "fixture-shard.npz",
            "layer": 0,
            "expert": 0,
            "weights_sha256": {
                name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for name, value in self.weights.items()
            },
        }
        self.identity_path.write_text(json.dumps(self.identity), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_validate_only_reads_headers_and_has_no_side_effects_or_openvino_import(self):
        output = self.root / "must-not-exist.json"
        original_import = builtins.__import__

        def deny_openvino(name, *args, **kwargs):
            if name == "openvino" or name.startswith("openvino."):
                raise AssertionError("validate-only imported OpenVINO")
            return original_import(name, *args, **kwargs)

        stdout = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=deny_openvino), \
                mock.patch("sys.stdout", stdout):
            rc = worker.main(["--weights", str(self.weights_path), "--identity", str(self.identity_path),
                              "--output", str(output)])
        self.assertEqual(rc, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "validated_only")
        self.assertFalse(result["openvino_imported"])
        self.assertFalse(result["device_queried"])
        self.assertFalse(result["file_written"])
        self.assertFalse(result["weights_header_info"]["payload_hashes_verified"])
        self.assertFalse(result["tensor_arrays_materialized"])
        self.assertFalse(result["payload_hashes_verified"])
        self.assertIn("may decompress a prefix", result["validation_note"])
        self.assertFalse(output.exists())

    def test_npz_header_metadata_and_real_run_payload_hash_check(self):
        header = worker.validate_npz_metadata(self.weights_path, self.identity)
        self.assertEqual(header["hidden"], 3)
        self.assertEqual(header["intermediate"], 4)
        loaded = worker.load_verified_weights(self.weights_path, self.identity, header)
        for name in self.weights:
            np.testing.assert_array_equal(loaded[name], self.weights[name])

    def test_rejects_wrong_tensor_shape_dtype_and_extra_member(self):
        wrong = self.root / "wrong.npz"
        np.savez(wrong, gate=self.weights["gate"].astype(np.float64), up=self.weights["up"],
                 down=self.weights["down"])
        with self.assertRaisesRegex(worker.WorkerError, "float32"):
            worker.validate_npz_metadata(wrong, self.identity)
        extra = self.root / "extra.npz"
        np.savez(extra, **self.weights, ignored=np.ones((1,), np.float32))
        with self.assertRaisesRegex(worker.WorkerError, "exactly"):
            worker.validate_npz_metadata(extra, self.identity)

    def test_rejects_big_endian_float32(self):
        wrong = self.root / "big-endian.npz"
        arrays = dict(self.weights)
        arrays["gate"] = arrays["gate"].astype(">f4")
        np.savez(wrong, **arrays)
        with self.assertRaisesRegex(worker.WorkerError, "native-endian float32"):
            worker.validate_npz_metadata(wrong, self.identity)

    def test_rejects_overdeclared_tensor_payload_from_small_header_without_allocating_it(self):
        huge_header = self._npy_header((50_000, 50_000), "<f4")
        oversized = self.root / "oversized-header.npz"
        with zipfile.ZipFile(self.weights_path, "r") as source, zipfile.ZipFile(oversized, "w") as target:
            target.writestr("gate.npy", huge_header)  # deliberately no tensor data
            for name in ("up", "down"):
                target.writestr(f"{name}.npy", source.read(f"{name}.npy"))
        self.assertLess(oversized.stat().st_size, 10_000)
        with self.assertRaisesRegex(worker.WorkerError, "tensor payload total exceeds"):
            worker.validate_npz_metadata(oversized, self.identity)

    def test_rejects_implausible_dimension_before_payload_load(self):
        huge_header = self._npy_header((worker.MAX_TENSOR_DIMENSION + 1, 1), "<f4")
        oversized = self.root / "oversized-dimension.npz"
        with zipfile.ZipFile(self.weights_path, "r") as source, zipfile.ZipFile(oversized, "w") as target:
            target.writestr("gate.npy", huge_header)
            for name in ("up", "down"):
                target.writestr(f"{name}.npy", source.read(f"{name}.npy"))
        with self.assertRaisesRegex(worker.WorkerError, "dimensions must be positive"):
            worker.validate_npz_metadata(oversized, self.identity)

    def test_payload_hash_mismatch_is_rejected(self):
        wrong_identity = dict(self.identity)
        wrong_identity["weights_sha256"] = dict(self.identity["weights_sha256"])
        wrong_identity["weights_sha256"]["gate"] = "0" * 64
        header = worker.validate_npz_metadata(self.weights_path, wrong_identity)
        with self.assertRaisesRegex(worker.WorkerError, "SHA-256"):
            worker.load_verified_weights(self.weights_path, wrong_identity, header)

    def test_identity_requires_explicit_expert_and_hashes(self):
        bad = dict(self.identity)
        bad.pop("expert")
        with self.assertRaisesRegex(worker.WorkerError, "expert"):
            worker._safe_identity(bad)

    def test_numpy_ffn_reference_is_finite_and_quality_metrics_gate(self):
        x = np.array([[0.2, -0.4, 0.1], [0.0, 0.3, -0.1]], dtype=np.float32)
        out = worker.numpy_ffn(x, self.weights)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(worker.quality_report(out, out)["pass"])
        altered = out.copy()
        altered[0, 0] += 0.2
        self.assertFalse(worker.quality_report(altered, out)["pass"])

    def test_device_cli_rejects_fallback_names(self):
        parser = worker.build_parser()
        for device in ("AUTO", "HETERO:GPU,CPU", "GPU"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--run", "--weights", "w.npz", "--identity", "i.json",
                                   "--device", device, "--output", "r.json"])

    def test_execution_device_must_match_requested_identifier_exactly(self):
        self.assertTrue(worker.execution_devices_match("GPU.0", ["GPU.0"]))
        self.assertFalse(worker.execution_devices_match("GPU.0", ["GPU"]))
        self.assertFalse(worker.execution_devices_match("GPU.0", ["GPU.0", "CPU"]))
        self.assertFalse(worker.execution_devices_match("CPU", []))

    def test_npu_full_name_accepts_intel_ai_boost_alias(self):
        self.assertTrue(worker.device_full_name_matches("NPU", "Intel(R) AI Boost"))
        self.assertTrue(worker.device_full_name_matches("NPU", "Intel NPU"))
        self.assertFalse(worker.device_full_name_matches("NPU", "Intel(R) Core(TM) Ultra CPU"))
        self.assertFalse(worker.device_full_name_matches("GPU.0", "NVIDIA GeForce RTX"))

    def test_precision_policy_unknown_and_mismatch_are_not_verified(self):
        self.assertEqual(worker.assess_precision_policy(None, "f32"),
                         {"precision_policy_status": "unknown", "precision_policy_matches": False})
        self.assertEqual(worker.assess_precision_policy(None, "f32", "query failed"),
                         {"precision_policy_status": "unknown", "precision_policy_matches": False})
        self.assertEqual(worker.assess_precision_policy("", "f32"),
                         {"precision_policy_status": "unknown", "precision_policy_matches": False})
        self.assertEqual(worker.assess_precision_policy("f16", "f32"),
                         {"precision_policy_status": "mismatch", "precision_policy_matches": False})
        self.assertEqual(worker.assess_precision_policy("f32", "f32"),
                         {"precision_policy_status": "match", "precision_policy_matches": True})

    def test_row_shapes_and_modes_are_explicit(self):
        self.assertEqual(worker.ROW_COUNTS, (1, 2, 4, 8, 16, 32, 64, 128, 256))
        args = worker.build_parser().parse_args(["--weights", "w.npz", "--identity", "i.json"])
        self.assertFalse(args.run)
        self.assertEqual(args.precision, "f32")

    @staticmethod
    def _npy_header(shape, descr):
        header = ("{'descr': " + repr(descr) + ", 'fortran_order': False, 'shape': " + repr(shape) + ", }")
        prefix = np.lib.format.magic(1, 0)
        padding = (-(len(prefix) + 2 + len(header) + 1)) % 16
        encoded = (header + " " * padding + "\n").encode("latin1")
        return prefix + struct.pack("<H", len(encoded)) + encoded


if __name__ == "__main__":
    unittest.main()
