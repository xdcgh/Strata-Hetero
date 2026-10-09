"""Pure activation contract tests; no model files or accelerator runtime."""
import contextlib
import io
import json
import struct
import unittest
from unittest import mock

import numpy as np
from tools import hetero_native_activation as activation


class ActivationTests(unittest.TestCase):
    def test_metadata_main_never_imports_array_runtime(self):
        original = __import__
        def forbid(name, *args, **kwargs):
            if name in ("numpy", "openvino"):
                raise AssertionError(name)
            return original(name, *args, **kwargs)
        stream = io.StringIO()
        with mock.patch("builtins.__import__", forbid), contextlib.redirect_stdout(stream):
            self.assertEqual(activation.main(), 0)
        self.assertFalse(json.loads(stream.getvalue())["production_route_enabled"])

    def test_q8k_matches_independent_scalar_source_for_signed_ties_and_halfway(self):
        x = np.zeros((2, 256), np.float32)
        x[0, :8] = [127, -127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
        x[1] = -x[0]
        got = activation.q8k_values(x)
        for i, row in enumerate(x):
            maximum, signed = np.float32(0), np.float32(0)
            for value in row:
                if abs(value) > maximum:
                    maximum, signed = abs(value), value
            inverse = np.float32(-127) / signed
            qs = [min(127, round(float(np.float32(inverse * value)))) for value in row]
            scale = np.float32(1) / inverse
            np.testing.assert_array_equal(got.quants[i, 0], qs)
            np.testing.assert_array_equal(got.values[i], np.array(qs, np.float32) * scale)
        np.testing.assert_array_equal(got.sums16[:, 0, 0], got.quants[:, 0, :16].sum(axis=1))

    def test_q81_affine_correction_matches_stored_sum_native_dot(self):
        x = np.linspace(-.073, .091, 32, dtype=np.float32).reshape(1, 32)
        q = activation.q81_values(x)
        nibbles = (np.arange(32) % 32).astype(np.float32)
        d, minimum = np.float32(.015625), np.float32(-.03125)
        weights = d * nibbles + minimum
        naive = np.sum(weights * q.values[0], dtype=np.float32)
        corrected = naive + minimum * q.affine_sum_correction[0, 0]
        expected = d * q.scale[0, 0] * np.dot(nibbles, q.quants[0, 0].astype(np.float32)) + minimum * q.stored_sum[0, 0]
        self.assertNotEqual(float(q.affine_sum_correction[0, 0]), 0)
        self.assertAlmostEqual(float(corrected), float(expected), places=7)
        self.assertGreater(abs(float(naive - expected)), 1e-7)

    def test_zero_blocks_and_q5_1_minimum_layout(self):
        self.assertTrue(np.all(activation.q8k_values(np.zeros((2, 256), np.float32)).values == 0))
        self.assertTrue(np.all(activation.q81_values(np.zeros((2, 32), np.float32)).affine_sum_correction == 0))
        raw = b"".join(struct.pack("<ee4s16s", .125, x, bytes(4), bytes(16)) for x in [.5, -.25, .75, -1])
        np.testing.assert_array_equal(activation.q5_1_minimums(raw, 2, 64), [[.5, -.25], [.75, -1]])
        with self.assertRaises(ValueError):
            activation.q5_1_minimums(raw[:-1], 2, 64)

    def test_nonfinite_bad_dimensions_and_scale_overflow_fail_closed(self):
        for function, width in ((activation.q8k_values, 256), (activation.q81_values, 32)):
            with self.assertRaises(ValueError):
                function(np.zeros((1, width + 1), np.float32))
            with self.assertRaises(ValueError):
                function(np.full((1, width), np.nan, np.float32))
            with self.assertRaises(ValueError):
                function(np.zeros((1, width), np.float64))
        with self.assertRaisesRegex(ValueError, "overflow"):
            activation.q81_values(np.full((1, 32), 1e10, np.float32))

    def test_closed_form_zero_gate_expert_and_shape_rejection(self):
        x = np.ones((2, 256), np.float32)
        gate = np.zeros((32, 256), np.float32)
        up = np.ones_like(gate)
        down = np.ones((256, 32), np.float32)
        minimums = np.zeros((256, 1), np.float32)
        np.testing.assert_array_equal(activation.native_activation_ffn(x, gate, up, down, minimums), np.zeros_like(x))
        with self.assertRaises(ValueError):
            activation.native_activation_ffn(x, gate, up, down, minimums.T)


if __name__ == "__main__":
    unittest.main()
