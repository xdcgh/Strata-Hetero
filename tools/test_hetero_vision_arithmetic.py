"""Pure, small arithmetic fixtures; no arrays, model files or device runtime."""
import unittest

from tools.hetero_vision_arithmetic import bf16_avx2_dot, bf16_rne, norm_scale_reference


class ArithmeticFixtures(unittest.TestCase):
    def test_bf16_ties_select_even_mantissa(self):
        self.assertEqual(bf16_rne(1 + 1 / 256), 1)
        self.assertEqual(bf16_rne(1 + 3 / 256), 1 + 1 / 64)
        self.assertEqual(bf16_rne(-1 - 1 / 256), -1)

    def test_avx2_tree_preserves_pair_cancellation_and_scalar_tail(self):
        # A sequential F32 sum gives 1. Native first pairs accumulators 0+2
        # and 1+3, giving 2. This fixture detects incorrect reduction trees.
        left = [0.0] * 32
        for index, value in ((0, 2**24), (8, 1), (16, -(2**24)), (24, 1)):
            left[index] = value
        self.assertEqual(bf16_avx2_dot(left, [1.0] * 32), 2)
        self.assertEqual(bf16_avx2_dot(left + [1], [1.0] * 33), 3)

    def test_native_vec_dot_rounds_both_operands_before_products(self):
        self.assertEqual(bf16_avx2_dot([1 + 1 / 256], [2]), 2)

    def test_invalid_vectors_and_norm_denominators_rejected(self):
        for left, right in (([], []), ([1], []), ([float("nan")], [1])):
            with self.assertRaises(ValueError):
                bf16_avx2_dot(left, right)
        with self.assertRaises(ValueError):
            norm_scale_reference(1, 0, reciprocal_first=True)


if __name__ == "__main__":
    unittest.main()
