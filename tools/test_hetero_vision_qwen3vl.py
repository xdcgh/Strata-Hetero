"""Small synthetic layout/math fixtures. No real payload, OpenVINO or device."""

from __future__ import annotations

import builtins
import copy
import io
import json
import math
import mmap
import sys
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from tools import hetero_vision_qwen3vl as port


def small_fixture():
    """Eight features, two heads, one block; all storage is synthetic F32."""
    spec = port.VisionSpec(
        asset_sha256="a" * 64, source_revision=port.SOURCE_REVISION,
        embedding=8, heads=2, blocks=1, feed_forward=12, patch_size=2, merge_size=2,
        projection=5, position_count=16, merger_hidden=32, layer_norm_epsilon=1e-6,
        image_size_metadata=8, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
        tensors=(), identity_kind="synthetic_fixture",
    )
    headers = tuple(port.TensorHeader(name, shape, "F32")
                    for name, shape in port.expected_tensor_shapes(spec).items())
    spec = replace(spec, tensors=headers)
    arrays = {header.name: np.zeros(header.numpy_shape, dtype=np.float32) for header in headers}
    for name in arrays:
        if ".ln" in name and name.endswith(".weight"):
            arrays[name].fill(1)
    for branch in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        for channel in range(spec.embedding):
            arrays[branch][channel, channel % 3, channel // 3 % 2, channel // 6 % 2] = 1
    arrays["mm.0.weight"][:] = np.eye(32, dtype=np.float32)
    for row, feature in enumerate((0, 8, 16, 24, 31)):
        arrays["mm.2.weight"][row, feature] = 1
    identity = {
        "schema_version": 1, "asset_sha256": spec.asset_sha256, "source_revision": spec.source_revision,
        "kind": "synthetic_fixture", "layout": port.ARRAY_LAYOUT, "decoded_dtype": "float32",
        "decoder": "IN_MEMORY_SYNTHETIC_FIXTURE_ONLY",
        "weights_sha256": {name: port.decoded_array_sha256(value) for name, value in arrays.items()},
    }
    return spec, arrays, identity


def scalar_source_rope(values, positions):
    """Independent scalar translation of ops.cpp mrope_cache + rotate_pairs.

    This uses the pinned source algorithm, not a compiled native encoder. The
    fixture checks the vector pairing, axis selection and section restarts.
    """
    tokens, heads, ne0 = values.shape
    sections = [ne0 // 4] * 4
    n_dims = ne0 // 2
    theta_scale = np.float32(math.pow(10000, float(np.float32(-2 / n_dims))))
    result = np.empty_like(values)
    for token in range(tokens):
        bases = [np.float32(positions[axis][token]) for axis in range(4)]
        theta = bases[:]
        cache = []
        for i0 in range(0, ne0, 2):
            sector = (i0 // 2) % sum(sections)
            if sector == 0:
                theta[0] = bases[0]
            elif sector == sections[0]:
                theta[1] = bases[1]
            elif sector == sum(sections[:2]):
                theta[2] = bases[2]
            elif sector == sum(sections[:3]):
                theta[3] = bases[3]
            selected = 0
            if sections[0] <= sector < sum(sections[:2]):
                selected = 1
            elif sum(sections[:2]) <= sector < sum(sections[:3]):
                selected = 2
            elif sector >= sum(sections[:3]):
                selected = 3
            cache.extend((np.float32(math.cos(float(theta[selected]))),
                          np.float32(math.sin(float(theta[selected])))))
            for axis in range(4):
                theta[axis] = np.float32(theta[axis] * theta_scale)
        for head in range(heads):
            for i0 in range(0, ne0, 2):
                j = i0 // 2
                left, right = values[token, head, j], values[token, head, j + n_dims]
                cc, ss = cache[i0], cache[i0 + 1]
                result[token, head, j] = left * cc - right * ss
                result[token, head, j + n_dims] = left * ss + right * cc
    return result


class ArrayNode:
    """Tiny host-array evaluator for graph plumbing; no OpenVINO imports."""

    def __init__(self, value):
        self.value = np.asarray(value)
        self.names = set()

    def output(self, index=0):
        if index != 0:
            raise AssertionError(index)
        return self

    def get_tensor(self):
        return self

    def set_names(self, names):
        self.names = set(names)

    def set_friendly_name(self, name):
        self.friendly_name = name


def value(node):
    return node.value if isinstance(node, ArrayNode) else np.asarray(node)


class ArrayOps:
    def __init__(self, image):
        self.image = image

    def constant(self, data, **kwargs):
        return ArrayNode(data)

    def parameter(self, shape, dtype, **kwargs):
        assert tuple(shape) == self.image.shape
        return ArrayNode(self.image)

    def reshape(self, data, shape, special_zero):
        assert not special_zero
        return ArrayNode(value(data).reshape(tuple(value(shape))))

    def transpose(self, data, order):
        return ArrayNode(value(data).transpose(tuple(value(order))))

    def gather(self, data, indices, axis):
        return ArrayNode(np.take(value(data), value(indices), axis=int(value(axis))))

    def split(self, data, axis, count):
        pieces = [ArrayNode(part) for part in np.split(value(data), count, axis=int(value(axis)))]
        return types.SimpleNamespace(output=lambda index: pieces[index])

    def convert(self, data, dtype):
        array = value(data)
        if dtype == "bf16":
            bits = array.astype(np.float32).view(np.uint32)
            rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
            return ArrayNode(rounded.view(np.float32))
        return ArrayNode(array.astype({"f16": np.float16, "f32": np.float32}[dtype]))

    def convolution(self, data, filters, strides, pads_begin, pads_end, dilations):
        assert pads_begin == pads_end == [0, 0] and dilations == [1, 1]
        image, kernels = value(data), value(filters)
        _, oc, kh, kw = (1, *kernels.shape[:1], *kernels.shape[2:])
        gh, gw = image.shape[2] // kh, image.shape[3] // kw
        assert strides == [kh, kw]
        out = np.zeros((1, oc, gh, gw), dtype=np.float32)
        for y in range(gh):
            for x in range(gw):
                patch = image[0, :, y * kh:(y + 1) * kh, x * kw:(x + 1) * kw]
                for c in range(oc):
                    out[0, c, y, x] = np.sum(patch * kernels[c], dtype=np.float32)
        return ArrayNode(out)

    def matmul(self, left, right, transpose_a, transpose_b):
        a, b = value(left), value(right)
        if transpose_a:
            a = a.swapaxes(-1, -2)
        if transpose_b:
            b = b.swapaxes(-1, -2)
        return ArrayNode(np.matmul(a, b))

    def reduce_mean(self, data, axes, keep_dims):
        return ArrayNode(np.mean(value(data), axis=tuple(value(axes)), keepdims=keep_dims))

    def softmax(self, data, axis):
        logits = value(data)
        numerator = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
        return ArrayNode(numerator / np.sum(numerator, axis=axis, keepdims=True))

    def concat(self, nodes, axis):
        return ArrayNode(np.concatenate([value(node) for node in nodes], axis=axis))

    def select(self, condition, left, right):
        return ArrayNode(np.where(value(condition), value(left), value(right)))


for method, function in (
    ("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply), ("divide", np.divide),
    ("minimum", np.minimum), ("maximum", np.maximum), ("less_equal", np.less_equal),
    ("greater_equal", np.greater_equal),
):
    setattr(ArrayOps, method, lambda self, left, right, _fn=function, **kwargs:
            ArrayNode(_fn(value(left), value(right))))
for method, function in (("sqrt", np.sqrt), ("tanh", np.tanh)):
    setattr(ArrayOps, method, lambda self, data, _fn=function, **kwargs: ArrayNode(_fn(value(data))))


class VisionPreparationTests(unittest.TestCase):
    def test_metadata_only_cli_does_not_import_array_or_device_runtime(self):
        original_import = builtins.__import__

        def no_compute_imports(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "openvino"):
                raise AssertionError("metadata-only path imported a compute runtime")
            return original_import(name, *args, **kwargs)

        stdout = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=no_compute_imports), mock.patch("sys.stdout", stdout):
            self.assertEqual(port.main([]), 0)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt["tensor_count"], 334)
        self.assertEqual(receipt["shape_plan"]["output"], [9, 2560])
        self.assertEqual(receipt["layer_norm_epsilon"], 9.999999974752427e-7)
        for flag in ("openvino_imported", "device_queried", "graph_constructed", "weights_materialized",
                     "compiled", "inference_run", "file_written", "production_route_enabled", "npu_accepted"):
            self.assertIs(receipt[flag], False)

    def test_cli_rejects_gguf_before_file_access(self):
        stdout = io.StringIO()
        with mock.patch.object(port.Path, "read_text", side_effect=AssertionError("asset opened")), \
                mock.patch("sys.stdout", stdout):
            self.assertEqual(port.main(["--inventory", "forbidden.gguf"]), 2)
        self.assertIn("JSON header inventory", json.loads(stdout.getvalue())["error"])

    def test_missing_semantics_and_deepstack_are_rejected(self):
        inventory = json.loads(port.DEFAULT_INVENTORY.read_text(encoding="utf-8"))
        for mutation in ("epsilon", "deepstack", "shape"):
            data = copy.deepcopy(inventory)
            if mutation == "epsilon":
                del data["vision_related_metadata"]["clip.vision.attention.layer_norm_epsilon"]
            elif mutation == "deepstack":
                data["vision_related_metadata"]["clip.vision.is_deepstack_layers"]["value"][0] = True
            else:
                data["tensor_descriptors"][3]["shape_numpy_order_reversed"].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(port.VisionPortError):
                port.spec_from_inventory(data)

    def test_merge_order_rectangular_grid_and_source_cont_permute(self):
        expected = (0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11,
                    12, 13, 18, 19, 14, 15, 20, 21, 16, 17, 22, 23)
        self.assertEqual(port.spatial_reorder_indices(4, 6), expected)
        # Fortran arrays express GGML's dim0-contiguous representation. Reproduce
        # the source's actual cont/reshape/permute chain independently of indices.
        e, gw, gh = 3, 6, 4
        inp = np.arange(e * gw * gh, dtype=np.float32).reshape((e, gw, gh), order="F")
        merged = np.reshape(inp, (2 * e, gw // 2, gh), order="F")
        merged = np.reshape(merged, (2 * e, gw // 2, 2, gh // 2), order="F")
        merged = np.asfortranarray(merged.transpose(0, 2, 1, 3))
        merged = merged.reshape((e, gw * gh), order="F").T
        row_major = inp.transpose(2, 1, 0).reshape(gh * gw, e)
        np.testing.assert_array_equal(merged, row_major[list(expected)])

    def test_positions_match_merge_coordinates_and_four_source_axes(self):
        ys, xs, duplicate_y, duplicate_x = port.vision_positions(2, 4)
        self.assertEqual(ys, (0, 0, 1, 1, 0, 0, 1, 1))
        self.assertEqual(xs, (0, 1, 0, 1, 2, 3, 2, 3))
        self.assertEqual((ys, xs), (duplicate_y, duplicate_x))
        with self.assertRaises(port.VisionPortError):
            port.spatial_reorder_indices(3, 4)

    def test_bilinear_align_corners_plane_endpoints_and_singletons(self):
        plan = port.bilinear_align_corners_plan(3, 4, 5, 7)
        table = np.asarray([10 * y + x for y in range(3) for x in range(4)], np.float32)
        dx, dy = np.array(plan["dx"], np.float32), np.array(plan["dy"], np.float32)
        actual = (table[list(plan["a"])] * (1 - dx) * (1 - dy)
                  + table[list(plan["b"])] * dx * (1 - dy)
                  + table[list(plan["c"])] * (1 - dx) * dy
                  + table[list(plan["d"])] * dx * dy).reshape(5, 7)
        expected = np.array([[10 * y / 2 + x / 2 for x in range(7)] for y in range(5)], np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        self.assertEqual(actual[0, 0], table[0])
        self.assertEqual(actual[-1, -1], table[-1])
        single = port.bilinear_align_corners_plan(3, 4, 1, 1)
        self.assertEqual(single["a"], (0,))
        self.assertEqual(single["dx"], (0.0,))
        expanded = port.bilinear_align_corners_plan(1, 1, 2, 4)
        self.assertEqual(set(expanded["a"] + expanded["b"] + expanded["c"] + expanded["d"]), {0})

    def test_vision_rope_matches_scalar_pinned_source_translation(self):
        positions = port.vision_positions(2, 4)
        # Exercise production head dimension without model weights: 1,152 floats.
        source = (np.arange(8 * 2 * 72, dtype=np.float32) - 110) / np.float32(137)
        source = source.reshape(8, 2, 72)
        cc, ss = (np.array(cache, np.float32)[:, None, :] for cache in port.vision_rope_cache(positions, 72))
        left, right = source[..., :36], source[..., 36:]
        actual = np.concatenate((left * cc - right * ss, left * ss + right * cc), axis=-1)
        np.testing.assert_array_equal(actual, scalar_source_rope(source, positions))
        # Token 4 is y=0,x=2: first section is identity, second starts at angle 2.
        self.assertEqual(cc[4, 0, 0], 1)
        self.assertAlmostEqual(float(cc[4, 0, 18]), math.cos(2), places=7)
        np.testing.assert_allclose(np.sum(actual ** 2, axis=-1), np.sum(source ** 2, axis=-1), rtol=2e-7)

    def test_linear_weight_direction_for_nonsquare_gguf_tensor(self):
        # GGUF [in=3,out=2] decodes into ordinary NumPy [out=2,in=3].
        weight = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        inp = np.array([[2, -1, 3]], dtype=np.float32)
        np.testing.assert_array_equal(inp @ port.linear_rhs(weight), [[9, 21]])
        self.assertTrue(port.linear_rhs(weight).flags.c_contiguous)

    def test_gelu_variant_is_tanh_and_cpu_rounding_is_separate(self):
        values = np.array([-10, -2.3, -0.7, 0, 0.7, 2.3, 10], np.float32)
        expected = np.array([0.5 * float(x) * (1 + math.tanh(math.sqrt(2 / math.pi) * float(x)
                             * (1 + 0.044715 * float(x) ** 2))) for x in values], np.float32)
        np.testing.assert_allclose(port.gelu_reference(values), expected, rtol=0, atol=2e-7)
        cpu = port.gelu_reference(values, cpu_rounding=True)
        self.assertEqual((cpu[0], cpu[-1]), (0, 10))
        self.assertGreater(float(np.max(np.abs(cpu - port.gelu_reference(values)))), 1e-5)

    def test_decoded_identity_shape_hash_dtype_and_bf16_gates(self):
        spec, arrays, identity = small_fixture()
        port.validate_decoded_weights(spec, arrays, identity)
        wrong_identity = dict(identity, layout="unidentified")
        with self.assertRaisesRegex(port.VisionPortError, "identity.layout"):
            port.validate_decoded_weights(spec, arrays, wrong_identity)
        arrays["mm.2.bias"][0] = 1
        with self.assertRaisesRegex(port.VisionPortError, "hash mismatch"):
            port.validate_decoded_weights(spec, arrays, identity)
        arrays["mm.2.bias"][0] = 0
        wrong_dtype = dict(arrays, **{"mm.2.bias": arrays["mm.2.bias"].astype(np.float64)})
        with self.assertRaisesRegex(port.VisionPortError, "float32"):
            port.validate_decoded_weights(spec, wrong_dtype, identity)
        # A decoder must not label arbitrary F32 values as exact BF16 payload.
        bf16_spec = replace(spec, tensors=tuple(replace(h, storage_type="BF16") if h.name == "mm.2.weight"
                                               else h for h in spec.tensors))
        arrays["mm.2.weight"][0, 1] = np.float32(0.1)
        identity["weights_sha256"]["mm.2.weight"] = port.decoded_array_sha256(arrays["mm.2.weight"])
        with self.assertRaisesRegex(port.VisionPortError, "exact decoded BF16"):
            port.validate_decoded_weights(bf16_spec, arrays, identity)

    def test_unsupported_merge_and_incomplete_norm_fail(self):
        spec, _, _ = small_fixture()
        with self.assertRaisesRegex(port.VisionPortError, "hardcodes 2x2"):
            replace(spec, merge_size=4).validate()
        with self.assertRaisesRegex(port.VisionPortError, "requires an identified norm"):
            replace(spec, tensors=spec.tensors + (port.TensorHeader("v.post_ln.bias", (8,), "F32"),)).validate()
        with self.assertRaisesRegex(port.VisionPortError, "divisible"):
            port.image_shape_plan(spec, 6, 8)

    def test_mapped_array_is_rejected_before_payload_hash(self):
        # Anonymous synthetic mapping; no file, model or real payload involved.
        with mmap.mmap(-1, 16) as buffer:
            array = np.ndarray((4,), dtype=np.float32, buffer=buffer)
            with mock.patch.object(port.hashlib, "sha256", side_effect=AssertionError("mapped payload hashed")):
                with self.assertRaisesRegex(port.VisionPortError, "mapped asset storage"):
                    port.decoded_array_sha256(array)
            del array

    def test_invalid_build_is_rejected_before_openvino_import(self):
        spec, arrays, identity = small_fixture()
        with mock.patch.dict("sys.modules", {"openvino": None}):
            with self.assertRaisesRegex(port.VisionPortError, "arithmetic_policy"):
                port.build_openvino_model(spec, arrays, identity, image_height=4, image_width=8,
                                          arithmetic_policy="guess")
            with self.assertRaisesRegex(port.VisionPortError, "output tap"):
                port.build_openvino_model(spec, arrays, identity, image_height=4, image_width=8,
                                          arithmetic_policy="graph_f32", output_taps=("unknown",))

    def test_entire_static_graph_with_tiny_host_evaluator_and_closed_form_projection(self):
        spec, arrays, identity = small_fixture()
        image = np.arange(3 * 4 * 8, dtype=np.float32).reshape(1, 3, 4, 8) / np.float32(128)
        fake_ops = ArrayOps(image)
        fake_ov = types.ModuleType("openvino")
        fake_ov.opset13 = fake_ops
        fake_ov.Model = lambda outputs, inputs, name: types.SimpleNamespace(outputs=outputs, inputs=inputs, name=name)
        modules = {"openvino": fake_ov, "openvino.opset13": fake_ops}
        with mock.patch.dict(sys.modules, modules):
            model, receipt = port.build_openvino_model(
                spec, arrays, identity, image_height=4, image_width=8, arithmetic_policy="graph_f32",
                output_taps=("patch_merge", "block.0", "merged"))
        # Attention/FFN weights are zero; both residuals preserve the explicit
        # temporal sum. Merger identity + selected columns has a closed form.
        patch_rows = []
        for y in (0,):
            for x in (0, 2):
                for dy in (0, 1):
                    for dx in (0, 1):
                        patch_rows.append([2 * image[0, c % 3, (y + dy) * 2 + c // 3 % 2,
                                                     (x + dx) * 2 + c // 6 % 2] for c in range(8)])
        patch_rows = np.array(patch_rows, np.float32)
        merged = patch_rows.reshape(2, 32)
        expected = port.gelu_reference(merged)[:, [0, 8, 16, 24, 31]]
        np.testing.assert_array_equal(model.outputs[0].value, expected)
        np.testing.assert_array_equal(model.outputs[1].value, patch_rows)
        np.testing.assert_array_equal(model.outputs[2].value, patch_rows)
        np.testing.assert_array_equal(model.outputs[3].value, merged)
        self.assertEqual(model.outputs[0].value.shape, (2, 5))
        self.assertFalse(receipt["compiled"] or receipt["inference_run"] or receipt["device_queried"])

    def test_nonzero_uniform_attention_ffn_residuals_and_both_norms(self):
        spec, arrays, identity = small_fixture()
        extra_headers = tuple(port.TensorHeader(prefix + suffix, (8,), "F32")
                              for prefix in ("v.pre_ln.", "v.post_ln.") for suffix in ("weight", "bias"))
        spec = replace(spec, tensors=spec.tensors + extra_headers)
        for prefix in ("v.pre_ln.", "v.post_ln."):
            arrays[prefix + "weight"] = np.arange(1, 9, dtype=np.float32) / np.float32(8)
            arrays[prefix + "bias"] = np.arange(-4, 4, dtype=np.float32) / np.float32(32)
        # Q=K=0 gives uniform attention, V is LN1's output plus a bias.
        arrays["v.blk.0.attn_qkv.weight"][16:] = np.eye(8, dtype=np.float32)
        arrays["v.blk.0.attn_qkv.bias"][16:] = np.arange(8, dtype=np.float32) / np.float32(16)
        arrays["v.blk.0.attn_out.weight"][:] = np.eye(8, dtype=np.float32)
        arrays["v.blk.0.attn_out.bias"][:] = np.arange(8, dtype=np.float32) / np.float32(64)
        arrays["v.blk.0.ffn_up.bias"][:] = np.arange(-6, 6, dtype=np.float32) / np.float32(8)
        for channel in range(8):
            arrays["v.blk.0.ffn_down.weight"][channel, channel] = np.float32(0.25)
        arrays["v.blk.0.ffn_down.bias"][:] = np.arange(8, dtype=np.float32) / np.float32(128)
        identity["weights_sha256"] = {name: port.decoded_array_sha256(array) for name, array in arrays.items()}
        image = np.arange(3 * 4 * 8, dtype=np.float32).reshape(1, 3, 4, 8) / np.float32(128)
        fake_ops = ArrayOps(image)
        fake_ov = types.ModuleType("openvino")
        fake_ov.opset13 = fake_ops
        fake_ov.Model = lambda outputs, inputs, name: types.SimpleNamespace(outputs=outputs)
        with mock.patch.dict(sys.modules, {"openvino": fake_ov, "openvino.opset13": fake_ops}):
            model, _ = port.build_openvino_model(
                spec, arrays, identity, image_height=4, image_width=8, arithmetic_policy="graph_f32",
                output_taps=("pre_norm", "block.0", "post_norm"))

        def scalar_norm(row, weights, biases):
            mean = sum(float(x) for x in row) / 8
            variance = sum((float(x) - mean) ** 2 for x in row) / 8
            return [(float(x) - mean) / math.sqrt(variance + spec.layer_norm_epsilon) * float(w) + float(b)
                    for x, w, b in zip(row, weights, biases)]

        patch_rows = [[2 * float(image[0, c % 3, dy * 2 + c // 3 % 2, (x + dx) * 2 + c // 6 % 2])
                       for c in range(8)] for x in (0, 2) for dy in (0, 1) for dx in (0, 1)]
        pre = [scalar_norm(row, arrays["v.pre_ln.weight"], arrays["v.pre_ln.bias"]) for row in patch_rows]
        normalized = [scalar_norm(row, [1] * 8, [0] * 8) for row in pre]
        average_v = [sum(row[c] + c / 16 for row in normalized) / 8 for c in range(8)]
        block = []
        for row in pre:
            after_ffn = []
            for c in range(8):
                bias = (c - 6) / 8
                ff = 0.5 * bias * (1 + math.tanh(math.sqrt(2 / math.pi) * bias * (1 + 0.044715 * bias ** 2)))
                after_ffn.append(row[c] + average_v[c] + c / 64 + ff / 4 + c / 128)
            block.append(after_ffn)
        post = [scalar_norm(row, arrays["v.post_ln.weight"], arrays["v.post_ln.bias"]) for row in block]
        np.testing.assert_allclose(model.outputs[1].value, pre, rtol=0, atol=4e-7)
        np.testing.assert_allclose(model.outputs[2].value, block, rtol=0, atol=6e-7)
        np.testing.assert_allclose(model.outputs[3].value, post, rtol=0, atol=6e-7)
        merged = np.array(post).reshape(2, 32)
        expected = [[0.5 * row[c] * (1 + math.tanh(math.sqrt(2 / math.pi) * row[c]
                     * (1 + 0.044715 * row[c] ** 2))) for c in (0, 8, 16, 24, 31)] for row in merged]
        np.testing.assert_allclose(model.outputs[0].value, expected, rtol=0, atol=6e-7)


if __name__ == "__main__":
    unittest.main()
