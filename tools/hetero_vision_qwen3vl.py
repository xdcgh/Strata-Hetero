#!/usr/bin/env python3
"""Preparation-only Qwen3VL vision graph port from pinned llama.cpp 3cf0325.

The CLI reads an existing JSON header inventory and prints a static graph plan.
It never reads a GGUF, decodes weights, imports OpenVINO, creates a device, or
runs inference. The Python graph builder accepts already decoded, identified
arrays; callers own admission, decoding, serialization, compilation and runs.
See bench/hetero/20261009-23-vision-port-preparation/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
ASSET_SHA256 = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
ASSET_BYTES = 907_543_008
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parent.parent
    / "bench/hetero/20261009-16-vision-encoder-headers/vision-header-inventory.json"
)
ARITHMETIC_POLICIES = ("graph_f32", "cpu_vec_dot_rounding")
ARRAY_LAYOUT = "numpy_reversed_gguf_c"


class VisionPortError(ValueError):
    """An identified input is unsupported, incomplete, or inconsistent."""


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VisionPortError(f"{label} must be a positive integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise VisionPortError(f"{label} must be a lowercase SHA-256")
    return value


def _f32(value: float) -> float:
    """Scalar float32 rounding without importing an array or device runtime."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class TensorHeader:
    name: str
    gguf_shape: tuple[int, ...]
    storage_type: str

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        # GGML dimension zero is contiguous; NumPy's last dimension is contiguous.
        return tuple(reversed(self.gguf_shape))


@dataclass(frozen=True)
class VisionSpec:
    asset_sha256: str
    source_revision: str
    embedding: int
    heads: int
    blocks: int
    feed_forward: int
    patch_size: int
    merge_size: int
    projection: int
    position_count: int
    merger_hidden: int
    layer_norm_epsilon: float
    image_size_metadata: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    tensors: tuple[TensorHeader, ...]
    identity_kind: str = "gguf_decoded"

    @property
    def head_dim(self) -> int:
        return self.embedding // self.heads

    @property
    def headers(self) -> dict[str, TensorHeader]:
        return {header.name: header for header in self.tensors}

    def validate(self) -> None:
        _digest(self.asset_sha256, "asset_sha256")
        if self.source_revision != SOURCE_REVISION:
            raise VisionPortError("only the pinned llama.cpp source semantics are implemented")
        if self.identity_kind not in ("gguf_decoded", "synthetic_fixture"):
            raise VisionPortError("identity_kind must identify real decoded weights or a synthetic fixture")
        if self.identity_kind == "gguf_decoded" and self.asset_sha256 != ASSET_SHA256:
            raise VisionPortError("real weights must identify the selected ed59f92 mmproj asset")
        for name in ("embedding", "heads", "blocks", "feed_forward", "patch_size", "merge_size",
                     "projection", "position_count", "merger_hidden", "image_size_metadata"):
            _positive_integer(getattr(self, name), name)
        if self.merge_size != 2:
            raise VisionPortError("the pinned Qwen3VL graph hardcodes 2x2 spatial reordering and merger")
        if self.embedding % self.heads or self.head_dim % 4:
            raise VisionPortError("embedding must divide into heads whose dimension is a multiple of four")
        if math.isqrt(self.position_count) ** 2 != self.position_count:
            raise VisionPortError("learned position count must form the source's square position table")
        eps = self.layer_norm_epsilon
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0:
            raise VisionPortError("layer norm epsilon must be an explicit finite non-negative number")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise VisionPortError("image normalization requires three RGB means and standard deviations")
        if not all(math.isfinite(x) for x in self.image_mean + self.image_std) or min(self.image_std) <= 0:
            raise VisionPortError("image normalization parameters must be finite with positive standard deviations")
        headers = self.headers
        if len(headers) != len(self.tensors):
            raise VisionPortError("duplicate tensor name")
        expected = expected_tensor_shapes(self)
        if set(headers) != set(expected):
            raise VisionPortError(f"unsupported tensor set: missing={sorted(set(expected) - set(headers))}, "
                                  f"extra={sorted(set(headers) - set(expected))}")
        for name, shape in expected.items():
            header = headers[name]
            if header.gguf_shape != shape or header.storage_type not in ("F32", "BF16"):
                raise VisionPortError(f"unsupported descriptor {name}: {header.gguf_shape}, {header.storage_type}")
            if len(shape) == 1 or name in ("v.position_embd.weight", "v.patch_embd.weight", "v.patch_embd.weight.1"):
                if header.storage_type != "F32":
                    raise VisionPortError(f"this port requires the selected asset's F32 storage for {name}")


def expected_tensor_shapes(spec: VisionSpec) -> dict[str, tuple[int, ...]]:
    """The non-deepstack, non-gated tensor contract; dimensions are GGUF order."""
    e, f, p = spec.embedding, spec.feed_forward, spec.patch_size
    shapes = {
        "v.patch_embd.weight": (p, p, 3, e),
        "v.patch_embd.weight.1": (p, p, 3, e),
        "v.patch_embd.bias": (e,),
        "v.position_embd.weight": (e, spec.position_count),
        "mm.0.weight": (4 * e, spec.merger_hidden),
        "mm.0.bias": (spec.merger_hidden,),
        "mm.2.weight": (spec.merger_hidden, spec.projection),
        "mm.2.bias": (spec.projection,),
    }
    for i in range(spec.blocks):
        prefix = f"v.blk.{i}."
        for suffix, shape in (
            ("ln1.weight", (e,)), ("ln1.bias", (e,)),
            ("attn_qkv.weight", (e, 3 * e)), ("attn_qkv.bias", (3 * e,)),
            ("attn_out.weight", (e, e)), ("attn_out.bias", (e,)),
            ("ln2.weight", (e,)), ("ln2.bias", (e,)),
            ("ffn_up.weight", (e, f)), ("ffn_up.bias", (f,)),
            ("ffn_down.weight", (f, e)), ("ffn_down.bias", (e,)),
        ):
            shapes[prefix + suffix] = shape
    # These norms are optional in the pinned graph. A bias without its weight
    # would be silently ignored by the source; reject that ambiguous contract.
    names = set(spec.headers)
    for prefix in ("v.pre_ln", "v.post_ln"):
        if prefix + ".bias" in names and prefix + ".weight" not in names:
            raise VisionPortError(f"{prefix}.bias requires an identified norm weight")
        if prefix + ".weight" in names:
            shapes[prefix + ".weight"] = (e,)
            if prefix + ".bias" in names:
                shapes[prefix + ".bias"] = (e,)
    return shapes


def spec_from_inventory(inventory: Mapping[str, Any]) -> VisionSpec:
    """Consume a JSON receipt only. Referenced model paths are never opened."""
    if inventory.get("schema_version") != 1:
        raise VisionPortError("header inventory must have schema_version 1")
    metadata = inventory.get("vision_related_metadata")
    if not isinstance(metadata, dict):
        raise VisionPortError("missing vision_related_metadata")

    def field(name: str) -> Any:
        entry = metadata.get(name)
        if not isinstance(entry, dict) or entry.get("value_truncated") is not False or "value" not in entry:
            raise VisionPortError(f"missing or truncated metadata: {name}")
        return entry["value"]

    if field("clip.projector_type") != "qwen3vl_merger" or field("clip.has_vision_encoder") is not True:
        raise VisionPortError("the identified asset must contain the Qwen3VL vision encoder")
    if field("clip.use_gelu") is not True or metadata.get("clip.use_silu", {}).get("value", False):
        raise VisionPortError("only the source's FFN_GELU variant is implemented")
    if any(field("clip.vision.is_deepstack_layers")):
        raise VisionPortError("deepstack is not implemented; do not silently drop its features")
    descriptors = inventory.get("tensor_descriptors")
    if not isinstance(descriptors, list):
        raise VisionPortError("missing tensor_descriptors")
    headers = []
    for entry in descriptors:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise VisionPortError("invalid tensor descriptor")
        shape = entry.get("shape_gguf_dimension_order")
        if not isinstance(shape, list) or not shape:
            raise VisionPortError(f"missing GGUF dimension order for {entry['name']}")
        shape = tuple(_positive_integer(x, entry["name"]) for x in shape)
        if entry.get("shape_numpy_order_reversed") != list(reversed(shape)):
            raise VisionPortError(f"inconsistent NumPy layout descriptor for {entry['name']}")
        headers.append(TensorHeader(entry["name"], shape, entry.get("type")))
    by_name = {header.name: header for header in headers}
    try:
        position_count = by_name["v.position_embd.weight"].gguf_shape[1]
        merger_hidden = by_name["mm.0.weight"].gguf_shape[1]
    except (KeyError, IndexError) as exc:
        raise VisionPortError("missing learned position or merger descriptor") from exc
    asset = inventory.get("asset", {})
    if asset.get("expected_bytes") != ASSET_BYTES:
        raise VisionPortError("header receipt does not identify the selected asset byte count")
    spec = VisionSpec(
        asset_sha256=asset.get("expected_sha256_from_verified_readback"),
        source_revision=SOURCE_REVISION,
        embedding=field("clip.vision.embedding_length"), heads=field("clip.vision.attention.head_count"),
        blocks=field("clip.vision.block_count"), feed_forward=field("clip.vision.feed_forward_length"),
        patch_size=field("clip.vision.patch_size"), merge_size=field("clip.vision.spatial_merge_size"),
        projection=field("clip.vision.projection_dim"), position_count=position_count,
        merger_hidden=merger_hidden, layer_norm_epsilon=field("clip.vision.attention.layer_norm_epsilon"),
        image_size_metadata=field("clip.vision.image_size"),
        image_mean=tuple(field("clip.vision.image_mean")), image_std=tuple(field("clip.vision.image_std")),
        tensors=tuple(headers),
    )
    if len(field("clip.vision.is_deepstack_layers")) != spec.blocks:
        raise VisionPortError("deepstack flags do not cover every identified block")
    spec.validate()
    return spec


def image_shape_plan(spec: VisionSpec, height: int, width: int) -> dict[str, Any]:
    spec.validate()
    _positive_integer(height, "image_height")
    _positive_integer(width, "image_width")
    if height % (spec.patch_size * 2) or width % (spec.patch_size * 2):
        raise VisionPortError("static input edges must be divisible by patch_size * 2")
    gh, gw = height // spec.patch_size, width // spec.patch_size
    return {
        "input": [1, 3, height, width], "input_dtype": "float32", "input_layout": "NCHW_RGB_normalized",
        "patch_grid": [gh, gw], "patches": gh * gw, "heads": spec.heads, "head_dim": spec.head_dim,
        "merger_grid": [gh // 2, gw // 2], "output": [gh * gw // 4, spec.projection],
        "output_layout": "token_major_row_major_merger_grid",
    }


def spatial_reorder_indices(grid_height: int, grid_width: int) -> tuple[int, ...]:
    """qwen3vl.cpp:18-30; each tile visits (dy, dx) = 00, 01, 10, 11."""
    _positive_integer(grid_height, "grid_height")
    _positive_integer(grid_width, "grid_width")
    if grid_height % 2 or grid_width % 2:
        raise VisionPortError("2x2 spatial merge requires even patch-grid edges")
    return tuple((y + dy) * grid_width + x + dx
                 for y in range(0, grid_height, 2) for x in range(0, grid_width, 2)
                 for dy in range(2) for dx in range(2))


def vision_positions(grid_height: int, grid_width: int) -> tuple[tuple[int, ...], ...]:
    """clip.cpp:4783-4806; axis-major positions are [y, x, y, x]."""
    order = spatial_reorder_indices(grid_height, grid_width)
    ys, xs = tuple(i // grid_width for i in order), tuple(i % grid_width for i in order)
    return ys, xs, ys, xs


def bilinear_align_corners_plan(source_height: int, source_width: int,
                                target_height: int, target_width: int) -> dict[str, tuple]:
    """Static gather/weight plan for ggml CPU's bilinear ALIGN_CORNERS rule.

    Preserve source float32 coordinates and reciprocal ratio, including its
    singleton-axis rule. Source: ops.cpp:7991-8004,8082-8119.
    """
    def axis(source: int, target: int) -> list[tuple[int, int, float]]:
        _positive_integer(source, "source edge")
        _positive_integer(target, "target edge")
        scale = _f32((target - 1) / (source - 1)) if target > 1 and source > 1 else _f32(target / source)
        output = []
        for i in range(target):
            coordinate = _f32(i / scale)
            low_unclamped = math.floor(coordinate)
            low = min(max(low_unclamped, 0), source - 1)
            high = min(max(low_unclamped + 1, 0), source - 1)
            fraction = min(max(_f32(coordinate - low), 0.0), 1.0)
            output.append((low, high, fraction))
        return output

    yy, xx = axis(source_height, target_height), axis(source_width, target_width)
    corners: list[list[int]] = [[], [], [], []]
    dxs, dys = [], []
    for y0, y1, dy in yy:
        for x0, x1, dx in xx:
            for indices, (y, x) in zip(corners, ((y0, x0), (y0, x1), (y1, x0), (y1, x1))):
                indices.append(y * source_width + x)
            dxs.append(dx)
            dys.append(dy)
    return dict(zip(("a", "b", "c", "d", "dx", "dy"),
                    (tuple(x) for x in (*corners, dxs, dys))))


def vision_rope_cache(positions: Sequence[Sequence[int]], head_dim: int) -> tuple[tuple, tuple]:
    """Scalar pinned VISION RoPE cache; this is not text IMROPE.

    qwen3vl.cpp:103-109; ops.cpp:5991-6057,6143-6158,6190-6215.
    n_dims=head_dim/2; sections=head_dim/4; full head is rotated by
    pairing channel j with j+head_dim/2. Independent sections restart
    their frequency sequence. YaRN ext_factor=0, scale=1, magnitude=1.
    """
    _positive_integer(head_dim, "head_dim")
    if head_dim % 4 or len(positions) != 4 or not positions[0]:
        raise VisionPortError("VISION RoPE requires four nonempty position axes and head_dim divisible by four")
    n = len(positions[0])
    if any(len(axis) != n for axis in positions):
        raise VisionPortError("position axes must have equal lengths")
    if any(isinstance(p, bool) or not isinstance(p, int) for axis in positions for p in axis):
        raise VisionPortError("positions must contain integer coordinates")
    section = head_dim // 4
    theta_scale = _f32(math.pow(10000.0, _f32(-2.0 / (head_dim // 2))))
    cosine, sine = [], []
    for token in range(n):
        bases = [_f32(float(axis[token])) for axis in positions]
        theta = bases[:]
        cc, ss = [], []
        for pair in range(head_dim // 2):
            sector = pair % head_dim
            if sector % section == 0:
                theta[sector // section] = bases[sector // section]
            axis = sector // section
            cc.append(_f32(math.cos(theta[axis])))
            ss.append(_f32(math.sin(theta[axis])))
            theta = [_f32(value * theta_scale) for value in theta]
        cosine.append(tuple(cc))
        sine.append(tuple(ss))
    return tuple(cosine), tuple(sine)


def linear_rhs(decoded_array: Any) -> Any:
    """Turn explicitly supplied [out,in] decoded weights into [in,out]."""
    import numpy as np

    if type(decoded_array) is not np.ndarray or decoded_array.ndim != 2:
        raise VisionPortError("linear weights must be an explicitly supplied 2-D ndarray")
    return np.ascontiguousarray(decoded_array.T)


def _require_in_memory_array(array: Any) -> None:
    """Reject mapped storage before any finite-value scan or payload hashing."""
    import mmap
    import numpy as np

    if type(array) is not np.ndarray or not array.flags.c_contiguous:
        raise VisionPortError("decoded arrays must be ordinary C-contiguous ndarrays, not mapped assets")
    base = array
    while base is not None:
        if isinstance(base, (np.memmap, mmap.mmap)):
            raise VisionPortError("decoded arrays must not reference mapped asset storage")
        base = base.obj if isinstance(base, memoryview) else getattr(base, "base", None)


def decoded_array_sha256(array: Any) -> str:
    """Hash explicitly supplied in-memory bytes, without opening an asset."""
    _require_in_memory_array(array)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def validate_decoded_weights(spec: VisionSpec, arrays: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Verify shape/dtype and decoded hashes. This does not reverify the GGUF."""
    spec.validate()
    if not isinstance(identity, Mapping) or identity.get("schema_version") != 1:
        raise VisionPortError("decoded identity must have schema_version 1")
    for field, expected in (("asset_sha256", spec.asset_sha256), ("source_revision", spec.source_revision),
                            ("layout", ARRAY_LAYOUT), ("decoded_dtype", "float32"),
                            ("kind", spec.identity_kind)):
        if identity.get(field) != expected:
            raise VisionPortError(f"decoded identity.{field} does not match the prepared contract")
    if not isinstance(identity.get("decoder"), str) or not identity["decoder"].strip():
        raise VisionPortError("decoded identity must identify its decoder")
    headers = spec.headers
    hashes = identity.get("weights_sha256")
    if not isinstance(arrays, Mapping) or set(arrays) != set(headers):
        raise VisionPortError("decoded arrays must contain exactly the identified tensor names")
    if not isinstance(hashes, Mapping) or set(hashes) != set(headers):
        raise VisionPortError("decoded hashes must contain exactly the identified tensor names")
    import numpy as np

    for name, header in headers.items():
        array = arrays[name]
        _require_in_memory_array(array)
        if type(array) is not np.ndarray or array.shape != header.numpy_shape:
            raise VisionPortError(f"{name} must be an ordinary ndarray with shape {header.numpy_shape}")
        if array.dtype != np.dtype("float32") or not array.dtype.isnative or not array.flags.c_contiguous:
            raise VisionPortError(f"{name} must be native-endian C-contiguous float32")
        if not np.isfinite(array).all():
            raise VisionPortError(f"{name} contains NaN or infinity")
        if header.storage_type == "BF16" and np.bitwise_and(array.view(np.uint32), 0xFFFF).any():
            raise VisionPortError(f"{name} contains values that are not exact decoded BF16 values")
        if decoded_array_sha256(array) != _digest(hashes[name], f"weights_sha256.{name}"):
            raise VisionPortError(f"decoded array hash mismatch: {name}")


def gelu_reference(values: Any, *, cpu_rounding: bool = False) -> Any:
    """Pinned tanh GELU; optional CPU LUT input/output rounding and tails.

    The CPU option emulates LUT math/rounding, not the table's exact libm bits.
    No claim of native reduction, dispatch or transcendental bitwise parity.
    """
    import numpy as np

    original = np.asarray(values, dtype=np.float32)
    # Tail selection uses the original F32 value, not its rounded F16 value.
    x = np.clip(original, -10, 10).astype(np.float16).astype(np.float32) if cpu_rounding else original
    value = np.float32(0.5) * x * (
        np.float32(1) + np.tanh(np.float32(0.7978845608028654) * x *
                                (np.float32(1) + np.float32(0.044715) * x * x)))
    if cpu_rounding:
        value = value.astype(np.float16).astype(np.float32)
        value = np.where(original <= -10, np.float32(0), np.where(original >= 10, original, value))
    return value


def build_openvino_model(spec: VisionSpec, arrays: Mapping[str, Any], identity: Mapping[str, Any],
                         *, image_height: int, image_width: int, arithmetic_policy: str,
                         output_taps: Sequence[str] = ()) -> tuple[Any, dict[str, Any]]:
    """Construct an uncompiled static Model from explicitly supplied weights.

    No Core, device query, compiler, serialization, inference, asset loader or
    production route exists here. Graph construction can use model-sized RAM,
    so admission and all real-payload access stay in a separately reviewed caller.
    arithmetic_policy is required: do not silently choose backend rounding.
    """
    plan = image_shape_plan(spec, image_height, image_width)
    if arithmetic_policy not in ARITHMETIC_POLICIES:
        raise VisionPortError(f"arithmetic_policy must be one of {ARITHMETIC_POLICIES}")
    known_taps = {"patch_merge", "positioned", "pre_norm", "post_norm", "merged", "projection"}
    known_taps.update(f"block.{i}" for i in range(spec.blocks))
    if len(set(output_taps)) != len(output_taps) or set(output_taps) - known_taps:
        raise VisionPortError("duplicate or unsupported diagnostic output tap")
    validate_decoded_weights(spec, arrays, identity)

    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    rounded_cpu = arithmetic_policy == "cpu_vec_dot_rounding"
    gh, gw = plan["patch_grid"]
    n, e, h, d = plan["patches"], spec.embedding, spec.heads, spec.head_dim
    headers = spec.headers
    constants: dict[str, Any] = {}
    taps: dict[str, Any] = {}

    def c(value: Any, dtype: Any = np.float32, name: str | None = None) -> Any:
        return ops.constant(np.asarray(value, dtype=dtype), name=name)

    def weight(name: str) -> Any:
        if name not in constants:
            constants[name] = ops.constant(arrays[name], name=name)
        return constants[name]

    def reshape(node: Any, shape: Sequence[int]) -> Any:
        return ops.reshape(node, c(shape, np.int64), False)

    def transpose(node: Any, order: Sequence[int]) -> Any:
        return ops.transpose(node, c(order, np.int64))

    def split(node: Any, count: int) -> list[Any]:
        result = ops.split(node, c(-1, np.int64), count)
        return [result.output(i) for i in range(count)]

    def roundtrip(node: Any, dtype: str) -> Any:
        return ops.convert(ops.convert(node, dtype), "f32")

    def linear(node: Any, name: str) -> Any:
        wname = name + ".weight"
        if rounded_cpu and headers[wname].storage_type == "BF16":
            node = roundtrip(node, "bf16")
        node = ops.matmul(node, weight(wname), False, True)
        if name + ".bias" in arrays:
            node = ops.add(node, weight(name + ".bias"))
        return node

    def norm(node: Any, name: str) -> Any:
        axis = c([-1], np.int64)
        centered = ops.subtract(node, ops.reduce_mean(node, axis, True))
        variance = ops.reduce_mean(ops.multiply(centered, centered), axis, True)
        node = ops.divide(centered, ops.sqrt(ops.add(variance, c(spec.layer_norm_epsilon))))
        node = ops.multiply(node, weight(name + ".weight"))
        if name + ".bias" in arrays:
            node = ops.add(node, weight(name + ".bias"))
        return node

    def gelu(node: Any) -> Any:
        original = node
        if rounded_cpu:
            # Bound intermediate values so the unused tail branch does not overflow.
            node = roundtrip(ops.minimum(ops.maximum(node, c(-10)), c(10)), "f16")
        cubic_term = ops.multiply(ops.multiply(c(0.044715), node), node)
        inner = ops.multiply(ops.multiply(c(0.7978845608028654), node), ops.add(c(1), cubic_term))
        result = ops.multiply(ops.multiply(c(0.5), node), ops.add(c(1), ops.tanh(inner)))
        if rounded_cpu:
            result = roundtrip(result, "f16")
            result = ops.select(ops.less_equal(original, c(-10)), c(0),
                                ops.select(ops.greater_equal(original, c(10)), original, result))
        return result

    inp = ops.parameter(plan["input"], np.float32, name="inp_raw")
    # ggml_conv_2d materializes F16 im2col for these F32 kernels (ggml.c:4763).
    patch_input = roundtrip(inp, "f16")
    branches = []
    for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        kernel = arrays[name]
        if rounded_cpu:
            # CPU F16 vec-dot converts the right operand (kernel) to F16 too.
            kernel = kernel.astype(np.float16).astype(np.float32)
        branches.append(ops.convolution(patch_input, ops.constant(kernel, name=name),
                                        [spec.patch_size] * 2, [0, 0], [0, 0], [1, 1]))
    hidden = reshape(transpose(ops.add(*branches), [0, 2, 3, 1]), [n, e])
    order = c(spatial_reorder_indices(gh, gw), np.int64)
    hidden = ops.gather(hidden, order, c(0, np.int64))
    hidden = ops.add(hidden, weight("v.patch_embd.bias"))
    taps["patch_merge"] = hidden

    side = math.isqrt(spec.position_count)
    if gh == side and gw == side:
        position = weight("v.position_embd.weight")
    else:
        resize = bilinear_align_corners_plan(side, side, gh, gw)
        a, b, cc, dd = [ops.gather(weight("v.position_embd.weight"), c(resize[key], np.int64),
                                  c(0, np.int64)) for key in ("a", "b", "c", "d")]
        dx, dy = c(np.asarray(resize["dx"]).reshape(n, 1)), c(np.asarray(resize["dy"]).reshape(n, 1))
        one_x, one_y = ops.subtract(c(1), dx), ops.subtract(c(1), dy)
        terms = [ops.multiply(ops.multiply(a, one_x), one_y),
                 ops.multiply(ops.multiply(b, dx), one_y),
                 ops.multiply(ops.multiply(cc, one_x), dy),
                 ops.multiply(ops.multiply(dd, dx), dy)]
        position = ops.add(ops.add(ops.add(terms[0], terms[1]), terms[2]), terms[3])
    hidden = ops.add(hidden, ops.gather(position, order, c(0, np.int64)))
    taps["positioned"] = hidden
    if "v.pre_ln.weight" in arrays:
        hidden = norm(hidden, "v.pre_ln")
    taps["pre_norm"] = hidden

    cosine, sine = vision_rope_cache(vision_positions(gh, gw), d)
    cosine, sine = c(np.asarray(cosine).reshape(1, n, d // 2)), c(np.asarray(sine).reshape(1, n, d // 2))

    def rope(node: Any) -> Any:
        left, right = split(node, 2)
        return ops.concat([ops.subtract(ops.multiply(left, cosine), ops.multiply(right, sine)),
                           ops.add(ops.multiply(left, sine), ops.multiply(right, cosine))], -1)

    for i in range(spec.blocks):
        prefix = f"v.blk.{i}."
        qkv = split(linear(norm(hidden, prefix + "ln1"), prefix + "attn_qkv"), 3)
        q, k, v = [transpose(reshape(value, [n, h, d]), [1, 0, 2]) for value in qkv]
        q, k = rope(q), rope(k)
        scale = _f32(1.0 / _f32(math.sqrt(d)))  # clip.cpp:264 uses 1.0f / sqrtf(d_head)
        attention = ops.softmax(ops.multiply(ops.matmul(q, k, False, True), c(scale)), -1)
        attention = reshape(transpose(ops.matmul(attention, v, False, False), [1, 0, 2]), [n, e])
        hidden = ops.add(hidden, linear(attention, prefix + "attn_out"))
        ff = gelu(linear(norm(hidden, prefix + "ln2"), prefix + "ffn_up"))
        hidden = ops.add(hidden, linear(ff, prefix + "ffn_down"))
        taps[f"block.{i}"] = hidden

    if "v.post_ln.weight" in arrays:
        hidden = norm(hidden, "v.post_ln")
    taps["post_norm"] = hidden
    hidden = reshape(hidden, [n // 4, 4 * e])
    taps["merged"] = hidden
    projection = linear(gelu(linear(hidden, "mm.0")), "mm.2")
    projection.set_friendly_name("image_embeddings")
    projection.output(0).get_tensor().set_names({"image_embeddings"})
    taps["projection"] = projection
    outputs = [projection]
    for name in output_taps:
        # An explicit identity op gives duplicate-stage taps distinct output names.
        tap = ops.add(taps[name], c(0), name="tap_" + name.replace(".", "_"))
        tap.output(0).get_tensor().set_names({"tap_" + name.replace(".", "_")})
        outputs.append(tap)
    model = ov.Model(outputs, [inp], "qwen3vl_vision_preparation")
    return model, {
        "schema_version": 1, "status": "constructed_uncompiled", "source_revision": spec.source_revision,
        "asset_sha256": spec.asset_sha256, "identity_kind": spec.identity_kind, "shape_plan": plan,
        "arithmetic_policy": arithmetic_policy,
        "gelu_variant": "ggml_tanh" if not rounded_cpu else "ggml_cpu_f16_lut_math_rounding",
        "openvino_imported": True, "device_queried": False, "compiled": False, "inference_run": False,
        "decoded_array_hashes_verified": True, "asset_payload_reverified": False,
        "output_taps": list(output_taps), "native_parity": "NOT_RUN", "npu_accepted": False,
        "native_arithmetic_unknowns": ["GEMM dispatch and reductions", "transcendental rounding", "device precision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    args = parser.parse_args(argv)
    try:
        if args.inventory.suffix.lower() != ".json":
            raise VisionPortError("the preparation CLI accepts only a JSON header inventory")
        spec = spec_from_inventory(json.loads(args.inventory.read_text(encoding="utf-8")))
        result = {
            "schema_version": 1, "status": "metadata_only", "source_revision": spec.source_revision,
            "asset_sha256_recorded": spec.asset_sha256, "asset_hash_reverified": False,
            "tensor_count": len(spec.tensors), "layer_norm_epsilon": spec.layer_norm_epsilon,
            "position_grid": [math.isqrt(spec.position_count)] * 2,
            "shape_plan": image_shape_plan(spec, args.height, args.width),
            "arithmetic_policies_require_explicit_choice": list(ARITHMETIC_POLICIES),
            "temporal_patch_size_metadata": "UNKNOWN_NOT_REQUIRED_FOR_TWO_BRANCH_STILL_IMAGE_GRAPH",
            "rotary_constants_origin": "PINNED_SOURCE_NOT_GGUF_METADATA",
            "openvino_imported": False, "device_queried": False, "graph_constructed": False,
            "weights_materialized": False, "compiled": False, "inference_run": False,
            "file_written": False, "production_route_enabled": False, "npu_accepted": False,
        }
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, VisionPortError, TypeError) as exc:
        print(json.dumps({"schema_version": 1, "status": "rejected", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
