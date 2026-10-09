"""Uncompiled native-activation FFN candidate; no Core or implicit asset access."""
from __future__ import annotations

from typing import Mapping


def build_model(weights: Mapping, minimums, identity: Mapping, rows: int):
    import hashlib
    import numpy as np
    from tools.hetero_xpu_worker import _safe_identity

    checked = _safe_identity(dict(identity))
    source = checked.get("source", {})
    if source.get("quantized_source_types") != {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"}:
        raise ValueError("candidate requires original Q4_K/Q5_1 weights")
    if type(rows) is not int or not 1 <= rows <= 256 or set(weights) != {"gate", "up", "down"}:
        raise ValueError("bounded explicit rows and gate/up/down weights required")
    for name, array in weights.items():
        if type(array) is not np.ndarray or array.dtype != np.float32 or not array.flags.c_contiguous or not np.isfinite(array).all():
            raise ValueError("finite owned F32 decoded matrices required")
        if hashlib.sha256(array.tobytes()).hexdigest() != checked["weights_sha256"][name]:
            raise ValueError("decoded weights hash mismatch")
    f, h = weights["gate"].shape
    if weights["up"].shape != (f, h) or weights["down"].shape != (h, f) or h % 256 or f % 32:
        raise ValueError("native activation block dimensions disagree")
    if type(minimums) is not np.ndarray or minimums.dtype != np.float32 or minimums.shape != (h, f // 32) or not np.isfinite(minimums).all():
        raise ValueError("identified finite Q5_1 minimum blocks required")
    # Caller must independently bind minimums to the original raw down bytes.
    import openvino as ov
    from openvino import opset13 as ops

    def c(value, dtype=np.float32):
        return ops.constant(np.asarray(value, dtype=dtype))

    def reshape(node, shape):
        return ops.reshape(node, c(shape, np.int64), False)

    def half(node):
        return ops.convert(ops.convert(node, "f16"), "f32")

    def nonzero_div(numerator, denominator):
        valid = ops.not_equal(denominator, c(0))
        safe = ops.select(valid, denominator, c(1))
        return ops.select(valid, ops.divide(numerator, safe), c(0))

    inp = ops.parameter([rows, h], np.float32, name="native_input")
    blocks = reshape(inp, [rows, h // 256, 256])
    absolute = ops.absolute(blocks)
    axis = c([2], np.int64)
    maximum = ops.reduce_max(absolute, axis, True)
    indices = c(np.arange(256).reshape(1, 1, 256), np.int64)
    # First signed absolute maximum, including exact ties, matches ggml Q8_K.
    matches = ops.equal(absolute, maximum)
    first = ops.reduce_min(ops.select(matches, indices, c(256, np.int64)), axis, True)
    signed = ops.gather_elements(blocks, first, 2)
    inverse = nonzero_div(c(-127), signed)
    quants = ops.minimum(c(127), ops.round(ops.multiply(blocks, inverse), "half_to_even"))
    scale = nonzero_div(c(1), inverse)
    quantized_x = reshape(ops.multiply(quants, scale), [rows, h])
    gate = ops.matmul(quantized_x, c(weights["gate"]), False, True)
    up = ops.matmul(quantized_x, c(weights["up"]), False, True)
    hidden = ops.multiply(ops.divide(gate, ops.add(c(1), ops.exp(ops.negative(gate)))), up)
    blocks = reshape(hidden, [rows, f // 32, 32])
    maximum = ops.reduce_max(ops.absolute(blocks), axis, True)
    unrounded_scale = ops.divide(maximum, c(127))
    inverse = nonzero_div(c(127), maximum)
    quants = ops.round(ops.multiply(blocks, inverse), "half_to_even")
    sums = ops.reduce_sum(quants, axis, True)
    scale = half(unrounded_scale)
    stored_sum = half(ops.multiply(unrounded_scale, sums))
    correction = reshape(ops.subtract(stored_sum, ops.multiply(scale, sums)), [rows, f // 32])
    quantized_h = reshape(ops.multiply(quants, scale), [rows, f])
    down = ops.matmul(quantized_h, c(weights["down"]), False, True)
    output = ops.add(down, ops.matmul(correction, c(minimums), False, True))
    output.set_friendly_name("native_output")
    return ov.Model([output], [inp], "native_q4k_q5_1_activation_candidate"), {
        "schema_version": 1, "status": "constructed_uncompiled", "rows": rows,
        "operator": "Q4_K_Q8_K_SILU_Q5_1_Q8_1", "core_created": False,
        "compiled": False, "inference_run": False, "raw_minimum_binding_owned_by_caller": True,
        "native_runtime_parity": "NOT_RUN", "production_route_enabled": False,
    }
