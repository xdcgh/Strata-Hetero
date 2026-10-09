#!/usr/bin/env python3
"""Host preparation for the native Q4_K/Q5_1 activation contract.

No asset or runtime is opened by the default CLI. These functions operate on
explicit caller-owned arrays; comparison with the native kernel is still needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

GGML_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
OPERATOR = "Q4_K_Q8_K_SILU_Q5_1_Q8_1"


@dataclass(frozen=True)
class Q8KValues:
    values: Any
    quants: Any
    scale: Any
    sums16: Any


@dataclass(frozen=True)
class Q81Values:
    values: Any
    quants: Any
    scale: Any
    stored_sum: Any
    affine_sum_correction: Any


def _blocks(values: Any, width: int) -> tuple[Any, tuple[int, int]]:
    import numpy as np

    if type(values) is not np.ndarray or values.dtype != np.dtype("float32"):
        raise ValueError("activation must be a native float32 ndarray")
    if values.ndim != 2 or min(values.shape) < 1 or values.shape[1] % width:
        raise ValueError(f"activation must have [rows, features] with features divisible by {width}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite activation")
    return values.reshape(values.shape[0], -1, width), values.shape


def q8k_values(values: Any) -> Q8KValues:
    """ggml-quants.c:2768: first signed absolute maximum, F32 scale, ties to even."""
    import numpy as np

    blocks, shape = _blocks(values, 256)
    # argmax's first tie matches the scalar source's strict ax > amax.
    selected = np.argmax(np.abs(blocks), axis=-1)[..., None]
    signed_max = np.take_along_axis(blocks, selected, axis=-1)
    inverse = np.divide(np.float32(-127), signed_max, out=np.zeros_like(signed_max), where=signed_max != 0)
    if not np.isfinite(inverse).all():
        raise ValueError("Q8_K inverse scale overflow; native agreement unavailable")
    quants = np.minimum(np.float32(127), np.rint(blocks * inverse))
    if (quants < -128).any():
        raise ValueError("Q8_K quant out of range")
    quants = quants.astype(np.int8)
    scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=inverse != 0)
    decoded = (quants.astype(np.float32) * scale).reshape(shape)
    sums = quants.reshape(*quants.shape[:-1], 16, 16).sum(axis=-1, dtype=np.int16)
    return Q8KValues(decoded, quants, scale[..., 0], sums)


def q81_values(values: Any) -> Q81Values:
    """x86/quants.c:400: RN-even quants, independently stored F16 d and s.

    Q5_1 uses its minimum times stored s. Sum(dequantized Q8_1) is different,
    so a float matmul also needs the returned per-block affine correction.
    """
    import numpy as np

    blocks, shape = _blocks(values, 32)
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    unrounded_scale = maximum / np.float32(127)
    inverse = np.divide(np.float32(127), maximum, out=np.zeros_like(maximum), where=maximum != 0)
    if not np.isfinite(inverse).all():
        raise ValueError("Q8_1 inverse scale overflow; native agreement unavailable")
    rounded = np.rint(blocks * inverse)
    if (np.abs(rounded) > 127).any():
        raise ValueError("Q8_1 quant out of range")
    quants = rounded.astype(np.int8)
    sums = quants.sum(axis=-1, keepdims=True, dtype=np.int32).astype(np.float32)
    with np.errstate(over="ignore"):
        scale = unrounded_scale.astype(np.float16).astype(np.float32)
        stored_sum = (unrounded_scale * sums).astype(np.float16).astype(np.float32)
    if not np.isfinite(scale).all() or not np.isfinite(stored_sum).all():
        raise ValueError("Q8_1 F16 scale/sum overflow; refuse a helper result")
    correction = stored_sum - scale * sums
    return Q81Values((quants.astype(np.float32) * scale).reshape(shape), quants,
                     scale[..., 0], stored_sum[..., 0], correction[..., 0])


def q5_1_minimums(raw_down: bytes, output_width: int, intermediate_width: int) -> Any:
    """Read only caller-supplied original block_q5_1 bytes (24 bytes per 32 values)."""
    import numpy as np

    if type(raw_down) is not bytes or type(output_width) is not int or type(intermediate_width) is not int:
        raise ValueError("raw bytes and integer dimensions required")
    if output_width < 1 or intermediate_width < 32 or intermediate_width % 32:
        raise ValueError("invalid Q5_1 dimensions")
    blocks = output_width * (intermediate_width // 32)
    if len(raw_down) != blocks * 24:
        raise ValueError("Q5_1 payload size mismatch")
    dtype = np.dtype([("scale", "<f2"), ("minimum", "<f2"),
                      ("high", "u1", (4,)), ("low", "u1", (16,))])
    result = np.frombuffer(raw_down, dtype=dtype)["minimum"].astype(np.float32)
    result = np.ascontiguousarray(result.reshape(output_width, -1))
    if not np.isfinite(result).all():
        raise ValueError("non-finite Q5_1 minimum")
    return result


def native_activation_ffn(x: Any, gate: Any, up: Any, down: Any, minimums: Any) -> Any:
    """Candidate algebra for one identified native expert, not a device worker.

    The caller must bind decoded weights and minimums to original raw bytes.
    Reduction/SiLU rounding and native-kernel agreement remain separate gates.
    """
    import numpy as np

    for value in (gate, up, down, minimums):
        if type(value) is not np.ndarray or value.dtype != np.dtype("float32") or value.ndim != 2:
            raise ValueError("weights/minimums must be float32 matrices")
        if not np.isfinite(value).all():
            raise ValueError("non-finite weight/minimum")
    if gate.shape != up.shape or down.shape != tuple(reversed(gate.shape)):
        raise ValueError("gate/up/down dimensions disagree")
    if minimums.shape != (down.shape[0], down.shape[1] // 32) or down.shape[1] % 32:
        raise ValueError("Q5_1 affine blocks disagree")
    if type(x) is not np.ndarray or x.ndim != 2 or x.shape[1] != gate.shape[1]:
        raise ValueError("activation/weight dimensions disagree")
    quantized_x = q8k_values(x).values
    g = quantized_x @ gate.T
    u = quantized_x @ up.T
    with np.errstate(over="ignore"):
        hidden = (g / (np.float32(1) + np.exp(-g))) * u
    quantized_h = q81_values(hidden)
    output = quantized_h.values @ down.T + quantized_h.affine_sum_correction @ minimums.T
    if not np.isfinite(output).all():
        raise ValueError("non-finite candidate expert output")
    return output


def main() -> int:
    print(json.dumps({"schema_version": 1, "status": "metadata_only", "operator": OPERATOR,
                      "ggml_revision": GGML_REVISION, "native_parity": "NOT_RUN",
                      "asset_opened": False, "array_runtime_imported": False,
                      "device_touched": False, "production_route_enabled": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
