"""Bounded scalar diagnostics for pinned GGML vision arithmetic.

No array or device runtime is imported. These helpers do not load any asset.
They describe the portable AVX2 source operations, not compiler fusion or an
unknown dispatched GEMM. Callers must identify their saved input/weight bytes.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Sequence

SOURCE_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_rne(value: float) -> float:
    """ggml-impl.h's finite F32 -> BF16 round-to-nearest-even conversion."""
    value = f32(value)
    if not math.isfinite(value):
        raise ValueError("finite diagnostic inputs required")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def bf16_avx2_dot(left: Sequence[float], right: Sequence[float]) -> float:
    """vec.cpp:176-202's four 8-lane BF16 accumulators and reduction tree.

    Both operands are rounded to BF16, as the native BF16 vec-dot contract
    requires. Multiplication/addition round separately. The source's leftover
    products accumulate in binary64; the final assignment rounds to F32.
    This is deliberately bounded to one real vision FFN-width vector.
    """
    if len(left) != len(right) or not 0 < len(left) <= 4304:
        raise ValueError("equal vector lengths in 1..4304 required")
    accumulators = [[0.0] * 8 for _ in range(4)]
    boundary = len(left) // 32 * 32
    for start in range(0, boundary, 32):
        for block in range(4):
            for lane in range(8):
                index = start + block * 8 + lane
                product = f32(bf16_rne(left[index]) * bf16_rne(right[index]))
                accumulators[block][lane] = f32(accumulators[block][lane] + product)
    lanes = [f32(f32(accumulators[0][i] + accumulators[2][i]) +
                 f32(accumulators[1][i] + accumulators[3][i])) for i in range(8)]
    four = [f32(lanes[i + 4] + lanes[i]) for i in range(4)]
    even, odd = f32(four[0] + four[2]), f32(four[1] + four[3])
    total = float(f32(even + odd))
    for index in range(boundary, len(left)):
        total += f32(bf16_rne(left[index]) * bf16_rne(right[index]))
    return f32(total)


def norm_scale_reference(centered: float, denominator: float, *, reciprocal_first: bool) -> float:
    """Isolate the port/native operation-order difference, without reductions."""
    centered, denominator = f32(centered), f32(denominator)
    if not math.isfinite(centered) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("finite centered value and positive denominator required")
    if reciprocal_first:
        return f32(centered * f32(1.0 / denominator))
    return f32(centered / denominator)
