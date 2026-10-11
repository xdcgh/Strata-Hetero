# Prefill non-finite diagnostics

This is a diagnostic for the v0.1.42 prefill path. It does not change model
arithmetic, repair NaNs, or establish a performance improvement. Run63 found
an all-NaN head in a failing v0.1.41 request. Run66 reproduced token-0 loops
with unmodified v0.1.42 upstream, without a logits readback. The first faulty
operator is still unknown.

Run69's instrumented v0.1.42 model has four matching first-bad records at
layer index 45, half 1, `moe_output`, row index 154: 2560 NaNs and no infinities.
The long_1k warmup and three formal outputs still loop on ID 0. Fifteen short
formal outputs match H4. This narrows the earliest scanned boundary without
identifying its faulty input or operator.

The diagnostic needs both CMake `-DSTRATA_PREFILL_FINITE_PROBE=ON` and the
exact environment value `STRATA_PREFILL_FINITE_PROBE=1`. Both default to off.
The build option accepts NVIDIA CUDA only; HIP and gfx906 builds reject it.
The separate SYCL prefill source has no instrumentation. With the build
option off, prefill does not include the diagnostic header or any probe call.
With the build option on and the environment option off, the probe makes no
CUDA call or allocation. This is a source contract, not measured binary or
model-output parity.

Each `Prefill::run_impl` invocation owns one record on its existing compute
stream. Read-only scan kernels run after input embedding/handoff, PLE residual
production, each half's hyper-connection mixed input, GDN/QSA output, MoE
combine, and residual writeback. Within the earliest scan that contains a
non-finite value, the record preserves the lowest element index and complete
NaN, positive-infinity, and negative-infinity counts. It also records chunk,
position, layer, half, row count, and row width. IEEE-754 exponent bits are
checked directly. A first bad scan identifies a boundary, not an individual
operator inside that boundary.

The small record is copied on the compute stream before an existing chunk
handoff/callback synchronization, and printed after that synchronization. A
run with no chunk callback reads it at the existing prompt-end synchronization.
An early return drains outstanding probe work before freeing its buffers.
There is no added per-layer host synchronization or device-wide synchronization.

Stderr records start with `strata pf_nonfinite: ` followed by schema-1 JSON.
The `begin` record gives a process-local run identifier and prompt size. Each
`checkpoint` record gives completed chunk coordinates and either `first_bad:
null` or the earliest non-finite scan seen in that invocation. Multiple GPU
stages have separate records; this does not order their failures globally.
Uninstrumented internal activations and KV/PLE/GDN state can fail earlier
than the first scanned output. A finite record is limited to scanned buffers.

Allocations, scan kernels, atomics, record copies, and logging add work. They
can perturb or mask a race even though they preserve activation bytes and
stream ordering. These runs are diagnostic only. A passing instrumented
request does not show that the original failure is fixed.

`prefill_finite_probe_test --disabled-only` tests the disabled branch before
creating a CUDA context. The ordinary synthetic fixture tests finite inputs,
NaN and both infinity signs, first-stage retention, the lowest index across
blocks, scan-before-overwrite ordering, input-byte preservation, and separate
records between runs. It loads no model. Build, fixture, and model receipts
belong under `bench/hetero/20261011-69-prefill-first-nonfinite/`; do not promote
runtime acceptance from source review alone.

For a selected layer, set `STRATA_PREFILL_DETAIL_LAYER=45` in addition to the
compile/runtime options above. The layer value must be decimal and within
the model's layer range. Absent: the original coarse probes only. The detail
path scans the FP16 and BF16 images of the mixed input, FP32 router logits and
weights, and shared expert gate/up, FP16 activation, down output and scalar
gate. It also scans routed products at their producers, and FP16 fallback
inputs/dequantized weights, before they are consumed or overwritten. The
fused path's byte intermediates are never treated as floats. Routed Dm is
scanned after producer waits/copies; peer-sums mode scans only the local
prefix because the peer rows are not written into Dm.

Schema-1 `moe_path` records provide the selected layer's observed native/MMQ/
fused/peer decisions, expert types, stream-all flag and CPU row count. They
are diagnostic metadata, not a non-finite finding. `first_bad` adds `format`,
`row_offset`, `source_row`, and `expert` when available. F32, IEEE binary16,
and BF16 have separate exponent-bit tests. Finite FP32 values can overflow
when converted to FP16. Packed int8/quantized intermediates have no floating
point finiteness scan here.

Routed group rows follow expert order. `source_row` is the group row offset
plus the local scan row, not automatically an original token position. For
dequantized weights the row/column coordinates identify the weight matrix,
with its expert number, not tokens. Mixed/shared/combined outputs use token
rows. Readers must recognize `moe_path`, new stages and typed coordinates
before running a detail experiment. Every JSON record is printed with one
stdio call, and readers still wait for complete LF-terminated lines.

Detail scans add more memory reads, kernel launches and logging, including
dequantized weights of the selected layer's FP16 experts. They can change
timing and mask races. Keep these experiments separate from performance
benchmarks. No detail runtime result is accepted from source review alone.
