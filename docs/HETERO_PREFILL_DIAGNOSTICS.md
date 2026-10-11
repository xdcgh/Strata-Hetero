# Prefill non-finite diagnostics

This is a diagnostic for the v0.1.42 prefill path. It does not change model
arithmetic, repair NaNs, or establish a performance improvement. Run63 found
an all-NaN head in a failing v0.1.41 request. Run66 reproduced token-0 loops
with unmodified v0.1.42 upstream, without a logits readback. The first faulty
operator is still unknown.

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
