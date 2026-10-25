# Verifier routed-output snapshot

This is an optional diagnostic, built only with `STRATA_VERIFY_FINITE_PROBE=ON`.
Set both `STRATA_VERIFY_FINITE_PROBE=1` and
`STRATA_VERIFY_ROUTE_SNAPSHOT_PREFIX=<existing run directory>/route` before
starting the engine. The prefix's parent directory must exist. Use a fresh
prefix for each process. The ordinary finite-probe JSON schema is unchanged.
No arithmetic, routing, sampling, or placement setting is changed.

The extra allocation and transfers are absent when the prefix is unset. An
entirely disabled finite probe makes no CUDA calls, even with a prefix supplied.
The current bound is an embedding width of 4096, divisible by 32. Unsupported
diagnostic shapes fail rather than truncating the captured row.

After a routed-parts scan finds the window's first bad checkpoint, one kernel
on the existing compute stream captures its group-local row before the next
layer can overwrite the input, router, or plan buffers. It copies the selected
expert ID, routing weight, plan counts and destination membership, mixed FP32
input, native packed Q8_1 input when that branch produced it, actual routed
part, and only the applicable raw output buffer. The grouped GPU destination
list contains local flattened token/slot rows, not expert IDs.

`gpu_hit_valid` identifies a row actually written into `hit_out`. All-resident
mode writes directly to `parts`, so its `gpu_hit_valid` is false.
`mapped_non_gpu_valid` identifies a row read from the mapped non-GPU output.
This alone does not prove CPU computation: the generic pool may use a peer or
remote backend, or skip a tail. `cpu_producer_proven` is always false. `remote`
reports only the verifier's optimized remote helper. The run's effective pool,
peer, remote, and tail-skip controls must establish the final attribution.
Unwritten scratch rows are neither copied nor reported as valid producers.

Input validity follows the producer paths, rather than the preceding finite
scan. `gr_read_group` passes `mixed_ + t*N` as `FusedGrArgs::mixed`; every
selected fused up/read kernel materializes that FP32 array. The optional
`gr_q8_tail` reads the materialized FP32 values and writes `q8_mixed`.
`gr_read_group(1)` targets `nat_xq_` for a native pack and returns whether this
fused Q8 image was written. If not, `pre` writes `nat_xq_` with
`quantize_q8_1_rows`: either in the QDEDUP branch or at the end of the native
branch. These paths precede `post` in all-resident and host-plan modes. The
same native buffer is then passed to `native_expert_grouped`. Non-native
packs report `native_q8_valid=false` and do not form or read its pointer.

The small original record and a separate persistent snapshot are allocated
before graph capture. A replay resets the snapshot's valid bit. A graph-tail
D2H copy uses pinned storage; file output occurs only after the existing
completion boundary. Runtime positions are taken from the completed window,
including explicit per-row positions, rather than captured graph constants.

For each captured window, the base name is
`<prefix>.owner-<owner>.window-<window>`. Files are created exclusively:

* `.mixed.f32` and `.parts.f32`: `width` native-endian FP32 values.
* `.native.q8_1`: when valid, `width / 32` blocks of 36 bytes, each with two
  FP16 fields (`d`, `s`) followed by 32 signed quantized bytes. This is the
  native grouped expert input, not its hidden activation or the CPU activation
  quantizer's output.
* `.gpu-hit.f32` or `.mapped-non-gpu.f32`: only when its validity flag is true.
* `.json`: completion metadata, written last, then logged with the prefix
  `strata verify_route_snapshot: `.

The metadata includes finite-input maximum magnitude and nonfinite count, plus
a typed census of the native Q8_1 FP16 header fields. Raw files retain the
original bytes. The metadata is not a file hash manifest; the harness should
hash all captured files and bind them to its source/binary/config identities.
An existing filename, invalid plan, or output error fails the verifier call.
Partial evidence is retained. Diagnostic output can perturb timing.

`verify_route_snapshot_graph_test --prefix <fresh run prefix>` is a synthetic
CUDA graph fixture. It exercises selected IDs and slots, mapped/GPU membership,
producer overwrite after capture, packed-byte preservation, replay reset,
runtime positions, invalid plan bounds, and exclusive file creation.
`--disabled-only` exercises the path that makes no CUDA calls. The source-only
checks are `python tools/test_verify_route_snapshot_source.py`; they do not
establish CUDA runtime or real-model correctness.
