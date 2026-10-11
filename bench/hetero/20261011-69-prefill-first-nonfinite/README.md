# Run69 first-nonfinite prefill diagnostic

The CUDA build/fixture gates pass. The model diagnostic reproduces the long_1k failure and identifies the first non-finite instrumented boundary at layer index 45, MoE output. No numerical fix is established.

- Diagnostic source: isolated branch `codex/69-prefill-first-nonfinite`, commit `1eb85d2a5d27cb2f054a9c757c3b074d7d3d7c86`; source worktree stays clean. This is an opt-in diagnostic, not a numerical fix or accepted optimization.
- Removing all 15 compile-guard blocks from prefill gives LF-normalized bytes identical to the pinned `b50ba770` file. This establishes source preservation with the build option off; binary/model parity has not been measured.
- CUDA build: Release, SM89, pinned CUDA 13.3.73 / MSVC 19.51.36260.0 / GGML 3cf03257. Configure and build exit 0. `STRATA_PREFILL_FINITE_PROBE=ON`; global tests remain OFF, with the single diagnostic fixture target built explicitly.
- Engine: 52,784,640 bytes, SHA256 `96b67b604e6b3b86f2a0bb865d9ffcc87be0a30232a9cfddb805387414243b83`.
- Fixture: 206,336 bytes, SHA256 `3495849bd3c8e002cde9a0d92e54fa115994026e576a8738a7aece822c2841cb`.
- All 436 actual build resource rows pass 12 GiB physical / 4 GiB PSAPI system commit. Minima 104.032 / 110.540 GiB. The immutable driver receipt reports 435 rows because its summary precedes the final sampler row; the actual row count is retained separately.
- Root independently rehashed all 55 copied build metadata/log files (9,650,084 bytes). No executable, object, debug symbol, or model payload is copied into Git evidence.
- The synthetic CUDA fixture passes 60 device cases across three repetitions. Root parsed 60 begin records and 114 checkpoint records (36 finite, 78 nonfinite), validating schema, first-stage/minimum-index/count fields, and independent run identifiers. The fixture itself checks unchanged input bytes and scan-before-overwrite ordering.
- Both disabled-only and synthetic fixture processes exit 0. The two actual fixture resource samples pass, with minima 109.449 / 116.326 GiB; these samples are not a peak-memory measurement. All recorded exact worker/sampler identities are terminal.
- Static lifecycle review found a missing fixed-executable guard and an unvalidated cleanup identity. Both were repaired before execution. Thirteen extracted-function mock assertions pass without a real process stop. Failed preliminary checker/metadata attempts are retained.

HIP/SYCL builds are not run. The extra translation unit is NVIDIA-only and guarded off by default; SYCL source is untouched. No throughput, latency, or full-model quality improvement is claimed. Probe allocations/kernels/atomics/readback/logging can perturb or mask races.

The model gate used the same five short prompts followed by natural long_1k, warmup 1 and formal 3. Root independently reparsed all 15 short formal SSE outputs: actual token IDs match H4 and visible UTF-8 bytes match the saved outputs. Long_1k warmup and all three formals each emit 128 actual ID-0 tokens and 128 bang bytes, finishing at the length cap.

All four long prefill records agree: ordinal 277, layer 45, half 1, stage moe_output, chunk 0 / pos0 0, T 1035, width 2560, index 394240 = row 154 / column 0. That checkpoint has exactly 2560 NaNs and no infinities. Indices are zero-based. Earlier scanned outputs were finite; this does not establish that every unscanned intermediate was finite. All 28 sampled head rows have 248320/248320 NaNs, finite count 0, finite argmax -1.

Strict quality stopped at long_1k. All 780 resource samples pass the physical/PSAPI gates, minima 61.974 / 42.551 GiB; one additional terminal row confirms sampler STOP. Cleanup exit 0, API loaded false / in-flight 0 / 24 requests, all exact owners and listener 8081 released.

The initial outer launch capture waited for inherited pipe EOF despite a successful inner launch. Root verified and adopted five exact active owners, explicitly forbade relaunch, continued on authorization 02, and closed the original capture after cleanup. The unrelated process that later reused a watch PID was left untouched. Source-only host prep mistakes and failed fixture runs remain in evidence; no extra model attempt was made.

Next: scan layer45 MoE routing/shared/routed inputs and reduction, including FP16 activation images separately from FP32 buffers. No unchanged costly retry; later long prompts and performance phases remain unrun.
