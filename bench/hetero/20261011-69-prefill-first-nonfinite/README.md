# Run69 first-nonfinite prefill diagnostic

Source-only design and the CUDA build/fixture gates are complete. Model diagnosis is not yet run.

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

Next: finish the one-side model-preparation host gates, freeze config/identity/provenance/scripts, perform strict fresh admission, run the original five short prompts followed by natural long_1k (warmup 1, formal 3), preserve both prefill and zero-head diagnostics, then exact sampler-first cleanup. No unchanged costly retry; later long prompts and performance phases remain unrun.
