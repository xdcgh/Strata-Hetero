# Opt-in stage trace draft for root review

Status: implemented locally; native compilation, C++ fixture execution and native/OpenVINO stage runtime are **NOT RUN**. Two host-only Python fixtures passed. No GGUF payload, installed OpenVINO, Core or device was used for this implementation review.

The helper extension adds `--enable-stage-taps`. Without it, normal `ENC` keeps its existing callback-free graph path, SVE1 bytes and response. With it, CPU-only `ENC_TAPS <image> <output> <prefix>` captures stages; regular `ENC` remains unarmed and is used for warmups. The output/prefix tokens follow the existing no-space protocol constraint. Prefix means a new exclusive directory, with an existing parent. Captures refuse existing prefix directories, stage files and SVE output files.

`tools/vision/vision_stage_trace.hpp` is 172 lines. It registers no global state, keeps no tensor pointers, and uses the existing `tools/hetero_native_cpu/sha256.hpp` for each raw stage hash. Every capture has 11 whitelisted files (13NE F32 values) plus a manifest. The manifest records source revision, selector/target names, actual F32 type ID, all four target and selector `ne`/`nb`, absolute paths, raw hashes, capture/aggregate byte counts and capture ordinal. Its `stages_captured` status describes stages only; caller acceptance still requires the helper's successful SVE response and independent final-SVE validation.

Guards check expected N=36/72, E=1152, F32, all four strides, allocation and the contiguous combined-QKV ADD-after-MUL_MAT parent. Merger capture validates its zero-offset post-norm reshape parent. Duplicate selectors, missing coverage, unexpected metadata, the 16 MiB aggregate cap, more than three formal captures for a canonical image, existing paths and write/close errors become sticky errors and prevent `OK`. Byte accounting includes failed partial writes. Callback exceptions are caught before returning through the C ABI; capture data and metadata are owned copies. No NN/ggml graph-source operation was changed.

The native helper diff is small and leaves normal SVE encoding unchanged. Tap SVE creation is exclusive and additionally checks close errors. A completed manifest is written before the SVE; any subsequent SVE failure still returns `ERR`, and the caller must retain that failure rather than treating the stages as accepted.

OpenVINO adds only optional aliases `ln1.0`, `ln1.1`, `ln1.26` and `qkv.0`. They refer to the same existing norm/QKV nodes; norm, affine projection and split execute once in the same order. The comparator's JSON-only tap whitelist accepts those aliases. Source math, default policies, tolerances and production routes are unchanged.

Validation performed:

- Host fake-OpenVINO alias fixture: projection is exactly equal with/without aliases; LN1 and QKV match independent small synthetic expectations.
- JSON-only alias contract fixture: selected names validate without payload/runtime access; an unselected `ln1.2` is rejected.
- `git diff --check` passed for the changed tracked files.

`tests/core/vision_stage_trace_test.cpp` is a standalone fake-GGML fixture source, not yet compiled or run. It defines the single copy API as a synthetic zero-byte fill and needs only the public GGML include headers, C++17 and standard libraries; do not link a real backend. It covers normal unarmed mode, all 11 selectors, full QKV-parent byte shape, merger shape/byte total, three-capture quota, duplicate/missing coverage, pre-read budget failure, noncontiguous/wrong-type/unallocated metadata, non-ADD QKV parent, reserved-file/SVE-output protection and existing-prefix refusal. It creates a new caller-owned fixture directory and preserves its files; it reads no GGUF and performs no model math.

Root review precedes compilation and execution. Future runs must use the verified actual worker/descendant tree guard, unchanged 12/4 gates, pinned source/build/input identities and CPU/flash-attn-off reference settings. Instrumented final native/OpenVINO embeddings must match the frozen fingerprints before any stage comparison is trusted. No NPU or vision API work is enabled.
