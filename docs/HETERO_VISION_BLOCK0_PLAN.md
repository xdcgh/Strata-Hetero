# Block-zero tap preparation

This is source preparation. The new profile and fixture additions have not been
compiled or executed. They establish no repaired encoder, NPU quality or gain.
The earlier arithmetic/QKV commit has separate retained test and replay receipts;
those receipts do not validate the later tap changes.

The retained selected-row comparison narrows the rounded-policy square-image gap
to operations after QKV. The rectangle also has earlier activation differences:
466 of 3,456 selected LN0 elements change BF16 bucket. Neither finding identifies
the responsible operation or establishes full-tensor accuracy.

The helper keeps its original profile by default. A separate CPU-only invocation
may select `--enable-stage-taps --stage-tap-profile block0`. Its nine selectors
come from pinned `qwen3vl.cpp` and `clip.cpp::build_ffn`, without modifying the
pinned source or adding model operators:

| Native callback | Port tap | Features per patch |
| --- | --- | ---: |
| ln1-0 | ln1.0 | 1152 |
| Qcur-0 combined parent | qkv.0 | 3456 |
| attn_out-0 | attn_out.0 | 1152 |
| ffn_inp-0 | ffn_inp.0 | 1152 |
| ffn_inp_normed-0 | ln2.0 | 1152 |
| ffn_up_b-0 | ffn_pre_gelu.0 | 4304 |
| ffn_gelu-0 | ffn_post_gelu.0 | 4304 |
| ffn_out-0 | ffn_out.0 | 1152 |
| layer_out-0 | block.0 | 1152 |

The profile replaces the original capture list rather than extending it. For
the 72-patch rectangle it writes 5,465,088 bytes per capture, or 16,395,264 bytes
across three captures, below the unchanged 16,777,216-byte per-image cap.
The same source enforces exclusive paths, contiguous F32 tensors, complete
coverage, at most three captures, and charging of partial failed writes.

Before any real capture, review and compile the synthetic fixture and the CPU
helper against pinned ggml. Execute the fixture in a new owned directory. Then
freeze helper/source hashes and use a fresh RunId. Preserve failures; do not
retry automatically. No execution is authorized by this document itself.

For a separately admitted capture window, run normal ENC controls and the new
profile on the same four frozen image inputs. Compare every complete SVE against
the original normal-ENC oracle before interpreting any stage. An instrumentation
fingerprint change invalidates the stage explanation; it never replaces the
oracle. Keep memory admission and exact owned-process terminal checks.

Only after native fingerprints pass, request the matching nine port taps under
both existing policies on CPU, with the original properties and warm1/formal3
schedule. Added OV outputs have previously changed projection fingerprints;
record those changes and retain original uninstrumented output comparisons.
Apply unchanged 0.01 absolute and 0.001 relative-RMSE/row-norm thresholds to full
arrays, without accepting selected points as a replacement.

Inspect the first failing boundary in order: QKV, post-attention, attention
residual, LN2, FFN pre-GELU, post-GELU, FFN output, final residual. A GELU jump
requires a same-input replay of native LUT semantics before changing that math.
A matmul jump requires the same decoded weights and identical incoming values.
For the rectangle, separately replay existing patch/position/LN0 captures and
count BF16 bucket changes before attributing its block gap to later operators.
The original profile remains available for those earlier stages. NPU stays off.
