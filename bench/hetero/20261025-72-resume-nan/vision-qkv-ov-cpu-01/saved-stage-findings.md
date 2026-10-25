# Retained CPU stage diagnosis

`compare_saved_stages.py` reads the frozen native and prior instrumented
OpenVINO CPU files, verifies their file/decoded hashes and compares three
selected full rows at four stages. It imports no array/device runtime and
does not run an encoder. The 32 comparisons are in `saved-stages-review.json`.

For the rounded policy, the three square fixtures have 0, 0 and 1 differing
BF16 buckets among 3,456 selected first-layer-norm values each. Their selected
full-width QKV relative RMSE is 1.47e-7..2.41e-7, while the first block output
is 3.72e-4..6.44e-4. This narrows the larger first-block gap on those selected
rows to operations after QKV; it does not identify attention, GELU or a
specific matmul as its cause.

The rectangular gradient already differs earlier: 466 of 3,456 selected
first-layer-norm values change BF16 bucket under either prior policy. Its
rounded QKV relative RMSE is 2.21e-4 and first-block output is 9.28e-4.
Pre-QKV input differences therefore remain relevant for this image.

The tiny OpenVINO replay in the neighboring receipt uses native layer-norm
inputs and only seven QKV output columns. It isolates a different geometry;
its small error does not establish a repaired full encoder. The prior full
instrumented encoder still fails its original final numerical gate. All
native oracle files and quality thresholds remain unchanged.

Next useful checks are identified captures within block zero (post-attention,
second norm, FFN pre/post-GELU), plus positioned/patch inputs for the rectangle.
Any new tap helper must first reproduce the frozen full native SVE outputs.
NPU execution, visual task quality and latency remain unaccepted.
