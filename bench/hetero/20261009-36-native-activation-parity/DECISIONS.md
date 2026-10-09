# Native activation algebra check

The original quantized blob, source down minimums, decoded weights, seeded inputs and frozen actual native CPU outputs are hash-bound. At rows 1/16/256, one warmup and three formals each pass the unchanged absolute/relative/row-norm thresholds. Relative RMSE spans about 2.3e-7..1.18e-4. The correction preserves the separately stored Q8_1 half sum used by Q5_1 affine dot products.

This accepts the sampled algebra as a candidate for device implementation. It is not a whole-model quality result or a performance comparison; process transport and scheduling remain separate gates.
