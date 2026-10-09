# Native CPU pool parity

The new single-thread path matches the frozen native CPU output for rows 1, 16 and 256. Pools with 1, 4 and 10 background workers, plus the participating host, match both the frozen and new-default outputs byte for byte at all three sizes. The twelve cases each have one warmup and three formal iterations. Root rechecks all output bytes, exact shapes and finite values. All owned child processes exit with code zero and resource gates pass.

This verifies the selected original Q4_K/Q5_1 expert and its existing Q8_K/Q8_1 activation arithmetic. The earlier mismatch with NumPy F32 describes a different operator and is not this reference. Affinity is `none`; no pin-success or hardware P/E classification claim is made. Worker/affinity sweeps, CPU/Arc contention and model-level scheduling improvement remain required before changing defaults.
