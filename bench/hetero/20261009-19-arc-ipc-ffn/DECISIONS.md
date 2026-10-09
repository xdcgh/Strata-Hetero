# Arc pipe transport measurement

One exact Intel Arc GPU.0 service completed nine static shapes with one warmup and five formal requests each. All 63 request IDs were consecutive, and all 45 float32 reference comparisons passed. Every shape recorded exact execution-device and precision-policy evidence, immutable input/output hashes, and a verified last-output file. Compilation/INIT, worker duration and full client RTT remain distinct.

Client RTT includes input serialization through copied output array; hashing and numerical checks are outside that timer. At 256 rows the medians are 17.6591 ms client and 5.2189 ms worker, showing substantial host/pipe overhead. A shared-memory transport may be worth testing, but is not implemented or accepted by this result.

Comparisons with the single-thread native Q4_K/Q5_1 plus Q8 activation pipeline remain informational. All shapes fail the unchanged native relative-error gate; the float32 helper is not interchangeable with that operator. Native CPU pool, concurrent memory contention and main-model 8% gain / automatic 2% regression gates are unrun. Keep CUDA/CPU routing unchanged.

The client preparation retained errors and review corrections: explicit tolerance arguments did not repair an API failure because the helper already supplied the same defaults. Actual import-path, precision representation, early-failure cleanup and RTT boundary issues were corrected and preflighted before the device run. Cleanup recorded the terminated venv launcher return code and separately verified the actual Python child and sampler descendants exited.
