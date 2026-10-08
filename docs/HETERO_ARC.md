# Arc helper: validation status

The primary text engine remains CUDA on the RTX 4090 D. On 2026-10-08 the isolated OpenVINO 2026.4.1 runtime enumerated `GPU.0` as Intel Arc 140T on this Windows host. Device enumeration is not inference or model validation. Upstream's experimental full SYCL engine currently documents a Linux path; it is separate from the requested helper and will not be used for a CUDA layer split.

`tools/hetero_xpu_worker.py` prepares an explicit-device FFN microbenchmark. It accepts identified, hashed gate/up/down float32 matrices and defaults to header validation without creating a device. Its explicit run includes input copies, inference and output retrieval in each latency. Static batches cover 1..256 rows, with separate compilation, warmup and at least three repeats. Device selection excludes AUTO/HETERO fallback. Precision hints and numerical checks are reported separately; a hint is not proof of every operator's precision.

The current FFN NumPy reference is not Strata's quantized CPU kernel. Actual expert extraction, native CPU comparison, Arc runs, CPU/Arc shared-memory contention, transport and CUDA integration remain pending. Synthetic results cannot establish a model speedup or a production crossover.

The next route decision requires real expert bytes and matching activations, finite outputs and error statistics, CPU and Arc timings including transport, and combined contention samples. Prefill precedes tiny decode jobs. Retain a helper route only after the affected model workload improves by at least 8%; automatic selection must avoid a stable regression over 2%. Unavailable, incorrect or slower routes remain disabled in the profile. This document makes no Arc model-performance claim.
