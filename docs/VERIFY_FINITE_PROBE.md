# Verifier finite probe

This is optional CUDA diagnostic source. It is not a NaN fix or model-quality
result. Both the CMake option `STRATA_VERIFY_FINITE_PROBE` and the runtime
environment variable of the same name default off. Build the option ON and set
the variable to exactly `1` before process startup to enable it. Off/auto/0/unset
do not enable runtime scans. HIP, gfx906 and non-CUDA builds reject the build
option; their default code paths remain unchanged. SYCL sources are untouched.

The independent `strata_verify_finite_probe` library only depends on CUDA runtime;
it adds no engine/prefill dependency cycle. The Verifier allocates its device
record and pinned receipt outside graph capture. Every captured window begins
with a device reset and ends with a small captured receipt copy. There is one
owner per Verifier/device, including pipeline parity verifiers. The host reports
the receipt only after the existing run synchronization, batch synchronization or
completion query, or pipelined done/finish boundary. No per-GEMM host waits are
added. Graphs are destroyed before the record is released and its stream dies.

Coarse scans cover initial materialized residual, both HC mixed outputs,
attention output, router logits and final route weights, shared output after the
existing side-stream join, actual combined CPU/GPU parts, MoE output, head mixed
and head logits. Deferred scalar-gate scratch is scanned only when the combine
consumes it (sg_gated). The BF16 scalar-gate input image is scanned only in the
nondeferred, non-native-BF16 branch that produces and consumes it. All successful
head branches produce head_mixed: canonical lm_head calls lm_head_mix with that
output pointer before quantizing it. No packed Q8 bytes are interpreted as
floats. All main scans use FP32; the shared image uses BF16. The utility also
supports IEEE FP16 without numeric conversion. Fused math, resident/device-plan
shortcuts, sampling and cache allocation decisions are not disabled.

The first nonfinite **checkpoint in compute-stream order** wins. Within that
checkpoint the lowest element index and full NaN/+Inf/-Inf census are retained.
With split groups this order is not a global sort by layer/token. A later
checkpoint cannot overwrite the first record. A finite receipt means the scanned
boundaries were finite, not that every internal intermediate or output was
correct. A finite repetitive output is still a quality failure.

Each `strata verify_nonfinite:` line contains one JSON receipt with device,
window counter, runtime position metadata, checkpoint ordinal/count, layer, half,
stage, format, scan shape, local/global row, token row and census. Position is
joined on the host from the actual replay's last_pos0 or batch positions; it is
never baked into capture. `routed_parts` rows are token-major K-slot rows, with
rows_per_token=K, not expert-order rows. Initial residual is the materialized
buffer; fused pending residual writes remain pending until their normal producer
applies them. A first bad boundary is not automatically the originating producer.

The probe reads activation memory but changes kernel count and timing. It may
perturb a race. Symptom disappearance under instrumentation is not proof of a
repair. Enabled scans, allocations and logs also affect latency and available
memory, so instrumented results are not performance acceptance measurements.

`verify_finite_probe_graph_test` is an explicit standalone target whenever the
probe build option is ON, even with global STRATA_BUILD_TESTS=OFF. Test-suite
registration remains conditional on the global flag. It is synthetic source:
it captures one graph and
replays it across FP32, FP16, BF16 and finite cases, checks reset/no leaked first
bad/census/index, and verifies runtime positions change on the same graph and
batch positions need not be consecutive. `--disabled-only` follows the disabled
API path without making CUDA calls. It does not load a model. This implementation
has not been built or run by its author; a clean diff check is source hygiene,
not runtime validation. Whole-model NaN diagnosis and upstream comparison remain
separate required work.
