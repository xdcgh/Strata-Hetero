# Arc helper: validation status

The primary text engine remains CUDA on the RTX 4090 D. On 2026-10-08 the isolated OpenVINO 2026.4.1 runtime enumerated `GPU.0` as Intel Arc 140T on this Windows host. Device enumeration is not inference or model validation. Upstream's experimental full SYCL engine currently documents a Linux path; it is separate from the requested helper and will not be used for a CUDA layer split.

`tools/hetero_xpu_worker.py` prepares an explicit-device FFN microbenchmark. It accepts identified, hashed gate/up/down float32 matrices and defaults to header validation without creating a device. Its explicit run includes input copies, inference and output retrieval in each latency. Static batches cover 1..256 rows, with separate compilation, warmup and at least three repeats. Device selection excludes AUTO/HETERO fallback. Precision hints and numerical checks are reported separately; a hint is not proof of every operator's precision.

The current FFN NumPy reference is not Strata's quantized CPU kernel. A real expert was extracted from the original UD-Q4_K_XL bytes using pinned official `gguf-py`: profile rank 0, layer 28, expert 288, Q4_K gate/up and Q5_1 down. Gate/up are 640 by 2560, down is 2560 by 640. The selected quantized payload totals 3,072,000 bytes; the decoded float32 matrices total 19,660,800 bytes. Current source-file identities match the prior complete model integrity receipt, and selected payload/decoded hashes are recorded separately.

On 2026-10-09 CPU and Intel Arc GPU.0 OpenVINO F32 tests each ran one warmup and five repeats per batch on those same weights and seeded inputs. All nine sampled batches passed finite/error checks and the reported precision policy; Arc execution-device evidence was exactly GPU.0. Arc was slower at 1, 2 and 4 rows, and faster at the sampled 8..256 row sizes. For example, median wall time including input copy/output retrieval was 0.575 ms CPU versus 0.427 ms Arc at 16 rows, and 1.661 versus 0.881 ms at 64 rows. See `bench/hetero/20261009-02-arc-real-ffn/comparison.json`.

These are single-expert in-process measurements against an OpenVINO F32 CPU reference. They do not include worker transport or Strata's native quantized CPU/Q8 activation arithmetic. A native CPU comparison is recorded below; CPU/Arc shared-memory contention, transport and CUDA integration remain pending. No model route or production crossover is accepted from this microbenchmark.

`20261009-10-native-cpu` now contains a linked Strata native CPU reference, not a substitute float32 implementation. Its original gate/up/down quantized slices total 3,072,000 bytes and match the earlier scoped hashes. Nine identified input files follow the previous seed-42 recipe. Each shape ran one warmup and five measured native iterations. This is a single CPU thread, with Q8_K activation quantization, Q4_K gate/up, SiLU-times-up, Q8_1 intermediate requantization and Q5_1 down; it does not use the engine worker pool.

Native medians were 0.1205/0.2418/0.4747/0.9458/1.9525/3.9053/7.8599/17.8075/34.5937 ms for 1/2/4/8/16/32/64/128/256 rows. These intervals exclude file I/O and allocations; the earlier Arc intervals include input copy/output retrieval. Descriptive timing ratios are retained, but are not a matched model-speedup claim. All native outputs were finite. Their relative RMSE against dequantized-weight NumPy F32 was 0.01107..0.01298, above the unchanged 0.001 limit, while max absolute error 0.000228..0.000503 passed. This measures the difference between native quantized activations and a float32 FFN; it does not establish a native model failure. The current float32 Arc result cannot be treated as an interchangeable native expert output. Quantization alignment, worker transport, contention and model quality remain required before routing.

The next route decision requires real expert bytes and matching activations, finite outputs and error statistics, CPU and Arc timings including transport, and combined contention samples. Prefill precedes tiny decode jobs. Retain a helper route only after the affected model workload improves by at least 8%; automatic selection must avoid a stable regression over 2%. Unavailable, incorrect or slower routes remain disabled in the profile. This document makes no Arc model-performance claim.

The standalone service in `tools/hetero_xpu_service.py` uses the bounded SXPU pipe transport. Default validation reads only identity/NPZ headers. Explicit serve mode selects Intel Arc GPU.0 with F32/ACCURACY hints, verifies execution and precision properties, and keeps one static-row graph at a time. INIT binds the fixed expert identity and a nonce; INFER accepts exact F32 row bytes and returns finite copied output with a worker timer. Failed initialization invalidates readiness, framed errors stay bounded valid JSON, and stdout is reserved for the binary protocol before device creation. Twenty-seven fake transport/service fixtures passed, including owned-child cleanup, blocked-write timeout and failed-init refusal.

The actual pipe experiment `20261009-19-arc-ipc-ffn` then ran the same identified expert and input bytes through one separate Arc service. Each of nine shapes had one warmup and five formal requests; all 45 formal results passed the unchanged float32 reference thresholds. Runtime execution was GPU.0 / Intel Arc 140T and reported precision matched F32. Client RTT starts before input serialization and ends after its copied output array; worker time covers its input copy, inference and output serialization. Compilation and INIT are separate.

| Rows | Client median RTT ms | Worker median ms |
|---:|---:|---:|
| 1 | 0.7395 | 0.4429 |
| 2 | 0.7146 | 0.4167 |
| 4 | 0.6882 | 0.4092 |
| 8 | 1.0478 | 0.4900 |
| 16 | 0.9065 | 0.4678 |
| 32 | 1.8752 | 0.7198 |
| 64 | 3.9998 | 1.4738 |
| 128 | 8.6461 | 3.1855 |
| 256 | 17.6591 | 5.2189 |

The duration difference is descriptive serialization/transport/client overhead, not a measured wire-only timer. All nine native-quantized CPU comparisons still fail the 0.001 relative-error limit; no limits were relaxed. Consequently this establishes a functioning, transport-measured float32 helper, not native arithmetic parity, a model route, a native pool crossover or the required model gain. CPU/Arc contention and quantization alignment remain pending. Six system resource samples passed the 12/4 GiB floors; the worker launcher, actual Python child and sampler tree were all confirmed exited in `terminal-audit.json`.
# Native activation contract preparation

The F32 helper and native quantized CPU expert are different operators. `tools/hetero_native_activation.py` prepares the actual selected Q4_K/Q5_1 contract from caller-owned arrays: signed-maximum Q8_K input blocks, SiLU, and Q8_1 intermediate blocks. Q8_1 stores its F16 scale and sum separately; Q5_1's affine minimum uses that stored sum. A decoded float matmul therefore needs a per-block sum correction rather than only rounding the activation. The original raw Q5_1 minimum is read from explicitly supplied bytes.

Six pure fixtures verify signed/tie behavior, zero blocks, half scale/sum correction against the native affine dot formula, payload layout, refusal of invalid/nonfinite/overflow inputs and a closed-form zero-gate expert. Default CLI is metadata-only. The functions have not yet been compared with the actual native outputs or ported to OpenVINO, and do not enable a helper route. This preserves the existing numerical tolerances and leaves native reduction/SiLU rounding as measured gates.
