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

The F32 helper and native quantized CPU expert are different operators. `tools/hetero_native_activation.py` implements the selected Q4_K/Q5_1 contract from caller-owned arrays: signed-maximum Q8_K input blocks, SiLU, and Q8_1 intermediate blocks. Q8_1 stores its F16 scale and sum separately; Q5_1's affine minimum uses that stored sum. A decoded float matmul therefore needs a per-block sum correction rather than only rounding the activation. The original raw Q5_1 minimum is read from explicitly supplied bytes.

Six pure fixtures verify signed/tie behavior, zero blocks, half scale/sum correction against the native affine dot formula, payload layout, refusal of invalid/nonfinite/overflow inputs and a closed-form zero-gate expert. Default CLI is metadata-only. These fixtures established the contract for the subsequent actual comparisons below; they alone did not enable a helper route.

The subsequent actual-expert experiments use the verified original layer 28/expert 288 blob and decoded weights, original seeded inputs, and frozen native CPU outputs at rows 1/16/256. Each has one warmup and three formals. The NumPy candidate passes all nine formals, with relative RMSE about 2.3e-7..1.18e-4. `tools/hetero_native_openvino.py` then constructs the same activation/affine contract: exact signed-max/tie selection, Q8_K scale, Q8_1 half scale/sum and raw-minimum correction. OpenVINO CPU under F32/ACCURACY passes nine formals, RMSE 2.10e-4..6.36e-4. Arc GPU.0, reported Intel Arc 140T, passes the same nine native-reference formals under F32/ACCURACY, RMSE 2.10e-4..6.92e-4. All remain below the unchanged 1e-3 relative/row-norm gates. CPU/Arc shape and finite checks pass.

These results close the sampled native-operator numerical gap for the selected expert and shapes. They do not establish equivalence for other quantizations or experts, transport-inclusive latency, CPU/Arc contention, full-model token/logit parity or the required 8% model gain. No production helper is enabled. Extend the worker to this identified contract and measure the complete route before changing scheduling defaults.

The service now accepts explicit `--operator native-q4k-q5_1 --native-blob <path>` while retaining its F32 default. Metadata validation checks only NPZ headers, identity metadata and the original blob's file metadata. Explicit serve verifies the three raw slices against the selected expert identity, derives Q5_1 minima from those bytes and hash-loads decoded weights before creating Core. The existing exact GPU.0/F32/ACCURACY checks and SXPU framing remain in force. INIT reports the operator, blob binding and weight readiness. Sixteen service fixtures passed; root also passed the 63 related service, transport, activation and vision fixtures. The default CLI was checked against the real asset metadata without loading tensor payloads or creating Core.

`20261009-42-native-q4k-q5_1-ipc` subsequently ran one identified Arc worker with one warmup and five formals per row count. All 45 formals pass the frozen native CPU reference at unchanged 0.01 maximum-absolute and 0.001 relative-RMSE/row-norm limits. Root independently rehashed the nine retained output files and recomputed those errors in float64; RMSE is 0.000210..0.000823. Each row's five output hashes match its retained output. The service graph actually compiled and inferred on GPU.0 under F32/ACCURACY. Its nested builder record is a snapshot taken before Core compilation, rather than the later service execution status.

| Rows | Native-contract client median RTT ms | Worker median ms |
|---:|---:|---:|
| 1 | 1.0406 | 0.8014 |
| 2 | 1.1541 | 0.8227 |
| 4 | 0.9528 | 0.5573 |
| 8 | 1.3867 | 0.8156 |
| 16 | 1.1788 | 0.6712 |
| 32 | 1.9779 | 1.0153 |
| 64 | 4.3252 | 1.9600 |
| 128 | 10.0494 | 3.7452 |
| 256 | 20.4807 | 6.4198 |

Client timing includes serialization, pipe exchange, validation before the output copy and the owned copy. INIT/compile and warmup are separate. It must be compared with a CPU route covering the same complete boundary before declaring a crossover. Twelve global samples pass the memory floors; the exact launcher, actual C Python service child and conhost are terminal. Eight host fixtures and a real small-artifact preflight preceded execution; the failed interpreter preflight is retained. A standalone profile successfully reloads these nine hash-bound observations, with no fabricated CPU cost and no model acceptance. It rejects the Arc route until model evidence exists. Shared-RAM contention, multi-expert/model integration and the required 8% model gain remain unmeasured; no automatic helper is enabled.
# Persistent native CPU comparison worker

`hetero_native_cpu_service` uses the same bounded SXPU stdio framing as the Arc worker. Its explicit `--serve` path verifies one original Q4_K/Q5_1 expert blob and identity, keeps weights and an optional pool resident, and rebuilds only row buffers and jobs after a shape change. Input copy, native activation quantization, the complete FFN and output serialization are included in its worker timing. Client measurements still need the same complete request boundary on both backends.

The CPU-only build `20261011-52-native-cpu-service-build` built five targets and passed four CTests. All 34 resource observations passed the physical RAM and PSAPI commit floors. Root independently rehashed all five binaries; the service SHA-256 is `486dedeeba69497dc5d3aebf771833278b8431fc29e0e0c33bab503e5a04caf9`. This build runs metadata, framing and affinity fixtures. Actual native IPC output parity, transfer-inclusive CPU/Arc comparisons, memory contention and model gains remain unmeasured by this build.

## Persistent native CPU comparison and paired calls

Run60 measures the same complete caller boundary for CPU10/all, CPU16/none and Arc GPU.0: serialization, framed exchange and owned output copy, with resource/hash/quality/file work excluded. All 135 formal and 27 warmup responses pass native numerical gates. CPU10/all has lower median caller RTT than Arc at each measured row count 1/2/4/8/16/32/64/128/256. At rows 1 and 256 the respective CPU/Arc medians are 0.2706/1.2111 ms and 16.0446/17.5180 ms. No crossover was established in this range.

Run61 executes only the two previously unrun synchronized joint arms against those reviewed run60 solo baselines. All 216 responses pass, including 108 independently replayed Arc comparisons; CPU outputs remain byte-identical to native gold. At rows 256, CPU10/all and Arc joint medians are 19.7789 and 19.5307 ms; CPU16/none and Arc are 27.9015 and 28.6046 ms. The solo and joint arms were not interleaved, and no memory-bandwidth counter was measured. These observations do not establish a full-model speedup or its cause. Arc model dispatch remains disabled by default. See bench/hetero/20261011-61-native-cpu-arc-ipc-joint-compare/DECISIONS.md and comparison.csv.
