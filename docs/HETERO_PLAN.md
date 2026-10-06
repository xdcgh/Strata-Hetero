# Strata-Hetero engineering plan

The aim is lower end-to-end latency, time to first token, startup time and P95 latency, and higher prompt, output and concurrent throughput on the GTi15. Device utilization alone is not an acceptance measure. The NVIDIA card remains the primary compute device. RAM precedes storage for hot data. Every change must preserve the model's computation and pass its quality gate before performance acceptance.

## Source and working boundaries

- Working checkout: `C:\Users\DC\Documents\ChatGPT\Strata-Hetero`.
- Origin: `git@github.com:xdcgh/Strata-Hetero.git`.
- Upstream: `https://github.com/Niko1221/Strata.git`.
- Initial upstream and fork base, fetched on 2026-10-06: `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`.
- Historical checkout: `C:\Users\DC\Documents\ChatGPT\Strata`, commit `6f32ec070f23ced9f50e704d854d775da52591ab`, engine 0.1.39. Its tracked source, installed environment and untracked benchmark artifacts are retained as a reference.
- Model sources are existing, integrity-verified files in `F:\Strata-data`. Copies to E must be hash verified before use. There is no authorization to remove unrelated files, replace the unique model, change drivers or terminate user applications.

The user requested a main architecture/review agent and GPT-6 Luna agents for bounded inventory, tests, benchmark execution and statistics. The main agent reviews code and evidence before merging. An unavailable worker is retried or its task is completed by the main agent; capacity failures are retained as coordination evidence.

## Initial facts and hypotheses

The first read-only Windows inventory reports Intel Core Ultra 9 285H (16 cores/threads), 127.56 GiB physical RAM, RTX 4090 D reporting 49,140 MiB VRAM, Arc 140T and Intel AI Boost devices. E is a Micron NVMe, F a USB SSD. At this snapshot E/F have about 192.5/341.0 GB free. These values are observations, not launch reservations; refresh them immediately before a costly run. No Strata, Python model or ComfyUI process was found in this Windows snapshot. WSL and device users need their own checks.

Upstream already has resident expert complements, a bounded PLE row cache, KV streaming, CPU core affinity, per-stage timings, MTP verification and RAM conversation snapshots. Windows `--ple-io ram` is explicitly refused. Intel SYCL is an independent experimental engine, not a CUDA helper. Arc 140T, NPU vision, storage placement and application striping need new validation. Reuse existing capabilities before adding competing implementations.

The first hypothesis is that the resident expert complement plus a RAM PLE table can fit with a useful context and system reserve on this host. The historical Q4 configuration reported 36.52 GiB resident CPU experts and 35.21 GiB GPU experts; the IQ4_NL PLE table needs about 26.82 GiB. This is a capacity estimate, not a demonstrated faster configuration. File cache, commit headroom, staging buffers, KV, MTP and other applications must all be measured.

## Stages and acceptance

| Phase | Deliverable | Validation before enabling |
|---|---|---|
| 0 | Source-grounded [data flow](HETERO_DATAFLOW.md), hardware identity, historical/current/hetero baseline, reproducible harness | Immutable source/config/engine/prompt identities; validate harness without a model, then admitted smoke and repeated baseline |
| 1 | RAM-first experts, PLE cache/full RAM/partial RAM, active KV in VRAM/RAM | Row byte/dequant parity; greedy, logits and long text; memory/commit pressure; startup and decode A/B |
| 2 | Physical-device profiler, workload placement, external PLE, parallel startup and bounded async reads | Copy/hash verification, random/sequential/QD latency profile, physical-device contention; affected workload gain at least 10% |
| 3 | CPU topology and worker role/affinity separation | CPU expert correctness, hybrid-core detection, repeated gain at least 5% |
| 4 | Independent `strata-xpu-worker`, Arc microbench, prefill then decode helper | Transfer-inclusive crossover, concurrent CPU/Arc memory contention, main workload gain at least 8%; auto mode avoids stable regression over 2% |
| 5 | NPU vision and measured static auxiliary candidates | Real device availability, exact model/preprocessing identity, quality parity; faster vision or lower GPU VRAM |
| 6 | Inventory -> measurements -> cost model -> placement -> scheduler | Phase-specific plans, transfer/sync-aware critical-path cost, runtime calibration, fail-safe fallback |
| 7 | `--hetero-calibrate`, profile persistence, final comparative benchmark | Repeat 3+, warmup, all quality gates, no unsupported settings enabled by default, rollback verified |

A measured negative result is a completed experiment: retain artifacts, state its scope in `DECISIONS.md` and leave that optional route disabled. A missing backend, unexecuted experiment or harness-only test is not a measured negative result.

## Benchmark contract

Use fixed model bytes, tokenizer, draft vocabulary, prompts, seed/sampling, context, KV type and expert placement controls. Measure natural-text prefill at 1K/4K/16K/32K and output at 256/512/1024 tokens. Extend natural text to 128K/256K/512K/1M only after context admission. Include NIAH, RULER, Pelican, code and repetition/NaN/Inf checks. Record actual prompt/output tokens, timeout/truncation and prompt-cache state.

Keep cold startup, warm startup, model-to-RAM, expert cache fill, PLE init, MTP init, server ready and first token separate. A file-cache-unknown restart is not a cold start. Collect RAM/commit, VRAM, GPU idle/power, Arc/NPU availability, disk read bytes/utilization and client latency. Report P95 and concurrent throughput separately from single-request output speed.

Each arm has a warmup and at least three repetitions. Interleave A/B when possible. Report all samples, median/mean/spread and confidence limits; do not claim a percentage from one sample or unlike workloads. Strict token equality requires fixed placement and deterministic controls, because upstream's CPU and GPU kernels can round differently. A placement optimization also needs logits/quality checks where those devices change arithmetic.

## Git and evidence

Each tested logical step is committed and pushed. Use phase branches with the `codex/` prefix, retain base SHA and rollback points, and fetch upstream before phase transitions. Do not force-push or replace the fork's main history. No secrets, full model copies or machine credentials go into Git.

Run artifacts live in `bench/hetero/<run>/`: hardware/storage/baseline JSON, results JSON/CSV, resource samples, logs, comparison and decisions. Raw large outputs may remain local with a tracked SHA/size manifest. Every retry has a new RunId. Preserve rejected launches and failed builds. Project state records phase, branch, commit/base, validated conclusions, hypotheses, rollback point and next experiment in `bench/hetero/STATE.json`.

## Final acceptance questions

The [benchmark report](HETERO_BENCHMARK.md) must answer RAM-first gain, full-RAM PLE, E/F roles, multi-disk startup gain, CPU gain, Arc prefill/decode value, NPU vision/auxiliary value, NVIDIA idle reduction, remaining bottleneck, best 32K/128K/256K/512K/1M configurations, daily/max-performance configurations and total gain over the historical 0.1.39 configuration. Missing measurements remain explicitly unverified. Final human acceptance belongs to the user.
