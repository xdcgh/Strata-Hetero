# Upstream v0.1.42 review for run51 quality degeneration

Date: 2026-10-11  
Scope: read-only source, tag, documentation, and preserved-run artifact review. No merge, source edits, build, test, or model/device call was made. This report is the only new file in this review.

## Source pins and checkout state

- Prior release base: v0.1.41 / fb58e0dbc8399662c0e47c76578c6e878b14f6cf (stage-buffer pinning made opt-in).
- Fetched upstream release: v0.1.42 / 61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406, tag v0.1.42, upstream/main.
- Compared range: fb58e0dbc8399662c0e47c76578c6e878b14f6cf..61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406. The range contains 661 paths, 162,449 insertions and 1,041 deletions; much of this is benchmark and provenance material.
- Local checkout remains on codex/45-upstream41-runtime at 8a23fe59f21b61bd090bf6ef8dc09a18ace7ac82. Existing untracked harnesses and artifacts were preserved.
- Run51 used the preserved cuda-hetero41-01/strata.exe, 52,433,920 bytes, SHA-256 6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb, built from clean source commit ffeb748d190ddeedff23587ce848596d4db0c752. See bench/hetero/20261011-51-worker-pool-sweep/matrix-tokenizer-rebased-05.json. The direct comparison records engine 0.1.41 and matching source/tree and engine identity, and leaves cause unresolved in bench/hetero/20261011-54-direct-regression-review/run51-vs-run41-evidence.json. These run pins precede and differ from upstream v0.1.42.

## Run evidence relevant to the comparison

The preserved run51 matrix uses the same 6bbc binary for the worker-pool arms. The long_1k capture contains one warm-up plus three formal requests; all four saved text artifacts are 128 exclamation marks with the same SHA-256. The formal quality report classifies all three as a single-character bang loop and truncated by length. These are captured degenerate outputs, not successful retrievals. The five preceding short prompts pass their three formal quality requests, with exact token-ID hashes matching H4. root-closeout-01.json and arms/w10-all/quality-on/cleanup/resource-summary.json show minimum available physical RAM of 62.851 GiB and system commit of 44.304 GiB; resource samples show the 12 GiB physical and 4 GiB commit gates passed. An admission-gate failure is not supported as an explanation.

One artifact discrepancy should remain visible: the top-level claim in run51-vs-run41-evidence.json labels smoke_three_key_retrieval as the stopping prompt, while its stored run51 request history and the more specific run51 quality-progress, failure, and long_1k quality files show that the three-key prompt passed and the matrix stopped at long_1k. This review follows the per-prompt run51 matrix artifacts for the chronology and does not treat the contradictory summary label as resolved.

Run51 and run41 have equal recorded environment, engine, model identity, and shared source/tree. The run51 arm adds explicit --pool-affinity all (the selected default) and a run-specific log path. Startup evidence records --pcie-frac 0, --prompt-cache 0, --adapt-swaps 0, --spec 4, --spec-min-p 0.5, INT8 KV, a 20 GiB resident budget, and 10 pool workers. The comparison explicitly leaves cause unresolved; this review does not upgrade correlation into causation.

## Changes closest to the symptom

| Upstream change | Relevance | Classification |
|---|---|---|
| b4e3407b: CUDA serve defaults STRATA_ROUTE_TAIL_SKIP to 7 on eligible runs; docs in DETAILS.md | This is the clearest new default that can alter verify/decode output. It skips missed experts routed at rank 7 or lower without renormalizing. It applies to eligible single-GPU, non-batch CUDA runs when experts are missing from VRAM. Docs report small KL/task checks and speed gains while acknowledging answer differences. | High-priority control for a v0.1.42 replay; ruled out as run51's cause because its binary/source are pinned to 0.1.41. Hold STRATA_ROUTE_TAIL_SKIP=0 fixed for a later comparison. |
| aa7b0cb1 and 0e9af370: refine default PCIe share using measured CPU-pool time | Could change the CPU/GPU split for missed experts. | Not applicable to recorded run51 argv: explicit --pcie-frac 0 wins. Also postdates the tested binary. |
| f71c5307 and 56354964: fence asynchronous copies before recycling pinned stage buffers, including lent-slot refills | Repairs a concrete IQ3 corruption path after a 4,096-token prompt with STRATA_STAGE_PIN=1. | Real quality fix, weak symptom match: run51 failed on a 1K retrieval request; no evidence stage pinning was enabled, and the tested binary predates the fix. Relevant to a separate long-prompt/pinned-buffer control. |
| 71eaee17: re-probe stale full MTP draft chains | Run51 enables MTP with --spec 4. This alters draft-size policy after stale-chain conditions. | Adjacent decoder-path change, absent from the tested binary. Control it in a v0.1.42 replay because speculation is enabled. The change is described as policy/performance recovery; no evidence connects it to the all-zero output. |
| 04dbe561 and eaaa237e: CPU prefill share memory fallback and don't arm it when all experts are in cache | Affects prompt GEMM work and buffer admission. | Low relevance to a failure during decode after a 1K prompt was accepted. Resource gates passed; no measured link to decode token identity. |
| d0336684: wait for KV residency-table upload during trim; e85567b6: bound MTP near context end | KV/spec boundary correctness changes. | Their stated triggers do not fit a 1K prompt within a 32K context. Neither was in run51's binary. |
| c81f9984: defer pipelined one-token commits to host verdict; dcc6f756: apply request logit bias in CUDA/HIP | Could touch sampling or verify behavior under those modes. | Review-only candidates: the recorded request did not enable pipelined one-token commit or request logit bias. No evidence ties either to the run. |

Other changes cover storage/residency (Windows resident reads use unbuffered experts.bin reads in f04ab568; Linux PLE direct I/O can fall back to mmap after a slow cold probe in 30017336), RAM/session guards (daa76552, c9680eef, 04dbe561), CPU kernel coverage (AVX1 expert kernels), helper-GPU cache sourcing, and layer-split lend sizing. These can change host I/O, admission, prompt speed, or CPU throughput in supported configurations. The recorded single-GPU run, passed resource gates, explicit --ple-io direct, and pinned 0.1.41 binary provide no affirmative causal path from them to the repeated 1K all-zero output.

## Broader release areas and overlap with Hetero work

- CPU capacity/calibration: PCIe share calibration uses pool timing; CPU prefill share has a memory-aware fallback; AVX1 kernels add CPU coverage. Run51 fixed PCIe share at zero and did not run the new tag.
- Routing/cache: tail skip is the output-affecting default. Look-ahead statistics and tools/replay_cache_policy.py are measurement/replay tools. STRATA_ROUTE_PRIOR and residency-biased routing remain opt-in and explicitly change outputs.
- RAM/storage: cgroup-aware memory guards, session cache/save admission, resident-file reads, stage-buffer fences, and PLE I/O behavior change memory safety or I/O lifecycle. Keep the RAM gate and I/O path as separate controls in a comparison.
- MTP/KV/stage: stale-chain re-probing, context-edge bounds, split MTP batching (opt-in), KV table synchronization, split pipeline, and pinned-buffer fences. Run51 enabled standard MTP, but none of these later fixes was in 6bbc.
- Arc/NPU/SYCL and planner: Alchemist/A770 fast paths, grouped XMX, and doorbell/mapped-transfer tuning are Intel-specific and mostly opt-in; gfx906/AMD/HIP work does not apply to this NVIDIA CUDA binary. Native CPU/Arc expert operators, CPU harnesses, vision-stage diagnostics, planner/ETA gates, and phase-oriented heterogeneity evidence are separate Hetero directions. Their build/harness receipts do not establish CUDA model task quality.

## Release/documentation evidence and limits

Upstream commit cdbf4586 labels the project version and setup minimum as 0.1.42. There is no single short changelog covering the whole merge range; release documentation is mainly in docs/DETAILS.md, including measured default-routing and PCIe-share sections, opt-outs, tested machines, and limitations. The tail-skip section states answers differ slightly from 0.1.41 and gives STRATA_ROUTE_TAIL_SKIP=0 as its control. The PCIe section says explicit --pcie-frac wins. These support classification, not a claim that either feature caused run51.

No causal fix for run51 is established by this diff. The actionable candidate for a later release replay is explicit STRATA_ROUTE_TAIL_SKIP=0 because it is a new output-affecting default. A same-binary 0.1.41 replay and a tag-pinned v0.1.42 A/B with that variable held fixed are needed before assigning a cause. Keep the all-zero output, resource admission, and timing/I/O evidence as separate claims.

## Source and artifact pointers

- Release pins and diff: git show v0.1.42, git show v0.1.41, and git diff fb58e0dbc8399662c0e47c76578c6e878b14f6cf..61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406.
- Tail-skip default: src/program/generate.cpp and src/core/expert_source.cpp at b4e3407b; measured limits and opt-out in docs/DETAILS.md.
- PCIe calibration: src/program/generate.cpp at aa7b0cb1; measurements/opt-outs in docs/DETAILS.md.
- Stage-buffer lifetime fixes: src/core/expert_source.cpp, include/strata/core/expert_source.hpp, src/program/generate.cpp, and src/core/peer_experts.cpp at f71c5307 and 56354964.
- MTP policy: src/spec/draft_policy.cpp and src/spec/draft_policy_test.cpp at 71eaee17.
- Run51 source/binary/matrix: bench/hetero/20261011-51-worker-pool-sweep/matrix-tokenizer-rebased-05.json; output and H4 artifacts under arms/w10-all/quality-on/.
- Run identity/request comparison: bench/hetero/20261011-54-direct-regression-review/run51-vs-run41-evidence.json.
- Resource floor/gates: bench/hetero/20261011-51-worker-pool-sweep/root-closeout-01.json, arms/w10-all/quality-on/cleanup/resource-summary.json, and arms/w10-all/quality-on/resource/samples.jsonl.
