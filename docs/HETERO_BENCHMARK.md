# Benchmark evidence and acceptance

This report starts with a historical reference. It does not yet claim an improvement from Strata-Hetero. Current-upstream and modified-engine measurements will be added with their immutable run identities.

## Historical 0.1.39 reference

Source: `C:\Users\DC\Documents\ChatGPT\Strata\bench\results\2026-10-06-local-rtx4090d-ud-q4-k-xl\README.md` and its JSON/raw/resource artifacts. Checkout `6f32ec070f23ced9f50e704d854d775da52591ab`, prebuilt engine 0.1.39, Windows 26300, Core Ultra 9 285H, 127.56 GiB RAM, RTX 4090 D reporting 49,140 MiB VRAM, F USB SSD, Unsloth UD-Q4_K_XL, 32K context, INT8 KV, resident budget 71 GiB, reserve 4096 MiB, 10 CPU workers, spec=4/min_p=0.5. Actual prompt sizes and draft/suffix behavior must be matched before comparing.

| Historical workload | Reported mean tok/s | Scope |
|---|---:|---|
| Prefill 1K | 355 | Historical report, not rerun by Hetero |
| Prefill 4K | 840 | Historical report, not rerun by Hetero |
| Prefill 16K | 1660 | Historical report, not rerun by Hetero |
| Prefill 30.7K | 1669 | Actual length below 32K |
| Output 256 | 69.4 | MTP/suffix aggregate drafting |
| Output 512 | 78.9 | MTP/suffix aggregate drafting |
| Output 1024 | 86.4 | MTP/suffix aggregate drafting |

Those rows are historical observations. The report's 65.5%-72.4% draft acceptance is aggregate, not a pure MTP rate. Pelican with reasoning disabled generated HTML; reasoning-enabled runs exhausted their budgets before an answer. A successful smoke, verified download and a performance sample are separate claims. The capacity/performance directory contains a separate later investigation and must not be silently combined with this matrix.

The later historical capacity report records a 523,572-token non-repeated natural-text prompt that retrieved all three keys (523.95 prefill tok/s, 26 generated tokens, 36.1 decode tok/s) and a 1,046,977-token prompt that produced 128 `!` tokens (532.26 prefill tok/s). Both are single samples. A subsequent 1M clean restart also failed short probes in that investigation. Capacity/prefill executed, but 1M retrieval and stable generation did not pass; the cause remains unresolved. Another chat is still running that historical investigation. These observations are preserved as historical failure shields, not as current-upstream or Hetero validation.

## Current upstream quality check on 2026-10-08

Run `bench/hetero/20261008-02-upstream-quality` uses the frozen upstream C++ source `82f46a8`, engine 0.1.40, the freshly verified four-shard UD-Q4_K_XL model, 32K context and deterministic placement controls. Five short tasks and four natural-text retrieval tasks each ran one warmup and three formal requests. All 27 formal requests passed their bounded task checks and finished normally. The four live natural prompt lengths were 1,036, 4,109, 16,396 and 30,712 tokens; all requests reported zero reused prompt tokens. Each task's three visible outputs were byte-identical.

This verifies those task outputs, not logits or emitted-token agreement: the initial client retained text and usage but did not capture the actual generated token IDs. The full-table RAM startup in `20261008-03-ram-quality` was stopped before inference because system commit headroom fell below 4 GiB. Its smaller expert complement also prevents a controlled timing comparison with this upstream run. Both records remain available; no speedup is accepted from them.

Quality runs can now set `hetero_capture_token_ids: true` in the server config. OpenAI SSE's final chunk then records the integer IDs actually yielded by the engine, including emitted stop/EOS tokens and excluding server-inserted wrap/forced-opening tokens. Heartbeats are excluded. This captures accepted output observed by the server, not speculative proposals that were discarded. The option defaults off and applies only to streamed OpenAI requests. Quality arms use it in both runs; performance acceptance uses a separately declared configuration. It does not capture logits.

The subsequent `20261008-05-upstream20-quality` run also passed all 27 formal tasks and captured their actual generated IDs. `20261008-07-ram20-buffered-quality` matched its short-task IDs but failed natural 1K retrieval with a single-token exclamation loop. Its later lengths were not run. Debug-instrumented diagnostic10 subsequently passed that retrieval; zero non-finite counts and one instrumented success do not prove the uninstrumented configuration stable. No RAM speedup is accepted.

Early Windows client artifacts used `Path.write_text` with automatic newline translation: model text hashes describe logical UTF8, while saved multiline text has Windows CRLF. Original artifacts remain intact, and per-file supplemental writer receipts verify that exact transformation. New client output files use UTF8 bytes without translation. The offline comparator separates model-text hashes from artifact hashes and requires proof for any historical conversion.

Upstream was fetched again on 2026-10-08 and is now `6674a00` (v0.1.40.4), 267 commits beyond the frozen `82f46a8` engine used above. These are frozen-reference measurements, not validation of the latest upstream. Integration and a fresh latest-upstream baseline are pending; the new upstream Windows PLE implementation will be reused while retaining the stricter Hetero reserve/complete-lock checks.

## Run contract

Store `hardware_manifest.json`, `storage_profile.json`, `baseline.json`, `results.json`, `results.csv`, `resource/`, `logs/`, `comparison.md` and `DECISIONS.md` under `bench/hetero/<run>/`. When an item is not measured, record its status and reason, not a zero value. Keep raw model outputs and engine/client times. Timed-out, truncated, cache-contaminated or admission-failed samples stay in the evidence and are excluded from accepted speedups.

Use warmup and repeat at least three times. Reuse the same natural text and model/tokenizer/draft bytes between arms. Record prompt caching, adaptive swaps and PCIe placement, including deterministic comparison controls. NIAH/RULER/Pelican/code quality gates precede performance acceptance. Monitor NaN/Inf, repetition and token loops; passing a scalar microbench alone does not demonstrate model quality.

## Final comparison (pending experiments)

| Metric | Historical | Current upstream | Hetero |
|---|---|---|---|
| Cold/warm startup and model-to-RAM | In historical logs; requires extraction | Pending | Pending |
| Expert fill / PLE / MTP initialization | In historical logs; requires extraction | Pending | Pending |
| Server ready / first token | In historical logs; requires extraction | Pending | Pending |
| TTFT 4K / 128K, P95 | Requires raw extraction / matching workload | Pending | Pending |
| Prefill 4K / 32K / 128K / 512K | 840 / 1669 at 30.7K / separate capacity study | Pending | Pending |
| Output 256/512/1024 and concurrent throughput | 69.4 / 78.9 / 86.4, single request | Pending | Pending |
| Pelican and other quality | Historical artifacts retained | Pending | Pending |
| RAM / VRAM / power / GPU idle | Historical samples retained | Pending | Pending |
| Arc / NPU / E / F | No helper performance baseline | Pending | Pending |

Final acceptance must answer all sixteen questions in [the plan](HETERO_PLAN.md). Unsupported 512K/1M configurations cannot be called stable because their resource estimate fits. Long natural-text generation, retrieval and state restore must actually run and pass.
