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
