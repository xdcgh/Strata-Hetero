# Full PLE RAM did not pass batched-prompt quality

The controlled upstream20 run passed all 27 formal requests. This full-table arm uses the same model bytes, 20 GiB expert complement, 32K INT8 context, deterministic placement and sampling. Expert file reads are explicitly buffered, matching the upstream run's observed effective mode. Startup logs prove all 28,800,138,240 PLE bytes locked and 20.00 GiB of pinned expert RAM.

Five short tasks passed, including code and three-key JSON retrieval. Their 15 formal outputs and actual generated-token sequences agree with upstream20. The 1,036-token natural-text retrieval then returned 128 exclamation marks in the warmup and each of three formal requests. The generated token sequence is 128 copies of ID 0. Generation stability and retrieval failed, despite the physical and system-commit floors passing. The 4K, 16K and 30.7K tasks were not run.

An independent short arithmetic probe after the failures still returned `4`. Therefore the observed failure is narrower than a permanently unusable model service. No underlying logits were captured in this arm, and the exclamation output alone does not prove NaN logits.

Decision: reject this configuration, preserve all outputs and counters, and leave full-table mode opt-in. No speedup or daily configuration is accepted. Diagnose the batched-prompt path before increasing context length.

The separate diagnostic10 arm enabled existing `STRATA_DBG_NAN=1` instrumentation. Its short probes returned `4`, and a later full-cap 1K retrieval passed with the same actual token sequence as upstream20. The printed non-finite counts were zero. Instrumentation changes timing and synchronization, and the request sequence differs; the cause and observer effect remain unresolved. A successful diagnostic request does not erase this arm's three reproducible failures.

The offline comparison in `offline-comparison-02-reviewed.json` verifies matching controls, confirms the short-task agreement, and rejects the failed/incomplete arm. The earlier comparison01 parser findings are retained. Supplemental writer receipts account exactly for historical Windows text newline translation without changing the raw artifacts.
