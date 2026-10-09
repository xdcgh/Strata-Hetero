# Same-binary diagnostic-off control

H21 uses the H18 diagnostic binary unchanged, with `STRATA_DIAG_ZERO_LOGITS` absent and no `STRATA_DBG_NAN`. The complete 28,800,138,240-byte PLE table is locked, resident expert budget is 20 GiB, and the fixed quality caps/request controls remain unchanged.

The first seven tasks pass all 21 formal requests with H4/H18 emitted IDs and visible text. Each 16K formal then produces 128 token IDs of zero and 128 visible exclamation marks from the first output token. The 30K task was not started. Three separate post-failure arithmetic requests, without a warmup and with cap 8, also emit eight zeros/exclamation marks. This confirms continuing degraded output in this process, not the cause of the degradation.

The 16K request window has minimum physical availability 38.302 GiB and system commit headroom 18.780 GiB. The whole sampled lifetime has minima 38.302/18.451 GiB over 1,004 samples. Gates pass. JSON unload returns 200; engine, sampler including its Python child, bridge and launcher exit, and 8081 is released. A reused sampler PID under another parent was not treated as ownership and was not stopped.

Full-table RAM remains unaccepted. The H18 pass does not prove a diagnostic-induced repair: no head-score readback branch fired in that run, and startup/request history still differ. No numeric logits are available from H21, and token ID zero is not a score value. Investigate state and cache restoration with bounded diagnostics; do not add global synchronization or discard the preserved failures. Continue the independent storage, CPU, Arc, NPU and planner stages.

Original provisional metadata and the initial null quality-status review remain unchanged. `postfailure-arithmetic8-review-final.json` supplies corrected classification; `root-terminal-review.json` independently validates raw failed SSE and terminal receipt hashes.
