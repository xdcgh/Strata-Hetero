# Run63 diagnostic decisions

The unchanged v0.1.41 Hetero binary `6bbc426b`, source `ffeb748d`, ran the same direct-PLE/20 GiB resident/32K INT8/10-worker/MTP4 request history as run51. The only numerical diagnostic switch was `STRATA_DIAG_ZERO_LOGITS=1`: it reads logits after sampling has returned ID0 and the existing window synchronization, without adding prefill GEMM synchronization.

The first five short prompts passed, with all 15 formal emitted token-ID sequences equal to H4. The next `long_1k` prompt failed: one warmup and all three formals emitted exactly 128 ID0 tokens, returned 128 `!` bytes, and stopped by length. Root independently parsed retained SSE IDs, reconstructed visible UTF8 bytes, and checked the failure report.

The captured verifier readbacks at positions 1035 through 1061 show every one of the 248,320 logits is NaN: finite=0, positive/negative infinity=0, finite_argmax=-1, selected ID0. This confirms non-finite head output during the degeneration. It does not locate the first faulty operator/layer or prove that PLE, CPU affinity, resident placement, MTP or any later upstream patch caused it. No unchanged v0.1.41 retry is warranted.

The three later natural prompts (4K, 16K, 30K) and the nine later matrix phases were not run. No performance result is accepted from this diagnostic.

All 268 physical-RAM and PSAPI system-commit samples pass 12/4 GiB floors; minima are 62.917 and 43.759 GiB. Aggregate cleanup still exits 1: its last post-stop resource row reports an unknown owner tree after the exact owner was removed. Preserve that rejected row and the non-pass cleanup result. POST unload confirmed loaded=false/in_flight=0; exact model/server/sampler identities and port8081 are terminal, documented in the separate addendum. A future lifecycle repair should stop and verify the sampler before the intentionally removed owner, without hiding any failed in-run sample.

Next: build the already reviewed matched upstream v0.1.42 and Hetero v0.1.42 sources, then replay fixed controls with `STRATA_ROUTE_TAIL_SKIP=0`. Do not treat an upstream fix as causal before measuring it. Real KV parking and model performance acceptance remain held until quality is valid. The original full project scope remains active.
