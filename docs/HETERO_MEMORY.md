# RAM-first memory experiments

Hot model data goes to VRAM or system RAM when it fits with the session, staging buffers and system reserve. SSD remains a cold capacity tier. The RTX 4090 D stays the primary compute device. A RAM budget is evaluated by end-to-end latency and memory pressure, not by allocated gigabytes.

## Existing expert and KV paths

Upstream already keeps GPU-missed experts in a resident CPU complement and supports budgeted file-backed experts. Preserve its native expert bytes and CPU arithmetic. The historical Q4 configuration reported 35.21 GiB of GPU experts and 36.52 GiB of page-locked CPU experts on this 127.56 GiB host. A requested resident budget of 71 GiB is not proof of a 71 GiB private CPU copy: complement selection, prefill borrowing and available-memory/commit guards affect the actual allocation.

The prompt stager and decode expert pool have different access patterns. Measure model-file reads and staging waits as well as cache hit rate. Warming the OS file cache is not the same guarantee as a locked expert arena. PLE residency can also change how much file cache remains for prompt experts; a faster lookup microbench alone cannot establish a faster full prompt.

Active KV remains in VRAM/RAM. Parked session snapshots currently use RAM; SSD persistence is a later cold-tier experiment. Do not add an SSD read to each active decode step just to use the drive.

## Windows full-table PLE mode

The base engine refused `--ple-io ram` on Windows. The experimental memory branch now reuses the existing `platform::lock_resident` working-set/VirtualLock helper for the mapped table. It keeps the original quantized file bytes and row dequantizers. It does not allocate a second table or register it as CUDA mapped host memory.

The mode checks physical RAM before warmup, warms 64 MiB pieces with up to four threads, requires the entire table to be locked, and refreshes the reserve check after locking. A partial lock or insufficient reserve is an explicit startup error; it is not reported as a successful resident table. `close()` unlocks the requested table region and clears the lock state. The new locked-byte counter is printed at startup. The process working-set helper raises a soft quota; process exit releases that quota, and a reused process can retain a raised quota after closing a table.

`--ple-ram-reserve-gib` defaults to 12 GiB on Windows and accepts a strictly parsed integer in 4..1024. This is a point-in-time physical-memory floor for the table load, not an allocation reservation or the engine's complete expert/KV budget. Keep the benchmark's 12 GiB available-memory stop condition and account for the subsequent expert, MTP, KV and staging allocations separately. The default table mode remains direct. POSIX keeps its existing mlock/warm-page behavior and reports whether locking actually succeeded.

For the documented IQ4_NL table, 320,001,536 rows × 90 bytes is 28,800,138,240 bytes, about 26.82 GiB. Read the real GGUF tensor metadata before applying that estimate to another variant. FP8/BF16/F32 tables have different row widths. Do not convert formats as part of a placement-only comparison.

## Verified scope so far

On 2026-10-07 the native Windows host build and CTest passed 2/2 targets. The PLE fixture now compares direct, ordinary mapped and fully locked mapped rows against each format's own dequantizer, including first/last rows, random rows and the sixteen-head gather. It checks complete locked bytes, cleared state on close, impossible-reserve refusal and reopening after refusal. The working-set memory check also passed. These tests run without CUDA or a model.

The full CUDA engine build also passed, with the same SM89/compiler/SDK/ggml settings as the unmodified upstream build. Eleven compiled CLI checks passed with GPU visibility disabled, including valid bounds and refusal of negative, malformed and overflowing reserve values. Real-table residency, greedy/logits quality, model memory-pressure behavior and performance are still pending. No new speedup or accepted model configuration is claimed by those build/fixture passes.

## Measurement matrix

| Arm | Expert budget | PLE | What must be recorded |
|---|---|---|---|
| Current upstream | Matched reference | Direct, existing row cache | Initial/warm startup, PLE wait, full prefill and output rates, file reads |
| Cache sweep | Matched reference | Direct, bounded row-cache sizes | Hit rate, CPU/cache metadata cost, physical read bytes, memory pressure |
| OS-cache reference | Matched reference | Mmap | Actual faults/file reads; never label it locked |
| Full table | Capacity-admitted budget | Locked RAM | Complete table lock, warmup cost, expert/file-cache tradeoff, all quality gates |
| Partial/table hot rows | Tuned jointly | RAM row cache + SSD | Admission, cache workload identity, miss latency and tail behavior |

Each arm needs warmup and three or more formal repetitions on the same natural-text prompts and output caps. Full-prefill experiments disable prompt-cache reuse. Strict output comparison fixes adaptive expert swaps and PCIe placement before evaluating table placement; performance profiles can then re-enable them and report that separate scope. Compare 32K first, then admitted 128K/256K/512K/1M with retrieval/stability checks. The historical 1M failures remain an unresolved quality constraint.
