# Storage profiling and placement

Storage is a measured cold/warm capacity tier, after VRAM and RAM for active data. A volume letter is not a physical device. The inventory joins volume, partition and disk: on this host E is Disk 1, a Micron NVMe; F is Disk 3, a Realtek USB SSD bridge. C/D are different small NVMe devices with little free capacity. Refresh the mapping and available space before any copy or profile.

## Read-only profiler

`strata-storage-profile` is built by `tools/hetero_host`. It uses the engine's `platform::DirectFile` unbuffered read path and aligned buffers. It generates an I/O workload on the selected existing file; it is not a passive observer. It never changes the input file. Without `--file`, or with `--validate-only`, it only checks metadata/parameters and does not instantiate a reader or write a profile.

Supported read blocks are 4 KiB, 16 KiB, 64 KiB and 1 MiB; requested queue depths are 1/4/8/16/32/64. Select random or sequential access, duration, warmup, seed and at least three repetitions. Every output path must be new. The file must have at least one complete block; the profiler records and excludes a partial tail. Short/zero completions inside the selected complete-block region are errors.

Record each repetition, submitted/completed/successful/failed/uncompleted reads, actual bytes and maximum QD, wall time, IOPS, MiB/s, mean latency, sampled P50/P95/P99 and caller-thread submit/wait CPU. Latency starts before the submit call and ends when completion is observed, so it includes software queue/submission/reaping time. It is not a drive-service-only timer. Caller-thread CPU excludes the issuer pool's CPU. The latency reservoir is bounded at 65,536 samples per repetition; the mean uses all successful samples, while percentiles are explicitly sampled. A separate sample RNG preserves the chosen read-offset sequence.

Unbuffered mode bypasses the OS file cache. Device/controller caches still exist; do not label a repeated direct read as a guaranteed cold flash read. Report file size, device mapping and measurement conditions. Failed phases retain counters/error evidence and have unavailable performance fields, not a fictional zero-speed disk result.

## Timeout and buffer ownership

Every read owns an aligned buffer until its completion. Normal shutdown drains all submitted requests before releasing buffers and the reader. The legacy DirectFile interface does not expose cancel-and-drain: Windows close joins issuer threads then closes IOCP/file handles; Linux close can wait on a blocking pread. That contract must be strengthened before runtime multi-device cancellation is integrated.

The standalone profiler has a 30-second drain bound. On a hard timeout it retains pending buffers/reader, writes and flushes a failed receipt, then exits only its own process with code 4. This tool-only escape is not a scheduler/engine cancellation design. Output failure is also nonzero. Never use a failed or uncompleted run to choose placement.

## Validation scope

The host fixture uses its own new 64 KiB file and deterministic fake readers/clocks. It checks real direct-read bytes at the last block, underlying EOF completion, alignment rejection, QD limits, submission-inclusive timing, bounded sampling/seed trace, read-tail exclusion, malformed/overflow arguments, failed/uncompleted records, no reader in validate-only mode and output preservation. Its temporary directory is checked by canonical identity and only its own files are removed. This is correctness evidence, not an E/F performance result.

Actual physical-device sweeps are pending until the other chat's timed inference completes. The orchestration, workload placement planner, exact external PLE extraction, parallel model startup and application-level striping remain to be implemented and measured. Startup and active lookup are different workloads and cannot share one unqualified disk score.

## Placement experiments

| Workload | Measure | Candidate if capacity/quality pass |
|---|---|---|
| PLE cache misses | 4 KiB random latency/IOPS at actual lookup depth | RAM first; best random-read device for the remainder |
| Expert staging/refill | Blob-sized and semi-random reads, loan size, cache hits | RAM complements/loan backups, then E warm tier |
| Model startup | Sequential throughput plus concurrent physical-device reads | Independent-device parallel loads with bounded buffers |
| Parked conversation | Sequential write/restore and first-token latency | E snapshot tier; active KV stays VRAM/RAM |
| Original models/archive | Capacity and integrity | Preserve F originals; copy/hash-verify before switching paths |

Prefill uses ordinary cache-resident weights in VRAM directly. Only misses and borrowed cache slots need another source. Budgeted complements currently omit the optional cache-loan RAM backup, and the file-cache estimate includes the non-RAM experts. Measure actual borrowed bytes and physical disk reads to test whether that estimate overstates the real file working set. A logical mapped read can be an OS-cache hit; it is not proof that the SSD was on the critical path.

Copies must follow copy -> full hash verification -> measured use. No unique model or unrelated user file is moved/deleted. Do not retain striping merely to exercise multiple drives if full RAM or a single NVMe is faster. Storage acceptance targets at least 10% gain in the affected workload with model quality unchanged.
