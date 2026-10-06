# Storage profiler preparation

- Build a standalone CPU-only read profiler from the existing DirectFile path. No original model or disk file is changed. Real E/F profiling waits for the other chat's timed inference and resource coordination.
- Require full selected blocks, bound QD and latency sample memory, and preserve every failed/uncompleted result. Failed phases have null performance fields. Percentile samples and caller-only CPU are labeled explicitly.
- Legacy DirectFile has no public cancel-and-drain. The profiler drains before normal cleanup; its own hard-timeout process exit retains pending buffers until OS teardown. Runtime/engine cancellation still needs a separate contract.
- Initial worker fixtures passed, but the authoritative cache showed Debug despite a Release label. The main review also hit a missing-vcvars include-path failure in a new shell; that failed log was retained. A fresh explicit Release build in `host-storage-release-02` then compiled and passed the fixture target, with GPU/WSL/model execution absent.
- These are tool correctness and build results. No physical-disk bandwidth, IOPS, P95/P99 or model speedup has been measured by this run.
