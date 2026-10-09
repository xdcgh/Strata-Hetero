# CPU worker-affinity sweep plan

This is a plan only. The E: output root is not created, row 6/10 inputs are not generated, and no native kernel or affinity setting has been changed. `sweep-contract.json` sets `execution_supported=false` pending root review.

The topology snapshot came from the read-only `GetSystemCpuSetInformation` query at 2026-10-09T17:58:22+08:00; its SHA-256 is `9d64b997de0d7b22dd3420a90648eeef43e23f796d4fea40df9106baf0a1baf`. All 16 CPU Set records are allowed by the observed process masks. Windows reported numeric `EfficiencyClass` 1 for CPU Set IDs 256, 257, 266–269 and class 0 for IDs 258–265, 270–271. The IDs are recorded as CPU Set IDs and `logical_processor_index` values. The topology tool leaves core labels unknown; this plan does not assert P/E/LP labels.

The snapshot marks class-0 CPU Set IDs 258–265 and 270–271, plus IDs 266 and 267 in class 1, as parked. They remain in the tool's allowed candidate ordering. This records current flags only; it does not predict whether workers will run on those sets during a future test.

The upstream candidate report lists 15 worker CPU Set IDs for `all` and `auto` (same order): 257, 266–269, 258–265, 270–271; host candidate 256. The `p-cores` affinity mode lists five worker candidate IDs (257, 266–269) and host candidate 256. These are candidate plans, not evidence of successful per-worker pinning. `none` leaves the host and workers unpinned.

The plan has two single-thread default reference gates for rows 6 and 10, followed by 144 worker/affinity/row cases. Rows 1, 2, 4, and 16 use frozen native outputs directly. Eighteen `p-cores` cases (workers 6, 10, and 16 for each row) are marked `not_run_strict_candidate_overflow`. `all` and `auto` at 16 workers remain planned with one unpinned overflow worker. Rows 6 and 10 have no frozen output; the contract requires a new same-binary default output before any pool case for those rows. Their row-seeded inputs are specified but not generated yet.

All scheduled cases use the same Q4_K/Q5_1 expert, hidden 2560, intermediate 640, one warmup, and five formal iterations. They compare output bytes exactly to the frozen native output when available and to the same-binary default output for each row. The earlier 12-case sweep established parity only for unpinned pools with worker counts 1, 4, and 10; it makes no performance claim for the planned matrix. Before execution, root review and a fresh 12 GiB physical / 4 GiB commit gate are still required.
