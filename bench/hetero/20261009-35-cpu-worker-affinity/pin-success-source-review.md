# CPU pool pin-success visibility

This is a read-only source review. No pool constructor, worker, affinity API, or native kernel was run.

- [`pool.cpp`](../../../src/kernels/cpu/pool.cpp#L347) defines `pin_this_thread` as a Boolean success result. On Windows it returns `detail::set_thread_group_affinity`; the Windows helper calls `SetThreadGroupAffinity` and returns success/failure in [`pool_affinity_win.hpp`](../../../src/kernels/cpu/pool_affinity_win.hpp#L17).
- The `ExpertPool` constructor launches each worker and calls `pin_this_thread(core, i)` without storing or exposing its Boolean in [`pool.cpp`](../../../src/kernels/cpu/pool.cpp#L461). The public `ExpertPool` interface exposes worker count, host participation, topology and affinity, but no worker pin-result accessor in [`pool.hpp`](../../../include/strata/kernels/cpu/pool.hpp#L136).
- The benchmark's `PoolRunInfo` has `host_pinned` but no worker result vector. `ScopedHostPin` throws if the host's `ThreadAffinity.valid` is false, so the existing `host_pin_applied` field is backed by a successful host API return. The native receipt explicitly emits `worker_pin_success_available=false` at [`native_expert_bench.cpp`](../../../src/hetero/native_expert_bench.cpp#L566).

The current runner therefore validates and records requested mode, reported host pin state, worker count, planned numeric CPU IDs and overflow. It leaves worker pin success unknown. Candidate IDs and a successful process exit are not evidence that every pool worker was pinned.

A future source change could store each `pin_this_thread` result in a per-worker atomic/vector and publish the result after a startup barrier. It should keep those booleans separate from planned IDs and retain `worker_pin_success_available=false` until such a measured receipt exists. This sweep preparation does not change the pool implementation.
