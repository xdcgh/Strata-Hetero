# Strata v0.1.42 Hetero capability inventory

Read-only source review. No tests, builds, hardware/device access, model reads, or runtime actions were performed. This inventory makes no performance or model-quality claim.

## Pins

- Upstream release: tag/commit `v0.1.42`, `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
- Hetero source worktree: clean branch `codex/62-upstream42-runtime`, merge commit `8cf0e89be5da8485cbeb6ac1b527089add25fbcf`, with parents `fb3fc9ce640b352c591263475085bc9cc019404c` and upstream `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
- The Hetero branch inherits most runtime code unchanged. The v0.42 and upstream blobs are identical for expert-source/residency, parallel pinned loading, peer/remote experts, conversation-memory guard, prefill, and calibration/profile producer files. The main program, CPU pool and server bridge differ at the Hetero pin; their refs below are branch-specific where noted.
- Current checkout planner/runtime files match the Hetero pin: `tools/hetero_planner.py` blob `413b7d4355d0526692778aaac08d06dbf5553216`; `tools/hetero_runtime.py` blob `7bda9a24fd2b7be1d0e9559981de141109b8ac11`.

## Implemented upstream capabilities that overlap the Hetero scope

| Capability | Source-level state and boundary |
|---|---|
| Expert tiers and memory fallback | The engine already places profiled experts across VRAM cache, a budgeted resident RAM complement, and mapped/file-backed expert shards. `--resident-budget-gib` ranks the non-VRAM complement by expert profile and leaves the rest file-backed. The resident-copy path observes RAM/commit limits and has mmap fallback behavior. This is the existing storage-tier mechanism to extend or configure, not duplicate. |
| Profile and heat inputs | `--expert-profile` supplies a ranked (layer, expert) list to cache fill and RAM placement. `tools/make_profile.py` builds that ranking from `--dump-routing` traces. The opt-in `--expert-profile-save` persists adaptive routing heat for a later start. These profiles encode expert order/frequency, not per-backend costs, hardware identity, per-shape acceptance, or quality results. |
| Parallel loading and file I/O | Startup expert loading and resident-complement copying use worker threads. `FileExpertSource` also exposes Linux file-prefetch controls and Windows/Linux direct-I/O paths. These cover data loading and file-page behavior; they do not decide which compute backend should execute a model operator. |
| Multiple GPUs | Existing `--peer-device`, remote expert-cache device slots, and layer split support place expert computation/cache on additional GPUs. `RemoteExperts` and `PeerExperts` own those paths. This is a substantial existing multi-device implementation; it is not the Hetero profile dispatcher. |
| CPU hybrid-core placement | `--pool-affinity all|auto|p-cores` and the CPU pool topology code handle Windows and Linux P/E layouts. Windows reads processor efficiency classes and group masks; Linux uses sysfs/topology data. The public roles are P/E, with optional SMT ordering; there is no separate LP-core role or E-only public policy in the enum. |
| Built-in calibration | `tools/calibrate.py` and setup's `--calibrate` tune engine settings such as PCIe share, speculative depth floor, CPU worker count, and adaptive-cache behavior. Setup stores selected settings for the hardware/model key. This calibrator measures engine decode throughput and restarts for some settings; it is not a generic CPU/Arc/NPU/storage cost-profile producer and has no per-candidate end-to-end model-quality acceptance gate. |
| KV prefetch and conversation parking | Prefill has optional async staging of the next streamed KV prefix. Separately, the conversation cache parks/restores session state in host memory with cgroup-aware RAM admission and fallback to ordinary prompt processing. The KV prefetch is a prefill pipeline optimization; conversation parking is session reuse. Neither is an external CPU/Arc/NPU scheduler or an automatically tiered persistent KV store. |

## Hetero-only planning/runtime code and limits

| Capability | Source-level state and boundary |
|---|---|
| Cost model | `tools/hetero_planner.py` defines an identity-bound, exact operator/phase/row key; it requires at least three full input-ready-to-output-ready timings and a known quality boolean. It checks profile expiry, exact hardware/runtime/model/engine identity, readiness/busy time, shared-memory contention and explicit acceptance evidence. Unknown shapes or mismatched identity retain the primary. This code is absent from upstream v0.1.42. |
| Serialized profile and evidence checks | `tools/hetero_runtime.py` writes/loads bounded, hash-checked profiles and evidence. Helper acceptance must bind to a separate model-end-to-end report; a kernel/IPC receipt alone cannot authorize a model route. This is Hetero-only support code. |
| Stateless dispatch wrapper | `HeteroRuntimeScheduler` accepts immutable bytes, allows one job in flight, requires ready helpers, verifies outputs, quarantines helper failures, performs exact helper cleanup, then retries the unchanged input on the primary. Its API excludes KV/session mutation. Source search found no production import from `serve/` or `src/`; the visible callers are its unit-test fixtures and the wrapper itself. Treat it as an unintegrated candidate API, not an active Strata runtime path. |
| CPU/Arc/NPU candidates | The Hetero tree has isolated native CPU and XPU/Arc tools. `tools/hetero_native_openvino.py` labels the native activation graph as uncompiled, uninferred and production-route-disabled; standalone XPU service code is separate from the Strata verifier. These files do not establish a production model route or task-quality acceptance. |
| Hetero KV work | `tools/hetero_kv_parking.py` / `hetero_kv_stdio.py` define authorized experiment adapters around the engine's private stdio/session behavior. They are experiment and evidence plumbing. The product-level conversation cache already exists upstream, so a future design should state which new KV tier or policy this adds before duplicating it. |

## Minimal future integration points

1. Keep `tools/calibrate.py` for product-setting calibration. A separate Hetero calibration command should emit the existing `HeteroCostModel` schema: hardware/runtime/model/engine identity, exact operator/phase/rows, at least three complete-boundary timings, evidence hashes, and independent model-quality acceptance. Do not reinterpret `--expert-profile` or the setup calibration settings as that profile.
2. A future opt-in `--hetero-profile` belongs beside the existing `Params`/argument parsing in `src/program/generate.cpp`; configuration forwarding belongs in `serve/server.py::engine_args()`. The loader should remain explicit and reject missing/stale/mismatched evidence before any helper is eligible.
3. Initially connect only a stateless operator boundary that owns input bytes, output validation and helper cleanup. `HeteroRuntimeScheduler.dispatch()` already expresses primary fallback semantics, but no product call site exists. Keep stateful verifier/KV mutation on the current engine path until there is a specific measured, quality-gated boundary.

The main gap is therefore integration and model-level evidence for alternate execution paths, not a lack of basic VRAM/RAM/file tiers, profile ranking, P/E affinity, multi-GPU placement, or session parking. All future routes should remain disabled when identity, exact shape, helper readiness, contention data, repeated timings, or model-quality acceptance is missing.
