# Cache-reachability delta review

Read-only inspection of the latest uncommitted `verify.cpp`, `expert_source.cpp`, `prefill.cpp`, and helper-header changes in `E:/Strata-Hetero-data/source/hetero-live-arc-20261025`. No test, compiler, Python, process, GPU, model, or WSL operation was run. No blocking defect identified in this delta by static inspection.

## Cached generation route

- `verify.cpp:719-724` disables both all-resident and device-plan shortcuts before the optional device-plan buffer setup. `ar_on()` therefore cannot bypass host service, including after refresh_ar().
- `verify.cpp:1420-1423` uses unconditional doorbell_publish in forced mode. This publishes actual F32 activation rows even when the device residency table says every selected expert is resident; it avoids stale/unwritten m_x_ inputs for the forced cached expert.
- `expert_source.cpp:3555-3557` limits the override to native, layer28, a usable host plan, and no peer/remote engine. `:3595` and `:3615` exclude expert288 from the PCIe miss accounting and tail-skip assignment. `:3634-3640` marks all its matching occurrences CPU and continues before creating any GPU group. Thus it is absent from both P.ptr/P.dst resident groups and the PCIe fetch groups.
- The subsequent actual native job retains its original row output location and actual input pointer. The original helper failure returns this same job to CPU execution. On the device, copy_rows_from_mapped treats rows absent from P.dst as CPU rows; moe_hit_add visits only P.dst, so the forced row is neither overwritten by GPU zeroing nor added twice. Other resident groups keep their existing zero/copy behavior.

The frozen expert can no longer be completely bypassed merely because it is cached, for the supported host-plan configuration. A real route selection and `hetero_live completed=1` remain necessary evidence. No-peer/no-remote and a usable plan remain explicit scope limits.

## Prefill route and lifetime

- `prefill.cpp:2248-2251` permits pool ownership under forced mode even with CPU share off; `:3068` forces layer28's activation-copy arm before the synchronous router readback. The cudaStreamSynchronize completes the F32 copy before the CPU future reads it.
- `:3193-3207` only adds expert288 when its actual count is 1..8 and a stable source blob exists. It does not insert a transient blob. The ordinary CPU-share CPU path already consumes this source stability contract while GPU uploads run concurrently.
- GPU order excludes every on_cpu expert. CPU experts are placed last while building off; slot/src are then regenerated from those off values. The CPU future writes offsets `p-r0c`, with `r0c=T*K-rows_cpu`; completion is awaited before the matching Dm tail copy. This is coherent for resident and nonresident forced experts, and for other CPU-share experts in the same batch.
- `:3267` disables resident sorting when an on_cpu vector exists, protecting that CPU-tail partition. Under helper off/auto, this extra condition does not alter an otherwise eligible standard all-resident sort: normal CPU-share candidates are nonresident, whereas resident-sort eligibility requires every expert resident.
- Source/gamma/shared weights remain the existing layer references; the delta does not introduce a new weight allocation or release. The CPU future is joined before leaving the layer. on_cpu, cpu_blob, cf, raw rows, and output rows therefore retain the existing layer lifetime.

Prefill still deliberately requires an attached pool, no active peer, available stable blob, eligible 1..8 routed rows, and !stream_all. A large or fused/stream-all prefill is not newly guaranteed to reach this helper. If no stable blob exists, it logs prefill_unavailable and retains the original GPU route rather than publishing an incomplete CPU row.

## Defaults and verification limits

The new route overrides require the environment to be exactly forced. Off/auto retain the previous resident/device-plan, conditional doorbell publication, CPU-share admission, and GPU plan construction. This statement assumes configuration is fixed at process startup; changing getenv values after verifier initialization/capture is not a supported transition established by this review.

No static evidence here substitutes for process lifecycle/fallback tests, actual frozen-expert telemetry, NaN repair, whole-model quality, or end-to-end machine comparison. In particular, no successful helper invocation has been observed by this reviewer.
