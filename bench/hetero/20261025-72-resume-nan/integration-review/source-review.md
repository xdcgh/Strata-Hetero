# Forced Arc helper source review

Scope: read-only inspection of uncommitted integration in `E:/Strata-Hetero-data/source/hetero-live-arc-20261025`, including new files. No Python, test, compiler, GPU, model, WSL, subprocess, or device operation was executed. This review does not establish runtime correctness or full-model acceptance.

No blocking defect was identified in the inspected actual-model multi-job path. This is a static result only.

## Verified source connections

- `expert_source.cpp:3798-3805` attaches current dispatch layer, actual expert ID, and raw model activation pointer; the existing native job blob and original output pointer remain intact. `prefill.cpp:3298-3307` does the corresponding wiring for the existing CPU-share path.
- `pool.cpp:797-815` only filters jobs when the environment is exactly forced. Off and auto do not enter the helper. Failed jobs remain in the native CPU pending batch; successful jobs keep their original output locations.
- `live_helper.cpp:110-118` restricts the frozen identity, quantization, dimensions, offsets, blob size, and 1..8 rows. It copies the actual job blob and actual input rows into the bridge payload.
- `hetero_live_bridge.py:33-52` checks the actual received gate/up/down segment hashes. It does not substitute the standalone fixture blob or golden activations. The Arc child computes from frozen artifacts only after their raw segment binding matches those actual model bytes. READY checks include identity hash, weight hashes, exact GPU.0, native operator signature, and segment binding.
- C++ validates response magic/version/kind/request ID/length and every output float before copying any original output row. A short, partial, malformed, failed, or nonfinite response leaves original output rows untouched before CPU fallback.
- Windows child starts suspended, then joins a kill-on-close Job Object before resume. Its handle whitelist contains only protocol pipe ends. Timeout/error cleanup targets the stored process and owned job, not a process-name search. The Arc service inherits the owned job; the bridge's OwnedPipeWorker also targets its exact Popen object.
- Both ordinary helper failures and protocol failures poison the helper and prevent restart. Exceptions within successful output validation trigger cleanup before returning false.

## Nonblocking limitations to keep explicit

1. `live_helper.cpp:135-138`: the 120-second future wait starts after `pipe.start()`, mutex acquisition, payload allocation, and copying. It bounds bridge request/Arc startup/compile/transfer, not the complete helper call. The wall_ms telemetry correctly starts earlier. Narrow the comment's startup wording to Arc startup, or use one absolute deadline if the requirement is a complete-call timeout. No evidence here establishes that ordinary CreateProcess stalls in practice.
2. `expert_source.cpp:3487-3488` and `generate.cpp:972-976`: the older single-token `expert_pool_dispatch` path still calls run_split/run and has no helper connection. The verifier multi path (`generate.cpp:1002-1007`) and prefill CPU-share path are connected. Keep the documented supported execution mode precise; any acceptance run must prove `hetero_live completed=1` in its actual engine mode rather than assume every generation entrypoint reaches the helper.

## Required runtime evidence, not supplied by this review

The host mock fixture does not exercise Windows process creation, job assignment, process-tree timeout cleanup, partial pipe writes, actual native CPU fallback output equivalence, or an actual model callsite. Those need targeted verification before full-model trials. Subsequent model acceptance must first resolve the independent NaN, then show actual frozen expert dispatch, real returned model output integration, quality/logit checks, and identical-driver whole-machine comparison. A static code review or isolated helper fixture cannot replace those results.
