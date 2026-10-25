# Experimental live expert helper

This source connects actual model jobs to the existing native Arc SXPU service.
It is not a full-model acceptance result. Four Python typed mock tests passed;
their raw logs and source/output SHA256 receipt are in
`bench/hetero/20261025-live-arc-hostmock-review/`. No C++ build, compiled fixture
run, device run, quality check or end-to-end A/B has been performed for this
integration yet.
Earlier isolated CPU/Arc measurements did not establish a stable joint speedup.

The default is off. `STRATA_EXPERT_HELPER=auto` also routes nothing: no model
quality or gain acceptance exists. Only `STRATA_EXPERT_HELPER=forced` attempts
the experimental route. On Windows set `STRATA_EXPERT_HELPER_COMMAND` to a
direct, quoted Python command invoking `tools/hetero_live_bridge.py` with:

```
--identity ABSOLUTE_IDENTITY_JSON --identity-sha256 PINNED_SHA256
--weights ABSOLUTE_WEIGHTS_NPZ --native-blob ABSOLUTE_NATIVE_BLOB
--stderr NEW_EXCLUSIVE_ABSOLUTE_ARC_LOG
```

Use the Python interpreter containing the already validated OpenVINO runtime.
The engine launches the command directly with CreateProcess, without a shell.
Do not use a shell wrapper. Paths containing spaces must be quoted by the caller.
The bridge imports no device runtime; its separately owned Arc child does.
Do not enable this during another device experiment or without fresh resource
admission. This source does not implement the system RAM admission controller.

## Reachable model path and limits

`expert_pool_dispatch_multi` attaches actual mapped model F32 rows and expert
identity to native CPU jobs during generation/decode/verify. The prefill CPU
share path attaches its actual CPU input rows. `run_split_multi_native` sends
only matching jobs to `live_helper`; successful results replace those original
job output rows and the existing model combine applies router weights.

Only layer 28, expert 288, Q4_K gate/up and Q5_1 down, hidden 2560,
intermediate 640 and 1..8 rows are supported. Forced generation on a single GPU
explicitly removes this expert from the host GPU plan when the actual router
selects it, even if resident in CUDA cache. The existing model source supplies
the actual blob; its native CPU job remains available as fallback. Other
experts retain existing placement. Peer/remote helper configurations are
excluded from the direct override.

Forced verification disables all-resident/device-plan shortcuts and always
publishes actual F32 activation rows to the host; otherwise an all-resident
layer could supply stale inputs. This overhead belongs in forced A/B timing.

Forced prefill takes the CPU pool when available, copies actual layer 28 input
rows even if the adaptive share would skip that layer, and removes expert 288
from GPU order when it has 1..8 rows and a stable source blob. It bypasses the
ordinary resident exclusion while retaining stream-all, buffer allocation,
stable-blob and no-peer limits. Other jobs follow the configured CPU share.
This does not make a large-batch prefill helper. NPU is not integrated.
Neither path changes router IDs to manufacture a hit.

For the first live demonstration, replay the actual natural-text corpus that
produced the expert profile (28/288 was selected from that profile), with the
existing single-GPU generate driver and at least 256 generated tokens. Keep
model, prompt token IDs and all ordinary generation parameters identical in
the off and forced arms. Use native experts and no peer/remote expert options.
The source route exists even if this expert is CUDA resident; the router must
still actually choose it. No new prompt has yet been demonstrated to select
it in this integration. A run with no helper completion is not acceptance.

Require engine logs in this order: `forced_host_plan=1`, a
`route=host_plan_override ... expert=288 cached=1` (or a prefill override),
then `attempt=1 completed=1 fallback=0`. Confirm the same run returns finite
model logits and usable output, then compare its output/logits and full wall
timing with the off arm. `fallback=1` proves fallback, not Arc participation.
The forced host-plan changes affect timing and belong to that comparison.

Every attempt copies the actual engine blob and F32 inputs to the bridge. The
bridge checks all three raw segment SHA256 values against the pinned identity,
then checks the Arc READY identity, raw segment binding, weights hashes, exact
device and native operator. No golden activation or test blob feeds the model.
SXPU version/kind/request ID, bounds, finite input/output and result dimensions
are checked. The entire output is validated before any original output changes.

The synchronous experimental route adds serialization and hashing costs and
does not overlap Arc with the remaining CPU batch. It may be substantially
slower. Larger batches, asynchronous overlap and accepted automatic decisions
remain future work requiring measured full-model quality and latency gates.

## Failure and logs

Unsupported jobs continue unchanged through native CPU execution. Failure of an
eligible helper attempt leaves outputs untouched and runs that original CPU
job. The engine owns a Windows kill-on-close Job Object containing the bridge
and Arc child. Handle inheritance is restricted to the bridge protocol pipes.
Process creation/assignment/I/O failure falls back. After process creation and
owned-job assignment, the IPC wait has a 120 second timeout covering worker
initialization/compile and request/response transfer. Process creation itself
is outside that timeout. Reported wall time includes startup, serialization
and hashing. Timeout cleanup can add a five second process wait plus pipe
thread drainage. It kills only the owned process tree; failed workers are not
restarted.

Each eligible attempt writes `hetero_live` to engine stderr with phase, layer,
expert, rows, attempt/completed/fallback, reason, serialized input bytes,
host wall time and worker inference nanoseconds. Worker nanoseconds exclude
compile, transport and hashing; host wall time includes them. `completed=1`
means the returned values were written into model job outputs, not that model
quality or performance passed. Jobs outside the frozen identity do not count
as attempted. Bridge error details accompany the fallback reason.

Pure-host fixture source is `tools/test_hetero_live_bridge.py`; it uses a mock
typed worker, synthetic blobs and rows, and cannot establish actual model
correctness. Required next checks are a targeted build, host lifecycle and
fallback tests, then real model telemetry, output/logit quality and identical
driver A/B after the independent NaN issue is repaired.

## Eligible Windows host fixture source

`live_helper_host_fixture` without arguments checks off/auto/unsupported
fallback without spawning anything. Its `eligible-success` and
`eligible-failure` arguments use the real production `live_helper`/Pipe with
synthetic two-row inputs, an eligible native format and a synthetic 3 MB blob.
The command is supplied by the host runner as a stdlib SXPU mock, so these
cases never import OpenVINO, read a model or perform expert math. They exercise
the C++ boundary and cannot establish actual model or Arc correctness.

After a separately authorized build, run:

```
python tools/hetero_live_host_lifecycle.py --fixture ABSOLUTE_EXE --output NEW_ABSOLUTE_RECEIPT_DIR
```

This runs success, explicit rejection, truncated wire frame, wrong output
length and last-element NaN. Both output rows must either become the expected
mock values or stay entirely at their original sentinels; failed transports
are called again to verify poison prevents restart. Every mock deliberately
spawns a sleeping descendant. At fixture exit the runner checks the mock and
descendant using PID plus creation FILETIME and a process-handle wait, with
five seconds allowed for exit propagation. Access errors are failures.

Add `--include-timeout` to exercise the unchanged production 120 second IPC
wait and descendant cleanup after timeout. The runner bounds this scenario
at 145 seconds; its own timeout is a fixture failure, even if cleanup succeeds.
All compiled eligible scenarios remain **NOT RUN / UNBUILT** in this revision.
Source additions alone provide no child lifecycle validation.
