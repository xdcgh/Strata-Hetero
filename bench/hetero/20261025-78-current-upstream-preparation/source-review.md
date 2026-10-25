# Current upstream token-ID bridge preparation

2026-10-25. **SOURCE PREPARED / EXECUTION NOT VALIDATED.** No Python, fixture,
compiler, device API, engine, model, model hashing, worker or benchmark ran in
this preparation. No dependency downloads. This is a separate source arm; it
does not replace the frozen diagnostic arm or finish the requested model A/B.

## Exact source identity

- Repository: `git@github.com:xdcgh/Strata-Hetero.git`.
- Branch: `codex/78-upstream-current-bridge`.
- Worktree: `E:\Strata-Hetero-data\source\upstream-7dd5cf18-bridge-20261025`.
- Exact upstream base: `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`.
- Product release: v0.2.4.2; engine project version: 0.1.49.2.
- Base `serve/server.py` Git blob: `71a55e1f0f9c8c95239396c2dd18297dee08e6df`.
- Local instrumentation reference: main `serve/server.py`, SHA256
  `32866083d6cb45872653790562adf5a27bc97395570ad8e3065e28a8ec298dbe`.
- Modified `serve/server.py`, SHA256
  `6420ea86c75b2209f3789336c9577e0fa42e5d1c994a87caa318836a0bcbefa7`.
- New `serve/test_hetero_token_ids.py`, SHA256
  `b10ffa5d7300576ea31cae6678764c68799af16cc81442f4f534363c75db3972`.

The full new checkout, including `strata_control`, is retained. Engine sources
are unchanged from the exact base. No source was copied into the historical
`C:\Users\DC\Documents\ChatGPT\Strata` checkout. No main index or main commit
was changed by this preparation.

## Small opt-in port

`hetero_capture_token_ids` must be literal JSON `true` in the local server
config. Default is false. A request cannot enable it. Capture is exposed only
when OpenAI chat-completions `stream` is literal `true`. With it off, there is
no extra token list and no diagnostics field. The old non-capture calls remain
the same, including current upstream's `declared_only=True` argument.

`Service._run` appends each non-None engine ID before EOS checks, detokenization,
stop-string filtering, repeat handling, or forced-call handling. It does not
encode visible text. The final SSE chunk carries
`strata_diagnostics.actual_generated_token_ids`, `include_stop: true`, and a
source description, matching `tools/hetero_compare_quality.py`'s evidence key.
Prompt IDs, server-inserted reasoning wraps, reasoning-close retries' inserted
IDs, and forced-call openings are excluded. IDs the native engine emitted as
accepted speculative tokens are included; speculative drafts that the engine
did not yield are excluded. This describes IDs consumed by the bridge, not
tokens possibly drained from native transport after the bridge stops early.

Current `StrataEngine.generate` dispatches both regular and batched generation
through this common iterator boundary. Its solo/admission/BT/BDONE transitions,
slot ownership and engine completion bookkeeping are untouched. Each service
request owns a separate capture list, including parallel requests and replicas.
Server continuations retain their accepted ID sequence. MCP rounds concatenate
their captured lists in round order, preserving the upstream declared-tool
behavior. Normal OpenAI SSE copies diagnostics only to the final chunk; no
Anthropic or Responses protocol changes were made.

The upstream nonfinite retry is permitted only at `n == 0`. Capture and `n`
advance together after the heartbeat branch, so the failed zero-token pass has
no captured IDs to clear. The existing failed-segment deletion remains intact.
An error after accepted IDs keeps only that prefix and the existing `length`
finish. A second empty failure still raises, without a successful final chunk.
Queue cancellation before generation retains upstream's empty cancel result;
it has no successful generated-token evidence and no added diagnostics.

## Checks and deferred execution

Completed here: textual diff inspection, `git diff --check`, source-file SHA256
metadata, and Git diff confirming no engine/backend files changed. These are
source checks only. Byte-identical default output is **not yet executed or
proven**. No runtime or performance claim is made.

Prepared 11 pure fixtures cover default off/request override rejection,
literal-stream gating, distinct IDs with identical decoded text, emitted EOS,
heartbeats, on/off non-diagnostic chunk equivalence, stop-string filtering,
empty nonfinite retry, partial nonfinite output, repeated nonfinite failure,
server-inserted reasoning wrappers, interleaved parallel request ownership,
and MCP aggregation. The batch fixture covers the service parallel branch;
it does not establish native BT/BDONE correctness or actual model output.

After the coordinator grants a runtime window, run from this full checkout:

```text
python -m unittest serve.test_hetero_token_ids -v
python -m unittest serve.test_server.NonFiniteLogits -v
```

Then compare normalized default-off fake-engine outputs with an unmodified
base, and run appropriate existing native-protocol pure fixtures. Fix any
fixture or implementation failure before launching the current engine.
Under a separately admitted model window, record actual binary identity,
resolved engine argv/env/profile state, explicit sampling/context/caps,
verified restored model hashes and configuration; execute the frozen OpenAI
SSE quality suite with capture on, then warm1/repeat3. Current upstream runtime
acceptance, real model quality and old/current A/B all remain NOT RUN here.
