# Run44 upstream41 direct F-placement quality resume

Status: `prepared_not_launched`. This is a new RunId and new model load following run40 infrastructure failure. Run40 stays failed with one possibly dispatched but uncaptured first request; none of its outputs count as quality results or are reused.

Candidate: upstream v0.1.41 binary from `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, SHA `a6e2763ffce668e213f12380389223674621c47e4e5cb94c0f724e6ea8e686fc`. Shared capture bridge is v1.41 ffeb748 with the existing pinned SHA.

The config retains the four original F dense shards, F shard2 `--ple-io direct`, resident expert budget 20 GiB, max context 32,768, int8 KV, prompt cache zero, deterministic seed 42 sampling, and explicitly disables `STRATA_PREFILL_CPU_SHARE` and `STRATA_STAGE_PIN`. Nine fixed prompts use one warmup plus three formal requests, caps 128 except `smoke_python_function` 256, with strict quality and H4 actual token-ID comparison.

The quality controller and startup watcher use the new bounded JSONL reader. They require exactly one complete JSON object from a newline-terminated record, preserve any trailing partial bytes in an exclusive SHA-addressed file, enforce the current run's sampler owner and fresh timestamp, and never fall back to an older RunId's evidence. Pure fixtures cover multi-record, CRLF, partial tails, malformed rows, oversized and stale rows, and owner mismatch.

Root review is required before launch and separately before quality execution.


Coordinator: `launch-quality-controller.ps1` sequences one launch, one bounded startup watch, read-only `-CheckReady`, then the fixed `-Run` matrix in the same PowerShell host/session. It stops on the first error and never retries or cleans up the model; the server and sampler remain for root review. Startup watcher and CheckReady return to the coordinator rather than exiting the PowerShell host.
