# Prepared v0.1.41 matched quality run 20261009-40-upstream41-direct-quality

Status: `prepared_not_launched`. This preparation performs no WSL shutdown, admission, server launch, API request, or model read.

Candidate: upstream41 at source fb58e0dbc8399662c0e47c76578c6e878b14f6cf (tree b5a210b6301395b703e3ed6224e0251c3ae5cc35); binary E:\Strata-Hetero-data\build\cuda-upstream41-03\strata.exe SHA-256 a6e2763ffce668e213f12380389223674621c47e4e5cb94c0f724e6ea8e686fc (52428800 bytes).

Bridge: shared E:\Strata-Hetero-data\source\hetero-0-1-41\serve\server.py, source checkout `ffeb748d190ddeedff23587ce848596d4db0c752`, SHA-256 `0ca2dd70485e5e03b0a6fcb5dfc1ee1a8863d2dc44947257e342ae608326b026`. The shared v1.41 bridge enables actual token-ID capture for both binaries.

Controls copied from reviewed F11: original F native dense GGUF shards (all four), original F PLE shard 2 with `--ple-io direct`, 20 GiB resident expert budget, 32,768 context, int8 KV, prompt cache 0, 10 workers, seed 42 / temperature 0 / top_p 1, no adaptation, `--pcie-frac 0`, same tokenizer/pack/MTP, same nine-prompt manifest.

New v1.41 behavior is explicitly frozen off in process-local config: `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_STAGE_PIN=0`. Other process-local environment entries retained individually from F11 are `PATH`, `CUDA_VISIBLE_DEVICES`, `STRATA_IQ_MT_MIN`, `STRATA_GGUF_PY`, and `STRATA_UNBUFFERED_LOAD`. Launcher clears inherited `STRATA_*` before server creation.

Planned sequence: one warmup plus three formal requests for each of the fixed nine manifest prompts; caps 128, except `smoke_python_function` 256. Compare every formal actual token-ID sequence with H4 direct baseline and apply the existing strict quality tool. Root must review fresh launch admission/ownership and separately authorize launch and quality. No controller is run by this preparation.

Comparison caveats: both runs use one shared bridge, prompts, and placement inputs. Only candidate engine source/binary differs between upstream and Hetero. Run sequentially. This is a task-quality comparison, not a performance acceptance or promotion result.
