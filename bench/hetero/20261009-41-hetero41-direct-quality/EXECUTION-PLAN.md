# Run41 Hetero v0.1.41 direct F-placement quality

Status: `prepared_not_launched`. This is a matched quality task against run44 using the integrated Hetero source/binary.

The coordinator `launch-quality-controller.ps1` uses one PowerShell 7 process to sequence one fresh-admission launch, exact-owner startup watch, read-only CheckReady, and then the fixed nine-prompt quality matrix. It leaves server and sampler live for root review; no retry is permitted.

Candidate binary SHA-256: `6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb`; source `ffeb748d190ddeedff23587ce848596d4db0c752`. Same shared capture bridge SHA, original four F dense shards and direct PLE on F shard2; resident 20 GiB; max context32768/int8 KV; prompt cache0; `STRATA_PREFILL_CPU_SHARE=0`; `STRATA_STAGE_PIN=0`; fixed nine prompts × warm1/formal3, caps128 except smoke_python_function256.

The controller and startup watcher use the reviewed bounded JSONL reader and keep `preflight-ready-binding.json` separate from the runtime `ready-binding.json`.
