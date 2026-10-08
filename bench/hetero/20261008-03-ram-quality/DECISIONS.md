# Full PLE startup: commit floor failed

The first real Windows full-table run reached ready. The log reports the original quantized PLE table fully locked, 28,800,138,240 bytes, with initialization taking 63.3 seconds. GPU experts occupied 35.68 GiB and the expert RAM complement was 27.89 GiB.

The requested 71 GiB expert budget was clamped by available commit capacity. At the ready check, physical available RAM remained about 43.6 GiB, but commit availability was only about 1.0-1.2 GiB. An independent PSAPI system-commit reading agreed with the GlobalMemoryStatusEx diagnostic. The 12 GiB physical check alone did not detect this pressure.

Decision: retain startup and resource evidence, send no inference request, unload this owned model, and retry with the same 20 GiB expert budget in both upstream and RAM arms. Add a 4 GiB system-commit floor to resource checks. Do not change the Windows pagefile or close unrelated applications.

No quality result or speedup is claimed. The upstream reference had 36.05 GiB of expert RAM, so even a completed timing from this arm would not isolate PLE placement. Both startups used an already accessed model; neither is a proven cold-start measurement.

Evidence: `ready-commit-gate.json`, `resource/samples.jsonl`, engine startup log and `shutdown-unload.json`. The stop receipt confirms zero inference requests and no model processes remaining.

Default PLE mode remains direct. Full-table mode remains opt-in pending quality and performance validation under a safe complete model budget.
