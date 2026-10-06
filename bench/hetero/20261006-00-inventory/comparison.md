# Preparation evidence

This run is inventory and offline validation. It is not an inference benchmark.

| Item | Historical reference | Current preparation |
|---|---|---|
| Source | Strata `6f32ec0`, engine 0.1.39 | New fork based on upstream `82f46a8` |
| Hardware | GTi15 / 127.56 GiB / 49,140 MiB RTX 4090 D | Same observed hardware; availability changed while collecting |
| GPU inference | Historical artifacts retained | Deferred while the other chat uses the model |
| CPU host reader / lock tests | Existing source | Native Windows CTest: 2/2 passed without CUDA SDK/GPU |
| Inventory / admission / SSE harness fixtures | No Hetero tooling | 6 / 11 / 14 tests passed |
| Storage throughput | Not supplied as a physical-device profile | Not measured; priorities remain hypotheses |
| Speedup and model correctness | Historical samples | No new model result or accepted speedup |

The original and later admission failures are kept. WSL can be started by external processes; the observer never boots a distro. It fails closed on a running distro with unknown device users. Another chat's loaded Strata is a separate actual reason to wait. The user authorized temporary WSL/ComfyUI shutdown; restoration remains required as [documented](../../../docs/HETERO_ROLLBACK.md).
