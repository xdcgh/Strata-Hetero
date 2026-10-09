# Run 20261009-42 decisions

- The executed worker was the standalone Q4_K/Q5_1 FFN IPC helper for Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL layer 28, expert 288, with the original selected Q4_K gate/up and Q5_1 down payload slices.
- The service reported Intel Arc 140T `GPU.0`, exact execution-device list `[GPU.0]`, F32 inference precision, and the requested F32/ACCURACY compile policy. It compiled and ran the service graph and returned 45 outputs; all 45 passed the unchanged native CPU comparison gates, with independent root rehash/review in `root-output-review.json`.
- `native_operator_build.status=constructed_uncompiled`, `compiled=false`, and `inference_run=false` describe the nested native-operator build snapshot. They do not mean the enclosing OpenVINO service graph was not compiled or that its formal inferences did not run. Native-runtime parity and production model-route integration remain untested/off.
- Timings cover only this isolated helper request from client F32LE serialization through response receipt and owned output copy. `worker_wall_ns` is separately recorded; its subtraction from RTT is descriptive. INIT compilation and warmup are excluded from the nine-row formal CSV.
- The profile contains nine exact row observations, each backed by five formal RTTs and bound to the actual `receipt.json` SHA-256. It has `acceptance=[]`; no CPU cost was invented; shared system-RAM contention remains unknown (`contention_multiplier=null`).
- Runtime round-trip proof shows an unaccepted Arc helper is rejected and primary CPU remains selected. Identity mismatch and profile expiry also fail closed. The profile is a standalone evidence artifact and was not installed into the app, engine, or configuration.
- This is not whole-model correctness, route acceptance, model-level phase timing, or model speedup evidence. The default model route remains disabled.

Artifacts: `receipt.json`, `root-output-review.json`, `row-timings.csv`, `executed-client.py`, `source-freeze.json`, `hetero-profile.json`, and `profile-proof.json`.
