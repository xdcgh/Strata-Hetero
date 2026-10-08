# NPU validation status

The isolated OpenVINO 2026.4.1 environment uses official-PyPI-verified binary wheels and does not replace the text engine's environment. On 2026-10-08 the runtime enumerated `NPU` as Intel AI Boost, architecture `3720`, and reported driver `2267`. The exact discovery and package receipts are under `bench/hetero/`.

A small synthetic FFN probe initially failed before inference because this NPU configuration rejected `EXECUTION_MODE_HINT=ACCURACY`. The device-specific worker then omitted unsupported hints explicitly and retried with the same synthetic weights and an F16 graph. Nine static row sizes compiled and ran with `EXECUTION_DEVICES` exactly `NPU`; no CPU/GPU fallback was allowed. This proves those tiny static compilations and inferences are available on this installed runtime/driver pair.

The probe did not pass the numerical quality gate. Its small output magnitudes produced relative RMSE about 0.0052..0.513 against the float32 reference, above 1e-3, despite small absolute errors. The NPU did not advertise the precision-hint property, so reported precision policy remains unknown. All crossover eligibility fields are false. These synthetic latencies are not vision or MoE performance claims. Failed and complete receipts are retained as runs 13 and 14.

Vision remains the first intended NPU integration. The selected UD-Q4_K_XL setup currently disables images. The existing helper consumes GGUF mmproj weights and emits SVE1 image embeddings; an OpenVINO implementation must preserve preprocessing, dimensions, positions and embedding values before connecting it. Actual vision conversion, query/compile coverage, visual task checks, CUDA comparison and GPU VRAM savings are pending.

Do not infer that the installed driver needs replacement from a preview-only dynamic-shape version threshold. The synthetic static probe actually ran; compatibility with the real encoder still needs measurement. NPU embedding, reranking, routing and MTP remain later candidates. No NPU route is enabled by default.
