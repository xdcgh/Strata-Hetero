# Prepared 32K experiment arms

These configurations are prepared, not launched or accepted. Both binaries use the same SDK/compiler/ggml/SM89 build settings. The upstream binary is from source 82f46a8; the RAM binary is the opt-in memory prototype. Both use loopback 8081 and existing F-drive weights, with no copied API credential.

Quality arms disable adaptive swaps/PCIe placement and enable the existing consistent CPU-row control. Performance arms retain the reference placement behavior but disable prompt-prefix reuse to measure full prefill. Do not compare quality-arm speed to the historical default placement. Source/model/config/engine identities must still be audited at launch, and raw outputs require quality review.

Create a fresh run directory, replace the engine log path with that RunId, regenerate its identity hash, run fresh admission, then launch only one arm. These files do not prove resource admission. The other chat's current benchmark process remains untouched; stopping its service requires the separate coordination requested from the user.
