# Run71 selected-layer typed MoE diagnostics

Diagnostic source is frozen and pushed at `3fbad60f479481333254f5ac1e6b04630ca37e49`, branch `codex/71-moe45-finite-detail`, isolated E71 worktree. Both compile/runtime options remain off by default. The optional `STRATA_PREFILL_DETAIL_LAYER=45` selects one layer for finer probes.

This follows Run69's four identical first-bad MoE output records (layer45/half1/row154, 2560 NaNs) and sampled all-NaN heads. It is not a numerical fix, accepted quality result, or performance improvement.

Root's final source review strips all 30 diagnostic compile blocks and verifies exact LF-normalized equality with the pinned original prefill. Static review covered same-stream producer/consumer ordering and typed buffer ranges. The final revision additionally scans consumed route weights after optional overrides and per-group down outputs after their producer. Fused byte intermediates are never read as floats; peer-sums Dm scans cover only the populated local prefix.

Typed scans distinguish FP32, IEEE binary16 and BF16. Records preserve format, expert when known, and packed row offset. Dequantized-weight matrix coordinates do not identify tokens. MMQ group scans may need bounds/order correlation to identify a member expert. Path flags are orthogonal predicates; mmq and fused can both be true.

The fixture retains all 60 original device cases and adds finite/NaN/Inf bit patterns, distinct FP16/BF16 exponent masks, first-format retention, input-bit preservation, and CUDA's host FP32-to-FP16 conversion at 65520. Layer parsing and single-line metadata formatting also have fixtures. Source review is complete; build/device/model acceptance is still pending. Initial mechanical inventory/driver transcription failures are preserved, with no device or model execution.

The upstream fetch still points to `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`. No upstream math fix is inferred. One serial CUDA build is authorized, with the inherited pinned toolchain, exact ownership, and physical/PSAPI resource floors. Tests and model execution require separate verified gates. Extra scans may perturb or mask races.
