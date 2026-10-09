# Head-only zero-token diagnostic run

The frozen diagnostic binary completed five short and four natural-text tasks, each with one warmup and three formals under the fixed cap contract. All 27 formals passed task checks and match H4's actual emitted IDs and visible text. Every natural request reported the expected prompt usage and zero reused prompt tokens.

The diagnostic does not enable whole-prefill DBG_NAN synchronization. It reads scores only after an already-sampled ID 0. No diagnostic line was emitted in this run; the readback branch did not fire. Therefore per-row score statistics remain unavailable, not zero or proven finite. Differences from H4/H5 include compiled source/layout, the optional CPU condition check, startup state and PLE mode; no causal repair or default performance acceptance is asserted.

H5's uninstrumented failure remains intact. A new same-binary diagnostic-off control is prepared separately to isolate the flag. Do not accept full-table RAM as stable merely from this run, and do not discard the earlier repeatable degeneration.

Exact owned unload/STOP/bridge cleanup finished, including the actual Python descendants. The first unload's HTTP 415 and the corrected JSON request's 200 response both remain. Shutdown-final records identity, sampled resource minima and all tracked process/listener exits.
