# CPU vision port numerical result

OpenVINO CPU construction, compilation and inference succeed at both shapes under both explicit arithmetic policies. Reported execution devices are CPU, with F32 and ACCURACY hints. Each of eight image/policy groups has one warmup and three formals. All 24 formal outputs have valid shape, finite values and identical hashes within their repeats. Maximum absolute error is below 0.01.

All 24 fail the original 0.001 relative-RMSE/row-norm gates. `graph_f32` relative RMSE spans 0.00581..0.01377; `cpu_vec_dot_rounding` spans 0.00272..0.00761. Preserve the outputs and unchanged tolerances. This is a CPU port arithmetic/parity failure, not an NPU support or compilation result. NPU and production vision remain disabled pending a passing selected policy.

All 416 internal checkpoints and 21 external system-memory samples pass. The external owner RSS/private bytes refer to the Windows venv launcher, so they do not measure the actual CPython worker peak; whole-system RAM/commit gates remain valid. The process exits normally with code one for the numerical failure. Root independently rehashes all 24 output files.

Next diagnose the earliest native/port intermediate divergence with a bounded stage oracle. Do not replace the reference, widen tolerances, infer an end-to-end speedup from these timings, or call NPU unsupported from this result.
