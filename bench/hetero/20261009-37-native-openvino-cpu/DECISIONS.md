# Native activation OpenVINO CPU

CPU constructs/compiles/infers the identified candidate at rows 1/16/256, with one warmup and three formals per shape. Reported execution is CPU, F32 hint and ACCURACY. All nine formals pass unchanged thresholds against frozen actual native CPU output; relative RMSE spans 2.10e-4..6.36e-4.

This validates the sampled device-graph arithmetic. It does not enable an engine helper or establish model throughput. Root still requires Arc execution, route latency, contention and model-level quality/performance.
