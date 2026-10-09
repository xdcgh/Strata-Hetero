# Native activation on Arc

Literal GPU.0 execution reports Intel Arc 140T. Both the F32 hint and ACCURACY property are checked. Rows 1/16/256 each run one warmup and three formals. All nine formals pass the unchanged native-reference thresholds; relative RMSE spans 2.10e-4..6.92e-4. Original Q4_K/Q5_1 bytes remain intact. Global physical/commit gates are recorded before and after each inference.

The recorded inference wall times exclude process transport and do not establish a crossover or the 8% main-workload gain. CPU/Arc contention, full-model outputs/logits and stable-regression acceptance remain required. Keep production routing disabled; continue the identified-worker transport integration.
