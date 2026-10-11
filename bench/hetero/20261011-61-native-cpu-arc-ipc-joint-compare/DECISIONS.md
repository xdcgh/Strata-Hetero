# Run61 decisions

The two previously unrun synchronized CPU/Arc arms completed on the GTi15, Ultra 9 285H, Arc 140T, OpenVINO 2026.4.1. Both use the identified layer-28/expert-288 native Q4_K/Q5_1 operator, CPU service build59, and the same nine seeded input shapes as run60. One warmup and five formal trials were performed per shape and service.

All 216 responses passed: 108 CPU outputs are byte-identical to frozen native CPU gold; 108 Arc outputs pass unchanged 0.01 maximum-absolute and 0.001 relative-RMSE/row-norm limits. Root independently rehashed and recomputed every response. All worker/controller processes are terminal. Nine PSAPI samples pass the 12 GiB physical and 4 GiB commit floors; minimum available reserves are 105.74 and 111.69 GiB.

The caller timer includes input serialization, framed exchange and owned output copy. Input hashes, numerical checks, resource reads and output-file writes occur outside that timer. Two barriers keep both paired exchanges ahead of quality/file work. Compilation/INIT and warmup are excluded from the five-trial medians.

Run60 solo baselines and run61 concurrent arms were run separately, not interleaved. Their latency deltas describe these observations and do not isolate a cause. No hardware memory-bandwidth counter was collected. CPU16/none is much slower than CPU10/all in this persistent-service boundary and substantially increases observed Arc latency. Its cause is unresolved.

Keep Arc model dispatch disabled by default. Run60 has no transfer-inclusive crossover against CPU10/all at rows 1 through 256; run61 establishes numerical validity under paired calls but no full-model token/logit parity, 8% model gain, or <=2% automatic-mode regression acceptance. Do not enable a helper based only on worker-kernel duration or the near-equal 256-row paired medians.

Preserve run60's pre-start log-collision failure and all successful solo measurements. Run61 changes arm-specific log names and validates that frozen baseline; it repeats no completed solo arm. The three root admission attempts and the agent's self-matching process scan failed before any worker started and remain retained.

Raw F32 response payloads remain exclusively on E:, outside Git. `comparison.csv` and `receipts/root-output-review.json` contain the independently checked statistics and hashes. Next: the already prepared one-phase model zero-logit diagnostic, then upstream v0.1.42 integration with route-tail skipping explicitly disabled for matched quality controls.
