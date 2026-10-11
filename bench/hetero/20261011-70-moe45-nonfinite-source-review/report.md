# Run70: layer-45 MoE source inventory

The inspected E69 source is clean at 1eb85d2a5d27cb2f054a9c757c3b074d7d3d7c86, tree 4fccda1c3a4a10952fa2e1079a43b98fb78ca548. File and metadata SHA pins are in report.json.

Run69 finite-probe metadata has four identical first-bad records: layer 45, half 1, chunk 0, moe_output, 1035 × 2560, row 154 / column 0, 2560 NaNs and no infinities. Sampled head rows show 0 finite and 248,320 NaNs. This is the earliest sampled boundary, not the upstream producer or cause.

At prefill.cpp:2845-2846, BF16-input projection writes FP32 router logits [T,E]; route writes ten int32 IDs and ten FP32 weights per token (kernels.cu:1491,2005-2012; N=2560 and K=10 at prefill.cpp:105). Shared expert gate/up are FP32 [T,640], SwiGLU writes IEEE FP16 sh_h, and down output plus scalar gate are FP32 (prefill.cpp:2907-2916).

The HC path has distinct images: m.mixed FP32 is scanned by the coarse probe; m.mixed_h is IEEE binary16 via hf(s); m.mixed_bf is BF16 on CUDA (kernels.cu:44-55,116-136,247-266). FP16 has 5 exponent bits; BF16 has 8. A finite m.mixed scan does not prove m.mixed_h finite. sh_h feeds shared down projection; Hh feeds only the FP16 fallback down projection. y_h and attn_h are not direct inputs to the layer-45 MoE branch.

Standard route mapping writes slot_h[i]=p and src_h[p]=i/K (prefill.cpp:3146-3151,3224-3252). FP16 fallback uses uint16 Xs, FP32 GU, IEEE FP16 Hh and FP32 Dm. MMQ uses FP32 GU/H/Dm and packed Q8 Xq/Hq. Fused uses GU grouping bytes and H int8 rows while Dm remains FP32 (prefill.cpp:2929-3012; moe_fused.cu:82-105,168-176,687-724). Native IQ fused kernels likewise write byte intermediates and FP32 Dm; their dispatch requires STRATA_PF_FUSED=1 plus supported types (moe_fused_iq.cu:426-432,1118-1148). Combine reads Dm[slot[t*10+k]*N+d], applies w[t*10+k], adds gated shared output, then writes bo (kernels.cu:1577-1613,2165-2176). bo aliases m.emb; the existing probe scans it after combine.

Captured config/provenance records CPU_SHARE=0, STAGE_PIN=0, ROUTE_TAIL_SKIP=0 and pcie-frac=0. The CUDA build receipt has MMQ_KQUANTS=OFF, but the CUDA CMake branch still adds strata_mmq (CMakeLists.txt:1354-1399). STRATA_PREFILL_MMQ=OFF is HIP-only and does not disable CUDA MMQ. Available logs have no per-layer MMQ/peer marker or fused banner, and do not identify layer 45's expert format. Effective path remains unknown.

A candidate m.mixed_h observation is after gr_mix/gr_mix_r (kernels.cu:136,266) before its consumer. A routed Dm observation is after producer waits/copies (prefill.cpp:3842-3870) before combine (3872-3876), with peer-sum layouts handled explicitly. Any new scan or copy can perturb timing/races; these are placement candidates, not no-effect guarantees or causal diagnosis.
