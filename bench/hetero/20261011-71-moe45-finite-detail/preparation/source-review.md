# Run71 source review: typed layer-45 MoE probes

## Source and default guards

E71 is HEAD 8b0cedac83f02b088861069d998112279f550792, tree 4e6ee481debbfc51048545cff3c46da31639d597. Its worktree was already dirty in docs/HETERO_PREFILL_DIAGNOSTICS.md, finite_probe.hpp/.cu, prefill.cpp, and tests/cuda/prefill_finite_probe_test.cpp; hashes are in source-review.json. No files were changed by this review.

The source has CMake option STRATA_PREFILL_FINITE_PROBE default OFF (CMakeLists.txt:72), and runtime requested() accepts only exact "1". STRATA_PREFILL_DETAIL_LAYER is parsed only after the runtime probe is enabled; absent means no detail scans. I stripped the 28 compile-guard blocks from E71 prefill.cpp in memory and compared LF bytes to Cmain b50ba770: both are 270,243 bytes with SHA-256 5e9153e991e7d91ef436818d531387a3ca4d0842e7c4e985763090c6d7145d70.

## Producer positions and extents

The HC typed scans follow gr_mix/gr_mix_r and distinguish m.mixed FP32, m.mixed_h IEEE FP16, and CUDA m.mixed_bf/low BF16 images (prefill.cpp:2575-2584). Router logits [T,E] follow bf16_proj and precede route; weights [T,K] follow route (2858-2866). Shared gate/up, activation, down, and scalar scans follow their producer kernels and precede their next consumers (2930-2955).

Per-group MMQ scans use GU+r0*1280 and H+r0*640, each with nr rows; H is scanned after SwiGLU and before quantize, and Dm down writes begin at r0*N (3774-3805). FP16 fallback scans cover Xs+o0*N for ne x N, FP16 dequant weights [1280,N]/[N,640], GU [ne,1280], and Hh [ne,640], with expert e and packed row offset o0 (3818-3845). These extents align with the identified producers and scratch windows.

Dm is scanned after CPU/peer producers converge and before combine. CPU placement requires no peer handle, so CPU tail rows cannot overlap peer-sums mode. Peer-sums scans only [0,back_at); ordinary/peer-row-copy paths scan T*K rows (3227-3263,3412,3627-3642,3936-3959).

## Record and log semantics

F16 and BF16 classifiers use their respective IEEE exponent masks; sign separates positive and negative infinity, and NaNs are counted separately (finite_probe.cu:13-42). Same-stream kernels atomically preserve the earliest bad scan and lowest index while counting all bad elements from that scan. Record copy/report occurs after the existing stream sync. row_offset/source_row are packed scan coordinates; per-expert fallback scans include expert=e, while grouped MMQ scans leave expert=-1 because a group spans experts.

The new moe_path record is one fprintf call and reports layout/dispatch predicates and peer/CPU fields. mmq and fused are not one-hot labels: use_mmq may coexist with fused_l. A detail run with STRATA_DBG_ROUTE_OVERRIDE set would scan route-produced m.w before the override rewrites m.ids/m.w (2863-2924); that would not describe the effective downstream weights. MMQ first_bad rows also cannot identify the member expert without the group bounds/order.

No source edits, compile/build, tests, engine or device work were performed.
