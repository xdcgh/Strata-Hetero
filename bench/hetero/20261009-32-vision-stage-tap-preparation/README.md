# Bounded native stage-tap feasibility

Status: read-only source review, not implemented or run. CPU run 31 accepted all four graph/CPU runtime arms on Intel Core Ultra 9 285H, OpenVINO 2026.4.1, reported float32/ACCURACY, but all 24 formal embeddings failed the unchanged relative gates. Both policies remain ineligible. Run 31 preserves the terminal receipt, 32-output hash index, 24 negative NPY files on E and resource evidence. This plan keeps the original native reference and neural-network operations.

## Public callback path

`mtmd_context_params` already exposes `ggml_backend_sched_eval_callback cb_eval` and `void * cb_eval_user_data` at [mtmd.h:97–113](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd.h#L97-L113). [mtmd.cpp:568–576](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd.cpp#L568-L576) forwards them to `clip_context_params`; [clip.cpp:220–225](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L220-L225) installs them with `ggml_backend_sched_set_eval_callback`.

Signature is `bool callback(ggml_tensor * t, bool ask, void * user_data)`. Return true during `ask` only for whitelisted taps; during observation copy data and return true to continue. The scheduler executes through the requested node and synchronizes its backend before observation, [ggml-backend.cpp:1805–1830](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-backend.cpp#L1805-L1830). No GGML/CLIP graph-source patch is needed to register this hook in an opt-in helper.

Copy with `ggml_backend_tensor_get(t, destination, 0, ggml_nbytes(t))`, after checking F32 type, expected `ne`, contiguous `nb`, allocated buffer and byte budget. `ne`, `nb`, `src`, `view_src` and `view_offs` are public [ggml.h:688–708](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/include/ggml.h#L688-L708); backend copy validates bounds and handles view buffers, [ggml-backend.cpp:350–362](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-backend.cpp#L350-L362). Copy inside the callback; retain bytes/metadata, not pointers after encoding.

## Exact selectors and axes

Use the pinned source `3cf03257f219afbe7334045ff7c6a06ac68c627d`, selected asset `b1a82259...` and the existing aligned PNGs. Let `N=36` or `72`, `E=1152`. Every selected activation is F32; GGML dimension 0 is the fastest feature axis. Preserve all four `ne`/`nb` values in metadata, and export contiguous rows in the listed NumPy axes.

| Stage | Native selector | Export axes | Source |
|---|---|---|---|
| Patch merge + bias | `patch_bias` | `[N,E]` | qwen3vl.cpp:18–36 |
| Positioned | `inp_pos_emb` | `[N,E]` | qwen3vl.cpp:39–52 |
| Affine LN1 | `ln1-0`, `ln1-1`, `ln1-26` | `[N,E]` | qwen3vl.cpp:75–77 |
| First combined QKV + bias, before RoPE | At `Qcur-0`, copy its `src[0]`/`view_src` parent, asserting `[3E,N,1,1]`, F32 contiguous, ADD after MUL_MAT | `[N,3E]`, contiguous Q then K then V | qwen3vl.cpp:81–101 |
| Block residual output | `layer_out-0`, `layer_out-1`, `layer_out-26` | `[N,E]` | qwen3vl.cpp:139–141 |
| Global post-norm | `norm_b-27` for this asset's present weight+bias | `[N,E]` | qwen3vl.cpp:163–165; clip.cpp:591–610 |
| Four-patch merger input | `norm_b-27 (reshaped)`, asserting RESHAPE, `view_offs=0`, parent post-norm and `[4E,N/4,1,1]` | `[N/4,4E]` | qwen3vl.cpp:168–170; ggml.c:3742–3756 |

Names are assigned by [clip.cpp:303–307](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L303-L307). The first Q view is `[72,16,N,1]` with token stride `3E*4`, so its raw span is **not** a packed `[N,E]` array; copying the full contiguous QKV parent avoids strided Q/K/V packing. The merger's generated name is explicit in [ggml.c:3742–3756](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml.c#L3742-L3756).

CPU split graphs preserve view nodes as graph ranges, ggml-backend.cpp:1303–1317,1460. CPU's graph-optimization hook is NULL, [ggml-cpu.cpp:199–209](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.cpp#L199-L209). Source therefore supports these view selectors; actual callback coverage remains a runtime gate. Missing/unexpected taps must fail, not substitute guessed tensors. `27` is the global post-norm suffix, not an additional transformer layer; transformer captures are restricted to 0/1/26.

## Bound and comparison

The whitelist totals `13*N*E*4` bytes per capture, counting separate post-norm and merger copies. At the larger fixture this is 4,313,088 bytes. Three formal captures total 12,939,264 bytes (12.34 MiB), below the strict 16,777,216-byte per-image cap. Disable tap capture for warmups. Enforce the aggregate cap before allocations/writes and retain partial failures; no weights or other layers are dumped.

Existing OpenVINO outputs already expose `patch_merge`, `positioned`, `block.0/1/26`, `post_norm` and `merged`. LN1/QKV outputs need opt-in aliases of the already computed nodes after root review; no re-computation or math changes. Compare in forward order to identify the first divergence. Keep every policy separate and preserve the original native SVE reference and unchanged tolerances.

Callbacks change scheduler batching, and extra OpenVINO outputs can affect fusion. A future instrumented run must verify its final embeddings against the frozen run-23 native/run-31 OpenVINO fingerprints; a changed final output is an instrumentation finding, not permission to switch references. Callback errors need a sticky helper error and explicit post-encode completeness check: returning false only breaks the current split's callback loop in this implementation, so it must not lead to an `OK` with partial/stale data. See ggml-backend.cpp:1829–1835.

## Current monitor limitation and future guard

Root identified run-31 owner PID 19424 as the E-venv launcher; the actual C-Python child performed runtime work. Original receipts are unchanged. Their RSS/private-byte fields describe the launcher and cannot establish runtime process memory. Current safety claims use the valid whole-system RAM/commit measurements: 416 internal gates and 21 external samples all passed, with minimum 105.2340 GiB RAM and 110.1423 GiB commit available. No memory stop fired.

Before another Python runtime run, the external guard must discover and bind the exact owned launcher **and runtime descendants** by PID, creation time, executable and argv, then stop/reap that verified tree if the 12/4 gate fails. Killing the launcher alone is insufficient. No process-name or generic Python kill is permitted. The source inventory records this limitation separately from immutable run-31 evidence.

No helper/source patch, compilation, stage runtime, GPU/NPU work or API change was made during this feasibility review. Root review precedes implementation and execution.
