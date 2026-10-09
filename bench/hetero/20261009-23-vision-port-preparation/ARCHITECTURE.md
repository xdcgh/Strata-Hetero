# Qwen3VL vision port preparation

Status: source port and synthetic fixtures prepared. Native encoder parity, real OpenVINO construction, CPU compilation, NPU compilation, image task quality and performance are **NOT RUN**. No vision route is enabled.

The port is `tools/hetero_vision_qwen3vl.py`. Its default CLI consumes the existing JSON header inventory only. It does not open the referenced GGUF, decode tensor payloads, import NumPy/OpenVINO, create a Core, query devices, compile or infer. The graph API requires caller-supplied decoded arrays, array hashes, explicit static image dimensions and an explicit arithmetic policy. Admission, asset decoding, serialization, devices and execution belong to a separately reviewed caller.

## Identity and source boundary

Selected asset, recorded by the earlier download/header receipts:

| Field | Recorded value |
|---|---|
| Asset | `E:\Strata-Hetero-data\models\vision-ed59f92\mmproj-Qwen3.8-Flash-Next-BF16.gguf` |
| Asset revision | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` |
| Asset bytes | 907,543,008 |
| Asset SHA-256 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| Header receipt | `bench/hetero/20261009-16-vision-encoder-headers/vision-header-inventory.json` |
| Source receipt | `bench/hetero/20261009-16-vision-encoder-headers/mtmd-source-pointers.json` |
| Authoritative source | `E:\Strata-Hetero-data\source\llama-3cf0325` |
| Source revision | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |

This preparation read source and JSON receipts. It did not reread the asset hash or any tensor payload. `source-hashes.json` records freshly checked source hashes; they match the earlier mtmd pointer receipt for the overlapping files. Source checkout status was clean during this preparation.

The JSON header identifies 334 descriptors, including 27 consecutive blocks. Vision width is 1,152, FFN width 4,304, heads 16 and head dimension 72. Patch size is 16 and spatial merge size is 2. The learned position table is GGUF `[1152,2304]`, a 48×48 grid. The projector produces width 2,560. The table's two patch kernels are GGUF `[16,16,3,1152]`. `clip.use_gelu=true`; norm epsilon is the actual stored FLOAT32 value `9.999999974752427e-07`. There are post-norm weight and bias tensors, no pre-norm tensors, no FFN gate and no deepstack tensors; all 27 deepstack flags are false. These are header facts, not numerical validation.

The port rejects a missing epsilon, wrong projector/activation, non-square learned-position table, inconsistent shapes, additional unknown tensors, gated FFNs, deepstack and merges other than 2. The selected asset has no temporal-patch-size or rotary metadata. The still-image graph does not invent those metadata values: it uses the two convolution branches and the constants in the pinned source. Video/two-frame routing is not implemented.

## Static graph

Let `GH=H/16`, `GW=W/16`, `N=GH*GW`, `T=N/4`, `E=1152`, `F=4304`, `heads=16`, `D=72`. Both static image edges must be divisible by 32. Batch is exactly one still image.

| Stage | Port representation | Pinned source |
|---|---|---|
| Input | F32 `[1,3,H,W]`, normalized RGB, planar NCHW | [clip.cpp:584–588](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L584-L588), [input packing:4527–4568](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L4527-L4568) |
| Temporal merge | Two independent stride-16 convolutions of the same still image, no padding, dilation 1; add the results | [qwen2vl.cpp:3–29](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen2vl.cpp#L3-L29) |
| Patch reorder | Flatten into `[N,E]`, gather tile order, then add patch bias | [qwen3vl.cpp:18–37](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L18-L37) |
| Learned positions | Resize 48×48 to `GH×GW`, bilinear ALIGN_CORNERS, then the same tile reorder; add to patches | [qwen3vl.cpp:39–52](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L39-L52), [clip.cpp:312–331](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L312-L331) |
| Optional pre-norm | Ordinary layer norm if an identified pre-norm weight exists | [qwen3vl.cpp:60–63](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L60-L63) |
| Block norm 1 | Centered population variance over feature axis; divide by `sqrt(variance+epsilon)`, multiply learned weight, add learned bias | [clip.cpp:591–613](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L591-L613), [CPU norm:3840–3888](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ops.cpp#L3840-L3888) |
| QKV | Combined linear projection + bias `[N,3E]`; split Q,K,V, reshape/transposes to `[heads,N,D]` | [qwen3vl.cpp:79–101](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L79-L101) |
| Vision M-RoPE | Q and K only; constant cosine/sine cache for the identified grid; pair channel `j` with `j+36` | [qwen3vl.cpp:103–109](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L103-L109), [CPU cache/pairs:5991–6077](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ops.cpp#L5991-L6077) |
| Attention | `softmax(Q @ K.T / sqrt(72), key_axis) @ V`; flatten heads, output projection + bias; add first residual | [clip.cpp:748–818](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L748-L818), [scale:260–264](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L260-L264), [residual:119–122](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L119-L122) |
| FFN | Norm 2; up `[E,F]` + bias; tanh GELU; down `[F,E]` + bias; add second residual | [qwen3vl.cpp:126–141](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L126-L141), [clip FFN:616–702](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L616-L702) |
| Optional post-norm | Present for this selected asset, applied before four-patch reshape | [qwen3vl.cpp:163–166](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L163-L166) |
| Merger | Consecutive tile rows reshape to `[T,4608]`, preserving four complete patch vectors; mm.0 + bias, tanh GELU, mm.2 + bias | [qwen3vl.cpp:168–176](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/models/qwen3vl.cpp#L168-L176), [projector names:2474–2480](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L2474-L2480) |
| Output | F32 `[T,2560]`; token rows follow row-major merger grid `[GH/2,GW/2]` | Source graph reshape above; existing SVE1 helper below |

For a 2×4 patch grid, the patch order is `0,1,4,5,2,3,6,7`. More generally the loops are `tile_y, tile_x, dy, dx`, with `dy,dx` each 0 then 1. Four patch vectors concatenate as top-left, top-right, bottom-left, bottom-right. Transformer attention runs over individual patches in that order; merging them before attention would change the model.

Static plans:

| Input height × width | Patch grid | Merger grid `(ny,nx)` | Output |
|---|---|---|---|
| 96 × 96 | 6 × 6 | 3 × 3 | `[9,2560]` |
| 96 × 192 | 6 × 12 | 3 × 6 | `[18,2560]` |
| 192 × 96 | 12 × 6 | 6 × 3 | `[18,2560]` |

The default CLI validates the first plan, not a model execution. It can print another plan using `--height` and `--width`. No `--execute` or device option exists.

## Position interpolation and RoPE details

Learned positions decode as `[2304,1152]`, with x varying fastest within the 48×48 grid. The port uses four static gathers and explicit bilinear arithmetic, rather than relying on a runtime interpolation mode to match GGML. Source interpolation uses F32 `sf=(target-1)/(source-1)`, coordinate `i/sf`, zero pixel offset, clamped neighbor indices and fractions; singleton edges retain the source's fallback ratio rule. The source's four-term multiplication/addition order is preserved. It does not apply antialiasing. See [ops.cpp:7991–8004](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ops.cpp#L7991-L8004) and [8082–8119](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ops.cpp#L8082-L8119).

Input RoPE positions are axis-major `[y,x,y,x]`, generated in the same 2×2 tile order by [clip.cpp:4783–4806](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L4783-L4806). Source `ggml_rope_multi` arguments are: `n_dims=36`, sections `[18,18,18,18]`, mode `GGML_ROPE_TYPE_VISION`, original context 32768, frequency base 10000, frequency scale 1, extrapolation factor 0, attention factor 1, beta-fast 32, beta-slow 1.

VISION mode initializes a cache for all 72 channels and rotates all channels, pairing the first and second 36-channel halves. Its independent-section cache chooses y for the first 18 pairs, x for the next 18 pairs; each section restarts its frequencies. Frequency progression is F32 repeated multiplication by `10000**(-2/36)`. Extra axes are supplied as the source supplies them, but are not reached by the 36-pair loop. This is not the interleaved text `GGML_ROPE_TYPE_IMROPE`. The scalar fixture independently translates the pinned CPU cache/pair loops; no compiled native oracle has run. Python and native libm transcendental rounding remain a later numerical check.

## Arithmetic policy

`arithmetic_policy` is a required graph API keyword. Both policies keep graph tensors F32 and materialize the input's F16 roundtrip for the selected F32 patch kernels, because [ggml_conv_2d:4753–4774](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml.c#L4753-L4774) explicitly emits an F16 im2col operand.

| Policy | Explicit behavior | Limit |
|---|---|---|
| `graph_f32` | Decoded weight values in F32, F16 patch-input materialization, full F32 linear/attention math, pinned tanh GELU | Does not reproduce every CPU backend storage/activation rounding decision |
| `cpu_vec_dot_rounding` | Also round F32 patch kernels through F16, BF16 activation roundtrips for BF16 linear weights, GELU F16 input/output roundtrips and original-input tail selection | Models the CPU vec-dot/LUT rounding path; does not prove native dispatch/reduction/libm parity |

The FFN activation is GGML's tanh GELU, `0.5*x*(1+tanh(0.7978845608028654*x*(1+0.044715*x*x)))`, not an unspecified GELU default. [clip.cpp:1368–1384](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L1368-L1384) selects FFN_GELU from this asset's metadata; [vec.h:963–1008](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/vec.h#L963-L1008) defines the formula and the CPU F16 LUT path. On that CPU path, original F32 values <= -10 produce zero, >= 10 are preserved, and interior values are rounded through the F16 lookup table. The port preserves the formula, rounding locations and tails, but does not claim exact lookup-table libm bits.

The CPU F16/BF16 vec-dot types are specified in [ggml-cpu.c:222–226](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c#L222-L226) and [395–399](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c#L395-L399). [Matmul:1273–1360](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c#L1273-L1360) can dispatch to llamafile before its vec-dot conversion path. Therefore that conversion path is an explicit comparison arm, not an assumed universal native behavior. Flash attention is not implemented in this port; later native reference runs must specify `--flash-attn off` because the flash branch casts K/V to F16.

No tolerance has been changed or claimed to pass. Device precision, backend reductions and transcendental rounding must be measured against the selected native oracle before accepting either policy.

## Preprocessing contract

The graph accepts an already preprocessed F32 tensor. It does not decode, resize or pad images. The external frontend must export and verify this input independently of the final embedding comparison.

Source [mtmd.cpp:694–703](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd.cpp#L694-L703) chooses dynamic image preprocessing for Qwen3VL. [Qwen loader:1651–1667](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip.cpp#L1651-L1667) chooses bicubic resize and source-default token limits 8..4096; with patch 16 and merge 2, pixel limits are 8,192..4,194,304 via [clip-model.h:190–195](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip-model.h#L190-L195). User overrides can replace those limits and must be recorded. Image-size metadata 768 does not force all images to 768×768.

The exact frontend order is:

1. Decode to RGB U8 with the same native image decoder or establish decoder parity separately.
2. Calculate a target preserving aspect ratio and aligned to 32 with the F32 source algorithm. Positive half ties use C++ `std::round`, not Python's ties-to-even `round`; oversized targets use floor scaling, undersized targets use ceil scaling. See [mtmd-image.cpp:112–156](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd-image.cpp#L112-L156) and [dynamic preprocessing:766–786](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd-image.cpp#L766-L786).
3. Resize with the source's **PAD_CEIL** default. Qwen3VL does not override that default: use the smaller edge scale, ceil resized inner edges, center with integer offsets and pad RGB black as needed. Identical dimensions use a direct copy. See [clip-model.h:63–68](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip-model.h#L63-L68) and [mtmd-image.cpp:41–92](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd-image.cpp#L41-L92).
4. The image resize is Pillow-style separable bicubic, alpha -0.5, filter support widened on downsampling, normalized coefficients converted to signed fixed-point with 22 fractional bits, horizontal then vertical integer passes with U8 clipping. It is distinct from learned-position interpolation and from GGML/PyTorch's bicubic alpha -0.75. See [mtmd-image.cpp:197–355](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/mtmd-image.cpp#L197-L355) and subsequent resampling passes. A library substitution needs byte-level parity before use.
5. Convert U8 with F32 `/255.0`, then F32 `(pixel-mean)/std` using stored RGB means/stds `[0.5,0.5,0.5]`. Pack planar NCHW in RGB order. See [clip-impl.h:707–734](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/tools/mtmd/clip-impl.h#L707-L734) and input packing above. Black padding therefore becomes -1, not 0, in this graph input.

Aligned 96×96 and 192×96 fixtures lie within default pixel limits and avoid resampling/padding; they are suitable first graph checks. They do not establish general preprocessing parity or grounding quality.

## Decoded weight and frontend export schemas

Arrays must be ordinary native-endian, C-contiguous NumPy F32 arrays in reversed GGUF dimension order. For a GGUF linear tensor `[in,out]`, the supplied NumPy shape is `[out,in]`; graph matmul uses `transpose_b=True`. Kernels supply `[out,3,patch_y,patch_x]`. Position embeddings supply `[position_count,E]`. The decoder must convert raw BF16 values exactly to F32; nonzero low 16 bits in those F32 values are rejected. The module contains no decoder.

The required in-memory identity object is:

```json
{
  "schema_version": 1,
  "kind": "gguf_decoded",
  "asset_sha256": "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
  "source_revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
  "decoder": "IDENTIFIED_DECODER_AND_VERSION_REQUIRED",
  "layout": "numpy_reversed_gguf_c",
  "decoded_dtype": "float32",
  "weights_sha256": {
    "EXACT_TENSOR_NAME": "SHA256_OF_NATIVE_F32_C_CONTIGUOUS_BYTES"
  }
}
```

The hash mapping and array mapping must each contain exactly every identified tensor name. Per-array hashes prove the caller's supplied arrays match its declared in-memory identity. They do not by themselves prove the decoder read the selected GGUF correctly: the caller's export receipt must independently bind decoder/version, source asset hash, raw tensor descriptor/storage type and exported decoded hashes. Synthetic fixtures use `kind=synthetic_fixture` and a separately identified synthetic digest.

A future preprocessed-image export receipt should include `schema_version`, source image bytes hash, decoder/version, original RGB shape, effective min/max token overrides, target `(height,width)`, resize/pad/normalization policy with pinned source revision, input shape `[1,3,H,W]`, F32 dtype, NCHW/RGB layout and SHA-256 of the exact normalized input bytes. Keep this receipt separate from model weights and embedding results.

Frontend output must preserve existing `tools/vision/strata_vision.cpp:183–226` behavior: `READY 2560`; `ENC` returns `OK n nx ny ms` after writing SVE1. File layout is little-endian int32 `[0x31455653,n,nx,ny,2560]`, then F32 `[n,2560]`. `nx=GW/2`, `ny=GH/2`, `n=nx*ny`, and token `i` has decoder position `x=i%nx,y=i/nx`. The helper currently derives its grid from the last decoder position and checks rectangularity. `serve/server.py:1711–1714,1925–1937` consumes this protocol and caches embeddings; no changes were made to those paths. New workers must fail on width/grid/nonfinite/short-write errors before claiming `OK`.

## Verification performed and remaining gates

`E:\Strata-Hetero-data\venv-intel\Scripts\python.exe -m unittest tools.test_hetero_vision_qwen3vl -v` passed 15 tests. Tests use small synthetic host arrays and a fake OpenVINO API backed by those arrays. The metadata-only test denies imports of NumPy and OpenVINO; graph tests never import the installed OpenVINO package. Checks cover GGUF weight direction, rectangular 2×2 ordering against the source's GGML Fortran-order reshape/permutation chain, four position axes, aligned-corner endpoints/singletons, full 72-channel RoPE versus an independent scalar translation, GELU variants, identity/hash/dtype/BF16 rejection, rejection of mapped arrays before hashing, full toy graph projection, nonzero uniform attention, FFN residuals and both optional norms. This validates preparation logic, not an OpenVINO runtime or a real encoder.

OpenVINO operation signatures were inspected read-only in the installed `venv-intel/Lib/site-packages/openvino/opset1/ops.py`, `opset8/ops.py` and `opset13/ops.py`. No real OpenVINO graph was constructed. Actual API acceptance remains untested until the root's review and fresh admission.

Required next gates, performed by the root or a separately authorized caller:

1. Review the new module and provenance schemas while the H21 text workload is active; keep all model execution quiescent until root admission permits it.
2. After admission, produce the selected native CPU oracle SVE1 for both aligned fixtures with explicit `--flash-attn off`, identified helper/build and thread count. Preserve failed runs.
3. Export and verify decoded arrays and preprocessed F32 inputs under new receipts; first construct/query the exact static graph and record all supported/unsupported ops without silently changing math.
4. Compile the exact CPU graph, compare shape/grid/finite values and named diagnostic stages against the native path when discrepancies require them. Compare both explicit arithmetic policies separately; never widen tolerances to label a failing arm a pass.
5. Only after CPU/native parity, query/compile/infer NPU with exact execution device evidence, no fallback, explicit precision evidence and static shape. No synthetic throughput or compile success establishes encoder quality.
6. Preserve SVE1 parity, then validate image tasks and measure end-to-end time/GPU VRAM savings before an opt-in production route. Default vision behavior remains unchanged through these preparation gates.
