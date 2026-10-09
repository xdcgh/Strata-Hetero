# Identified vision payload and CPU comparison preparation

Status: two small tools and nine synthetic tests prepared. Both default CLIs passed JSON-only validation. The first fixture-suite run passed seven checks and reported one resolvable dependency error: the normal pinned GGUF package import requires `yaml` through `gguf/metadata.py`; `venv-intel` did not yet contain PyYAML. `dependency-gap-01.json` retains this historical failure. Root subsequently reported official-SHA-verified isolated PyYAML 6.0.3 and Pillow 12.3.0 installation and all eight original tests passing. The reader cleanup/repeat revision passed three targeted tests, including the added cleanup test. No package-import bypass, global environment change, real asset read, installed OpenVINO import, Core, compilation or inference was performed by this preparation agent.

The source port and its fifteen fixtures from run 23 remain frozen. New files are `tools/hetero_vision_payload.py`, `tools/hetero_vision_cpu_compare.py` and `tools/test_hetero_vision_preparation.py`. The exporter and comparator are each under 300 lines. Their default modes read JSON contracts only; a requested output path requires the corresponding explicit `--prepare` or `--run` flag. Root review and a fresh gate after the F quality arm becomes terminal precede every real payload/runtime invocation.

## Exporter

The existing header inventory identifies all 334 F32/BF16 tensor ranges. The exporter validates descriptor sizes/non-overlap/bounds without opening the asset by default. Explicit preparation creates a new directory under `E:\Strata-Hetero-data`, verifies the source file size, stable path/FD identity and complete selected SHA-256 before reading tensors, and uses the pinned `gguf.quants.dequantize` for F32/BF16. It creates a fresh pinned GGUFReader only for matching descriptors and vision metadata; it does not access `ReaderTensor.data`. Header values are copied into plain lists/integers/strings, all reader/tensor references and exception tracebacks are released, and only then is the owned mapping closed. The same cleanup runs after a metadata error. Tensor bytes are read from the identified already-hashed file descriptor at checked offsets.

Each tensor artifact binds source raw offset, byte length, storage type and raw SHA-256 to reversed-GGUF-order F32 shape, decoded array SHA-256, NPY file size/hash and an independent decoded NPY readback hash. A second complete source-file hash and unchanged path/FD stat close the export. The receipt includes pinned decoder file hashes and the port's decoded-identity schema. The normal source-package imports remain intact.

Decoder source is the pinned llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`:

| Component | SHA-256 |
|---|---|
| `gguf-py/gguf/quants.py` | `123b9d5741ede6ca8a321d7e08d43c8a0bd9fb53ec4f786323f5d9b9a214b4e8` |
| `gguf-py/gguf/constants.py` | `0bfc8c29cbcc3228fae0c64ae210cda9b14077abd72581961f7fda28c11b33e9` |
| `gguf-py/gguf/gguf_reader.py` | `495f86fe509eddef80b22e0d861de14d51ce5df01c5a584c63ddc302166eda8b` |

[quants.py:68–75](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/gguf-py/gguf/quants.py#L68-L75) selects F32 view/typed dequantization; [BF16:205–218](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/gguf-py/gguf/quants.py#L205-L218) reconstructs BF16 by shifting its bits into F32. These source files were freshly hashed read-only in this preparation.

Only the four identified RGB PNG fixtures from run 22 are supported. Their dimensions are 96×96 or height 96/width 192, so the source's effective token limits require no resize or padding. The exporter verifies PNG hashes before and after Pillow decode, refuses color/shape conversion, applies F32 `/255`, then stored mean/std, and packs NCHW RGB. Pillow/version and normalization policy are recorded. This deliberately limited frontend is not a general image-preprocessing replacement.

Artifacts are separate `<tensor-name>.npy` files, `<fixture-name>.input.npy` files, a fsynced `progress.jsonl` and an exclusive terminal `export-receipt.json`. Partial artifacts are retained on failure. No serializer uses pickle. No file is overwritten or deleted.

## CPU comparator

The default comparator validates the JSON native output index, completed pinned-helper CPU receipt (`--flash-attn off`, no `--gpu`), selected asset identity and formal-1 shape/grid/body-size/hash declarations. It does not read SVE bodies. An optional export receipt must bind the current inventory/fixture manifest hashes and every raw→decoded descriptor/hash; NPY reads remain restricted to `--run`.

Explicit run rechecks each NPY size/hash/header/decoded hash and native SVE1 body size/hash/header/finite values. It runs the two arithmetic policies separately at each of the two static shapes. Device selection is literal `CPU`; the compiled execution device must be exactly CPU. Both F32 inference precision and ACCURACY execution-mode hints are mandatory and must be reported back as matching. No all-device enumeration, NPU, GPU, AUTO, HETERO or fallback path exists.

Each image/policy receives one excluded warmup and three formal inferences. Each formal result has its own quality report, inference wall time, output file/hash and optional diagnostic taps. Formal output filenames end in `formal-1`, `formal-2` or `formal-3`; no formal result overwrites another. Warmup time/hash is kept separately and excluded from the three quality comparisons. Repeated decoded hashes and their equality are reported; equality is diagnostic, not a silently added bitwise acceptance threshold.

Quality uses `hetero_xpu_worker.quality_report` with its unchanged `DEFAULT_TOLERANCES`: max absolute error 1e-2, relative RMSE 1e-3 and row-norm relative error 1e-3. There are no tolerance-override flags. Negative formal comparisons continue through all fixtures/policies and are saved with their outputs. An image passes only if all three formal comparisons pass; arm and policy acceptance require all their images. Arms retain device/property/build/compile/quality/error metadata; per-policy acceptance fields are separate from the aggregate result. No arm automatically enables NPU or a production route.

Comparison artifacts include each formal F32 output/tap, the fsynced stage/resource journal and an exclusive `comparison-receipt.json`. The run provides eight image/policy groups, eight warmups and 24 formal results. This comparison is a correctness gate with repeated inference timings, not an accepted end-to-end performance benchmark: preprocessing, export, compilation and native helper timing have different measurement boundaries.

## Memory and failure boundary

Both tools reuse `hetero_resources.memory_snapshot`, requiring known available physical RAM >=12 GiB and available commit >=4 GiB. Unknown/error telemetry fails closed. Gates run before opening the asset, during whole-file hash chunks, before each tensor/image load, before/after graph build/compile/inference and at completion. A failed gate stops further cost and records its stage; already created artifacts stay in the new owned directory.

A checkpoint cannot cancel a native compile/inference call while it is blocked in the runtime. The root's external monitor of the exact owned worker process remains necessary for prompt termination during those calls if RAM crosses the gate. The tools do not kill processes by name, close user apps, replace drivers or alter settings.

The pure tests cover JSON-only/no-runtime defaults, exact known F32/BF16 source decoder bits/layout, new NPY artifact readback/corruption, SVE1 validation, all 334 metadata bindings, GPU/flash reference refusal, failing admission before asset/Core access, weak-reference checks proving header views are released before mapping closure on success/failure, and retaining all eight negative image groups/24 formal outputs across four fake host arms with unchanged tolerances. Fake Core/model objects used for state-machine tests are host-only Python fixtures; they do not import the installed OpenVINO package or exercise a device.

## Root-owned next steps

1. Retain the first dependency failure and root's isolated dependency verification; the original eight tests have passed after that installation.
2. Review the reader cleanup/repeat revision and its three targeted fixtures, then invoke default validation again. No real payload or Core is needed for this review.
3. After the F quality arm is terminal and fresh admission passes, explicitly export into a new E artifact directory. Preserve the complete binding receipt and export failures.
4. Validate that exported JSON receipt, then explicitly run the CPU comparator in another new E directory with the external resource monitor. Review all positive and negative policy arms.
5. Only a selected policy that passes every native/CPU fixture can proceed to root-reviewed NPU coverage/compile/quality work. General preprocessing and actual visual task quality remain separate gates.
