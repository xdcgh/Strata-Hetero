# Calibration and benchmark tools

These preparation tools run independently of setup and do not enable a new backend. The final `--hetero-calibrate` installer integration is pending model measurements.

The profile API in `tools/hetero_runtime.py` now supports exclusive `write_profile` and bounded `load_profile` with explicit evidence paths keyed by their SHA-256. Loading checks those receipts again. Helper acceptance additionally requires a `strata-hetero-route-acceptance-v1` report scoped to `model_end_to_end`, bound to the exact hardware/runtime/model/engine identity and matching the acceptance fields. Its producer must perform the benchmark and quality checks; changing the JSON schema cannot create that evidence. An IPC microbenchmark profile without model acceptance retains the primary route. This API does not run calibration, launch workers or edit setup settings.

The native CPU pool affinity experiment `20261009-35-cpu-worker-affinity` completes 128 processes: two new single-thread reference gates for rows 6/10 and 126 pool cases covering six row counts, six worker counts and four affinity modes. Eighteen `p-cores` overflow points are excluded. Each executed case has one warmup and five formals; all outputs are finite and bit-identical to the required reference. All 257 global resource samples pass. Root independently rehashed/recompared all 128 outputs. Requested affinity and successful host pinning are recorded, while per-worker pin success and P/E/LP labels remain unknown. This is isolated expert evidence; the required 5% model gain is not established.

The first input preparation failed before writing because a disk-floor key was read at the wrong contract level. During fixture validation, a test using the live authorized contract then created the two approved seeded inputs. Its producer source differed from the first authorization hash; original source/contract and failed test evidence are retained, and root independently replays the seed recipe and verifies both outputs. The fixture now uses a temporary disabled contract and fails if real preparation or execution is reached. No input was regenerated and no kernel ran during those preparations. The subsequent sweep has its own explicit source/contract binding and authorization.

`ExpertPool::worker_affinity()` now provides an immutable startup observation per worker, published with release/acquire independently of job/park counters. It separates pending, globally requested pinning, unpinned overflow, API application and actual Windows group/mask readback; the startup processor is a point-in-time observation. Readback failure does not imply verified placement. Worker numerical kernels and phase/epoch scheduling remain unchanged. Native harness receipts contain these records; the engine's added startup INFO fields still require its separate CUDA build/runtime verification.

The separate CPU-only observation build passes three CTests, including an owned two-thread pin/readback fixture. Actual expert run49 then passes 15 cases and 75 formals at rows 1/16/256: all outputs are bit-identical to the original frozen native CPU references. Root independently checks all output bytes and the 36 pinned worker observations: call success, known readback and matching group/single-bit mask. The unpinned cases request/apply no affinity. All 31 resource samples pass. This closes the sampled worker-pin visibility gap without assigning P/E/LP labels or claiming a model gain; run35's older unavailable-pin reports remain intact.

## Inspect and validate before using the model

```powershell
python tools/hetero_inventory.py --output bench/hetero/<new-run>/hardware_manifest.json
python tools/hetero_admission.py --validate-only --output bench/hetero/<new-run>/admission.json
python tools/hetero_admission.py --observe --output bench/hetero/<new-run>/admission.json
```

The observer's default requirement is 46,000 MiB free VRAM and 12 GiB available RAM for this large-model host. Those are admission bounds, not final memory reservations. It reports Strata/Python/ComfyUI or unknown GPU process identities and running WSL device users. It never stops an application or boots a distro. Exit 3 means rejected; do not launch a model from that receipt. Repeat observation immediately before each model arm; old receipts expire with the resource state. Inventory/receipt paths must be new and existing evidence is never overwritten.

## CUDA-free host correctness gate

In a Visual Studio x64 compiler environment on Windows, or a C++20 environment on Linux:

```text
cmake -S tools/hetero_host -B <new-build> -DCMAKE_BUILD_TYPE=Release
cmake --build <new-build> -j 1
ctest --test-dir <new-build> --output-on-failure
```

This builds the existing PLE reader and working-set memory check without CUDA. The optional `STRATA_SOURCE_ROOT` points at a specific reference tree. The fixture directory is under the build directory. A reader/lock fixture pass is not model correctness or performance acceptance.

## Isolated native CUDA SDK

`tools/hetero_prepare_cuda.py --manifest <pinned-manifest> --root <new-sdk-root> --validate-only` validates the manifest and target without downloads or writes. Execution verifies each NVIDIA Windows component's exact byte count and SHA-256 before extraction. Reuse requires a matching receipt and fresh per-file hashes. Failed partials and staging folders remain; different SDK files never overwrite each other. Root-level component license/version metadata stays under `component-metadata/<component>/`.

For this host, the verified SDK is `E:\Strata-Hetero-data\toolchains\cuda-13.3.1`, with seven archives, 2,183 extracted files and nvcc 13.3.73. It uses existing VS 2026 MSVC 14.51. Driver, registry and global PATH were not changed. NVIDIA's [CUDA 13.3 Windows compiler table](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-installation-guide-microsoft-windows/index.html) supports MSVC 195x. A compile or inference pass must still be recorded separately.

## Repeated API measurements

`tools/hetero_native_inputs.py` prepares a Strata native CPU comparison from an already completed real-expert extraction. Its default mode checks metadata and the original shard identities without reading tensor payloads or writing files. Explicit `--prepare --source-receipt ... --output-dir <new-absolute-directory> --receipt <new-absolute-path>` reads only the selected gate/up/down slices, verifies their prior scoped hashes, and concatenates the original quantized bytes. It does not requantize the float32 NPZ. Nine little-endian float32 input files use the same seeded recipe as the OpenVINO expert probe, with independent small-artifact readback hashes. Preparation is separate from a kernel or model measurement.

After admitting and starting an isolated loopback server, validate the client without a request:

```text
python tools/hetero_bench.py --prompt-file <natural-prompt> --output bench/hetero --run-id <new-run> --max-tokens 256 --validate-only
```

Supply `--identity-json` with `schema`, `source`, `config`, `model` and `engine` identifiers for execution. Defaults are one warmup, three formal batches, concurrency 1, temperature 0, seed 42, reasoning disabled and `http://127.0.0.1:8081`. The API key is read only from `STRATA_HETERO_API_KEY`. The client bypasses inherited proxies and refuses redirects and non-loopback addresses. A live model or engine identity mismatch saves a failed preflight receipt and makes no benchmark request.

`--allow-capped-performance` includes clean `finish_reason=length` samples when intentionally measuring 256/512/1024-token output caps. The length cap remains visible and is not a quality pass. Missing `[DONE]`, timeout, API errors and unknown finish states are excluded from eligible timing statistics. Raw generated text, reasoning, events, usage and content hashes remain separate artifacts. Warmups stay outside formal statistics.

First generated token timing includes reasoning; first visible content is recorded separately. Client latency is measured from the serialized request to completed SSE reading. Concurrent output speed uses formal batch wall time, including failures, and known service token counts; missing counts make it a lower bound. No character-to-token estimate is used. Full-prefill rates only use requests whose engine reports `cache_n=0`; cached repeats cannot become full-prefill evidence. Disable the engine prompt cache for a full-prefill experiment, and record adaptive/PCIe/determinism controls with the identity.

Results remain `not_accepted_pending_quality` until source/config/model identity, sample validity, resource isolation and model quality have been reviewed. P95 from three samples is the largest observed sample and has limited statistical precision. More repetitions are needed for a credible tail-latency estimate.
