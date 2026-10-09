# Stage-tap CPU build and synthetic fixture

Status: the fake-tensor fixture compiled and passed; the stage-tap helper compiled but has **NOT RUN**. No GGUF/model, ENC, OpenVINO Core, GPU/NPU or benchmark was started. Root review is required before real stage captures.

All four owned phases exited 0: fixture compilation, fixture execution, configuration and the 131-step helper build. Helper compilation ended at `2026-10-09T10:56:55.7822898Z`; a subsequent compiler-overlap snapshot reported zero compilers. Root was notified immediately for the waiting model pair.

The execution used pwsh 7.6.5, MSVC 19.51.36260.0, C++17, `/MT`, portable AVX2 CPU flags and single-job Ninja at inherited/enforced Idle priority. CMake 4.4.3 and the exact run-20 CPU configuration were reused with a new independent build directory. Cache checks confirm CUDA, HIP, SYCL, Vulkan, OpenCL, native ISA, OpenMP and dynamic backend loading OFF; AVX2/FMA/F16C/BMI2 ON. Actual compilation commands preserve `/MT` and CPU AVX2. The only added CMake option exports compile commands for evidence.

| Artifact | Path | SHA-256 |
|---|---|---|
| Synthetic fixture executable | `E:\Strata-Hetero-data\build\vision-stage-fixture-20261009-39\vision-stage-trace-fixture.exe` | `52417b9dcfc16776f5f636fdf045add092b9aebcf55818129b2eaf51d08843ff` |
| Stage helper, 5,069,824 bytes | `E:\Strata-Hetero-data\build\vision-stage-taps-cpu-20261009-39\bin\strata-vision.exe` | `f01aa1f111931e4f24c2003d65bd3ab853370c7c1b7f7a8f62900d6b2c7a7c6d` |
| Original oracle, unchanged | `E:\Strata-Hetero-data\build\vision-oracle-cpu-20261009-20\bin\strata-vision.exe` | `7285dd08f24c3985fd6b97aaff7d9917b65e5ea8b263ae5567985d0dc04ba612` |

The fixture used a new owned directory `E:\Strata-Hetero-data\vision-fixtures\stage-metadata-20261009-39`. Three complete synthetic captures each contain 11 stages and 2,156,544 bytes (13NE at N=36), alongside preserved failing-case directories. It exercised selector/layout/parent guards, quota, pre-read byte limits, duplicate/missing coverage, noncontiguous/wrong-type/unallocated tensors, existing files/directories and exclusive SVE creation without reading real tensors or executing model operations.

All 58 global RAM/commit samples passed the unchanged 12/4 GiB gates; minimum available RAM was 109.5314 GiB and commit was 114.4783 GiB. No source file changed during compilation. Pinned llama.cpp head remains `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

`run-build.ps1`, the four actual executed `.cmd` scripts, process/terminal receipts, original stdout/stderr logs, source-before/after hashes, process-tree snapshots, resource JSONL, CMake cache, build.ninja and compile_commands are preserved here. `build-receipt.json` and `verification.json` bind binaries, exits, flags and hashes. CMake's CMP0194 developer warning and the pinned llama.h code-page C4819 warning remain in their original logs; neither caused a failed phase or a source modification.

No commit or push was created by this agent. Real native/helper or OpenVINO stage runs stay gated until root approves them after run 40/41 becomes terminal.
