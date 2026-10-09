# Upstream v0.1.41 isolated merge review

The worktree `E:\Strata-Hetero-data\source\hetero-0-1-41` is based on clean `47fded1cfc6f50b8fa861e0c545925a7e25b34fd`, branch `codex/33-upstream-0-1-41`. The verified remote `upstream` tag `v0.1.41` resolves to `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. The pinned merge is staged with `--no-commit --no-ff`, no conflicts, and no commit. No source files in the main checkout were changed.

The merge changes 235 paths (30,018 insertions, 394 deletions). Its full staged binary diff, status, name-status, merge log, before-log, and empty pre-merge diff are preserved beside this note. `git diff --cached --check` result is in `merge.check.txt`.

CMake now reports engine version 0.1.41. The ggml FetchContent pin remains `3cf03257f219afbe7334045ff7c6a06ac68c627d`; the matching H4 build used the external checkout `E:\Strata-Hetero-data\source\llama-3cf0325`. No submodule changed. `--ple-ram-reserve-gib` still defaults to 12 GiB, and the `serve/server.py` actual token-ID capture opt-in path/config key was not altered by this upstream diff.

The main matching build cache is `E:\Strata-Hetero-data\build\cuda-hetero-upstream04-01\CMakeCache.txt`. Required flags to reproduce that build are listed in `review-summary.json`: Release, CUDA ON with architecture 89 and the pinned CUDA 13.3 compiler; portable AVX2 and static MSVC runtime; native experts ON; tests, MMQ K-quants, Q6K expert kernels, HIP, SYCL, gfx906, HIP prefill MMQ, and Orca Q4K_S MMQ OFF; empty ISA floor; and the pinned external ggml source. `GGML_CUDA` is forced OFF by native experts even though Strata's own CUDA backend is ON.

Runtime behavior needs an explicit comparison choice. In 0.1.41 the CPU prefill share is on by default only for CUDA, one GPU, no batch slots; unset `STRATA_PREFILL_CPU_SHARE` takes the share path, and `STRATA_PREFILL_CPU_SHARE=0` disables it to retain pre-1.41 behavior. Layer-split, batch-slot, and HIP paths stay off. Stage-buffer pinning now defaults off (`STAGE_PIN_DEFAULT=0`); do not force `STRATA_STAGE_PIN=1` for the matched validation because upstream identifies prior long-prompt IQ3_S corruption with forced pinning.

This is static merge/build preparation only: no configure/build, tests, model, GPU, or source conflict resolution was performed.

git diff --cached --check reports 109 added rows with trailing whitespace in upstream ench/results/2026-10-07-community-arc-b65-v01402/results.csv; no other file was reported. The incoming data file is left untouched for root review.
