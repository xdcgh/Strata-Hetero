# v0.1.41 CUDA build evidence

This package preserves the successful sequential build pair, exact build receipts, actual driver script copies, configure/build logs and command batches, CMake caches, resource samples, and available first-failure evidence. Every listed file was copied byte-for-byte from the pinned E: source/build evidence. The manifest records each copied relative path, byte count, and SHA-256.

The compiled executables and generated object/library files remain in their pinned E: build directories and were intentionally not duplicated. Their external paths, byte sizes, and SHA-256 values are recorded in `pair/cuda-build-pair-41.json` and each build receipt. No engine/model CLI, tests, or GPU work was run.

The first two upstream attempts did not reach successful CMake/configure: attempt 01 failed in the build-driver Git helper; attempt 02 failed because the CMD wrapper serialized separate tool commands into one line. Their available process/controller logs, receipts, and complete build-directory text evidence are preserved under `failed-attempts/` and `pair/controller-evidence/`. The pair record points to the executed upstream02 script and current reviewed build driver.
