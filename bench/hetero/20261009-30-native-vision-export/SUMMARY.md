# Native vision payload export

Status: exported and verified. This is artifact-integrity evidence. OpenVINO Core, real vision graph construction/compilation/inference, vision API integration and NPU acceptance remain NOT RUN by this export task.

The owned hidden Python process was PID 21584, created at `2026-10-09T09:09:44.882596Z`, and returned exit code 0. `process.json` records exact argv, interpreter/base-interpreter/source hashes and authorization scope; `exit.json` records the successful owned-handle exit. The original process is no longer present and stderr is empty. Root reported a separate CPU helper compilation during this export, so the export is not a storage or inference performance result.

The selected 907,543,008-byte source asset stayed at its original E path. The frozen exporter completed both whole-file SHA-256 assertions against `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`, before and after tensor export. Source path/FD device, file ID, size and mtime remained stable; a post-run stat still matches the terminal receipt.

The new exclusive directory is:

`E:\Strata-Hetero-data\vision-payloads\20261009-30-native-vision-export`

It contains 334 tensor NPY files, four normalized input NPY files, the terminal export receipt and the durable progress journal. There are 224 F32 and 110 BF16 source tensors. Source tensor payloads total 907,523,008 bytes; decoded tensor data totals 1,795,724,224 bytes and their NPY files total 1,795,766,976 bytes. Every tensor has raw offset/bytes/type/SHA, decoded F32 C-order shape/SHA, NPY size/SHA and successful independent decoded-array readback binding. All declared file sizes match; the decoded identity mapping contains exactly the same 334 hashes.

The pinned source decoder is `gguf` 0.19.0 from llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, with its three decoder file hashes in the export receipt. Runtime package records are Python 3.14.8, NumPy 2.5.3, PyYAML 6.0.3 and Pillow 12.3.0. Reader metadata and descriptor matching completed with release-before-close cleanup.

All 354 runtime resource gates passed. Minimum observed available physical RAM was 110.1484 GiB and available system commit was 115.0992 GiB, above the unchanged 12/4 GiB thresholds.

| Input | F32 NCHW shape | SHA-256 of normalized input bytes |
|---|---|---|
| gray-96 | `[1,3,96,96]` | `f2b65f387f6e5bcdfcec1e7396ce8168850b437774768f377747d812308b407d` |
| quadrants-96 | `[1,3,96,96]` | `eddcd454d209b77d0406cd98cfb25b4d4c787790b1742e798f270712c79dcee1` |
| checkerboard-96 | `[1,3,96,96]` | `cff3654e176cede5b12b39f7dea94986d85384f74f78bf8a805e8d4fb120dd31` |
| gradient-192x96 | `[1,3,96,192]` | `e0dd097b35c61c3c9fb9f3daa9ba1ddb14a0ed575fdc0547b6532df37a947691` |

`export-receipt.json` is a metadata copy of the E receipt; `tensor-input-index.json` preserves all 334 tensor bindings and four input records. The E receipt SHA-256 is `9293d413d80a89bea202d9c79c480fb3e8c07ca79e8fae740ca52cfca0ff4e01`. Captured stdout is JSON-equal to that receipt. `verification.json` records stat/count/size/identity/package/resource checks and log hashes.

The default CPU-comparison contract validator accepted this exact export receipt with exit code 0, while reporting no exported-array/SVE-body read, no Core, no compilation and no inference. Its receipt is `comparison-contract.json`. Real CPU graph execution remains for root review and separate authorization. No source asset/shard was moved or rewritten, no production route was changed, and this agent created no commit.
