# Native CPU vision oracle

Accept the saved outputs as the oracle for these four small aligned PNG fixtures, with identified pinned mtmd source, CPU-only build, fresh mmproj/model hash checks, ten threads and flash attention off. Each fixture has one client warmup and three formal encodes; all 16 files have exact SVE1 shape/body size and finite values, with identical hashes within each image. The root independently rechecks the external bytes and the repository integrity manifest.

Do not extrapolate this result to grounding quality, arbitrary resize preprocessing, full multimodal requests, a CUDA comparison or NPU acceptance. The helper skips its internal CPU pre-READY warmup; client warmups are separate. Its READY time is a single startup observation. The helper's full `GetProcessTimes` creation timestamp is in the original receipt; the provisional summary lost fractional seconds, as recorded in `root-review.json`.

Next: verify decoded weight/input identity and compare the static OpenVINO CPU graph with this oracle under separately declared arithmetic policies and unchanged numerical tolerances. Preserve failures. Only a passing native comparison permits NPU execution and later task/performance validation. Production vision defaults remain unchanged.
