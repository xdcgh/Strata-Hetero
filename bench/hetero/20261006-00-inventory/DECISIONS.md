# Initial decisions

- 2026-10-06: Preserve the old Strata working tree and its untracked benchmark results. Clone the existing fork into Strata-Hetero; fetch upstream. Both new remote main tips are `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. The old local shallow tip is `6f32ec070f23ced9f50e704d854d775da52591ab` and is not present in the new history; compare source trees and preserve its separate identity instead of assuming a shared ancestry.
- 2026-10-06: SSH initially failed host-key verification. Fetch GitHub's published Ed25519 key, verify `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`, and keep the pinned known-hosts file under the new checkout's `.git`. A batch-mode authenticated `ls-remote` then succeeded. Strict verification remains enabled. No global SSH credentials or known-hosts file was changed.
- 2026-10-06: Main architecture/review work and Luna inventory delegation started. One Luna startup hit model capacity and was retried; this is not benchmark or engine evidence.
- 2026-10-06: Reuse upstream expert RAM complement, PLE row cache, KV streamer and CPU affinity. The first missing host feature is Windows full-table PLE RAM residency. Arc/NPU device discovery is not proof that a model/runtime can execute on those devices.
- 2026-10-06: Do not enable SSD active-KV spill. Investigate parked snapshots only after the active VRAM/RAM path has evidence.
