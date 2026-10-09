# Prefill cache borrowing and restore: source inventory

Static source review at `53d60c3`; no runtime, GPU, or code change. H21's first seven prompts matched H4/H18 IDs and text; `long_16k` and the post-failure arithmetic probe emitted token ID 0 repeatedly. That is generated-output evidence, not a cause or logit readback.

## Mapping, loan, and source

`src/program/generate.cpp:4524-4611` fills the expert cache once in loaded profile order: `ExpertCache::admit(layer, expert)` assigns a slot and the matching expert bytes are copied there. `slot_of` is a bounds-checked lookup in `residency_[layer*n_expert+expert]` (`src/core/expert_cache.cpp:633-655`). `generate.cpp:5459-5485` creates a layer-major `host_res` table containing those slot IDs or `-1`, selecting the owning stage cache on a layer split, then uploads it to device(s). Default admission is sequential; `--expert-cache-per-layer` uses per-layer slot ranges. The temporary loan changes `host_res`/`d_res`, not `ExpertCache::residency_`.

With a profile, borrowing is enabled unless `--no-prefill-borrow` is set (`generate.cpp:3151`). `plan_lend` prices `Prefill::bytes_needed` against actual slot sizes and preserves a 128-slot floor (`generate.cpp:5644-5689`). In serve mode, persistent `PfPart` objects identify each cache, device, layer range and tail slots (`generate.cpp:6103-6125,6164-6171`). For resident-RAM mode, `FileExpertSource::pin_cache_complement` also tries to retain experts belonging to lendable tail slots in the RAM complement (`src/core/expert_source.cpp:2174-2188,2559-2570`). Refill source `blob_stable` uses a resident complement/override when present, otherwise mapped-file fallback (`expert_source.cpp:2741-2791`); with H21's buffered-file setting, fallback may still require OS file-cache or storage reads.

## Request lifecycle and lifetime

For a batched segment, `lend` lays Prefill buffers over the cache tail and marks only matching `(layer, expert)` rows in that participant's layers as `-1`, then uploads `d_res` (`generate.cpp:9455-9504`). The segment runs through `sp.run`; before verify-window segments the loop refills first, and after prompt segments it unconditionally refills before generation (`generate.cpp:9698-9750`). Restore decodes each saved layer-major index, fetches the source blob, copies it into the same slot on its owning device, sets `host_res[i]=slot`, waits, clears `p.lent`, then uploads the full map to every device (`generate.cpp:9396-9449). Windows releases file pages only after copies land (`9413-9418`). `fill_slot_queued` uses the default stream and `sync_queued` synchronizes it (`src/core/expert_cache.cpp:723-746); `res_put` also waits for the residency-table copy (`generate.cpp:5463-5465,7256-7263).

There is no dedicated CUDA event for cache restoration. Prefill's ring events persist with the `Prefill` object, but normal `run_impl` exit synchronizes compute, expert-copy and optional KV-copy streams; `Prefill::run` drains pipeline successors (`src/prefill/prefill.cpp:3568-3580,3635-3667`). `relayout` fences those streams before rebinding a borrowed range (`1114-1144`). The `Prefill` object/layout persists across requests, but its tail bytes are expected to be overwritten with experts after each prompt. The MMQ identity scratch is rewritten each run because refill overwrites it (`prefill.cpp:1085-1088,1816-1818).

One caveat for interpreting the arithmetic probe: `windows_ok` routes segments no longer than the default `--short-read 64` through verify windows, refilling before them (`generate.cpp:611,9215-9221`). Thus a 27-token probe may exercise no new loan; existing trace output (`lent`/`refilled`, `generate.cpp:9443-9500`) can falsify this.

## Falsifiable checks, not conclusions

- On a separately authorized reproducer, log each segment's `windows_ok`, requested chunk and lent `(layer, expert, slot)` list to establish whether the failing segment actually borrowed slots.
- At restore, check sampled source bytes against D2H bytes from the same slot, plus `host_res[i] == slot == cache.slot_of(layer, expert)`; record RAM-complement versus file fallback. This separates payload, routing-table and source-path questions.
- Before the next prompt, record that every loan list is empty and relevant streams are idle. Compare the post-failure prompt with a fresh instance only under separate authorization; do not treat an unloaned short prompt as a cache-restore test.

Local comparison: these borrow/restore files are unchanged from H4 source revision `e87d74c`. In the locally available upstream diff, the inspected files differ only in PLE RAM-reserve handling in `generate.cpp`; the loan path is unchanged. This inventory does not explain H21's output or establish a repair.
