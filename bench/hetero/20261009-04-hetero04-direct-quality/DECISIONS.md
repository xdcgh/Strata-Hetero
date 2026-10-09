# Integrated 0.1.40.4 direct quality control

The final reviewed comparison accepted all 27 required formal prompt-index pairs against the complete latest-upstream run `20261009-03-upstream04-quality`. Actual emitted IDs, SSE visible text and stored output bytes agree, all task checks pass, and controls and launch evidence match. Every task has one warmup and three formal requests. This is a direct-mode quality control, not a full-RAM or performance acceptance.

The first four natural tasks used max_tokens=1024 rather than the requested 128. Their entire raw request groups are preserved in `quality/attempts-max1024/`; `direct04-vs-upstream03-full.json` remains incomplete. All four were rerun at the actual cap 128 before producing `reviewed-final.json`. No original metadata was changed to make an attempt appear aligned.

The complete owned instance was unloaded and stopped after exact identity, listener and zero-in-flight checks. Resource sampling ended through the owned STOP file. See `shutdown-final.json` and the referenced unload/bridge/sampler receipts. Earlier malformed or incomplete shutdown preflight captures remain with correction notes.

Next: latest integrated full-table RAM quality at matched expert budget, then controlled placement/performance experiments. Preserve the previous full-RAM bang-loop failure until a matched, uninstrumented experiment actually establishes the latest behavior.
