# Run71 harness review (static only)

Scope: `prepare_moe71_model.py`, `test_prepare_moe71_model.py`, and Run69 `prepare_probe69_model.py` reference. Both target Python files parse with `ast.parse`; no script was imported or executed and no tests, prep, API, device, model, or build actions were run.

## Finding

1. **Validator disagreement for `moe_path.layer == -1` — fix before relying on diagnostics.** In `prepare_moe71_model.py`, `valid_moe_path()` accepts `layer == -1` (`layer < -1` is the reject condition, line 248). The generated PowerShell `Convert-DiagnosticLine` rejects the same record because it requires `$record.layer -ge 0` (line 325). Thus a record accepted by the Python classifier becomes `fatal_invalid` in the actual quality controller. Since the event's legal range is not established by the two scripts alone, align both validators to the frozen C++ contract (or require nonnegative in both); add parity fixtures for `-1`, `0`, and a value below the allowed minimum. The current Python moe-path unit cases do not exercise layer bounds, and the PowerShell classifier fixture does not include a `moe_path` record at all (`test_prepare_moe71_model.py` lines 129-145, 234-268).

## Requested contract checks

- Python stage allowlist is checked against both C++ `Stage` enum and `stage_name()` labels. Its first-bad validation binds `f32`/`f16`/`bf16`, index/row/column, and `source_row == row_offset + row`. The generated PowerShell classifier applies corresponding checks. Unit fixtures cover every stage/format in Python and every stage in PowerShell; format variants are not individually exercised in PowerShell.
- `moe_path` is explicitly classified as normal metadata, not a failure; invalid flags, peer-sums dependency, integer fields and CPU-row bound fail closed. The layer boundary discrepancy above remains.
- Quality rewrite selects the first five short prompts plus `long_1k`, records all nine manifest entries and verifies their file length/SHA and tokenizer; it pins one warmup and three formal repeats and stops after a failed prompt while saving its prompt outputs/diagnostics. Diagnostic reader advances only through LF-complete records and preserves partial bytes. Before-request fatal diagnostics are written before throw; after-request diagnostic receipt is saved before result evaluation and the stop.
- Preparation binds the reviewed E71 source HEAD and build receipt, verifies clean source identity, successful build-only flags, binary/test-binary identities and finite-probe CMake control. It takes complete identity/provenance templates from Run66 hetero42 while separately validating the upstream42 failure history.
- Default preparation path creates request directory but no per-prompt request leaves; writes are exclusive and execution remains disabled. Test has a temporary complete-preparation fixture and checks no request leaves.
- Tests pin the exact PowerShell 7 executable and SHA and include generated classifier execution plus PowerShell AST parse checks. These are test-source observations only; tests were not run.

## Evidence limits

This is a static script review, not runtime validation. The supplied assignment states the E71 source/build identity and that neither tests nor device/model work has run; this report does not independently verify those external artifacts. It does not establish full original comparison or model quality. The existing six-prompt preparation is a diagnostic triage harness, not completion of the user's larger heterogeneous-scheduling and full-original-comparison goal.