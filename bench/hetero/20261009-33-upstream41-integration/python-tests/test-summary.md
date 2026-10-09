# Isolated v0.1.41 pure-test results

No-commit merge stays on `codex/33-upstream-0-1-41`. Commands and complete outputs are in the sibling receipts/logs.

Final count: 770 tests run, 769 passed, 1 skipped, 0 failed.

- Serve API and mock-engine suites: 400 run, 399 passed, 1 skipped, using the existing Strata venv.
- Calibrate/setup suites: 163 passed under venv.
- Tokenizer piece-cache: 2 passed under venv.
- Hetero synthetic suites: 192 passed across 16 modules.
- CUDA preparation suite: 13 passed; it patches `urlopen`/`build_opener` and uses synthetic temporary archives, with no real SDK download.

Initial C14 attempts are retained. The global install lacks `jinja2` and `regex`; affected tests passed under the venv. The first Hetero unittest-module invocation missed tools-directory imports; direct-script execution with process-local repository/tools `PYTHONPATH` passed all modules.

The narrow `.gitattributes` rule disables whitespace checking only for the upstream CSV. Its raw worktree blob and index blob still match. After this rule, `git diff --cached --check` exits 0. The CSV bytes were not edited.

No merge commit was made; root review is pending.
