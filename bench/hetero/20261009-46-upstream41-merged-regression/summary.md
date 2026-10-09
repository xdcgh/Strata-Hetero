# Upstream41 merged pure regression

All 23 test commands exited zero. The runner then failed only while aggregating its result because StrictMode accessed `tests_run` on the custom PowerShell fixture entry, which records `fixture_cases`. Results below were reconstructed from the already completed command ledger and suite logs; no tests were rerun.

Source HEAD: `167689590a66d4ad620b5e8cc0b7893190fc19d5` on `codex/45-upstream41-runtime`. Tracked diff paths: 5; staged diff paths: 0.

Python unit totals: 792 run, 791 passed, 1 skipped, 0 failed. Additional PowerShell fixture cases: 10.

Reused run33 command suites: serve (400), calibrate/setup (163), tokenizer (2), 16 Hetero modules (198 on the merged tree), and mocked CUDA-preparation tests (13). Added planner (9) and runtime (7) suites. Commands and complete logs are in `commands.jsonl` and `logs/`.

No compiler, model, API request, GPU, OpenVINO Core/device initialization, or real download was performed.
