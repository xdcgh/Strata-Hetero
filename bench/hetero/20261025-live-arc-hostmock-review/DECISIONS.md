# Live helper source review backup

The four Python typed mock tests ran on Windows using
`C:\Python314\python.exe -m unittest -v tools.test_hetero_live_bridge`.
They passed, exit 0, with unittest reporting 0.033 seconds. The stdout file is
empty because unittest wrote its complete report to stderr; both raw files
are preserved. `python-unittest.receipt.json` records exact command, UTC times,
exit status, tested source hashes and output hashes.

These tests cover actual payload validation with synthetic bytes, framed mock
READY/INFER responses, row-graph reuse, changed engine-blob rejection before
worker invocation, wrong device/output dimensions, nonfinite input and
unsupported rows. They do not launch a child, import OpenVINO or load a model.

AST parsing of the four new Python sources passed. `git diff --check` passed.
The Windows eligible C++ fixture, production Pipe lifecycle, worker rejection,
partial writes, all-or-nothing output update and timeout/descendant cleanup
have executable fixture source but remain **UNBUILT / NOT RUN**.

No C++ compiler, CUDA/Arc/NPU workload, model or end-to-end benchmark was run.
No model quality or gain acceptance is claimed. Automatic routing stays off;
forced mode is experimental and may be slower because it serializes the helper
and adds host publication, hashing and transport overhead.
