# CPU vision stage diagnostic

One CPU-only OpenVINO diagnostic completed all four arms (two arithmetic policies and two static shapes), with eight warmups and 24 formal projections. Reported execution is CPU, float32 and ACCURACY. All formal projections and 264 stage outputs are finite, have the required shape and repeat identically across three formals. Both policies fail the unchanged final numerical gate; all 24 negative projection reports remain in the receipt.

Adding the 11 diagnostic outputs changes every formal projection fingerprint from frozen run 31. This is recorded for all 24 formals. These instrumented projections are diagnostic outputs and do not replace the original native oracle or the original run-31 outputs.

The saved native and OpenVINO arrays produced 264 stage comparisons, including 153 negative reports. The following are the first stages that fail the unchanged gate, in graph order; smaller differences before these stages are still preserved in the full reports.

| Fixture | graph_f32 | cpu_vec_dot_rounding |
| --- | --- | --- |
| gray-96 | block.0 | block.1 |
| quadrants-96 | qkv.0 | ln1.1 |
| checkerboard-96 | block.0 | block.1 |
| gradient-192x96 | qkv.0 | ln1.1 |

Patch merge, learned-position addition and the first layer norm pass for both policies on every fixture. The rounded policy also passes the first combined QKV and first block on every fixture. These checks locate the first gate failures in the sampled stages; they do not yet establish the cause of the later error or authorize a math change.

The main-venv launcher/actual C-Python controller and Intel-venv launcher/actual C-Python worker were bound by executable, creation time and argv. The outer observer normalizes only Windows interpreter argv[0]; the ordered remaining argv is exact. PIDs 16060, 7512, 444 and 16764 are gone. Outer controller and observer exit zero; the comparison worker exits one for retained numerical negatives, and its owned tree is proven terminal.

All 24 external one-second memory samples and 416 internal gates pass. External sampled minima are 104.87 GiB RAM and 109.39 GiB commit availability; the maximum observed sample gap is 1.058 seconds. The fixed worker deadline is 1,800 seconds. Three host fixtures pass the deadline, exact child-before-launcher timeout cleanup/partial preservation and Windows argv-normalization checks. The pre-edit driver and contract bytes remain in the preparation directory; the native client and previously executed native bytes are unchanged.

Receipts, all comparisons, projection-change rows, indexes, logs, executed driver/contract bytes and process/resource evidence are preserved here. The 288 formal NPY files remain only under the new E directory. No inference retry, NPU/GPU execution, C++ compilation, affinity change, production route or commit/push occurred. Native/CPU numerical acceptance remains false.
