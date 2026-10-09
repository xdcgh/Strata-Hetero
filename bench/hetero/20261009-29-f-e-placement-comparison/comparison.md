# F/E PLE placement comparison

Created: 2026-10-09T09:07:23.993168+00:00

Mechanical summary: 27 formal requests per arm across nine prompts; warmups excluded.

Normalized configurations equal (`828e33d1df933df84cc0b16823d9945facc22d0b9ca7ccf8548f0fca4e36729a`). Removed only the run-specific `log` path and the value following `--ple-gguf` (sentinel `<PLE_SOURCE>`). Four original F native-dense paths remain present and identical.

Engine source `e87d74c452f7c0bdcb46201fa0a9bcd5453cd10c`; binary SHA-256 `fc3bab3f2797b51407cf3afabaf4ac7fd198cc07bbf88e62cc40b5a639ab2494`; bridge SHA-256 `e0210a9a12ccf423e29b4d85f2ef4963b9850f0a4fce84bc36705dac3d131f5b`; expert-profile SHA-256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.

All formal prompt-cache counts are zero. Actual output token IDs and text bytes match H4 for every F/E formal. Timing cells show mean / median / min–max across three formals. TTFT uses `first_generated_s`; rates are reported engine fields.

| Prompt | F E2E s | E E2E s | F TTFT s | E TTFT s | F prompt ms | E prompt ms | F prompt tok/s | E prompt tok/s | F decode tok/s | E decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke_arithmetic (27 prompt tokens) | 0.706 / 0.697 / 0.665–0.757 | 0.637 / 0.667 / 0.574–0.671 | 0.490 / 0.493 / 0.466–0.510 | 0.430 / 0.438 / 0.396–0.455 | 444.700 / 446.500 / 425.700–461.900 ms | 397.633 / 403.200 / 359.000–430.700 ms | 60.800 / 60.500 / 58.500–63.400 | 68.300 / 67.000 / 62.700–75.200 | 8.167 / 8.000 / 6.900–9.600 | 9.067 / 8.700 / 8.200–10.300 |
| smoke_unicode (32 prompt tokens) | 0.777 / 0.749 / 0.666–0.915 | 0.838 / 0.811 / 0.758–0.945 | 0.384 / 0.377 / 0.376–0.400 | 0.426 / 0.413 / 0.411–0.454 | 359.567 / 354.100 / 346.200–378.400 ms | 398.733 / 391.000 / 375.400–429.800 ms | 89.133 / 90.400 / 84.600–92.400 | 80.500 / 81.800 / 74.500–85.200 | 25.700 / 25.900 / 18.800–32.400 | 23.667 / 23.800 / 19.600–27.600 |
| smoke_json_arithmetic (50 prompt tokens) | 1.295 / 1.292 / 1.283–1.310 | 1.326 / 1.328 / 1.234–1.417 | 0.591 / 0.608 / 0.546–0.618 | 0.573 / 0.540 / 0.540–0.639 | 568.800 / 579.600 / 528.300–598.500 ms | 542.000 / 522.400 / 521.200–582.400 ms | 88.133 / 86.300 / 83.500–94.600 | 92.500 / 95.700 / 85.900–95.900 | 26.333 / 27.100 / 24.400–27.500 | 24.933 / 24.100 / 23.900–26.800 |
| smoke_python_function (39 prompt tokens) | 1.336 / 1.344 / 1.277–1.387 | 1.291 / 1.289 / 1.257–1.327 | 0.434 / 0.430 / 0.407–0.465 | 0.426 / 0.423 / 0.413–0.442 | 411.933 / 401.600 / 390.500–443.700 ms | 404.933 / 406.100 / 396.100–412.600 ms | 94.967 / 97.100 / 87.900–99.900 | 96.333 / 96.000 / 94.500–98.500 | 19.700 / 20.100 / 18.100–20.900 | 20.533 / 20.200 / 20.200–21.200 |
| smoke_three_key_retrieval (114 prompt tokens) | 4.643 / 4.617 / 4.594–4.719 | 4.600 / 4.607 / 4.500–4.693 | 1.057 / 1.065 / 1.020–1.085 | 1.068 / 1.073 / 1.031–1.099 | 1016.000 / 1014.400 / 983.600–1050.000 ms | 1006.100 / 1004.300 / 978.800–1035.200 ms | 112.300 / 112.400 / 108.600–115.900 | 113.367 / 113.500 / 110.100–116.500 | 23.233 / 23.600 / 22.500–23.600 | 23.467 / 23.600 / 22.600–24.200 |
| long_1k (1036 prompt tokens) | 6.796 / 6.697 / 6.677–7.016 | 6.768 / 6.758 / 6.672–6.874 | 3.513 / 3.513 / 3.441–3.583 | 3.524 / 3.501 / 3.488–3.584 | 3470.800 / 3457.300 / 3405.600–3549.500 ms | 3479.100 / 3468.200 / 3429.200–3539.900 ms | 298.600 / 299.700 / 291.900–304.200 | 297.833 / 298.700 / 292.700–302.100 | 25.400 / 26.000 / 23.300–26.900 | 25.667 / 26.100 / 24.700–26.200 |
| long_4k (4109 prompt tokens) | 7.042 / 7.011 / 6.949–7.167 | 7.006 / 7.022 / 6.688–7.307 | 4.112 / 4.092 / 4.074–4.170 | 4.140 / 4.230 / 3.913–4.276 | 4053.467 / 4044.900 / 4014.500–4101.000 ms | 4075.067 / 4158.100 / 3841.300–4225.800 ms | 1013.767 / 1015.800 / 1002.000–1023.500 | 1010.100 / 988.200 / 972.400–1069.700 | 28.433 / 29.100 / 27.000–29.200 | 29.033 / 29.700 / 27.400–30.000 |
| long_16k (16396 prompt tokens) | 12.491 / 12.568 / 12.318–12.588 | 12.035 / 12.006 / 11.877–12.222 | 10.061 / 10.141 / 9.890–10.152 | 9.714 / 9.745 / 9.607–9.790 | 9950.600 / 10019.700 / 9797.500–10034.600 ms | 9624.033 / 9650.300 / 9521.000–9700.800 ms | 1647.933 / 1636.400 / 1633.900–1673.500 | 1703.767 / 1699.000 / 1690.200–1722.100 | 33.900 / 33.900 / 33.800–34.000 | 35.733 / 36.500 / 34.000–36.700 |
| long_30k7 (30712 prompt tokens) | 20.110 / 19.946 / 19.922–20.463 | 19.598 / 19.640 / 19.270–19.884 | 17.509 / 17.505 / 17.335–17.688 | 17.563 / 17.598 / 17.248–17.844 | 17389.100 / 17386.100 / 17220.200–17561.000 ms | 17438.600 / 17473.000 / 17123.600–17719.200 ms | 1766.300 / 1766.500 / 1748.900–1783.500 | 1761.500 / 1757.700 / 1733.300–1793.500 | 32.700 / 34.200 / 26.700–37.200 | 40.833 / 40.800 / 40.700–41.000 |

## Scope and limits

Saved formal single-request timings; three formals per task, warmups excluded. Mean/median/min/max/range only; no pooled p95, cold-start, hardware-counter, or throughput extrapolation.
Actual token-ID capture was enabled; timings reflect this quality-evidence configuration.
All F tasks preceded all E tasks; not randomized/counterbalanced; cache, thermal and system state were uncontrolled.
E long_30k7 had no /v1/status polling during generation; resource sampler was observed separately. F monitoring used its frozen quality controller.
Exploratory placement comparison only; not a >=10% acceptance result and not daily promotion evidence.

Controller evidence is separate from prompt outcomes. The first E matrix controller failed during cleanup binding and masked its original observer exception; root independently confirmed long_16k artifacts complete and H4-matched. No completed task was rerun.
