## Table III

Multi-generator RAG comparison.

| Generator | No retrieval F1 (ZH/EN) | Approved-hybrid F1 (ZH/EN) | Gain (ZH/EN) | Holm-adjusted *p* (ZH/EN) |
|---|---:|---:|---:|---:|
| Original Qwen | 0.262 / 0.306 | 0.402 / 0.440 | +0.141 / +0.133 | 1.89e-39 / 2.28e-42 |
| Original GLM | 0.230 / 0.269 | 0.297 / 0.318 | +0.067 / +0.048 | 1.13e-22 / 9.56e-31 |
| Qwen QLoRA | 0.630 / 0.573 | 0.660 / 0.651 | +0.030 / +0.078 | 1.34e-06 / 1.41e-11 |
| GLM QLoRA | 0.398 / 0.331 | 0.476 / 0.351 | +0.077 / +0.020 | 3.16e-09 / 9.37e-03 |
| Qwen3-14B | 0.295 / 0.336 | 0.442 / 0.454 | +0.147 / +0.118 | 6.66e-32 / 7.58e-29 |
