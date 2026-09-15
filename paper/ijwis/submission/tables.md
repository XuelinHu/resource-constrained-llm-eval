# Tables

Tables are numbered with Roman numerals and supplied separately in accordance with the IJWIS author guidelines.

## Table I

Formal implementation and experimental environment.

| Component | Formal setting |
|---|---|
| Database | PostgreSQL 16.14; pgvector 0.8.4 |
| Embedding | BAAI/bge-m3; 1,024 dimensions |
| Adapted generators | Qwen2.5-7B-Instruct; GLM-4-9B-Chat-HF |
| Reference generator | Qwen3-14B through Ollama 0.19.0 |
| Training | NF4 QLoRA; rank 64; one epoch; seed 42 |
| Hardware boundary | One RTX 3090 24 GB; 32 GB RAM; no multi-GPU or required cloud inference |
| Statistical analysis | 2,000 bootstrap resamples; paired Wilcoxon; Cohen's *d*<sub>z</sub>; Holm correction |
| Software | Linux; Python 3.11.15; PyTorch 2.11.0; Transformers 5.12.1; PEFT 0.18.1; bitsandbytes 0.49.2; Ollama 0.19.0 |

## Table II

Formal cross-source evidence-equivalent retrieval on 400 knowledge pairs.

| Retrieval | Language | Evidence R@1 | Evidence R@3 | Evidence R@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | Chinese | 0.520 | 0.595 | 0.620 | 0.560 | **69.5** |
| Vector | Chinese | 0.518 | 0.653 | 0.690 | 0.588 | 134.4 |
| Hybrid approved | Chinese | **0.570** | **0.675** | **0.715** | **0.628** | 217.6 |
| BM25 | English | **0.570** | **0.670** | 0.685 | 0.620 | **59.0** |
| Vector | English | 0.463 | 0.553 | 0.580 | 0.509 | 134.4 |
| Hybrid approved | English | **0.570** | 0.668 | **0.708** | **0.620** | 209.2 |

## Table III

Multi-generator RAG comparison.

| Generator | No retrieval F1 (ZH/EN) | Approved-hybrid F1 (ZH/EN) | Gain (ZH/EN) | Holm-adjusted *p* (ZH/EN) |
|---|---:|---:|---:|---:|
| Original Qwen | 0.262 / 0.306 | 0.402 / 0.440 | +0.141 / +0.133 | 1.89e-39 / 2.28e-42 |
| Original GLM | 0.230 / 0.269 | 0.297 / 0.318 | +0.067 / +0.048 | 1.13e-22 / 9.56e-31 |
| Qwen QLoRA | 0.630 / 0.573 | 0.660 / 0.651 | +0.030 / +0.078 | 1.34e-06 / 1.41e-11 |
| GLM QLoRA | 0.398 / 0.331 | 0.476 / 0.351 | +0.077 / +0.020 | 3.16e-09 / 9.37e-03 |
| Qwen3-14B | 0.295 / 0.336 | 0.442 / 0.454 | +0.147 / +0.118 | 6.66e-32 / 7.58e-29 |

## Table IV

Supplementary information-system validation results.

| Validation | Chinese | English | Operational result |
|---|---:|---:|---|
| Bilingual-field hybrid index, Evidence Recall@5 | 0.718 | 0.675 | Highest balanced mean (0.696) |
| Original Qwen hybrid, supported-claim proxy | 0.878 | 0.836 | Citation precision 0.955/0.970 |
| Qwen QLoRA hybrid, supported-claim proxy | 0.964 | 0.905 | Citation recall 0.000/0.002 |
| Governance history | — | — | 1,337 events; 82 edits; two recorded reviewers |
