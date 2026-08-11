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
| Hardware | RTX 3090 24 GB; 32 GB RAM; single workstation |
| Statistical analysis | 2,000 bootstrap resamples; paired Wilcoxon; Holm correction |

## Table II

Formal cross-source retrieval on 400 knowledge pairs.

| Retrieval | Language | R@1 | R@3 | R@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | Chinese | 0.520 | 0.595 | 0.620 | 0.560 | **69.5** |
| Vector | Chinese | 0.518 | 0.653 | 0.690 | 0.588 | 134.4 |
| Hybrid approved | Chinese | **0.570** | **0.675** | **0.715** | **0.628** | 217.6 |
| BM25 | English | **0.570** | **0.670** | 0.685 | 0.620 | **59.0** |
| Vector | English | 0.463 | 0.553 | 0.580 | 0.509 | 134.4 |
| Hybrid approved | English | **0.570** | 0.668 | **0.708** | **0.620** | 209.2 |

## Table III

Supplementary information-system validation results.

| Validation | Chinese | English | Operational result |
|---|---:|---:|---|
| Bilingual-field hybrid index, Recall@5 | 0.718 | 0.675 | Highest balanced mean (0.696) |
| Original Qwen hybrid, supported-claim proxy | 0.878 | 0.836 | Citation precision 0.955/0.970 |
| Qwen QLoRA hybrid, supported-claim proxy | 0.964 | 0.905 | Citation recall 0.000/0.002 |
| Governance history | - | - | 1,337 events; 82 edits; two recorded reviewers |
| BM25 retrieval, concurrency 10 | - | - | P95 0.515 s; 21.57 requests/s; 0/12 failures |
| Hybrid retrieval, concurrency 10 | - | - | P95 1.125 s; 9.51 requests/s; 0/12 failures |
