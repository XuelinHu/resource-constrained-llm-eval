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
