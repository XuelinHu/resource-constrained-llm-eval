# SCI Paper Workplan

## Objective

Produce a journal-ready paper instead of a course-style benchmark report.

## Core Deliverables

1. A reproducible evaluation pipeline for 7 open models under a single RTX 3090 constraint.
2. A QLoRA adaptation pipeline for selected models.
3. A domain benchmark or test set with clear annotation rules.
4. Publication-quality tables and figures.
5. A complete LaTeX manuscript under `paper/`.

## Execution Order

### Phase 1. Baseline Infrastructure

1. Verify `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`.
2. Verify `lm-evaluation-harness` locally.
3. Finalize benchmark list.
4. Freeze prompt templates and decoding settings.

### Phase 2. Baseline Experiments

1. Run all 7 models on the same benchmark suite.
2. Record raw task metrics.
3. Record peak VRAM, latency, and throughput.
4. Export all results to CSV.

### Phase 3. QLoRA Experiments

1. Select 2-3 models based on baseline tradeoff.
2. Build domain training and validation sets.
3. Run QLoRA with a fixed training recipe.
4. Save adapters and training logs.

### Phase 4. Re-evaluation and Analysis

1. Re-run domain and general benchmarks for adapted models.
2. Compare before/after QLoRA.
3. Analyze where small models saturate or fail.
4. Analyze whether reasoning-style models justify inference overhead.

### Phase 5. Writing

1. Fill `paper/sections/*.tex`.
2. Add tables from `results/`.
3. Add figures for tradeoff and ablation analysis.
4. Finalize references and submission formatting.

## Minimum Publishable Experiment Set

If time becomes tight, keep this reduced scope:

1. Models: `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`, `Gemma-3-4B`
2. Tasks: `MMLU`, `GSM8K`, `HumanEval or MBPP`, `C-Eval`, one domain set
3. Optimization: `QLoRA` only
4. Efficiency metrics: `peak_vram_gb`, `tokens_per_second`, `mean_latency_s`
