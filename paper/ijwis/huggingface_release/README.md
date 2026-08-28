---
license: other
language:
  - zh
  - en
tags:
  - railway
  - bilingual
  - retrieval-augmented-generation
  - web-information-system
  - resource-constrained-llm
---

# IJWIS bilingual railway RAG reproducibility release

This dataset repository contains the non-copyrighted derived artifacts supporting the manuscript **A Knowledge-Enhanced Large Language Model Web Information System for Bilingual Railway Vocational Education**.

## Contents

- `analysis/`: aggregate retrieval, RAG, translation, efficiency, statistical and error-category results.
- `metadata/`: dataset statistics, experiment manifest and supplementary validation metadata.
- `configs/`: the frozen experiment and model configuration files.

The release intentionally excludes source passages, textbook or regulation text, raw prompts and answers, model checkpoints, adapters, tokens, `.env` files and runtime logs. Restricted source passages and additional reproducibility materials can be requested from the authors through the journal submission system, subject to the applicable rights and access conditions.

## Experimental scope

- 400 bilingual railway knowledge pairs and 800 formal language-specific queries.
- Pair-grouped QLoRA split: 12,178 training, 1,524 validation and 1,526 test records.
- QLoRA models: Qwen2.5-7B-Instruct and GLM-4-9B-Chat-HF.
- Retrieval: BM25, BGE-M3/pgvector and hybrid reciprocal-rank fusion.
- Reference generator: Qwen3-14B through Ollama.
- Fixed seed: 42; hardware: one RTX 3090 24 GB workstation.

## Interpretation

The three authors jointly reviewed the knowledge records and checked the training and evaluation data. This was an author-led governance procedure, not an independent inter-rater study. Automated citation and evidence-support values are proxies and must not be interpreted as factual entailment or expert consensus.

The source Git commit and checksums are recorded in `metadata/experiment_manifest.json`. This repository is intended to remain private until the authors decide how to share the derived artifacts. Results are released for research reproducibility and should not be used as authority for live safety-critical railway operation.
