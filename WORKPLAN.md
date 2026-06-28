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

## Current Railway Corpus Review Dataset

The annotation system currently stores the review corpus in PostgreSQL and serves it through `annotation_system/backend` APIs. The Vue frontend does not read JSONL files directly; it reads `/api/items`, `/api/stats`, `/api/options`, and related endpoints. Source files are imported into the database with stable `external_id` de-duplication.

Current review database status:

- Total review items: 19,316
- Review status: 19,316 `pending`, 0 `approved`, 0 `needs_revision`, 0 `rejected`
- RAG retrieval index size: 13,634 documents
- Main source families:
  1. Terminology records
  2. Regulation QA records
  3. Textbook QA records
  4. Textbook source/OCR records

Task type distribution:

| Task type | Count |
| --- | ---: |
| `terminology_pair` | 11,857 |
| `regulation_requirement_qa` | 883 |
| `regulation_extractive_qa` | 789 |
| `regulation_multiple_choice` | 672 |
| `textbook_extractive_qa` | 635 |
| `concept_explanation_qa` | 632 |
| `textbook_multiple_choice` | 632 |
| `textbook_source` | 552 |
| `regulation_clause_qa` | 486 |
| `regulation_inspection_qa` | 457 |
| `textbook_judgment` | 455 |
| `textbook_operation_qa` | 308 |
| `regulation_standard_qa` | 267 |
| `regulation_judgment` | 141 |
| `regulation_prohibition_qa` | 131 |
| `terminology_explanation` | 126 |
| `terminology_translation` | 116 |
| `textbook_definition_qa` | 94 |
| `regulation_definition_qa` | 60 |
| `regulation_responsibility_qa` | 21 |
| `regulation_principle_qa` | 2 |

## Current RAG System Status

The current RAG system is functionally usable but not yet complete as a research-grade or production-grade subsystem.

RAG technology stack:

| Layer | Current choice |
| --- | --- |
| Frontend | Vue 3 + Vite |
| Backend API | FastAPI |
| Database | PostgreSQL |
| ORM / database access | SQLAlchemy |
| Retrieval | Project-local BM25-style inverted index |
| Tokenization | Chinese character tokens + Chinese bigrams + Latin/number regex tokens |
| Index storage | In-memory Python structures rebuilt from PostgreSQL |
| Generation runtime | Local Ollama HTTP API |
| Default generation model | `qwen3:14b` |
| RAG API endpoints | `/api/rag/stats`, `/api/rag/rebuild`, `/api/rag/ask` |
| Fallback behavior | Return top retrieved evidence when local generation fails |
| Not currently used | LangChain, LlamaIndex, vector database, embedding model, reranker |

Implemented:

1. Builds an in-memory character-level BM25-style index from PostgreSQL `corpus_items`.
2. Excludes `generated_eval_review` and test split records to reduce evaluation leakage.
3. Searches across question, answer, evidence, domain, knowledge category, and chapter text.
4. Returns source metadata including document, task type, domain category, page number, and review status.
5. Supports retrieval-only mode.
6. Supports generation mode through local Ollama model `qwen3:14b`.
7. Falls back to the most relevant retrieved evidence when the local generation model is unavailable.
8. Provides frontend controls for question input, top-k evidence count, generation toggle, examples, and evidence display.

Still missing before calling it complete:

1. RAG quality evaluation set and metrics, such as retrieval recall, answer faithfulness, citation accuracy, and hallucination rate.
2. Manual review workflow for RAG answers and failed retrieval cases.
3. Persistent or rebuildable index artifact instead of only in-memory indexing.
4. Hybrid retrieval with dense embeddings or reranking for better semantic matching.
5. Stronger source filtering by task type, domain category, status, and approved-only records.
6. Citation-level answer formatting that binds each answer sentence to evidence IDs.
7. Monitoring/logging of user questions, retrieval results, generation errors, and latency.
8. Clear policy on whether pending records can be used for final RAG answers.

Near-term RAG TODO:

1. Add an approved-only retrieval switch for final-facing answers.
2. Create a small RAG validation set from reviewed railway questions.
3. Add retrieval and generation logs to PostgreSQL.
4. Add answer feedback labels in the frontend.
5. Evaluate BM25-only retrieval against a dense or reranked baseline.
