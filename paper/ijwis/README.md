# IJWIS manuscript workspace

Target journal: *International Journal of Web Information Systems* (Emerald).

Working title: **A Knowledge-Enhanced Large Language Model Web Information System for Bilingual Railway Vocational Education**

The manuscript follows an Emerald-style structured abstract and a system-oriented argument. The former generic single-GPU benchmark draft has been removed; this directory is the authoritative draft for the IJWIS submission.

The authoritative execution checklist is `experiment_todo.md`. The formal experiment matrix and automated system validation were completed on 2026-08-01. Knowledge review was performed jointly by the three authors; no independent inter-rater statistic or new human semantic rating is claimed.

The authoritative text is `manuscript.md`. `IJWIS__Copy_/Main.tex` is the anonymous IJWIS template wrapper, and `IJWIS__Copy_/manuscript_body.tex` is regenerated from Markdown. The three authors jointly checked the training and evaluation data and reviewed domain correctness, educational suitability, bilingual expression and source consistency. The manuscript does not claim independent external-expert agreement.

## Reproducible evidence

- Formal bilingual retrieval: `data/exports/retrieval_eval_railway_bilingual_400.json`
- Top-k retrieval ablation: `data/exports/retrieval_eval_top{1,3,5,8}.json`
- Approved QLoRA data statistics: `data/qlora_bilingual_approved/statistics.json`
- Dedicated regulation RAG test: `data/rag_eval/regulation_test_120.jsonl`
- Formal cross-source RAG test: `data/rag_eval/railway_bilingual_400.jsonl`
- Formal RAG test statistics: `data/rag_eval/railway_bilingual_400_statistics.json`
- Frozen environment: `results/ijwis_single_gpu_3090/manifest/experiment_manifest.json`
- Original/QLoRA generations: `results/ijwis_single_gpu_3090/{baseline,qlora_eval}/`
- Multi-generator RAG: `results/ijwis_single_gpu_3090/rag/`
- Analysis CSV: `results/ijwis_single_gpu_3090/analysis/`
- Traceable asset hashes: `results/ijwis_single_gpu_3090/analysis/asset_manifest.json`
- Supplementary validation hashes: `results/ijwis_single_gpu_3090/analysis/supplementary_asset_manifest.json`
- Paper-ready tables and figures: `paper/ijwis/{tables,figures}/`
- Editable architecture sources: `paper/ijwis/figures/{system_architecture,knowledge_governance_lifecycle}.drawio`
- Translation metrics: direction- and subtask-separated SacreBLEU, chrF++ and COMET using `Unbabel/wmt22-comet-da` revision `2760a223ac957f30acfb18c8aa649b01cf1d75f2`.

All numerical claims in `manuscript.md` are regenerated from these artifacts. Independent human semantic evaluation remains outside the current protocol. In particular, citation-format coverage is not treated as citation entailment or factual correctness. A public release should contain only non-copyrighted derived metrics and identifiers; restricted source passages and reproducibility materials can be requested from the authors through the journal submission system.

## Core commands

```bash
conda run -n rc-llm-eval python scripts/freeze_experiment_manifest.py
conda run -n rc-llm-eval python -m src.rc_llm_eval.cli run-qlora --experiment configs/experiments/ijwis_single_gpu_3090.yaml --model qwen2_5_7b_instruct --dataset bilingual_approved_qa
conda run -n rc-llm-eval python -m src.rc_llm_eval.cli run-qlora --experiment configs/experiments/ijwis_single_gpu_3090.yaml --model glm_4_9b_chat_hf --dataset bilingual_approved_qa

conda run -n rc-llm-eval python scripts/evaluate_rag_generators.py --prepare --top-k 3
conda run -n rc-llm-eval python scripts/evaluate_rag_generators.py --backend hf --model-key qwen2_5_7b_instruct --label qwen2_5_original
conda run -n rc-llm-eval python scripts/evaluate_rag_generators.py --backend ollama --ollama-model qwen3:14b --label qwen3_14b_reference

HF_ENDPOINT=https://hf-mirror.com conda run -n rc-llm-comet python scripts/evaluate_comet.py GENERATIONS.json --checkpoint ~/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt

conda run -n rc-llm-eval python scripts/export_ijwis_design_tables.py
conda run -n rc-llm-eval python scripts/analyze_ijwis_results.py
conda run -n rc-llm-eval python scripts/plot_ijwis_architecture.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n rc-llm-eval python scripts/evaluate_bilingual_index_ablation.py --batch-size 8 --max-seq-length 512
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n rc-llm-eval python scripts/evaluate_rag_faithfulness.py --backend embedding --batch-size 8
conda run -n rc-llm-eval python scripts/analyze_governance_history.py
conda run -n rc-llm-eval python scripts/load_test_rag_api.py --concurrency 1 5 10 --requests 12
conda run -n rc-llm-eval python scripts/build_ijwis_validation_figure.py
conda run -n rc-llm-eval python scripts/export_ijwis_latex.py
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error IJWIS__Copy_/Main.tex
cp IJWIS__Copy_/Main.pdf output/pdf/ijwis_bilingual_railway_manuscript_latex.pdf
```
