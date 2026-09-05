# Bilingual Railway RAG and Resource-Constrained LLM Evaluation

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10-3776AB?logo=python&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/pytorch-2.3%2B-EE4C2C?logo=pytorch&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/cuda-12.1-76B900?logo=nvidia&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/transformers-4.51%2B-FFD21E?logo=huggingface&amp;logoColor=black" />
  <img height="20" src="https://img.shields.io/badge/lm--eval-0.4.7%2B-8A2BE2" />
  <img height="20" src="https://img.shields.io/badge/QLoRA-4--bit-CC0000" />
  <img height="20" src="https://img.shields.io/badge/GPU-RTX%203090%2024GB-0099FF" />
</p>

This repository contains the experiments and manuscript assets for the IJWIS paper:

`Knowledge-Enhanced Large Language Models for Bilingual Railway Vocational Education under Resource Constraints`

Target hardware:

- 1 x RTX 3090 24 GB
- CUDA-enabled Linux or Windows + WSL

Final evaluated model matrix:

- `Qwen2.5-7B-Instruct`
- `GLM-4-9B-Chat-HF`
- `Qwen3-14B` through Ollama as the unadapted reference generator
- `BAAI/bge-m3` with PostgreSQL + pgvector for bilingual retrieval

Planned paper scope:

- Unified baseline evaluation
- 4-bit deployment comparison
- QLoRA domain adaptation on selected models
- Post-adaptation re-evaluation
- Performance-efficiency trade-off analysis and automated system validation

## Repository Layout

```text
configs/         Experiment, model, and task configuration
scripts/         Bash and PowerShell entry scripts
src/             Python package for orchestration
paper/ijwis/     Authoritative IJWIS manuscript, evidence and submission package
IJWIS__Copy_/    Anonymous IJWIS LaTeX template wrapper
results/         Generated outputs, metrics, and tables
```

## File Path Index

### Runtime Entry Paths

- CLI entry: `src/rc_llm_eval/cli.py`
- Python package marker: `src/__init__.py`
- Package version marker: `src/rc_llm_eval/__init__.py`

### Script Paths

- Environment setup: `scripts/setup_conda.sh`
- Model prefetch wrapper: `scripts/run_prefetch_models.sh`
- Model prefetch implementation: `scripts/prefetch_models.py`
- Pilot baseline run: `scripts/run_baseline_pilot.sh`
- Full baseline run: `scripts/run_baseline_all.sh`
- Single evaluation PowerShell entry: `scripts/run_eval.ps1`
- Single QLoRA PowerShell entry: `scripts/run_qlora.ps1`
- Batch QLoRA training: `scripts/run_qlora_all.sh`
- Batch adapted evaluation: `scripts/run_qlora_eval_all.sh`
- Result summarization: `scripts/summarize_results.ps1`
- Paper table export: `scripts/export_paper_tables.sh`
- Paper table export for PowerShell: `scripts/export_paper_tables.ps1`
- End-to-end formal pipeline: `scripts/run_formal_pipeline.sh`

### Source Paths

- Baseline pipeline: `src/rc_llm_eval/pipelines/baseline.py`
- QLoRA pipeline: `src/rc_llm_eval/pipelines/qlora.py`
- Reporting pipeline: `src/rc_llm_eval/pipelines/reporting.py`
- Config utilities: `src/rc_llm_eval/utils/config.py`
- Data IO utilities: `src/rc_llm_eval/utils/io.py`
- Model loading utilities: `src/rc_llm_eval/utils/modeling.py`
- System utilities: `src/rc_llm_eval/utils/system.py`
- Text normalization utilities: `src/rc_llm_eval/utils/text.py`

### Configuration Paths

- Main experiment config: `configs/experiments/ijwis_single_gpu_3090.yaml`
- Pilot experiment config: `configs/experiments/pilot_single_gpu_3090.yaml`
- Dataset registry: `configs/datasets/tasks.yaml`
- Model registry: `configs/models/models.yaml`

### Dataset Paths

- Domain train set: `data/domain/train.jsonl`
- Domain validation set: `data/domain/valid.jsonl`
- Domain test set: `data/domain/test.jsonl`
- Efficiency prompts: `data/efficiency/prompts.jsonl`
- Domain dataset notes: `data/domain/README.md`

### Paper And Project Notes

- Authoritative manuscript: `paper/ijwis/manuscript.md`
- IJWIS project README: `paper/ijwis/README.md`
- Anonymous submission package: `paper/ijwis/submission/`
- LaTeX template wrapper: `IJWIS__Copy_/Main.tex`
- Final evidence tables and figures: `paper/ijwis/{tables,figures}/`
- Final full manuscript PDF: `output/pdf/ijwis_manuscript_full.pdf`
- Final anonymous manuscript PDF: `output/pdf/ijwis_manuscript_anonymous.pdf`
- Final Chinese review PDF: `output/pdf/ijwis_manuscript_zh.pdf`
- Private derived-artifact release: `paper/ijwis/huggingface_release/`
- Formal experiment checklist: `paper/ijwis/experiment_todo.md`

## Output And Snapshot Paths

The experiment output roots are defined in config files:

- Main experiment output root: `results/ijwis_single_gpu_3090`
- Pilot experiment output root: `results/pilot_single_gpu_3090`

Typical baseline output paths under `results/ijwis_single_gpu_3090/baseline/<model_key>/`:

- Run plan snapshot: `<model_key>_<precision>_plan.json`
- lm-eval raw snapshot: `<model_key>_<precision>_lm_eval.json`
- Local domain snapshot: `<model_key>_<precision>_domain_qa.json`
- Domain generation snapshot: `<model_key>_<precision>_domain_generations.json`
- Efficiency snapshot: `<model_key>_<precision>_efficiency.json`
- Efficiency generation snapshot: `<model_key>_<precision>_efficiency_generations.json`
- Combined summary snapshot: `<model_key>_<precision>_summary.json`
- Combined summary CSV: `<model_key>_<precision>_summary.csv`

Typical adapted evaluation paths under `results/ijwis_single_gpu_3090/qlora_eval/<model_key>/`:

- Adapted run plan snapshot: `<model_key>_int4_<label>_plan.json`
- Adapted lm-eval snapshot: `<model_key>_int4_<label>_lm_eval.json`
- Adapted summary snapshot: `<model_key>_int4_<label>_summary.json`
- Adapted summary CSV: `<model_key>_int4_<label>_summary.csv`

Typical QLoRA training paths under `results/ijwis_single_gpu_3090/qlora/<model_key>/`:

- Run configuration snapshot: `run_config.json`
- Trainer checkpoints: `checkpoint/`
- Training metrics: `train_metrics.json`
- Evaluation metrics: `eval_metrics.json`
- Saved adapter weights: `adapter/`

Aggregated result and export paths:

- Baseline aggregated metrics: `results/ijwis_single_gpu_3090/baseline/all_metrics.csv`
- Baseline aggregated efficiency: `results/ijwis_single_gpu_3090/baseline/all_efficiency.csv`
- Adapted aggregated metrics: `results/ijwis_single_gpu_3090/qlora_eval/all_metrics.csv`
- Adapted aggregated efficiency: `results/ijwis_single_gpu_3090/qlora_eval/all_efficiency.csv`
- Generated result tables: `results/ijwis_single_gpu_3090/baseline/tables/`
- IJWIS paper tables: `paper/ijwis/tables/*.csv`
- IJWIS paper figures: `paper/ijwis/figures/*.pdf`

## Script Guide

The repository keeps experiment orchestration split by responsibility so single-GPU runs can be resumed or repeated safely:

- `scripts/setup_conda.sh`: create or update the Conda environment from `environment.yml`.
- `scripts/run_prefetch_models.sh` and `scripts/prefetch_models.py`: pre-download baseline models into the local Hugging Face cache.
- `scripts/run_baseline_pilot.sh`: run a lightweight pilot baseline before the full sweep.
- `scripts/run_baseline_all.sh` and `scripts/run_eval.ps1`: launch baseline evaluation jobs.
- `scripts/run_qlora_all.sh` and `scripts/run_qlora.ps1`: launch QLoRA training jobs.
- `scripts/run_qlora_eval_all.sh`: evaluate trained adapters with the baseline pipeline.
- `scripts/summarize_results.ps1`: aggregate generated result files into summary CSV files.
- `scripts/export_paper_tables.sh` and `scripts/export_paper_tables.ps1`: export paper-ready LaTeX tables.
- `scripts/run_formal_pipeline.sh`: run the end-to-end formal experiment sequence with logging and GPU-budget checks.

## Commenting Convention

- Python files and script files now use UTF-8 encoded Chinese comments.
- Comments are focused on module responsibility, key control flow, resource assumptions, and non-obvious implementation choices.
- Trivial statements are intentionally left uncommented so the code remains readable during later maintenance.

## Readme Variants

- English README: `README.md`
- Chinese README: `README-CN.md`

## Finalization Status

The formal experiment matrix, manuscript, figures, tables and anonymous submission files have been generated. Remaining journal-side work is limited to final author metadata, journal submission fields and any requested supplementary-material upload.

## Environment Setup

Recommended for Ubuntu + CUDA:

```bash
conda env create -f environment.yml
conda activate rc-llm-eval
```

Or update/create explicitly:

```bash
bash scripts/setup_conda.sh
conda activate rc-llm-eval
```

If you prefer pip inside an existing environment:

```powershell
pip install -r requirements.txt
```

## Example Commands

List planned tasks and models:

```powershell
python -m src.rc_llm_eval.cli print-plan
```

Run an lm-eval job from config:

```bash
python -m src.rc_llm_eval.cli run-eval \
  --experiment configs/experiments/ijwis_single_gpu_3090.yaml \
  --model qwen2_5_7b_instruct
```

Run the full baseline sweep:

```bash
bash scripts/run_baseline_all.sh
```

Or use Make:

```bash
make baseline MODEL=qwen2_5_7b_instruct
make baseline-all
make summarize
make export-paper-tables
make qlora MODEL=qwen2_5_7b_instruct DATASET=bilingual_approved_qa
make paper
```

Launch a QLoRA experiment:

```bash
python -m src.rc_llm_eval.cli run-qlora \
  --experiment configs/experiments/ijwis_single_gpu_3090.yaml \
  --model qwen2_5_7b_instruct \
  --dataset bilingual_approved_qa
```

## Recommended Run Order

Before the full paper sweep, optionally prefetch baseline models into the local Hugging Face cache:

```bash
conda activate rc-llm-eval
bash scripts/run_prefetch_models.sh
```

Then run a pilot on a real target model:

```bash
conda activate rc-llm-eval
bash scripts/run_baseline_pilot.sh
```

Use `paper/ijwis/experiment_todo.md` as the authoritative record of the completed formal experiments.

## Notes

- Large local datasets, model caches, generated results and competition materials are intentionally kept outside the Git submission surface.
- The private Hugging Face repository contains derived metrics and metadata only; raw copyrighted source passages and model weights are excluded.
- Keep all generated outputs under `results/` so tables can be imported into LaTeX cleanly.

## format
- https://peerj.com/articles/cs-3773/
- https://peerj.com/articles/cs-3762/
- https://www.overleaf.com/latex/templates/latex-template-for-peerj-journal-and-pre-print-submissions/ptdwfrqxqzbn
- https://peerj.com/about/policies-and-procedures/#discipline-standards
- https://peerj.com/about/author-instructions/#reference-format

<!-- codex-runtime-notes:start -->

## Runtime Ports And Database Configuration

### Database
- Core benchmark pipeline does not require a database.
- The optional annotation system uses PostgreSQL.
- Annotation database name: `railway_annotation`.
- Default PostgreSQL host and port: `localhost:5432`.
- Default annotation DB user: `deipss`; password must come from local `.env` and must not be committed.
- Annotation RAG helper uses local Ollama by default at `http://127.0.0.1:11434`.

### Default Ports
- Annotation backend FastAPI service: `8000`.
- Annotation frontend local Vite port: `5173`; project frontend mode also uses `4005` for local/FRP exposure.
- PostgreSQL: `5432`.
- Ollama: `11434`.

### Notes
- This repository may contain large local datasets and generated outputs; do not stage unrelated data changes with documentation edits.

### Source Files Checked
- `annotation_system/backend/.env.example`
- `annotation_system/backend/app/config.py`
- `annotation_system/frontend/vite.config.js`
- `annotation_system/README.md`

<!-- codex-runtime-notes:end -->
