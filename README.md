# Resource-Constrained LLM Evaluation

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10-blue" />
  <img height="20" src="https://img.shields.io/badge/pytorch-2.3+-ee4c2c" />
  <img height="20" src="https://img.shields.io/badge/cuda-12.1-76b900" />
  <img height="20" src="https://img.shields.io/badge/transformers-4.51+-yellow" />
  <img height="20" src="https://img.shields.io/badge/lm--eval-0.4.7+-purple" />
  <img height="20" src="https://img.shields.io/badge/QLoRA-4bit-red" />
  <img height="20" src="https://img.shields.io/badge/GPU-RTX3090%2024GB-0099ff" />
  <img height="20" src="https://img.shields.io/badge/latex-paper-green" />
</p>

This repository is organized for a publishable paper on:

`Evaluation and QLoRA-based Adaptation of Open LLMs up to 8B Parameters under Single-GPU Resource Constraints`

Target hardware:

- 1 x RTX 3090 24 GB
- CUDA-enabled Linux or Windows + WSL

Target model pool:

- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `Qwen3-4B`
- `Qwen3-8B`
- `Qwen2.5-7B-Instruct`
- `DeepSeek-R1-Distill-Qwen-7B`
- `Gemma-3-4B`

Planned paper scope:

- Unified baseline evaluation
- 4-bit deployment comparison
- QLoRA domain adaptation on selected models
- Post-adaptation re-evaluation
- Performance-efficiency tradeoff analysis

## Repository Layout

```text
configs/         Experiment, model, and task configuration
scripts/         Bash and PowerShell entry scripts
src/             Python package for orchestration
paper/           Standalone LaTeX manuscript workspace
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

- Main experiment config: `configs/experiments/single_gpu_3090.yaml`
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

- Main paper workspace: `paper/main.tex`
- Paper notes: `paper/todo.md`
- Paper README: `paper/README.md`
- Bibliography: `paper/bib/references.bib`
- Formal run checklist: `FORMAL_EXPERIMENT_CHECKLIST.md`
- Work plan: `WORKPLAN.md`

## Output And Snapshot Paths

The experiment output roots are defined in config files:

- Main experiment output root: `results/single_gpu_3090`
- Pilot experiment output root: `results/pilot_single_gpu_3090`

Typical baseline output paths under `results/single_gpu_3090/baseline/<model_key>/`:

- Run plan snapshot: `<model_key>_<precision>_plan.json`
- lm-eval raw snapshot: `<model_key>_<precision>_lm_eval.json`
- Local domain snapshot: `<model_key>_<precision>_domain_qa.json`
- Domain generation snapshot: `<model_key>_<precision>_domain_generations.json`
- Efficiency snapshot: `<model_key>_<precision>_efficiency.json`
- Efficiency generation snapshot: `<model_key>_<precision>_efficiency_generations.json`
- Combined summary snapshot: `<model_key>_<precision>_summary.json`
- Combined summary CSV: `<model_key>_<precision>_summary.csv`

Typical adapted evaluation paths under `results/single_gpu_3090/qlora_eval/<model_key>/`:

- Adapted run plan snapshot: `<model_key>_int4_<label>_plan.json`
- Adapted lm-eval snapshot: `<model_key>_int4_<label>_lm_eval.json`
- Adapted summary snapshot: `<model_key>_int4_<label>_summary.json`
- Adapted summary CSV: `<model_key>_int4_<label>_summary.csv`

Typical QLoRA training paths under `results/single_gpu_3090/qlora/<model_key>/`:

- Run configuration snapshot: `run_config.json`
- Trainer checkpoints: `checkpoint/`
- Training metrics: `train_metrics.json`
- Evaluation metrics: `eval_metrics.json`
- Saved adapter weights: `adapter/`

Aggregated result and export paths:

- Baseline aggregated metrics: `results/single_gpu_3090/baseline/all_metrics.csv`
- Baseline aggregated efficiency: `results/single_gpu_3090/baseline/all_efficiency.csv`
- Adapted aggregated metrics: `results/single_gpu_3090/qlora_eval/all_metrics.csv`
- Adapted aggregated efficiency: `results/single_gpu_3090/qlora_eval/all_efficiency.csv`
- Generated result tables: `results/single_gpu_3090/baseline/tables/`
- Generated paper tables: `paper/tables/generated_main_results.tex`
- Generated paper efficiency table: `paper/tables/generated_efficiency_results.tex`
- Generated paper QLoRA table: `paper/tables/generated_qlora_results.tex`

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

## Immediate Work Items

1. Install dependencies and verify `lm-evaluation-harness`.
2. Download and cache all selected base models.
3. Run baseline evaluation for all models on the same task set.
4. Measure latency, throughput, and peak VRAM.
5. Choose 2-3 models for QLoRA.
6. Build a small domain benchmark set.
7. Re-run evaluation after QLoRA and prepare paper tables/figures.

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
  --experiment configs/experiments/single_gpu_3090.yaml \
  --model qwen3_4b
```

Run the full baseline sweep:

```bash
bash scripts/run_baseline_all.sh
```

Or use Make:

```bash
make baseline MODEL=qwen3_4b
make baseline-all
make summarize
make export-paper-tables
make qlora MODEL=qwen3_4b DATASET=domain_qa
make paper
```

Launch a QLoRA experiment:

```bash
python -m src.rc_llm_eval.cli run-qlora \
  --experiment configs/experiments/single_gpu_3090.yaml \
  --model qwen3_4b \
  --dataset domain_qa
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

Use `FORMAL_EXPERIMENT_CHECKLIST.md` as the gate before starting the full baseline and QLoRA runs.

## Notes

- The current codebase is a scaffold focused on experiment orchestration and paper production.
- Benchmark execution still depends on local model access, datasets, and your final prompt/evaluation policy.
- Keep all generated outputs under `results/` so tables can be imported into LaTeX cleanly.
