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

## Notes

- The current codebase is a scaffold focused on experiment orchestration and paper production.
- Benchmark execution still depends on local model access, datasets, and your final prompt/evaluation policy.
- Keep all generated outputs under `results/` so tables can be imported into LaTeX cleanly.
