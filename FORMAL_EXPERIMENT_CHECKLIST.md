# Formal Experiment Checklist

This checklist is the gate before running paper-grade experiments.

## 1. Freeze The Protocol

- Confirm the final model list for the baseline sweep.
- Confirm the final task list and splits.
- Freeze decoding settings:
  - `num_fewshot`
  - `batch_size`
  - `precision`
  - `max_new_tokens`
  - `temperature`
  - `top_p`
- Freeze the QLoRA recipe:
  - candidate models
  - target modules
  - learning rate
  - epochs
  - batch size
  - accumulation steps
  - max sequence length
- Record the exact experiment config file used for the paper.

## 2. Verify The Runtime

- Use a single Python environment for all formal runs.
- Verify `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, `trl`, and `lm_eval` import successfully.
- Verify `torch.cuda.is_available()` is `True`.
- Verify the GPU is the intended device and has enough free VRAM before each long run.
- If Hugging Face downloads go through a proxy, set `HF_HUB_DISABLE_XET=1`.
- If available, set `HF_TOKEN` before long download sessions to reduce rate-limit risk.
- If model weights are not cached yet, prefetch them before the pilot run.

Recommended prefetch command:

```bash
conda activate rc-llm-eval
bash scripts/run_prefetch_models.sh
```

## 3. Run A Pilot Before The Full Sweep

- Run a small pilot on at least one real target model with `configs/experiments/pilot_single_gpu_3090.yaml`.
- Confirm the model downloads fully.
- Confirm `lm_eval` finishes on the limited subset.
- Confirm local `domain_qa` evaluation finishes.
- Confirm efficiency JSON is written.
- Confirm summary JSON and CSV are written.

Recommended command:

```bash
conda activate rc-llm-eval
bash scripts/run_baseline_pilot.sh
```

## 4. Baseline Sweep Requirements

- Run the same benchmark suite for every baseline model.
- Keep all raw outputs under `results/`.
- Do not change prompts, decoding settings, or task definitions mid-sweep.
- Save aggregated outputs with:
  - per-model summary JSON
  - per-model summary CSV
  - `all_metrics.csv`
  - `all_efficiency.csv`
- Record any failed runs and reruns in a text log.

## 5. QLoRA Readiness

- Select 2 to 3 models from baseline tradeoff results.
- Freeze the domain train and validation files.
- Write down annotation rules and dataset construction notes.
- Confirm training logs, adapter checkpoints, and tokenizer files are saved.
- Re-run both domain and general benchmarks after adaptation.

## 6. Paper Readiness

- Generate the final aggregated CSV files.
- Export LaTeX tables from results.
- Produce at least:
  - a main benchmark table
  - an efficiency table
  - a before/after QLoRA comparison table
  - one tradeoff figure
- Fill all `paper/sections/*.tex` files with final numbers, not placeholders.
- Record the final git commit used for reported experiments.

## 7. Stop Conditions

Do not start the full paper sweep until all of the following are true:

- Pilot run completes on a real target model.
- Summary JSON and CSV include both `lm_eval` and `domain_qa` rows.
- Efficiency metrics are present.
- The experiment config is frozen.
- The download path for Hugging Face models is stable.
