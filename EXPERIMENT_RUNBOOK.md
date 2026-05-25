# Experiment Runbook

This runbook defines the foreground-only experiment sequence. Do not start these
commands until the current code and dataset are confirmed.

## Scope

The formal baseline model pool is defined in `configs/experiments/single_gpu_3090.yaml`.
Models that are not fully cached locally, such as `llama_3_1_8b_instruct`,
`gemma_2_9b_it`, and `gemma_3_4b`, are intentionally excluded from the formal
baseline list for now.

QLoRA candidates are also defined in `configs/experiments/single_gpu_3090.yaml`.
The current conservative set is:

- `qwen3_4b`
- `qwen2_5_7b_instruct`
- `phi_3_mini_4k_instruct`

## Step 0: Readiness Check

This command only checks local files and configs. It does not download models or
run experiments.

```bash
python scripts/check_experiment_readiness.py --experiment configs/experiments/single_gpu_3090.yaml
```

Expected time: under 1 minute.

## Step 0.5: Full Smoke Baseline on 1% Data

This command rebuilds the stratified 1% smoke dataset, checks local model and
dataset readiness, and then runs the configured smoke baseline in the foreground.

```bash
bash scripts/run_smoke_full.sh
```

Expected time: several hours if all ten smoke models are executed. It is a
smoke test, not a formal result source.

## Step 1: Pilot Baseline

This validates the baseline evaluation path with the pilot config.

```bash
bash scripts/run_baseline_pilot.sh
```

Expected time: 30 minutes to half a day, depending on GPU availability and task
runtime.

## Step 2: QLoRA Smoke

This validates the training path with one model and the railway `domain_qa`
dataset.

```bash
bash scripts/run_qlora_smoke.sh
```

Expected time: several hours for a full pilot-config run. If this is too long,
reduce `num_train_epochs`, `max_seq_length`, or train subset handling before
launching it.

## Step 3: Formal Baseline

This runs baseline evaluation for every configured formal model in the foreground.

```bash
bash scripts/run_baseline_all.sh
```

Expected time: 1 to 3 days, depending on task count and GPU throughput.

## Step 4: Formal QLoRA

This trains the configured QLoRA candidate models in sequence.

```bash
bash scripts/run_qlora_all.sh
```

Expected time: 2 to 4 days for the current candidate set.

## Step 5: Adapted Evaluation

This evaluates trained adapters after QLoRA.

```bash
bash scripts/run_qlora_eval_all.sh
```

Expected time: several hours to 1 day.

## Step 6: Paper Tables

```bash
python -m src.rc_llm_eval.cli export-paper-tables --experiment configs/experiments/single_gpu_3090.yaml
```

Expected time: under 10 minutes once result files exist.

## Notes

- All scripts run in the foreground.
- Do not use `nohup`, `&`, `tmux`, or background execution unless explicitly
  agreed before the run.
- Use `Ctrl+C` to stop a run if GPU pressure, logs, or output paths look wrong.
- Rerun `scripts/check_experiment_readiness.py` after any model or dataset
  change.
