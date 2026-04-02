param(
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml"
)

python -m src.rc_llm_eval.cli summarize-results --experiment $Experiment
python -m src.rc_llm_eval.cli export-paper-tables --experiment $Experiment
