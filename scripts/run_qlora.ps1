param(
  [string]$Model = "qwen3_4b",
  [string]$Dataset = "domain_qa",
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml"
)

python -m src.rc_llm_eval.cli run-qlora `
  --experiment $Experiment `
  --model $Model `
  --dataset $Dataset
