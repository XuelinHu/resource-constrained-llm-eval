param(
  [string]$Model = "qwen3_4b",
  [string]$Dataset = "domain_qa",
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml"
)

# 运行单个模型的 QLoRA 训练任务。
python -m src.rc_llm_eval.cli run-qlora `
  --experiment $Experiment `
  --model $Model `
  --dataset $Dataset
