param(
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml"
)

# 聚合 baseline 结果文件，生成总表 CSV。
python -m src.rc_llm_eval.cli summarize-results --experiment $Experiment
