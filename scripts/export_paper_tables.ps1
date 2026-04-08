param(
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml"
)

# 先聚合结果，再生成论文表格，避免直接从零散文件导出。
python -m src.rc_llm_eval.cli summarize-results --experiment $Experiment
python -m src.rc_llm_eval.cli export-paper-tables --experiment $Experiment
