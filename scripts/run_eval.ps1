param(
  [string]$Model = "qwen3_4b",
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml",
  [string]$Precision = ""
)

# 使用数组拼接参数，避免 PowerShell 中字符串转义带来的歧义。
$cmd = @(
  "-m", "src.rc_llm_eval.cli", "run-eval",
  "--experiment", $Experiment,
  "--model", $Model
)
if ($Precision -ne "") {
  # 仅在用户显式传参时覆盖实验默认精度。
  $cmd += @("--precision", $Precision)
}
python @cmd
