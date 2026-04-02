param(
  [string]$Model = "qwen3_4b",
  [string]$Experiment = "configs/experiments/single_gpu_3090.yaml",
  [string]$Precision = ""
)

$cmd = @(
  "-m", "src.rc_llm_eval.cli", "run-eval",
  "--experiment", $Experiment,
  "--model", $Model
)
if ($Precision -ne "") {
  $cmd += @("--precision", $Precision)
}
python @cmd
