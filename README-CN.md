# 资源受限大模型评测项目

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

本仓库面向如下论文实验主题：

`在单卡资源约束下，对 8B 以内开源大模型进行评测与基于 QLoRA 的适配`

目标硬件：

- 1 张 RTX 3090 24 GB
- 支持 CUDA 的 Linux，或 Windows + WSL

目标模型池：

- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `Qwen3-4B`
- `Qwen3-8B`
- `Qwen2.5-7B-Instruct`
- `DeepSeek-R1-Distill-Qwen-7B`
- `Gemma-3-4B`

计划覆盖的实验范围：

- 统一 baseline 评测
- 4-bit 部署对比
- 选定模型的 QLoRA 领域适配
- 微调后再评测
- 性能与效率权衡分析

## 仓库结构

```text
configs/         实验、模型、任务配置
scripts/         Bash 与 PowerShell 脚本入口
src/             Python 编排代码
paper/           独立论文 LaTeX 工作区
results/         生成的结果、指标与表格
```

## 文件路径索引

### 运行入口路径

- CLI 主入口：`src/rc_llm_eval/cli.py`
- Python 包入口标记：`src/__init__.py`
- 包版本标记：`src/rc_llm_eval/__init__.py`

### 脚本路径

- 环境初始化：`scripts/setup_conda.sh`
- 模型预下载封装脚本：`scripts/run_prefetch_models.sh`
- 模型预下载实现：`scripts/prefetch_models.py`
- Pilot baseline 运行：`scripts/run_baseline_pilot.sh`
- 全量 baseline 运行：`scripts/run_baseline_all.sh`
- 单次评测 PowerShell 入口：`scripts/run_eval.ps1`
- 单次 QLoRA 训练 PowerShell 入口：`scripts/run_qlora.ps1`
- 批量 QLoRA 训练：`scripts/run_qlora_all.sh`
- 批量适配器评测：`scripts/run_qlora_eval_all.sh`
- 结果汇总：`scripts/summarize_results.ps1`
- 论文表格导出：`scripts/export_paper_tables.sh`
- 论文表格导出 PowerShell 入口：`scripts/export_paper_tables.ps1`
- 正式实验总控：`scripts/run_formal_pipeline.sh`

### 源码路径

- Baseline 流水线：`src/rc_llm_eval/pipelines/baseline.py`
- QLoRA 流水线：`src/rc_llm_eval/pipelines/qlora.py`
- 报表导出流水线：`src/rc_llm_eval/pipelines/reporting.py`
- 配置工具：`src/rc_llm_eval/utils/config.py`
- 数据读写工具：`src/rc_llm_eval/utils/io.py`
- 模型加载工具：`src/rc_llm_eval/utils/modeling.py`
- 系统工具：`src/rc_llm_eval/utils/system.py`
- 文本规范化工具：`src/rc_llm_eval/utils/text.py`

### 配置文件路径

- 主实验配置：`configs/experiments/single_gpu_3090.yaml`
- Pilot 实验配置：`configs/experiments/pilot_single_gpu_3090.yaml`
- 数据集注册表：`configs/datasets/tasks.yaml`
- 模型注册表：`configs/models/models.yaml`

### 数据集路径

- 领域训练集：`data/domain/train.jsonl`
- 领域验证集：`data/domain/valid.jsonl`
- 领域测试集：`data/domain/test.jsonl`
- 效率测试提示词：`data/efficiency/prompts.jsonl`
- 领域数据说明：`data/domain/README.md`

### 论文与项目说明路径

- 论文主文件：`paper/main.tex`
- 论文待办：`paper/todo.md`
- 论文说明：`paper/README.md`
- 参考文献：`paper/bib/references.bib`
- 正式实验检查清单：`FORMAL_EXPERIMENT_CHECKLIST.md`
- 工作计划：`WORKPLAN.md`

## 输出与快照路径

实验输出根目录由配置文件定义：

- 主实验输出根目录：`results/single_gpu_3090`
- Pilot 实验输出根目录：`results/pilot_single_gpu_3090`

`results/single_gpu_3090/baseline/<model_key>/` 下常见输出：

- 运行计划快照：`<model_key>_<precision>_plan.json`
- lm-eval 原始结果快照：`<model_key>_<precision>_lm_eval.json`
- 本地域问答结果快照：`<model_key>_<precision>_domain_qa.json`
- 本地域问答生成快照：`<model_key>_<precision>_domain_generations.json`
- 效率指标快照：`<model_key>_<precision>_efficiency.json`
- 效率生成快照：`<model_key>_<precision>_efficiency_generations.json`
- 汇总结果快照：`<model_key>_<precision>_summary.json`
- 汇总结果 CSV：`<model_key>_<precision>_summary.csv`

`results/single_gpu_3090/qlora_eval/<model_key>/` 下常见输出：

- 适配后运行计划快照：`<model_key>_int4_<label>_plan.json`
- 适配后 lm-eval 快照：`<model_key>_int4_<label>_lm_eval.json`
- 适配后汇总快照：`<model_key>_int4_<label>_summary.json`
- 适配后汇总 CSV：`<model_key>_int4_<label>_summary.csv`

`results/single_gpu_3090/qlora/<model_key>/` 下常见输出：

- 训练运行配置快照：`run_config.json`
- Trainer 检查点目录：`checkpoint/`
- 训练指标：`train_metrics.json`
- 评估指标：`eval_metrics.json`
- 保存的适配器权重：`adapter/`

聚合结果与论文导出路径：

- Baseline 聚合指标：`results/single_gpu_3090/baseline/all_metrics.csv`
- Baseline 聚合效率：`results/single_gpu_3090/baseline/all_efficiency.csv`
- QLoRA 评测聚合指标：`results/single_gpu_3090/qlora_eval/all_metrics.csv`
- QLoRA 评测聚合效率：`results/single_gpu_3090/qlora_eval/all_efficiency.csv`
- 结果表中间文件目录：`results/single_gpu_3090/baseline/tables/`
- 论文主结果表：`paper/tables/generated_main_results.tex`
- 论文效率表：`paper/tables/generated_efficiency_results.tex`
- 论文 QLoRA 对比表：`paper/tables/generated_qlora_results.tex`

## 脚本说明

仓库将单卡实验编排拆成多个职责明确的脚本，便于恢复、重跑和分阶段执行：

- `scripts/setup_conda.sh`：根据 `environment.yml` 创建或更新 Conda 环境。
- `scripts/run_prefetch_models.sh` 与 `scripts/prefetch_models.py`：预先下载 baseline 模型到本地 Hugging Face 缓存。
- `scripts/run_baseline_pilot.sh`：先跑一个轻量 Pilot baseline。
- `scripts/run_baseline_all.sh` 与 `scripts/run_eval.ps1`：启动 baseline 评测。
- `scripts/run_qlora_all.sh` 与 `scripts/run_qlora.ps1`：启动 QLoRA 训练。
- `scripts/run_qlora_eval_all.sh`：对训练后的适配器执行评测。
- `scripts/summarize_results.ps1`：把零散结果聚合成汇总 CSV。
- `scripts/export_paper_tables.sh` 与 `scripts/export_paper_tables.ps1`：导出论文用 LaTeX 表格。
- `scripts/run_formal_pipeline.sh`：串联完整正式实验流程，并带失败日志与显存预算检查。

## 当前优先工作

1. 安装依赖并验证 `lm-evaluation-harness`。
2. 下载并缓存所有基础模型。
3. 在相同任务集上运行 baseline 评测。
4. 记录时延、吞吐与峰值显存。
5. 选择 2 到 3 个模型做 QLoRA。
6. 构建小规模领域基准集。
7. 微调后重新评测并导出论文表格。

## 环境准备

推荐在 Ubuntu + CUDA 环境中执行：

```bash
conda env create -f environment.yml
conda activate rc-llm-eval
```

或者显式创建/更新：

```bash
bash scripts/setup_conda.sh
conda activate rc-llm-eval
```

如果使用已有环境中的 `pip`：

```powershell
pip install -r requirements.txt
```

## 示例命令

查看当前实验计划：

```powershell
python -m src.rc_llm_eval.cli print-plan
```

按配置运行单次 lm-eval：

```bash
python -m src.rc_llm_eval.cli run-eval \
  --experiment configs/experiments/single_gpu_3090.yaml \
  --model qwen3_4b
```

运行完整 baseline：

```bash
bash scripts/run_baseline_all.sh
```

或者使用 `Makefile`：

```bash
make baseline MODEL=qwen3_4b
make baseline-all
make summarize
make export-paper-tables
make qlora MODEL=qwen3_4b DATASET=domain_qa
make paper
```

启动一次 QLoRA 实验：

```bash
python -m src.rc_llm_eval.cli run-qlora \
  --experiment configs/experiments/single_gpu_3090.yaml \
  --model qwen3_4b \
  --dataset domain_qa
```

## 推荐运行顺序

在正式大规模运行前，可以先把 baseline 模型预下载到本地缓存：

```bash
conda activate rc-llm-eval
bash scripts/run_prefetch_models.sh
```

然后先跑一个目标模型的 Pilot：

```bash
conda activate rc-llm-eval
bash scripts/run_baseline_pilot.sh
```

开始完整 baseline 与 QLoRA 之前，先对照 `FORMAL_EXPERIMENT_CHECKLIST.md` 检查一遍。

## 注释规范

- 现在所有 Python 文件和脚本文件都采用 UTF-8 编码的中文注释。
- 注释重点说明模块职责、关键控制流程、资源假设和不直观的实现选择。
- 明显的简单语句不会强行逐行加注释，避免影响后续维护阅读。

## README 版本

- 英文版：`README.md`
- 中文版：`README-CN.md`

## 备注

- 当前代码库重点是实验编排和论文产出流程。
- 基准实际执行仍然依赖本地模型访问、数据集准备以及最终提示词和评测策略。
- 建议将生成输出统一保存在 `results/` 下，方便后续导入 LaTeX 表格。
