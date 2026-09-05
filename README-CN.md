# 双语铁路 RAG 与受限资源大模型评测项目

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

本仓库面向 IJWIS 论文及其可复现实验：

`资源约束下双语铁路职业教育的知识增强型大语言模型`

目标硬件：

- 1 张 RTX 3090 24 GB
- 支持 CUDA 的 Linux，或 Windows + WSL

最终论文使用的模型矩阵：

- `Qwen2.5-7B-Instruct`
- `GLM-4-9B-Chat-HF`
- `Qwen3-14B`（通过 Ollama 作为未适配参照生成器）
- `BAAI/bge-m3`（与 PostgreSQL + pgvector 配合完成双语检索）

计划覆盖的实验范围：

- 统一 baseline 评测
- 4-bit 部署对比
- 选定模型的 QLoRA 领域适配
- 微调后再评测
- 性能与效率权衡分析、自动化系统验证

## 仓库结构

```text
configs/         实验、模型、任务配置
scripts/         Bash 与 PowerShell 脚本入口
src/             Python 编排代码
paper/ijwis/     IJWIS 权威论文、证据与投稿包
IJWIS__Copy_/    IJWIS 匿名 LaTeX 模板封装
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

- 主实验配置：`configs/experiments/ijwis_single_gpu_3090.yaml`
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

- 权威论文正文：`paper/ijwis/manuscript.md`
- IJWIS 项目说明：`paper/ijwis/README.md`
- 匿名投稿包：`paper/ijwis/submission/`
- LaTeX 模板封装：`IJWIS__Copy_/Main.tex`
- 论文证据表和图：`paper/ijwis/{tables,figures}/`
- 最终英文全量稿：`output/pdf/ijwis_manuscript_full.pdf`
- 最终英文匿名稿：`output/pdf/ijwis_manuscript_anonymous.pdf`
- 最终中文核对稿：`output/pdf/ijwis_manuscript_zh.pdf`
- Hugging Face 私有派生结果包：`paper/ijwis/huggingface_release/`
- 正式实验清单：`paper/ijwis/experiment_todo.md`

## 输出与快照路径

实验输出根目录由配置文件定义：

- 主实验输出根目录：`results/ijwis_single_gpu_3090`
- Pilot 实验输出根目录：`results/pilot_single_gpu_3090`

`results/ijwis_single_gpu_3090/baseline/<model_key>/` 下常见输出：

- 运行计划快照：`<model_key>_<precision>_plan.json`
- lm-eval 原始结果快照：`<model_key>_<precision>_lm_eval.json`
- 本地域问答结果快照：`<model_key>_<precision>_domain_qa.json`
- 本地域问答生成快照：`<model_key>_<precision>_domain_generations.json`
- 效率指标快照：`<model_key>_<precision>_efficiency.json`
- 效率生成快照：`<model_key>_<precision>_efficiency_generations.json`
- 汇总结果快照：`<model_key>_<precision>_summary.json`
- 汇总结果 CSV：`<model_key>_<precision>_summary.csv`

`results/ijwis_single_gpu_3090/qlora_eval/<model_key>/` 下常见输出：

- 适配后运行计划快照：`<model_key>_int4_<label>_plan.json`
- 适配后 lm-eval 快照：`<model_key>_int4_<label>_lm_eval.json`
- 适配后汇总快照：`<model_key>_int4_<label>_summary.json`
- 适配后汇总 CSV：`<model_key>_int4_<label>_summary.csv`

`results/ijwis_single_gpu_3090/qlora/<model_key>/` 下常见输出：

- 训练运行配置快照：`run_config.json`
- Trainer 检查点目录：`checkpoint/`
- 训练指标：`train_metrics.json`
- 评估指标：`eval_metrics.json`
- 保存的适配器权重：`adapter/`

聚合结果与论文导出路径：

- Baseline 聚合指标：`results/ijwis_single_gpu_3090/baseline/all_metrics.csv`
- Baseline 聚合效率：`results/ijwis_single_gpu_3090/baseline/all_efficiency.csv`
- QLoRA 评测聚合指标：`results/ijwis_single_gpu_3090/qlora_eval/all_metrics.csv`
- QLoRA 评测聚合效率：`results/ijwis_single_gpu_3090/qlora_eval/all_efficiency.csv`
- 结果表中间文件目录：`results/ijwis_single_gpu_3090/baseline/tables/`
- IJWIS 论文表格：`paper/ijwis/tables/*.csv`
- IJWIS 论文图片：`paper/ijwis/figures/*.pdf`

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

## 当前定稿状态

正式实验矩阵、论文正文、图表和匿名投稿文件均已生成。剩余工作主要是投稿系统中的作者信息、期刊字段和按要求上传补充材料。

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
  --experiment configs/experiments/ijwis_single_gpu_3090.yaml \
  --model qwen2_5_7b_instruct
```

运行完整 baseline：

```bash
bash scripts/run_baseline_all.sh
```

或者使用 `Makefile`：

```bash
make baseline MODEL=qwen2_5_7b_instruct
make baseline-all
make summarize
make export-paper-tables
make qlora MODEL=qwen2_5_7b_instruct DATASET=bilingual_approved_qa
make paper
```

启动一次 QLoRA 实验：

```bash
python -m src.rc_llm_eval.cli run-qlora \
  --experiment configs/experiments/ijwis_single_gpu_3090.yaml \
  --model qwen2_5_7b_instruct \
  --dataset bilingual_approved_qa
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

正式实验记录和完成状态统一以 `paper/ijwis/experiment_todo.md` 为准。

## 注释规范

- 现在所有 Python 文件和脚本文件都采用 UTF-8 编码的中文注释。
- 注释重点说明模块职责、关键控制流程、资源假设和不直观的实现选择。
- 明显的简单语句不会强行逐行加注释，避免影响后续维护阅读。

## README 版本

- 英文版：`README.md`
- 中文版：`README-CN.md`

## 备注

- 大型本地数据集、模型缓存、生成结果和比赛材料有意不进入 Git 提交面。
- Hugging Face 私有仓库只包含派生指标和元数据，不包含受版权保护的原始语料和模型权重。
- 建议将生成输出统一保存在 `results/` 下，方便后续导入 LaTeX 表格。
