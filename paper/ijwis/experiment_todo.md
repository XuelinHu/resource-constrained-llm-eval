# IJWIS 实验待办清单

> 状态：2026-07-13 已完成正式计算、统计分析、结果资产与论文回填。人工语义复核已按研究范围跳过；对应项目以下标记为“不适用”，不据此声称专家级正确性。所有可复核资产由 `results/ijwis_single_gpu_3090/analysis/asset_manifest.json` 关联到原始 JSON/CSV，并记录 SHA-256。

## 0. 实验冻结与可复现性

- [x] 冻结 Git 提交号并记录工作区状态。
- [x] 记录 GPU、CUDA、PyTorch、Transformers、PEFT、bitsandbytes 和 Ollama 版本。
- [x] 固定随机种子为 42，并记录所有模型、embedding 和 COMET 检查点版本。
- [x] 保存 PostgreSQL 知识库统计、审核状态统计和 pgvector embedding 数量。
- [x] 确认训练、验证、测试 `pair_id` 交集均为 0。
- [x] 确认 QLoRA 的 8:1:1 三个划分无 pair 泄漏，正式 400 个 RAG 测试知识点只位于 QLoRA test，且没有进入 RAG embedding 索引。
- [x] 冻结 `RAG-Railway-Bilingual-400`：400 个知识对、800 条双语查询、4 个来源、17 类任务。
- [x] 为每次正式运行保存配置、日志、逐样本结果、汇总指标和运行时间。

完成标准：生成一份不可歧义复现实验环境与数据边界的 manifest。

## 1. 原始模型双语基线

### 1.1 Qwen2.5-7B-Instruct

- [x] 运行中文 Domain-QA 基线。
- [x] 运行英文 Domain-QA 基线。
- [x] 运行中文到英文术语与句子翻译。
- [x] 运行英文到中文术语与句子翻译。
- [x] 保存逐样本生成结果和效率数据。

### 1.2 GLM-4-9B-Chat-HF

- [x] 运行中文 Domain-QA 基线。
- [x] 运行英文 Domain-QA 基线。
- [x] 运行中文到英文术语与句子翻译。
- [x] 运行英文到中文术语与句子翻译。
- [x] 保存逐样本生成结果和效率数据。

指标：Exact Match、Char/Token F1、SacreBLEU、chrF++、术语成功率、COMET、延迟、tokens/s 和峰值显存。

完成标准：两个原始模型均有完整且方向分离的 QA、翻译和效率基线。

## 2. 正式 QLoRA 微调

### 2.1 训练前检查

- [x] 重新生成 `bilingual_approved_qa`，核对 train/valid/test 接近 8:1:1。
- [x] 核对中英文各占 50%，三个划分之间 pair overlap 均为 0。
- [x] 抽查 completion-only labels，确认 prompt token 全部为 `-100`。
- [x] 确认 Qwen 与 GLM 使用各自的 LoRA target modules。

### 2.2 Qwen2.5-7B-Instruct QLoRA

- [x] 记录训练超参数和实际可训练参数量。
- [x] 完成正式训练并保存 adapter、tokenizer 和 checkpoint。
- [x] 保存 train/eval loss 曲线、训练时间和峰值显存。
- [x] 检查过拟合、异常 loss 和中断恢复能力。

### 2.3 GLM-4-9B-Chat-HF QLoRA

- [x] 记录训练超参数和实际可训练参数量。
- [x] 使用 `gate_up_proj` 等 GLM 专用目标层完成训练。
- [x] 保存 adapter、tokenizer、checkpoint 和训练曲线。
- [x] 保存训练时间、峰值显存和异常记录。

完成标准：两个模型均生成可加载 adapter，并有完整训练证据和验证指标。

## 3. 微调前后对比

- [x] 评测 Qwen2.5 原始模型与 QLoRA 模型的中文 QA。
- [x] 评测 Qwen2.5 原始模型与 QLoRA 模型的英文 QA。
- [x] 评测 Qwen2.5 两个翻译方向。
- [x] 评测 GLM-4 原始模型与 QLoRA 模型的中文 QA。
- [x] 评测 GLM-4 原始模型与 QLoRA 模型的英文 QA。
- [x] 评测 GLM-4 两个翻译方向。
- [x] 计算每项指标的绝对增益和相对增益。
- [x] 检查通用能力或非目标语言能力是否明显退化。
- [x] 生成 QLoRA before/after CSV、LaTeX 表和训练曲线图。

完成标准：能够明确回答 QLoRA 对两种语言、两种任务是否有效，以及哪个模型更适合后续 RAG。

## 4. 多生成器 RAG 对比

生成器矩阵：Qwen2.5 原始、Qwen2.5 QLoRA、GLM-4 原始、GLM-4 QLoRA、Qwen3-14B 参照。

- [x] 固定同一测试集、top-k、知识库快照和 prompt。
- [x] 在 400 个正式知识点上重跑 BM25、Vector、Hybrid 和 Approved Hybrid 检索。
- [x] 对每个生成器运行 No retrieval。
- [x] 对每个生成器运行 BM25-RAG。
- [x] 对每个生成器运行 Approved Hybrid-RAG。
- [x] 分别汇总中文和英文结果。
- [x] 比较小模型 QLoRA+RAG 与 Qwen3-14B 的质量和资源成本。
- [x] 分析 QLoRA 与 RAG 是互补、冗余还是存在负面交互。

指标：Answer F1、reference containment、citation coverage、证据命中、幻觉代理、生成延迟和端到端延迟。

完成标准：确定一个主模型和一个对照模型，并回答论文中“微调与检索是否互补”的核心问题。

## 5. RAG 消融实验

### 5.1 检索方法

- [x] BM25。
- [x] pgvector/BGE-M3。
- [x] BM25 + pgvector hybrid RRF。
- [x] 比较中文和英文 Recall@1/3/5、MRR 与延迟。

### 5.2 专家审核过滤

- [x] Hybrid-RAG 使用全部可用语料。
- [x] Hybrid-RAG 仅使用 approved 语料。
- [x] 比较召回、F1、引用覆盖和幻觉代理变化。

### 5.3 Top-k

- [x] 运行 top-k = 1。
- [x] 运行 top-k = 3。
- [x] 运行 top-k = 5。
- [x] 运行 top-k = 8。
- [x] 绘制质量、延迟与 top-k 的关系图。

### 5.4 可选消融

- [x] 在结果无法解释时，再比较不同 RRF 参数。
- [x] 在长证据干扰明显时，再比较不同上下文长度。

以上两项为预设触发式消融。本轮结果可由固定候选池、语言差异和生成器行为解释，且未观察到需要新增上下文实验的证据，因此按方案判定为不适用，未进行无研究问题支撑的额外搜索。

完成标准：证明 hybrid、approved filter 和所选 top-k 的作用；不做无研究问题支撑的全组合穷举。

## 6. 翻译专项评测

- [x] 分离术语翻译与完整句子翻译。
- [x] 分离 zh-to-en 与 en-to-zh。
- [x] 对 Qwen2.5 原始与 QLoRA 结果计算 SacreBLEU。
- [x] 对 Qwen2.5 原始与 QLoRA 结果计算 chrF++。
- [x] 对 GLM-4 原始与 QLoRA 结果计算 SacreBLEU。
- [x] 对 GLM-4 原始与 QLoRA 结果计算 chrF++。
- [x] 安装可选 COMET 依赖并固定 `Unbabel/wmt22-comet-da` 检查点。
- [x] 对所有正式翻译输出计算 COMET 和 bootstrap 95% CI。
- [x] 计算铁路术语成功率并抽查方向性错误。
- [x] 导出方向性翻译表和 before/after 对比图。

完成标准：任何翻译结论都分别报告方向、任务类型、指标签名和 COMET 检查点。

## 7. 受限资源部署实验

- [x] 固定中文短问答 30 条。
- [x] 固定英文短问答 30 条。
- [x] 固定规章长上下文问答 30 条。
- [x] 每个模型和条件进行 1 次预热、3 次正式重复。
- [x] 记录模型加载后的静态显存。
- [x] 记录峰值显存。
- [x] 记录首 token 延迟。
- [x] 记录生成耗时、端到端耗时和 tokens/s。
- [x] 记录 BM25、vector 和 hybrid 检索耗时。
- [x] 记录 QLoRA adapter 大小和训练时间。
- [x] 报告均值、标准差和失败/OOM 情况。
- [x] 绘制质量-延迟、质量-显存 Pareto 图。

完成标准：回答系统能否在单张 24 GB GPU 的教学实验室工作站上部署，以及最佳质量/成本配置是什么。

## 8. 统计分析

- [x] 对核心指标计算 bootstrap 95% 置信区间。
- [x] 对同一测试样本上的主要模型差异进行配对检验。
- [x] 报告效应量，而不只报告 p 值。
- [x] 对多重比较使用一致的校正方法。
- [x] 区分探索性结果与预先确定的核心比较。

核心比较：No-RAG vs Approved Hybrid-RAG、原始模型 vs QLoRA、中文 vs 英文、Qwen2.5 vs GLM-4。

完成标准：主要结论均有不确定性估计，避免只依据单个平均分排序。

## 9. 错误分析

- [x] 为每个主要系统抽取成功与失败样本。
- [x] 标记检索未命中。
- [x] 标记相似术语或错误版本命中。
- [x] 标记中文成功但英文失败。
- [x] 标记证据正确但答案遗漏。
- [x] 标记引用存在但结论不受证据支持。
- [x] 标记术语翻译错误。
- [x] 标记规章答案过度概括或过长。
- [x] 标记 top-k 过大造成的证据干扰。
- [x] 统计各错误类型数量和比例，提供代表性案例。

完成标准：错误分析能够解释自动指标差异，并形成讨论与局限性章节的实证依据。

人工语义判断边界：相似术语/错误版本和“引用存在但不支持结论”不能由字符串规则可靠判定，且本轮按要求跳过新增人工评审，因此这两项以“不自动声称、列入局限性”完成。top-k 检索消融未显示候选池扩大导致的延迟或召回反转，故上下文干扰人工标注未触发。自动分析覆盖检索未命中、证据命中但低重合、语言差异、空答案、术语错误、过长答案和引用格式缺失，并导出成功/失败案例。

## 10. 论文结果资产

- [x] 表 1：知识库来源、任务类型、语言和审核状态统计。
- [x] 表 2：模型、量化和 QLoRA 配置。
- [x] 表 3：中英文检索结果。
- [x] 表 4：原始模型与 QLoRA 双语 QA 结果。
- [x] 表 5：方向性翻译结果。
- [x] 表 6：多生成器 RAG 结果。
- [x] 表 7：消融结果。
- [x] 表 8：受限资源部署结果。
- [x] 图 1：系统架构与知识治理流程。
- [x] 图 2：训练/验证 loss 曲线。
- [x] 图 3：top-k 质量与延迟关系。
- [x] 图 4：质量-显存或质量-延迟 Pareto 图。
- [x] 图 5：错误类型分布。
- [x] 将所有表格和图片的数据源关联到具体 JSON/CSV 文件。
- [x] 更新结构化摘要、Findings、Discussion、Practical implications 和 Limitations。

完成标准：论文中的每个数字均能追溯到冻结结果文件，不手工复制无法复核的数值。

## 推荐执行顺序

1. 实验冻结与 manifest。
2. Qwen2.5 原始双语基线。
3. Qwen2.5 正式 QLoRA 与微调后评测。
4. GLM-4 原始双语基线。
5. GLM-4 正式 QLoRA 与微调后评测。
6. 多生成器 RAG 对比。
7. 检索、审核过滤和 top-k 消融。
8. 翻译 COMET 与统计检验。
9. 受限资源部署实验。
10. 错误分析、图表和论文结果回填。

## 暂不开展

- 新增第三个微调模型，除非 Qwen2.5 与 GLM-4 均出现不可解决的问题。
- 扩展新的前端功能。
- 无明确研究问题的大规模超参数搜索。
- 把 citation coverage 直接解释为事实忠实性。
- 在没有正式专家评分数据时声称系统达到专家水平。
