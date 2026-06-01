# 论文工作总方案：单卡资源约束下铁路领域本地大模型评测与适配

## 1. 论文定位

本论文建议定位为：

> 单卡 RTX 3090 资源约束下，面向铁路领域问答与翻译任务的本地开源大模型评测、量化推理与低成本 QLoRA 适配研究。

不建议把论文写成“提出一个新的铁路大模型”。当前更稳妥、更符合已有条件的主线是：

1. 在真实单卡 24GB VRAM 条件下，系统比较多个本地开源模型的可部署性。
2. 分析 bf16、int8、int4 量化对通用能力、领域能力和效率的影响。
3. 构建并使用铁路领域数据集，验证通用 benchmark 高分模型是否一定适合铁路领域任务。
4. 使用 QLoRA 进行低成本领域适配，比较不同训练数据组合对领域问答/翻译任务的影响。

## 2. 当前已有条件

### 2.1 硬件与运行环境

- GPU：单张 NVIDIA RTX 3090
- 显存：24GB
- 实验约束：单卡、本地部署、资源受限
- 主要框架：
  - PyTorch
  - Hugging Face Transformers
  - PEFT
  - BitsAndBytes
  - Datasets
  - lm-evaluation-harness

该硬件约束是论文的问题背景，不是实验缺陷。论文应强调这是高校实验室、企业内网、边缘工作站等场景中常见的实际部署条件。

### 2.2 当前本机可用模型池

当前本机可用模型如下：

| 模型 | 缓存位置 | 状态 | 建议用途 |
|---|---|---|---|
| `qwen3_4b` | home | 可用 | 小模型效率基线 |
| `qwen3_8b` | home | 可用 | 中等规模强基线，已完成 QLoRA 训练 |
| `qwen2_5_7b_instruct` | home | 可用 | 主力中文/多语指令模型，已完成 QLoRA 训练 |
| `deepseek_r1_distill_qwen_7b` | home | 可用 | 推理型蒸馏模型对照 |
| `mistral_7b_instruct_v0_3` | home | 可用 | 英文/多语指令模型对照 |
| `phi_3_mini_4k_instruct` | home | 可用 | 小参数高效模型对照 |
| `yi_6b_chat` | home | 可用 | 中文模型对照 |
| `gemma_3_4b` | `/ds1` | 可用 | 小模型对照，需纳入正式配置 |
| `gemma_2_9b_it` | `/ds1` | 可用 | 9B 级模型对照，需纳入正式配置 |
| `glm_4_9b_chat_hf` | `/ds1` | 可用 | 中文强模型对照，需纳入正式配置 |

当前 `configs/experiments/single_gpu_3090.yaml` 中正式 baseline 已包含 7 个模型：

- `qwen3_4b`
- `qwen3_8b`
- `qwen2_5_7b_instruct`
- `deepseek_r1_distill_qwen_7b`
- `phi_3_mini_4k_instruct`
- `yi_6b_chat`
- `mistral_7b_instruct_v0_3`

待补入正式实验配置的模型：

- `gemma_3_4b`
- `gemma_2_9b_it`
- `glm_4_9b_chat_hf`

### 2.3 当前数据集

#### 2.3.1 `domain_qa`

路径：`data/domain/`

用途：

- 铁路术语中英互译
- 铁路规章段落中英翻译

规模：

| split | 数量 |
|---|---:|
| train | 22,164 |
| valid | 2,770 |
| test | 2,770 |

类别分布：

| 类别 | 说明 |
|---|---|
| `terminology_zh_to_en` | 中文铁路术语到英文术语 |
| `terminology_en_to_zh` | 英文铁路术语到中文术语 |
| `zh_to_en_translation` | 中文规章段落到英文 |
| `en_to_zh_translation` | 英文规章段落到中文 |

该数据集适合作为“铁路领域术语与翻译适配”数据。

#### 2.3.2 `domain_regqa`

路径：`data/domain_regqa/`

用途：

- 铁路规章制度中文问答
- 基于规章条款进行答案抽取
- 支撑法规问答能力评估和任务定向微调

规模：

| split | 数量 |
|---|---:|
| train | 1,600 |
| valid | 200 |
| test | 200 |

特点：

- 总量 2,000 条
- 规则抽取生成，不是 LLM 生成
- 每条样本带 `evidence`
- 答案可在 evidence 中定位
- 适合后续人工核对

典型类别：

| 类别 | 说明 |
|---|---|
| `regulation_clause_qa` | 条款原文问答 |
| `regulation_requirement_qa` | 要求类问答 |
| `regulation_prohibition_qa` | 禁止/不得/严禁类问答 |
| `regulation_inspection_qa` | 检查、检测、维护类问答 |
| `regulation_standard_qa` | 标准类问答 |
| `regulation_responsibility_qa` | 职责类问答 |
| `regulation_definition_qa` | 定义类问答 |

当前状态：

- 数据文件已经存在。
- 还没有加入 `configs/datasets/tasks.yaml`。
- 后续应新增 `domain_regqa` 任务配置。

建议配置：

```yaml
domain_regqa:
  suite: local_jsonl
  train_file: data/domain_regqa/train.jsonl
  valid_file: data/domain_regqa/valid.jsonl
  test_file: data/domain_regqa/test.jsonl
  metric: exact_match
  prompt_field: prompt
  answer_field: answer
  text_field: text
```

#### 2.3.3 公开 benchmark

当前已配置的公开任务：

| 任务 | 来源 | 主要衡量能力 |
|---|---|---|
| `mmlu` | lm-eval | 多学科通用知识 |
| `gsm8k` | lm-eval | 数学文字题推理 |
| `humaneval` | lm-eval | Python 代码生成 |
| `ceval` | lm-eval | 中文综合知识与推理 |

这些 benchmark 是论文中的通用能力基线，不能替代铁路领域任务。

#### 2.3.4 效率测试数据

路径：`data/efficiency/prompts.jsonl`

用途：

- 延迟测试
- token throughput 测试
- 显存峰值测试

规模：5 条 prompt。

## 3. 已完成工作状态

### 3.1 已完成的 QLoRA 训练

已完成正式训练的 adapter：

| 模型 | 训练数据 | 状态 | 说明 |
|---|---|---|---|
| `qwen2_5_7b_instruct` | `domain_qa` | 完成 | 可作为 Adapter-A |
| `qwen3_8b` | `domain_qa` | 完成 | 可作为 Adapter-A 对照 |

`qwen2_5_7b_instruct` 训练结果：

| 指标 | 数值 |
|---|---:|
| train runtime | 17,987.45 s |
| train loss | 0.4641 |
| eval loss | 0.4938 |
| eval accuracy | 0.8938 |
| perplexity | 1.6385 |

`qwen3_8b` 训练结果：

| 指标 | 数值 |
|---|---:|
| train runtime | 24,052.40 s |
| train loss | 0.4878 |
| eval loss | 0.5071 |
| eval accuracy | 0.8890 |
| perplexity | 1.6605 |

注意：

- 这两个 adapter 只使用了 `domain_qa`。
- 它们没有使用新生成的 `domain_regqa`。
- 因此不能称为法规问答微调模型。
- 它们仍然可以作为“仅基于术语/翻译数据训练后的跨任务迁移模型”。

### 3.2 已完成的正式 adapter 评测

`qwen2_5_7b_instruct` 的 `int4 + adapter` 正式评测已完成。

关键结果：

| 指标 | 数值 |
|---|---:|
| MMLU | 0.7112 |
| C-Eval | 0.7853 |
| HumanEval | 0.5915 |
| GSM8K | 0.0000 |
| Domain QA exact match | 0.0000 |
| Domain QA char F1 | 0.1634 |
| Domain QA token F1 | 0.1028 |
| Domain QA reference contained | 0.4202 |
| Domain QA length ratio | 7.60 |
| 平均延迟 | 14.09 s |
| 平均生成速度 | 18.17 tokens/s |
| 峰值显存 allocated | 12.22 GB |

解释：

- 通用 benchmark 保持较好。
- 领域 QA 的 exact match 很差。
- `reference_contained=0.4202` 说明有部分输出包含正确答案片段。
- `length_ratio=7.60` 说明输出明显过长。
- 当前训练方式在答案格式控制上不足。

### 3.3 当前已清理内容

之前的冒烟测试数据和脚本已从活跃路径移除，并归档到：

`archive/smoke_artifacts_20260601.txt`

论文后续不再使用 smoke 数据作为实验依据。

## 4. 核心研究问题

建议围绕以下研究问题组织全文：

### RQ1：哪些开源大模型能在单张 RTX 3090 上稳定部署？

关注：

- 是否能成功加载
- 是否能完成推理
- 显存峰值
- 延迟
- tokens/s

对应实验：

- 10 个本地可用模型的加载与效率测试
- bf16、int8、int4 不同精度对比

### RQ2：量化是否显著损害模型的通用能力和领域能力？

关注：

- bf16 vs int8 vs int4
- 通用 benchmark 分数变化
- 领域任务分数变化
- 显存节省与速度变化

对应实验：

- 公开 benchmark 量化对比
- `domain_qa` 和 `domain_regqa` 量化对比

### RQ3：通用 benchmark 表现强的模型，是否也适合铁路领域任务？

关注：

- MMLU / C-Eval 高分是否等价于领域 QA 高分
- 中文能力、推理能力、领域术语能力之间是否一致
- 英文/多语模型在铁路术语翻译上的表现

对应实验：

- 所有模型的公开 benchmark
- 所有模型的 `domain_qa`
- 所有模型的 `domain_regqa`
- 排名相关性分析

### RQ4：QLoRA 是否能在 24GB 显存内有效提升铁路领域能力？

关注：

- QLoRA 是否提升 `domain_qa`
- QLoRA 是否提升 `domain_regqa`
- 训练成本是否可接受
- 通用 benchmark 是否退化

对应实验：

- base vs adapter
- `domain_qa` adapter vs `domain_regqa` adapter vs mixed adapter
- 通用 benchmark 保持率

### RQ5：法规问答数据应单独训练，还是与原领域数据混合训练？

关注：

- 只训 `domain_qa` 是否能迁移到 `domain_regqa`
- 只训 `domain_regqa` 是否会损害术语/翻译能力
- mixed 训练是否能取得更好的综合表现

对应实验：

- Adapter-A：`domain_qa`
- Adapter-B：`domain_regqa`
- Adapter-C：`domain_qa + domain_regqa`
- 可选 Adapter-D：从 Adapter-A 继续补训 `domain_regqa`

## 5. 实验总体设计

### 5.1 实验一：模型可部署性与效率评测

对象：

- 10 个本机可用模型

推荐先跑：

- int4

如资源允许，再补：

- bf16
- int8

指标：

| 指标 | 说明 |
|---|---|
| load success | 是否能加载 |
| peak allocated VRAM | 峰值实际分配显存 |
| peak reserved VRAM | 峰值预留显存 |
| mean latency | 平均延迟 |
| median latency | 中位延迟 |
| tokens/s | 生成吞吐 |
| failure reason | 失败原因 |

论文作用：

- 支撑“资源受限部署”的实验背景。
- 说明哪些模型实际能在 3090 上运行。

### 5.2 实验二：公开 benchmark 能力评测

对象：

- 10 个模型

任务：

- MMLU
- C-Eval
- GSM8K
- HumanEval

建议：

- 主表使用 int4 结果，因为最贴近资源受限部署。
- 附表展示 bf16/int8/int4 对比。

论文作用：

- 给出通用能力基线。
- 解释不同模型族的能力差异。

### 5.3 实验三：铁路领域零样本/原始模型评测

对象：

- 10 个模型

任务：

- `domain_qa`
- `domain_regqa`

指标：

| 指标 | 说明 |
|---|---|
| exact match | 完全匹配 |
| char F1 | 中文/混合文本字符级相似度 |
| token F1 | token 级相似度 |
| reference contained | 输出是否包含标准答案 |
| length ratio | 输出长度与参考答案长度比例 |

论文作用：

- 证明通用能力强不等于领域能力强。
- 找出最适合铁路任务的 base model。

### 5.4 实验四：量化影响实验

推荐选择 5 个代表模型：

| 模型 | 代表性 |
|---|---|
| `qwen3_4b` | 小模型 |
| `qwen2_5_7b_instruct` | 主力中文/多语指令模型 |
| `qwen3_8b` | 中等规模强模型 |
| `glm_4_9b_chat_hf` | 中文强模型 |
| `gemma_2_9b_it` 或 `mistral_7b_instruct_v0_3` | 9B 或英文/多语对照 |

精度：

- bf16
- int8
- int4

指标：

- benchmark 分数变化
- 领域任务分数变化
- 显存变化
- 速度变化

论文作用：

- 量化带来的收益和代价。
- 给出 3090 上的推荐部署精度。

### 5.5 实验五：QLoRA 适配实验

这是论文的核心实验之一。

#### Adapter-A：已有旧 adapter

| 字段 | 内容 |
|---|---|
| 初始化 | base model |
| 训练数据 | `domain_qa` |
| 当前状态 | `qwen2_5_7b_instruct`、`qwen3_8b` 已完成 |
| 作用 | 术语/翻译适配；测试对 `domain_regqa` 的跨任务迁移 |

#### Adapter-B：法规问答单独训练

| 字段 | 内容 |
|---|---|
| 初始化 | base model |
| 训练数据 | `domain_regqa` |
| 作用 | 测试法规问答数据本身的有效性 |

#### Adapter-C：混合训练，推荐作为主模型

| 字段 | 内容 |
|---|---|
| 初始化 | base model |
| 训练数据 | `domain_qa + domain_regqa` |
| 作用 | 同时覆盖术语、翻译、法规问答 |
| 推荐性 | 论文主线最推荐 |

混合策略不建议简单拼接，因为 `domain_qa` 有 22,164 条训练样本，`domain_regqa` 只有 1,600 条，会导致法规问答被淹没。

推荐采样比例：

```text
domain_qa : domain_regqa = 3 : 1
```

或：

```text
domain_qa 采样 8,000 到 12,000 条
domain_regqa 1,600 条重复 3 到 5 次
```

#### Adapter-D：继续补训，可选

| 字段 | 内容 |
|---|---|
| 初始化 | Adapter-A |
| 训练数据 | `domain_regqa` |
| 作用 | 检验继续训练是否有效，以及是否产生遗忘 |
| 推荐性 | 可选消融，不作为主结果 |

不建议把 Adapter-D 作为主模型，因为顺序微调容易出现遗忘，论文解释也不如 mixed 训练清晰。

### 5.6 实验六：loss 形式对比

当前训练方式：

- full-sequence causal LM loss
- prompt 和 answer 都参与 loss

当前代码逻辑：

```text
labels = input_ids.copy()
```

问题：

- 模型同时学习问题和答案。
- 对 QA 任务不够理想。
- 已观察到输出偏长问题。

建议新增：

```text
completion-only loss
```

即：

```text
Question / instruction 部分 label = -100
Answer 部分正常计算 cross entropy
```

对比：

| 方法 | 说明 |
|---|---|
| Full-sequence SFT | 当前已有训练方式 |
| Completion-only SFT | 只对答案计算 loss，推荐后续使用 |

论文价值：

- 这是方法层面的改进。
- 可解释 `qwen2_5_7b_instruct` 当前输出过长的问题。
- 有望改善 exact match 和 length ratio。

## 6. 推荐实验优先级

### 第一优先级：必须完成

1. 把 `domain_regqa` 加入任务配置。
2. 把 `gemma_3_4b`、`gemma_2_9b_it`、`glm_4_9b_chat_hf` 加入正式实验配置。
3. 对 10 个模型跑 int4 效率评测。
4. 对 10 个模型跑公开 benchmark。
5. 对 10 个模型跑 `domain_qa` 和 `domain_regqa`。
6. 保留已有 Adapter-A 结果。
7. 为至少一个主模型训练 Adapter-C mixed。

### 第二优先级：强烈建议完成

1. 在 `qwen2_5_7b_instruct` 上做 completion-only SFT。
2. 对 `qwen2_5_7b_instruct` 做 Adapter-B 和 Adapter-C。
3. 对 `qwen3_8b` 做 Adapter-C。
4. 对 `glm_4_9b_chat_hf` 或 `gemma_2_9b_it` 做 Adapter-C。
5. 补完整 bf16/int8/int4 量化对比。

### 第三优先级：可选增强

1. Adapter-D：在已有 Adapter-A 上继续补训 `domain_regqa`。
2. 分析 `domain_qa` 与 `domain_regqa` 的交叉泛化。
3. 对输出错误进行人工分类。
4. 对 `domain_regqa` 随机抽样人工核对并报告质量。

## 7. 最小可发表实验组合

如果时间有限，最低限度建议完成以下实验：

1. 10 个模型 int4 推理效率评测。
2. 10 个模型公开 benchmark。
3. 10 个模型 `domain_qa` 与 `domain_regqa` 测试。
4. `qwen2_5_7b_instruct`：
   - base
   - Adapter-A：`domain_qa`
   - Adapter-C：mixed
5. `qwen3_8b`：
   - base
   - Adapter-A：`domain_qa`
   - Adapter-C：mixed
6. 一个新增强模型：
   - `glm_4_9b_chat_hf` 或 `gemma_2_9b_it`
   - base
   - Adapter-C：mixed

这样至少能形成：

- 多模型对比
- 量化与效率分析
- 领域任务分析
- 微调前后分析
- 数据组合消融

## 8. 论文实验表格设计

### Table 1：模型池信息

列：

- Model
- Family
- Parameters
- Context length
- Instruction tuned
- Thinking/reasoning style
- Cache location
- Status

### Table 2：数据集统计

列：

- Dataset
- Task type
- Train
- Validation
- Test
- Generation method
- Evidence available
- Main categories

### Table 3：公开 benchmark 主结果

列：

- Model
- Precision
- MMLU
- C-Eval
- GSM8K
- HumanEval
- Average

### Table 4：领域任务主结果

列：

- Model
- Precision
- Domain QA EM
- Domain QA char F1
- Domain QA token F1
- RegQA EM
- RegQA char F1
- RegQA token F1
- Length ratio

### Table 5：效率结果

列：

- Model
- Precision
- Peak allocated VRAM
- Peak reserved VRAM
- Mean latency
- Median latency
- Tokens/s

### Table 6：QLoRA 训练结果

列：

- Model
- Adapter type
- Training data
- Runtime
- Train loss
- Eval loss
- Eval accuracy
- Perplexity

### Table 7：微调前后对比

列：

- Model
- Version
- Training data
- Domain QA
- RegQA
- MMLU
- C-Eval
- Efficiency

### Table 8：数据组合消融

列：

- Model
- Base
- `domain_qa`
- `domain_regqa`
- Mixed
- Continue training

## 9. 图设计

建议至少做 5 张图：

### Figure 1：实验流程图

内容：

```text
Model pool -> Quantized inference -> Public benchmark
           -> Domain benchmark -> QLoRA adaptation -> Adapter evaluation
```

### Figure 2：准确率-显存 trade-off

横轴：

- Peak VRAM

纵轴：

- Average score 或 domain score

点：

- 不同模型/精度

### Figure 3：速度-质量 trade-off

横轴：

- tokens/s

纵轴：

- domain score

### Figure 4：QLoRA 前后提升

柱状图：

- base
- Adapter-A
- Adapter-B
- Adapter-C

### Figure 5：数据集消融结果

展示：

- `domain_qa`
- `domain_regqa`
- mixed

在两个测试集上的表现差异。

## 10. 论文结构建议

### 10.1 Abstract

应包含：

- 单卡资源受限背景
- 铁路领域任务
- 多模型评测
- 量化推理
- QLoRA 适配
- 主要发现

不应过度声称：

- 不要说提出了新的大模型。
- 不要说完全解决铁路问答。
- 不要夸大 `domain_regqa` 的人工质量，除非完成核对。

### 10.2 Introduction

核心逻辑：

1. 大模型在专业领域有潜力。
2. 铁路领域涉及大量术语、规章、双语材料。
3. 实际使用常受限于单卡资源和本地部署要求。
4. 现有研究更多关注云端大模型或通用 benchmark，缺少单卡条件下的系统评测。
5. 本文研究在 24GB RTX 3090 上如何选择、量化和适配本地 LLM。

### 10.3 Related Work

建议分四类：

1. Open-source LLM evaluation
2. Quantized LLM inference
3. Parameter-efficient fine-tuning and QLoRA
4. Domain adaptation for technical QA and railway/engineering NLP

### 10.4 Problem Setting

应说明：

- 目标用户
- 任务类型
- 单卡限制
- 评价目标

任务类型：

- 术语翻译
- 规章段落翻译
- 规章制度问答

评价目标：

- 准确率
- 显存
- 延迟
- 吞吐
- 适配成本

### 10.5 Datasets

应详细写：

- `domain_qa` 数据来源
- `domain_regqa` 生成方式
- 数据划分
- 字段设计
- 质量控制
- 局限性

`domain_regqa` 应强调：

- 规则抽取
- evidence grounded
- 非 LLM 生成
- 后续人工核对

### 10.6 Methodology

应写：

- 模型池
- 精度设置
- 评测流程
- QLoRA 设置
- loss 设置
- adapter 设计

QLoRA 当前正式配置：

| 参数 | 值 |
|---|---:|
| LoRA rank | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| learning rate | 2e-4 |
| epochs | 3 |
| per-device batch size | 1 |
| gradient accumulation | 16 |
| max sequence length | 2048 |

### 10.7 Experiments

按 5 组实验写：

1. Public benchmark evaluation
2. Domain benchmark evaluation
3. Quantization and efficiency evaluation
4. QLoRA adaptation evaluation
5. Dataset composition ablation

### 10.8 Results

建议按研究问题回答：

- RQ1：哪些模型可部署？
- RQ2：量化损失多少？
- RQ3：通用能力和领域能力是否一致？
- RQ4：QLoRA 是否有效？
- RQ5：mixed 是否优于单一数据训练？

### 10.9 Discussion

建议讨论：

- 小模型是否更适合资源受限环境
- reasoning 模型是否值得额外开销
- int4 是否足够
- QLoRA 的收益与局限
- 输出过长和 exact match 低的问题
- completion-only loss 的必要性
- `domain_regqa` 的数据质量和人工核对需求

### 10.10 Conclusion

应给出：

- 最推荐模型/精度组合
- 领域适配建议
- 数据集贡献
- 后续工作

## 11. 建议论文题目

候选题目：

1. Resource-Constrained Evaluation and Adaptation of Local Large Language Models for Railway-Domain Question Answering
2. Quantization and QLoRA Adaptation of Open-Source LLMs for Railway-Domain Tasks on a Single RTX 3090
3. Evaluating Local Large Language Models for Railway-Domain Bilingual QA under 24GB GPU Constraints

中文内部题目：

1. 单卡资源约束下铁路领域本地大模型评测与低成本适配研究
2. 面向铁路领域问答的开源大模型量化推理与 QLoRA 适配研究
3. 基于 RTX 3090 的铁路领域本地大模型部署、评测与微调研究

## 12. 关键写作边界

可以写：

- 本文系统评估多个本地开源 LLM 在单卡 RTX 3090 上的表现。
- 本文构建铁路领域术语/翻译数据和规章问答数据。
- 本文比较量化推理和 QLoRA 适配的效果。
- 本文分析通用 benchmark 与领域任务表现之间的不一致。

不建议写：

- 本文提出了全新的铁路大模型。
- 本文完全解决铁路规章问答。
- 已有 Adapter-A 是法规问答微调模型。
- `domain_regqa` 是人工完全标注数据集。
- exact match 很低时仍声称问答效果很好。

## 13. 下一步具体执行清单

### 13.1 配置层

- [ ] 将 `domain_regqa` 加入 `configs/datasets/tasks.yaml`。
- [ ] 将 `domain_regqa` 加入正式实验任务列表。
- [ ] 将 `gemma_3_4b`、`gemma_2_9b_it`、`glm_4_9b_chat_hf` 加入 `configs/experiments/single_gpu_3090.yaml`。
- [ ] 增加 mixed 训练数据构建脚本。
- [ ] 增加 completion-only loss 训练路径。

### 13.2 数据层

- [ ] 对 `domain_regqa` 抽样人工核对。
- [ ] 记录人工核对通过率。
- [ ] 构建 mixed 数据：
  - `domain_qa`
  - `domain_regqa`
  - controlled ratio
- [ ] 输出 mixed 数据统计表。

### 13.3 实验层

- [ ] 跑 10 模型 int4 公开 benchmark。
- [ ] 跑 10 模型 int4 `domain_qa`。
- [ ] 跑 10 模型 int4 `domain_regqa`。
- [ ] 跑 5 模型 bf16/int8/int4 量化对比。
- [ ] 跑 `qwen2_5_7b_instruct` mixed QLoRA。
- [ ] 跑 `qwen3_8b` mixed QLoRA。
- [ ] 跑 `glm_4_9b_chat_hf` 或 `gemma_2_9b_it` mixed QLoRA。
- [ ] 跑 adapter 评测。
- [ ] 导出论文表格。

### 13.4 写作层

- [ ] 更新 `paper/sections/problem_setting.tex`。
- [ ] 更新 `paper/sections/experimental_setup.tex`。
- [ ] 更新 `paper/sections/methodology.tex`。
- [ ] 更新 `paper/sections/results.tex`。
- [ ] 更新 `paper/sections/discussion.tex`。
- [ ] 更新 `paper/tables/` 下所有结果表。
- [ ] 生成正式图。

## 14. 预期主要结论形式

最终论文应尽量形成以下类型的结论：

1. 在单张 RTX 3090 上，int4 是多数 7B 到 9B 模型更现实的部署精度。
2. 公开 benchmark 表现与铁路领域任务表现并不完全一致。
3. 铁路术语/翻译数据训练得到的 adapter 对法规问答可能存在有限迁移，但不是最佳方案。
4. mixed 训练比单独 `domain_qa` 或单独 `domain_regqa` 更适合作为综合领域适配方案。
5. completion-only loss 更适合答案短、格式严格的铁路问答任务。
6. 对于资源受限场景，应综合考虑准确率、显存、延迟和领域适配成本，而不是只看模型参数规模或公开榜单分数。

## 15. 当前最推荐路线

当前最推荐的工作路线如下：

1. 保留已有 `qwen2_5_7b_instruct` 和 `qwen3_8b` 的 `domain_qa` adapter，作为 Adapter-A。
2. 将 `domain_regqa` 正式接入任务配置。
3. 对 10 个可用模型完成 int4 baseline + domain evaluation。
4. 从 base 重新训练 mixed adapter，作为论文主模型。
5. 优先在 `qwen2_5_7b_instruct` 上做 completion-only mixed QLoRA。
6. 再补 `qwen3_8b` 和一个 9B 中文/强模型。
7. 用已有 Adapter-A 证明“仅术语/翻译适配对法规问答迁移有限”。
8. 用 mixed adapter 证明“加入法规问答数据后领域 QA 更稳”。

一句话总结：

> 论文主实验应选择 mixed QLoRA 作为最终路线，已有 `domain_qa` adapter 不废弃，而是作为跨任务迁移对照；继续补训 `domain_regqa` 可以作为可选消融，但不作为主模型。
