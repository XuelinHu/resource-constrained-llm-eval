# 铁道教育语料人工校正系统

系统将专业术语、规章问答和 DeepSeek OCR 教材页统一导入 PostgreSQL，并提供 Vue 审核工作台。

## 数据范围

- `terminology_pair`：中英文专业术语及具体专业类别
- 规章问答：`data/domain_regqa_refined/{train,valid,test}.jsonl`
- `textbook_source`：两本教材逐页 OCR 原文，可人工整理章节并抽取问答

## 初始化

启动 PostgreSQL + pgvector：

```bash
cd annotation_system
RAILWAY_DB_PASSWORD=change-me docker compose --env-file backend/.env.example -f docker-compose.pgvector.yml up -d
```

pgvector 没有独立端口，它是 PostgreSQL 扩展；服务端口仍然使用 PostgreSQL 默认端口 `5432`。生产或正式实验环境应把 `change-me` 替换为本地 `backend/.env` 中的真实密码；不要提交真实密码。

```bash
cd annotation_system/backend
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.init_database
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.import_data
```

导入以 `external_id` 去重，重复执行不会重复写入。

## 启动

```bash
./annotation_system/start_backend.sh
./annotation_system/start_frontend.sh
```

- 本机前端：`http://localhost:4005`
- FRP 外网前端：`http://47.120.48.245:14005`
- API 文档：`http://localhost:8000/docs`

前端支持三种访问模式，默认启动脚本已启用 FRP：

```bash
FRONTEND_MODE=local ./annotation_system/start_frontend.sh      # API: http://127.0.0.1:8000
FRONTEND_MODE=lan ./annotation_system/start_frontend.sh        # API: http://192.168.1.9:8000
FRONTEND_MODE=frp ./annotation_system/start_frontend.sh        # API: http://47.120.48.245:18000
```

数据库密码只保存在被 Git 忽略的 `backend/.env` 中；配置模板见 `backend/.env.example`。

审核台支持筛选、全文检索、来源证据对照、问题与答案校正、质量标记、审核状态、修改历史和已通过数据导出。导出文件写入 `data/exports/`。

## RAG 问答

前端右上角可切换到“RAG 问答”。当前流程为：

1. 从 PostgreSQL 语料库构建字符级 BM25 索引。
2. 排除 `generated_eval_review` 和所有 `split=test` 数据，避免测试泄漏。
3. 检索规章、教材 OCR、术语及训练语料中的相关证据。
4. 使用本地 Ollama `qwen3:14b` 根据证据生成回答并标注来源编号。
5. 本地模型不可用时自动返回最相关原始证据。

论文正式路线使用 PostgreSQL + pgvector 作为专业知识库底座：PostgreSQL 保存结构化语料、审核状态、来源证据与会话记录，pgvector 保存知识块向量。当前 BM25 检索保留为关键词基线，后续实验应比较 BM25、pgvector 语义检索和混合检索。

`RAILWAY_EMBEDDING_DIMENSION` 必须与正式实验使用的 embedding 模型输出维度一致；默认值为 1024。若后续改用 768、1536 等维度模型，应在初始化数据库前调整该环境变量。

默认 embedding 模型为 `BAAI/bge-m3`，用于中英双语铁路教育知识检索。构建 pgvector 向量索引：

```bash
cd annotation_system/backend
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.build_embeddings --rebuild
```

调试时可先限制样本数：

```bash
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.build_embeddings --limit 100 --rebuild
```

RAG API：

- `GET /api/rag/stats`：索引状态
- `POST /api/rag/rebuild`：重建索引
- `POST /api/rag/ask`：检索或生成回答，`retrieval_mode` 支持 `bm25`、`vector` 和 `hybrid`，`approved_only` 可限制仅检索专家核验通过语料

命令行验证 pgvector 语义检索：

```bash
cd annotation_system/backend
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.search_embeddings "What is contact wire maintenance?" --mode vector --top-k 3
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.search_embeddings "接触网检修要求是什么" --mode hybrid --approved-only --top-k 3
```

运行初始检索评测：

```bash
cd annotation_system/backend
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.evaluate_retrieval --top-k 5
```

运行初始问答生成评测：

```bash
cd annotation_system/backend
/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python -m app.evaluate_qa --limit 10 --top-k 3 --include-no-retrieval
```

## PM2

```bash
pm2 start annotation_system/ecosystem.config.cjs
pm2 save
```

PM2 进程：

- `railway-corpus-api`
- `railway-corpus-frontend`
