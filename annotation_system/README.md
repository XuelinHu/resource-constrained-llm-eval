# 铁道教育语料人工校正系统

系统将专业术语、规章问答和 DeepSeek OCR 教材页统一导入 PostgreSQL，并提供 Vue 审核工作台。

## 数据范围

- `terminology_pair`：中英文专业术语及具体专业类别
- 规章问答：`data/domain_regqa_refined/{train,valid,test}.jsonl`
- `textbook_source`：两本教材逐页 OCR 原文，可人工整理章节并抽取问答

## 初始化

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

RAG API：

- `GET /api/rag/stats`：索引状态
- `POST /api/rag/rebuild`：重建索引
- `POST /api/rag/ask`：检索或生成回答

## PM2

```bash
pm2 start annotation_system/ecosystem.config.cjs
pm2 save
```

PM2 进程：

- `railway-corpus-api`
- `railway-corpus-frontend`
