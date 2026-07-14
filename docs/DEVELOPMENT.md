# 开发、测试与发布

本文档面向参与 CircuitMind 开发的成员。学生使用方法见根目录 [README](../README.md)。

## 开发环境

推荐：

- Python 3.12；
- Node.js 20 或更新版本；
- LM Studio，或其他 OpenAI Chat Completions 兼容服务；
- 可选 Redis；
- macOS 扫描 PDF 可使用 Vision OCR，其他平台推荐配置外部解析器。

初始化：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_embedding_model.py

cd frontend
npm ci
cd ..

cp .env.example .env
```

开发时可分别启动：

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm run dev
```

## 后端模块边界

```text
backend/app/main.py
    API、SSE、模型选择、生命周期与依赖组装

backend/app/agents/workflow.py
    总路由、答疑、四智能体解题、出题与会话引导

backend/app/rag/
    文档解析、清洗、切片、题目抽取、关联和混合检索

backend/app/services/
    附件、会话记忆、问题状态、模型客户端、错题本、知识图谱
```

新增能力时优先保持这些边界：API 层不写教学决策，智能体不直接读取文件系统，RAG 不负责数值计算，知识图谱不修改原始索引。

## 路由规则

所有新输入先经过 `_route_intent`。修改路由时应同时添加边界测试，至少覆盖：

- 一个明确概念问题；
- 一个完整求解请求；
- 一个学生步骤检查；
- 一个出题请求；
- 一个含“练习题”但实际要求求解的冲突样例；
- 一个省略式上下文追问；
- 一个离题或低信息输入；
- 一个提示注入样例。

不要把模糊输入默认送进四智能体解题链。没有题目、拓扑或待求量时应补问，而不是生成假设并继续计算。

## 模型客户端

页面所选模型作为默认客户端。下列环境变量可以为逻辑智能体指定异构模型：

```dotenv
UNDERSTANDING_MODEL=
SOLVER_MODEL=
DIAGNOSIS_MODEL=
TUTOR_MODEL=
```

异构模型通过同一 LM Studio OpenAI 兼容端点访问。题目理解模型需要图片能力；其余节点不应收到无关图片。模型私有 reasoning 不返回前端。

## API 概览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 模型、记忆和知识库状态 |
| POST | `/api/chat` | POST SSE 对话 |
| POST | `/api/attachments` | 上传本轮题目附件 |
| GET/DELETE | `/api/sessions/{id}` | 恢复或删除会话 |
| POST | `/api/kb/ingest` | 批量导入知识库 |
| GET | `/api/kb/status` | 构建状态 |
| GET | `/api/knowledge-graph` | 知识图谱视图 |
| GET/POST/PATCH/DELETE | `/api/wrong-questions...` | 错题和分类管理 |

请求结构以 `backend/app/schemas.py` 为准。

## 测试

后端：

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

前端类型检查与生产构建：

```bash
cd frontend
npm run build
```

`vite.config.ts` 是唯一的 Vite 配置源。编译产物和 `*.tsbuildinfo` 不应提交；生产构建会将学生端、教师端、学习空间与第三方依赖拆分为可缓存资源。

提交前还应执行：

```bash
git diff --check
```

涉及模型的冒烟测试需要本地模型服务在线；常规单元测试不应依赖外网或真实 API Key。

## 数据和提交边界

以下内容禁止进入公共提交：

- `.env` 和任何 API Key；
- 新增教材、试卷和 OCR 原文；
- `data/uploads/`；
- `data/session_memory/`、`data/problem_sessions/`、`data/wrong_notebook/`；
- 向量模型权重和解析缓存；
- 包含个人学习记录的截图或日志。

若本地默认知识库发生变化，提交代码时应显式选择文件，不要使用无检查的 `git add -A`。

## Git 分支和提交

建议：

```text
功能分支：feature/<short-name>
修复分支：fix/<short-name>
提交信息：一句话说明用户可见结果
```

PR 描述应说明改了什么、为什么改、对用户的影响、验证命令，以及仍未覆盖的风险。
