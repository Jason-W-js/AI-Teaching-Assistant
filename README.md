# CircuitMind：面向电路课程的多智能体 AI 助教

CircuitMind 是一个本地优先的电路课程智能学习平台。项目聚焦学生解题过程，而不是只做“输入题目、输出答案”的聊天机器人：系统会根据用户意图选择知识答疑、完整求解、步骤诊断、同类出题、学习规划或会话引导流程，并将课程知识库、题库、错题本和知识图谱关联起来。

本仓库基于 [yishi170336/AI-Teaching-Assistant](https://github.com/yishi170336/AI-Teaching-Assistant) 的 `a5ff57b` 版本继续开发。新增能力和设计变化见 [与上游版本的差异](docs/UPSTREAM_CHANGES.md)。

> 当前阶段以学生端为验收重点。教师端仅保留入口和状态接口；电路仿真、正式身份认证和生产级多租户隔离尚未实现。

## 已实现能力

### 学生学习工作台

- 逐步引导与完整解答两种教学模式；
- 文本、图片、PDF、Word、Excel 和 Markdown 附件输入；
- SSE 流式回答、历史会话恢复和失败回答重新生成；
- 重新生成时保留原题图片等附件；
- KaTeX 数学公式渲染与常见 LaTeX 格式修复；
- 错题本支持选择多条对话形成一道错题、自定义分类和重命名；
- 知识图谱按课程主题分组，节点可查看定义、知识要点、教材来源、相关章节和关联错题。

### 分层语义路由

LangGraph 总编排不会让所有输入机械执行同一条多智能体链：

| 路由 | 典型输入 | 执行方式 |
|---|---|---|
| `qa` | “为什么这个结果是这样”“我不理解叠加定理” | 单个上下文答疑 Agent 直接解释 |
| `answer` | 完整题目、计算请求、学生步骤、验算或错因诊断 | 四智能体过程型解题链 |
| `quiz` | “根据二极管知识出一道题” | 独立出题、校验与修正链 |
| `plan` | “帮我安排一周复习计划”“如何补齐前置知识” | 学情提取、课程资料检索与可执行学习计划链 |
| `chat` | 寒暄、离题内容、乱码、提示注入或信息不完整 | 会话引导 Agent，不检索、不建立题目状态 |

四智能体解题链包括：

1. **题目理解智能体**：解析题干、公式、图片、已知量、待求量和节点—支路拓扑；
2. **领域求解智能体**：选择方法并生成只在后端保存的内部参考解；
3. **验证与错因诊断智能体**：对比参考解、SymPy 结果和学生当前步骤；
4. **教学辅导智能体**：按 L1–L5 控制提示深度，避免过早泄露完整答案。

详细状态图见 [LANGGRAPH_MULTI_AGENT.md](LANGGRAPH_MULTI_AGENT.md)。

### 课程知识工程

- 教材与试卷自动分类；
- PDF/Word/Markdown/TXT/Excel/JSON 导入；
- 章节语义切分和页码、章节、知识点元数据保留；
- 扫描 PDF 的 macOS Vision OCR 回退，以及外部 PDF-Extract-Kit、MinerU 或 Docling 适配边界；可选 Qwen 多模态清洗与图文嵌入；
- 试卷题目候选抽取和结构化题库；
- 向量检索、BM25、元数据过滤和规则重排；默认本地 FAISS，可选 Qdrant 与 Neo4j；
- 教材片段与题库分开索引，常规答疑只检索教材证据，题库只在明确出题流程中使用，避免标准答案意外泄露；
- 题目与教材片段的 `supported_by` 关联；
- 检索质量门控：题意不完整、知识点无证据或相关度不足时不展示引用；
- 知识图谱内容经过 OCR 噪声过滤，基础定义可使用校核卡片替代不可靠公式 OCR。

知识库格式、构建流程和数据产物见 [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md)。

## 技术栈

```text
前端           React 19 + TypeScript + Ant Design + Zustand + KaTeX
后端           FastAPI + Pydantic + SSE
工作流         LangGraph
本地模型       LM Studio / Ollama
远程模型       DeepSeek / 通义千问 / OpenAI 兼容 API
知识检索       SentenceTransformers + FAISS/Qdrant + BM25 + Jieba + 可选 Neo4j
数学验证       SymPy
会话存储       Redis；不可用时回退本地 JSON
文档解析       PyMuPDF + python-docx + openpyxl + PDF-Extract-Kit/可插拔 OCR
```

## 快速开始

### 1. 准备环境

推荐 Python 3.12、Node.js 20+。macOS 本地模型方案还需要安装并启动 LM Studio，加载支持图片输入的 `qwen/qwen3.5-9b` 或兼容模型。

```bash
git clone https://github.com/Jason-W-js/AI-Teaching-Assistant.git
cd AI-Teaching-Assistant

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_embedding_model.py

cd frontend
npm ci
npm run build
cd ..

cp .env.example .env
```

`.env` 只用于本机配置，已经被 Git 忽略。不要把 API Key 写入 `.env.example` 或提交到仓库。

### 2. 启动服务

macOS + LM Studio：

```bash
./scripts/start.sh
```

其他环境或已经手动启动模型服务时：

```bash
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

访问：

- 学生端：<http://127.0.0.1:8000/student>
- 健康检查：<http://127.0.0.1:8000/api/health>

Windows 仍可使用 `scripts/start.ps1`。

### 3. 导入自己的课程资料

可在学生端点击“添加教材 / 新建知识库”，也可以将有权使用的资料放入 `RAG_Resources/` 后执行：

```bash
.venv/bin/python scripts/build_knowledge_base.py --full
```

非默认知识库：

```bash
.venv/bin/python scripts/build_knowledge_base.py --knowledge-base circuits-101 --full
```

教材、试卷、OCR 缓存、上传附件、会话记录和错题本属于本地运行数据，不应直接提交到公共仓库。

## 模型配置

默认模型配置在 `.env.example` 中。常用变量：

```dotenv
DEFAULT_MODEL_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=qwen/qwen3.5-9b

# 可选的异构智能体模型；留空时复用页面所选模型
UNDERSTANDING_MODEL=
SOLVER_MODEL=
DIAGNOSIS_MODEL=
TUTOR_MODEL=

MEMORY_TURNS=24
CHAT_FIRST_TOKEN_TIMEOUT_SECONDS=600
```

`CHAT_FIRST_TOKEN_TIMEOUT_SECONDS` 只限制“多久仍未出现第一个可见 token”；开始输出后不设置总回答时限。

前端也支持 Ollama、DeepSeek、通义千问和自定义 OpenAI Chat Completions 兼容接口。使用云端模型时，当前问题、最近对话和必要检索上下文会发送到对应服务商。

## 测试与构建

```bash
PYTHONPATH=. .venv/bin/pytest -q

cd frontend
npm run build
```

当前回归集覆盖路由边界、上下文记忆、题目状态、检索质量门控、知识库构建、知识图谱、错题本、模型客户端和数学验证。

## 项目结构

```text
backend/app/
  agents/workflow.py          # LangGraph 总路由与各智能体工作流
  rag/                        # 解析、切片、题库关联和混合检索
  services/                   # 模型、记忆、问题状态、知识图谱、错题本
frontend/src/
  pages/StudentPage.tsx       # 学生工作台
  pages/LearningViews.tsx     # 知识图谱与错题本
  store/chatStore.ts          # 会话、附件和流式状态
scripts/                      # 启动、嵌入模型下载、知识库构建和冒烟测试
tests/                        # 后端回归测试
data/                         # 本地索引与运行数据
```

## 文档

- [多智能体工作流与状态模型](LANGGRAPH_MULTI_AGENT.md)
- [与上游版本的差异](docs/UPSTREAM_CHANGES.md)
- [知识库构建与数据格式](docs/KNOWLEDGE_BASE.md)
- [开发、测试与发布](docs/DEVELOPMENT.md)
- [安全和数据边界](SECURITY.md)
- [当前能力与路线图](PROJECT_REVIEW.md)

## 数据与版权说明

请只导入你有权使用的教材、试卷和课程资料。项目代码可以公开，并不意味着用户导入的教材、OCR 文本、题目或向量索引可以公开再分发。本仓库的公开提交应排除本地资料和运行数据。

上游仓库当前未包含明确的开源许可证文件；在添加许可证或对代码进行再许可前，应先确认上游作者和本 fork 的授权边界。
